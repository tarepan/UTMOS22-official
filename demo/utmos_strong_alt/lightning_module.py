"""Cloned from `https://huggingface.co/spaces/sarulab-speech/UTMOS-demo`."""

import torch
import torch.nn as nn
import lightning as L # pyright: ignore [reportMissingTypeStubs]

from model import SSL_model, DomainEmbedding, LDConditioner, Projection


class BaselineLightningModule(L.LightningModule):
    """The UTMOS model."""

    def __init__(self, cfg):
        super().__init__()

        self.save_hyperparameters()
        feat_ssl, feat_domain_emb, feat_judge_emb = 768, 128, 128
        feat_cat = feat_ssl + feat_domain_emb + feat_judge_emb

        # wave-to-unit SSL / dataDomain-to-emb Embedding
        self.feature_extractors = nn.ModuleList([SSL_model(), DomainEmbedding(3, feat_domain_emb)])

        # Transform
        blstm = LDConditioner(feat_i=feat_cat, judge_dim=feat_judge_emb)
        prjct = Projection(input_dim=blstm.get_output_dim())
        self.output_layers = nn.ModuleList([blstm, prjct])

        self._prepared = False

    def prepare(self):
        """Prepare smart model."""

        # Move SSL model
        self.wav2vec2 = self.feature_extractors[0].ssl_model

        # Preprocess DataDomain embedding :: (B=1,) -> (B=1, Feat)
        domain_ids = torch.zeros(1, dtype=torch.int).to(self.device)
        domain_emb = self.feature_extractors[1].embedding(domain_ids)
        self.domain_emb = nn.Parameter(data=domain_emb, requires_grad=False)

        # Preprocess JudgeID embedding :: (B=1,) -> (B=1, Feat)
        judge_ids = torch.ones(1, dtype=torch.int).to(self.device) * 288
        judge_emb = self.output_layers[0].judge_embedding(judge_ids)
        self.judge_emb = nn.Parameter(data=judge_emb, requires_grad=False)

        # Move BLSTM
        self.blstm      = self.output_layers[0].decoder_rnn

        # Reconstruct Dropout-less Projection
        self.projection = self.output_layers[1].net
        del self.projection[2]

        # Clear remnants
        del self.feature_extractors
        del self.output_layers

        self._prepared = True

    def forward(self, wave):
        """ :: (B, T) -> (B, Frame, Feat=1) """

        if not self._prepared:
            raise RuntimeError("BaselineLightningModule should be prepared before 1st forward.")

        # Feature extraction :: (B, T) -> (B, Frame, Feat)
        ssl_feats = self.wav2vec2(wave)
        bsz, frm, _ = ssl_feats.size()

        # Embedding Batch/Time expansion :: (B=1, Feat) -> (B=bsz, Frame=frm, Feat)
        domain_emb = self.domain_emb.unsqueeze(1).expand(bsz, frm, -1)
        judge_emb  =  self.judge_emb.unsqueeze(1).expand(bsz, frm, -1)

        # Feature concatenation - SSL/DataDomainEmb/JudgeEmb :: (B, Frame, Feat=f1) + (B, Frame, Feat=f2) + (B, Frame, Feat=f3) -> (B, Frame, Feat=f1+f2+f3)
        x = torch.cat([ssl_feats, domain_emb, judge_emb], dim=2)

        # Frame-scale score estimation :: (B, Frame, Feat) -> (B, Frame, Feat=1) - BLSTM/Projection
        x = self.blstm(x)[0]
        x = self.projection(x)

        return x

    @property
    def device(self):
        """Current device."""
        return next(self.parameters()).device


