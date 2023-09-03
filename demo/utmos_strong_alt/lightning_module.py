"""Cloned from `https://huggingface.co/spaces/sarulab-speech/UTMOS-demo`."""

import torch
import torch.nn as nn
import lightning as L # pyright: ignore [reportMissingTypeStubs]

from model import SSL_model, DomainEmbedding, LDConditioner, Projection


class BaselineLightningModule(L.LightningModule):
    """The UTMOS model."""

    def __init__(self, cfg):
        super().__init__()
        # self.cfg = cfg
        self.save_hyperparameters()

        # wave-to-unit SSL / dataDomain-to-emb Embedding
        self.feature_extractors = nn.ModuleList([SSL_model(), DomainEmbedding(3, 128)])
        output_dim = sum([ feature_extractor.get_output_dim() for feature_extractor in self.feature_extractors])

        # Transform
        blstm = LDConditioner(input_dim=output_dim, judge_dim=128, num_judges=3000)
        prjct = Projection(input_dim=blstm.get_output_dim(), hidden_dim=2048)
        self.output_layers = nn.ModuleList([blstm, prjct])

    def forward(self, inputs):
        """
        Args:
            inputs
                wav      :: Tensor
                domains  :: Tensor
                judge_id :: Tensor
        """

        # Feature extraction :: -> {"ssl-feature"::Tensor, "domain-feature"::Tensor}
        outputs = {}
        for feature_extractor in self.feature_extractors:
            outputs.update(feature_extractor(inputs))
        x = outputs

        # Transform :: -> (..., 1)
        for output_layer in self.output_layers:
            x = output_layer(x, inputs)

        return x
