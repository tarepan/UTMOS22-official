"""Cloned from `https://huggingface.co/spaces/sarulab-speech/UTMOS-demo`."""

import os
from typing import Any

import torch
from torch import nn, Tensor
import fairseq
import hydra


def load_ssl_model():
    """Load SSL model with wrapper."""

    SSL_OUT_DIM = 768
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(["wav2vec_small.pt"])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
    wrapped_model = SSL_model(ssl_model, SSL_OUT_DIM)

    return wrapped_model


class SSL_model(nn.Module):
    """SSL model wrapper."""

    def __init__(self, ssl_model, ssl_out_dim: int) -> None:
        super().__init__()

        self.ssl_model, self.ssl_out_dim = ssl_model, ssl_out_dim

    def forward(self, batch):
        """
        Args:
            batch
                wav         :: Tensor
                domains     :: Tensor
                judge_id    :: Tensor
        Returns:
            {
                ssl-feature :: Tensor
            }
        """

        # Data preparation
        wav = batch['wav'] 
        wav = wav.squeeze(1) # [batches, audio_len]

        # Forward
        x = self.ssl_model(wav, mask=False, features_only=True)["x"]

        return { "ssl-feature": x }

    def get_output_dim(self):
        return self.ssl_out_dim


class DomainEmbedding(nn.Module):
    def __init__(self, n_domains: int, domain_dim: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(n_domains, domain_dim)
        self.output_dim = domain_dim

    def forward(self, batch):
        """
        Args:
            batch
                wav         :: Tensor
                domains     :: Tensor
                judge_id    :: Tensor
        Returns:
            {
                domain-feature :: Tensor
            }
        """
        return {"domain-feature": self.embedding(batch['domains'])}

    def get_output_dim(self):
        return self.output_dim


class LDConditioner(nn.Module):
    '''
    Conditions ssl output by listener embedding
    '''
    def __init__(self, input_dim: int, judge_dim: int, num_judges: int):
        super().__init__()

        self.judge_embedding = nn.Embedding(num_judges, judge_dim)
        self.decoder_rnn = nn.LSTM(
            input_size = input_dim + judge_dim,
            hidden_size = 512,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )
        self.out_dim = self.decoder_rnn.hidden_size * 2

    def get_output_dim(self):
        return self.out_dim

    def forward(self, x, batch):
        """
        Args:
            x
                ssl-feature    :: Tensor
                domain-feature :: Tensor
            batch
                wav            :: Tensor
                domains        :: Tensor
                judge_id       :: Tensor
        Returns:
        """

        judge_ids = batch['judge_id']

        # Feature concatenation - SSL/DataDomainEmb/
        concatenated_feature = x['ssl-feature']
        concatenated_feature = torch.cat(
            (
                concatenated_feature,
                x['domain-feature'].unsqueeze(1).expand(-1, concatenated_feature.size(1), -1),
            ),
            dim=2,
        )
        concatenated_feature = torch.cat(
            (
                concatenated_feature,
                self.judge_embedding(judge_ids).unsqueeze(1).expand(-1, concatenated_feature.size(1), -1),
            ),
            dim=2,
        )

        # Forward
        decoder_output, _ = self.decoder_rnn(concatenated_feature)

        return decoder_output


class Projection(nn.Module):
    """Projection, SegFC-ReLU-Do-SegFC."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, _: Any) -> Tensor:
        return self.net(x)
