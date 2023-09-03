"""Cloned from `https://huggingface.co/spaces/sarulab-speech/UTMOS-demo`."""

from typing import Any

import torch
from torch import nn, Tensor

from fairseq_alt import Wav2Vec2Model


class SSL_model(nn.Module):
    """Wav2Vec2 model wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.ssl_model = Wav2Vec2Model()

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
        unit_series = self.ssl_model(wav)

        return { "ssl-feature": unit_series }

    def get_output_dim(self):
        return 768


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
