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


class DomainEmbedding(nn.Module):
    def __init__(self, n_domains: int, domain_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_domains, domain_dim)


class LDConditioner(nn.Module):
    '''
    Conditions ssl output by listener embedding
    '''
    def __init__(self, feat_i: int, judge_dim: int):
        super().__init__()

        self.judge_embedding = nn.Embedding(3000, judge_dim)
        # 1-layer BLSTM_512
        self.decoder_rnn = nn.LSTM(input_size=feat_i, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = self.decoder_rnn.hidden_size * 2

    def get_output_dim(self):
        return self.out_dim


class Projection(nn.Module):
    """Projection, SegFC-ReLU-Do-SegFC."""
    def __init__(self, input_dim: int):
        super().__init__()
        feat_h = 2048
        self.net = nn.Sequential(nn.Linear(input_dim,  feat_h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(feat_h, 1))
