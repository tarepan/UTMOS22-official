"""Cloned from `https://huggingface.co/spaces/sarulab-speech/UTMOS-demo`."""

import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import hydra

from model import load_ssl_model, PhonemeEncoder, DomainEmbedding, LDConditioner, Projection


class BaselineLightningModule(pl.LightningModule):
    """The model."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
        self.save_hyperparameters()
    
    def construct_model(self):
        """Init."""

        # wave-to-unit SSL / dataDomain-to-emb Embedding
        self.feature_extractors = nn.ModuleList([
            load_ssl_model(),
            DomainEmbedding(3, 128),
        ])
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
