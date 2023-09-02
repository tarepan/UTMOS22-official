"""Cloned from `https://huggingface.co/spaces/sarulab-speech/UTMOS-demo`."""

import torch
import torchaudio
import unittest

import lightning_module


class Score:
    """Predicting score for each audio clip."""

    def __init__(
        self,
        ckpt_path: str = "epoch=3-step=7459.ckpt",
        input_sample_rate: int = 16000,
        device: str = "cpu"):
        """
        Args:
            ckpt_path: path to pretrained checkpoint of UTMOS strong learner.
            input_sample_rate: sampling rate of input audio tensor. The input audio tensor
                is automatically downsampled to 16kHz.
        """
        print(f"Using device: {device}")
        self.device = device
        self.in_sr = input_sample_rate
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=input_sample_rate,
            new_freq=16000,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        ).to(device)
        # The model (same as `app.py`)
        self.model = lightning_module.BaselineLightningModule.load_from_checkpoint(ckpt_path).eval().to(device)
    
    def score(self, wavs: torch.tensor) -> torch.tensor:
        """
        Args:
            wavs: audio waveform to be evaluated. When len(wavs) == 1 or 2,
                the model processes the input as a single audio clip. The model
                performs batch processing when len(wavs) == 3. 
        """

        # Reshape
        if len(wavs.shape) == 1:
            out_wavs = wavs.unsqueeze(0).unsqueeze(0)
        elif len(wavs.shape) == 2:
            out_wavs = wavs.unsqueeze(0)
        elif len(wavs.shape) == 3:
            out_wavs = wavs
        else:
            raise ValueError('Dimension of input tensor needs to be <= 3.')

        # Resampling
        if self.in_sr != 16000:
            out_wavs = self.resampler(out_wavs)

        # Data preparation - domain0 / judgeID288 (same as `app.py`)
        bs = out_wavs.shape[0]
        batch = {
            'wav': out_wavs,
            'domains': torch.zeros(bs, dtype=torch.int).to(self.device),
            'judge_id': torch.ones(bs, dtype=torch.int).to(self.device)*288
        }

        # Forward - (same as `app.py`)
        with torch.no_grad():
            output = self.model(batch)
        calculated_score = output.mean(dim=1).squeeze(1).cpu().detach().numpy()*2 + 3

        return calculated_score
