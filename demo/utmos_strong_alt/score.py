
import os

import torch
from torch import nn, Tensor

from lightning_module import BaselineLightningModule


UTMOS_CKPT_URL = "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/epoch%3D3-step%3D7459.ckpt"

class UTMOSScore(nn.Module):
    """Predicting score for each audio clip.

    Work as wrapper.
    """

    def __init__(self, ckpt_path: str = "epoch=3-step=7459.ckpt"):
        super().__init__()

        filepath = ckpt_path
        # filepath = os.path.join(os.path.dirname(__file__), ckpt_path)
        if not os.path.exists(filepath):
            download_file(UTMOS_CKPT_URL, filepath)
        self.model = BaselineLightningModule.load_from_checkpoint(filepath, map_location="cpu").eval()

    @property
    def device(self):
        """Current device."""
        return next(self.parameters()).device

    def score(self, wavs: Tensor) -> Tensor:
        """
        Args:
            wavs: audio waveform to be evaluated. When len(wavs) == 1 or 2,
                the model processes the input as a single audio clip. The model
                performs batch processing when len(wavs) == 3.
        """
        # :: -> (B, )
        if len(wavs.shape) == 1:
            out_wavs = wavs.unsqueeze(0).unsqueeze(0)
        elif len(wavs.shape) == 2:
            out_wavs = wavs.unsqueeze(0)
        elif len(wavs.shape) == 3:
            out_wavs = wavs
        else:
            raise ValueError("Dimension of input tensor needs to be <= 3.")

        bs = out_wavs.shape[0]
        batch = {
            "wav": out_wavs,
            "domains": torch.zeros(bs, dtype=torch.int).to(self.device),
            "judge_id": torch.ones(bs, dtype=torch.int).to(self.device) * 288,
        }
        with torch.no_grad():
            output = self.model(batch)

        return output.mean(dim=1).squeeze(1).cpu().detach() * 2 + 3


def download_file(url, filename):
    """
    Downloads a file from the given URL

    Args:
        url (str): The URL of the file to download.
        filename (str): The name to save the file as.
    """

    import requests
    from tqdm import tqdm

    print(f"Downloading file {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            progress_bar.update(len(chunk))
            f.write(chunk)

    progress_bar.close()


if __name__ == '__main__':
    """Inference."""

    import torch
    import torchaudio
    import librosa

    utmos_model = UTMOSScore()

    wave, sr_org = librosa.load("sample.wav", sr=None)
    wave = torch.from_numpy(wave).unsqueeze(0)
    wave_pred_16k = torchaudio.functional.resample(wave, orig_freq=sr_org, new_freq=16000)

    # :: (B, T) -> (B, 1, T) -> [scoring] -> ?
    utmos_score = utmos_model.score(wave_pred_16k.unsqueeze(1)).mean()
    print(utmos_score)