"""Using a trained model: load it and clean audio with it.

Long files are cleaned in overlapping 4 second windows. The time attention
compares every frame with every other frame, so its memory grows with the
length squared, and a 16 second file does not fit at once. The windows overlap
by 0.5 s and are cross-faded, so there is no click where two windows meet.
Files shorter than 4 s (almost all of the test set) go through in one piece.
"""

from pathlib import Path

import numpy as np
import soundfile
import torch
from tqdm import tqdm

from noisereducer.config import ROOT_DIR, Config, ModelConfig
from noisereducer.data import read_audio
from noisereducer.model import NoiseReducer

AUDIO_TYPES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4", ".aac", ".mov"}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def best_checkpoint(run_name):
    """The path of runs/<run_name>/best.pt, with a clear error if it is missing."""
    path = ROOT_DIR / "runs" / run_name / "best.pt"
    if not path.exists():
        runs = []
        for folder in sorted((ROOT_DIR / "runs").iterdir()):
            if (folder / "best.pt").exists():
                runs.append(folder.name)
        raise FileNotFoundError(f"{path} does not exist. Runs with a best.pt: {runs}")
    return path


def load_model(checkpoint_path, device="cpu"):
    """Build the network and load the trained weights from a .pt file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Take the network size from the checkpoint, in case it was trained with
    # other values than the ones in config.py now.
    # Older checkpoints also saved "axes" and "topology", which no longer
    # exist, so we only take the fields ModelConfig still has.
    saved = checkpoint["config"]["model"]
    settings = {name: saved[name] for name in vars(ModelConfig())}

    model = NoiseReducer(Config(model=ModelConfig(**settings)))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print(f"loaded {checkpoint_path} (epoch {checkpoint['epoch']})")
    return model


def enhance(model, waveform, sample_rate=16_000, chunk_seconds=4.0, overlap_seconds=0.5):
    """Clean one waveform of any length. Returns a CPU tensor of the same length."""
    device = next(model.parameters()).device
    chunk = int(chunk_seconds * sample_rate)
    overlap = min(int(overlap_seconds * sample_rate), chunk // 2)
    total = waveform.shape[-1]

    with torch.no_grad():
        if total <= chunk:
            return model(waveform.to(device)).waveform.cpu()

        # Window start positions. The last window is moved back so it ends
        # exactly at the end of the file, so every window has the same length.
        starts = [0]
        while starts[-1] + chunk < total:
            starts.append(min(starts[-1] + chunk - overlap, total - chunk))

        output = torch.zeros(total)
        weight = torch.zeros(total)
        ramp = torch.linspace(0.0, 1.0, overlap)
        for i, start in enumerate(starts):
            cleaned = model(waveform[start:start + chunk].to(device)).waveform.cpu()

            # Fade in at the start and fade out at the end of every window
            # (except the very first start and the very last end).
            taper = torch.ones(chunk)
            if i > 0:
                taper[:overlap] = ramp
            if i < len(starts) - 1:
                taper[-overlap:] = ramp.flip(0)

            output[start:start + chunk] += cleaned * taper
            weight[start:start + chunk] += taper

    return output / weight.clamp_min(1e-6)


def list_audio_files(folder):
    """All the audio files in a folder, sorted by name."""
    files = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in AUDIO_TYPES:
            files.append(path)
    return files


def enhance_folder(model, noisy_dir, clean_dir, sample_rate=16_000):
    """Clean the audio files in noisy_dir one by one, and save every cleaned
    waveform in clean_dir as <name>_clean.wav."""
    files = list_audio_files(noisy_dir)
    for path in tqdm(files, desc="cleaning"):
        noisy = read_audio(path, sample_rate)
        cleaned = enhance(model, noisy, sample_rate)
        # A wav file can only hold values between -1 and 1.
        waveform = np.clip(cleaned.numpy(), -1.0, 1.0)
        soundfile.write(Path(clean_dir) / f"{path.stem}_clean.wav", waveform, sample_rate)
    return len(files)
