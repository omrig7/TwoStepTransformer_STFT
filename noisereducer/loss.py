"""The loss: one term on the spectrogram and one on the waveform.

    loss = 0.4 * MSE(enhanced waveform, clean waveform)
         + 0.6 * MSE(mask * |Y|^0.3, |S|^0.3)

The magnitude term compares *compressed* magnitudes, the same thing the mask
multiplies, so loud and quiet bins count more equally. The waveform term is
there because we reuse the noisy phase: after the inverse STFT the audio does
not always match the magnitude we asked for, and this term punishes that.

Padded frames and samples are left out of both averages. A padded frame is
silence in and silence out, so it would push the mask towards 1 for no reason.
"""

from typing import NamedTuple

import torch
from torch import nn

from noisereducer.dsp.features import frame_mask


class LossTerms(NamedTuple):
    total: torch.Tensor
    time: torch.Tensor  # waveform MSE (before the weight)
    magnitude: torch.Tensor  # compressed magnitude MSE (before the weight)


def masked_mean(values, mask):
    """Mean over the real entries only (mask = True where the entry is real)."""
    if mask is None:
        return values.mean()
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1)


class EnhancementLoss(nn.Module):
    """Use the model's own STFT and features, so the target is computed exactly
    like the input was:

        loss_fn = EnhancementLoss(model.stft, model.features, config.loss)
        terms = loss_fn(model(noisy, lengths), clean, lengths)
        terms.total.backward()
    """

    def __init__(self, stft, features, config):
        super().__init__()
        self.stft = stft
        self.features = features
        self.time_weight = config.time_weight
        self.magnitude_weight = config.magnitude_weight
        self.error = {"mse": torch.square, "l1": torch.abs}[config.error]

    def forward(self, enhanced, clean, lengths=None):
        # The target: compressed magnitude of the clean audio.
        clean_magnitude, _ = self.stft.to_magnitude_phase(self.stft.analyze(clean))
        target = self.features.compress(clean_magnitude)

        samples = frames = None
        if lengths is not None:
            positions = torch.arange(clean.shape[-1], device=lengths.device)
            samples = positions.unsqueeze(0) < lengths.unsqueeze(-1)
            frames = frame_mask(self.stft.frame_lengths(lengths), target.shape[-2])
            frames = frames.unsqueeze(-1)

        # mask * compressed noisy magnitude, i.e. still in the compressed domain
        estimate = enhanced.mask * enhanced.compressed

        magnitude = masked_mean(self.error(estimate - target), frames)
        time = masked_mean(self.error(enhanced.waveform - clean), samples)
        total = self.time_weight * time + self.magnitude_weight * magnitude
        return LossTerms(total, time, magnitude)
