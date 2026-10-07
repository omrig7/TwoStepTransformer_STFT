"""Preparing the magnitude for the network: compression and standardisation.

1. Compression, ``|Y| ** 0.3``. Speech magnitudes go from tiny to huge, and a
   loss on the raw values would only care about the loudest bins. After the
   power 0.3 quiet and loud bins are much closer, so the quiet details (which
   matter for intelligibility) count too.
2. Standardisation. Subtract the mean and divide by the std of each utterance,
   so the model does not depend on how loud the recording is.

The mask multiplies the *compressed* magnitude (not the standardised one), so
a mask value M is a gain of ``M ** (1 / 0.3) = M ** 3.33`` on the real
magnitude. The standardisation is only for the network input; a mask has no
units, so there is nothing to undo on the way out.

Why ``magnitude_floor``: the derivative of ``x ** 0.3`` is ``0.3 * x ** -0.7``,
which goes to infinity at x = 0. The loss compresses the clean target and our
output, so without a floor the silent bins (there are many) would get the
biggest gradients. With a floor of 1e-5 the derivative is at most ~949.
"""

from typing import NamedTuple

import torch
from torch import nn


class PreparedMagnitude(NamedTuple):
    features: torch.Tensor  # compressed and standardised: the network input
    compressed: torch.Tensor  # only compressed: what the mask multiplies


def frame_mask(lengths, num_frames):
    """``(batch, num_frames)`` bool mask, True for real frames and False for padding."""
    positions = torch.arange(num_frames, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(-1)


class MagnitudeFeatures(nn.Module):
    """Compression and standardisation. It has no weights; it is an
    ``nn.Module`` only so that it is part of the model."""

    def __init__(self, exponent=0.3, magnitude_floor=1e-5, std_floor=1e-5):
        super().__init__()
        self.exponent = exponent
        self.magnitude_floor = magnitude_floor
        self.std_floor = std_floor  # so a silent file does not divide by zero

    def compress(self, magnitude):
        """Magnitude -> compressed magnitude. Always float32, also under autocast."""
        return magnitude.to(torch.float32).clamp_min(self.magnitude_floor).pow(self.exponent)

    def decompress(self, compressed):
        """Compressed magnitude -> magnitude."""
        return compressed.to(torch.float32).clamp_min(0.0).pow(1.0 / self.exponent)

    def standardise(self, features, lengths=None):
        """Subtract the mean and divide by the std, for each utterance separately.

        ``features`` is ``(batch, frames, bins)`` and ``lengths`` the number of
        real frames of each utterance (None = no padding).

        The mean and std are computed on the real frames only. If the padding
        was included, a short file in a batch with long files would get a
        different mean each epoch, depending on who it was batched with.
        Padded frames come out as exact zeros.
        """
        features = features.to(torch.float32)
        num_bins = features.shape[-1]

        if lengths is None:
            mean = features.mean(dim=(-2, -1), keepdim=True)
            centred = features - mean
            variance = centred.pow(2).mean(dim=(-2, -1), keepdim=True)
        else:
            mask = frame_mask(lengths, features.shape[-2]).unsqueeze(-1).to(torch.float32)
            valid_elements = mask.sum(dim=(-2, -1), keepdim=True) * num_bins

            mean = (features * mask).sum(dim=(-2, -1), keepdim=True) / valid_elements
            centred = (features - mean) * mask
            variance = centred.pow(2).sum(dim=(-2, -1), keepdim=True) / valid_elements

        std = variance.clamp_min(self.std_floor ** 2).sqrt()
        return centred / std

    def forward(self, magnitude, lengths=None):
        """Compress, then standardise."""
        compressed = self.compress(magnitude)
        return PreparedMagnitude(self.standardise(compressed, lengths), compressed)

    def enhanced_magnitude(self, mask, compressed):
        """Apply the mask to the compressed magnitude and decompress.
        This is the only place where the mask is applied."""
        return self.decompress(mask * compressed)
