"""The whole model: noisy waveform in, clean waveform out.

    waveform -> STFT -> |Y|, phase
    |Y| -> compress (^0.3) and standardise -> encoder (convolutions)
        -> group 4x4 -> 4 dual-path blocks -> ungroup -> decoder -> mask
    mask * |Y|^0.3 -> decompress -> with the noisy phase -> inverse STFT -> waveform

Two choices about the mask:

* It is a ReLU with no upper limit. Noise and speech add as complex numbers,
  so sometimes the noise partly cancels the speech and the noisy magnitude is
  *smaller* than the clean one. To fix those bins the mask must be above 1,
  which a sigmoid mask (0 to 1) cannot do.
* It starts at 1 (bias = 1, small weights), so the untrained model just passes
  the noisy audio through, and learns from there what to remove. With a
  normal init about half the mask would be negative, the ReLU would make it
  zero, and the model would start by outputting silence.

The STFT is inside the model and not in the training loop, so cleaning a file
is a single call: ``model(waveform).waveform``.
"""

from typing import NamedTuple

import torch
from torch import nn

from noisereducer.config import Config
from noisereducer.dsp.features import frame_mask
from noisereducer.model.attention import DualPathBlock
from noisereducer.model.convolution import AxisDown, AxisUp, ConvBridge, coarsen_mask


class Enhanced(NamedTuple):
    """What the model returns. The loss needs ``mask`` and ``compressed`` too,
    because it compares ``mask * compressed`` with the clean compressed
    magnitude."""

    waveform: torch.Tensor
    magnitude: torch.Tensor
    compressed: torch.Tensor
    mask: torch.Tensor


class NoiseReducer(nn.Module):
    """The speech enhancement network.

    ``forward(waveform, lengths)`` takes ``(samples,)`` or ``(batch, samples)``.
    ``lengths`` is the real number of samples of each file in a padded batch,
    so the padding is ignored everywhere (``None`` if nothing is padded).
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = Config() if config is None else config
        model = self.config.model

        self.stft = self.config.stft.build()
        self.features = self.config.features.build()

        self.encoder = ConvBridge(1, model.width, channels=model.channels)
        # Grouping only happens around the attention: the convolutions are
        # cheap, but attention costs (length of its axis)^2, so that is where
        # we need fewer positions.
        self.resample = nn.ModuleDict()
        for axis, stride in (("frequency", model.frequency_stride),
                             ("time", model.time_stride)):
            self.resample[f"{axis}_down"] = AxisDown(model.width, stride, axis)
            self.resample[f"{axis}_up"] = AxisUp(model.width, stride, axis)
        self.time_stride = model.time_stride
        # A ModuleList and not a Sequential, because every block also needs
        # the padding mask as a second argument.
        self.blocks = nn.ModuleList(
            DualPathBlock(model.width, model.heads) for _ in range(model.blocks)
        )
        self.decoder = ConvBridge(model.width, 1, channels=model.channels)

        # Start the mask at ~1 (see the top of the file). The bias alone is not
        # enough: the random weights would still spread the mask by about
        # +-0.6, which is a gain of 3 to 11 after the ^3.33. Small weights (not
        # zero) keep the start close to 1 and still let gradients flow.
        nn.init.ones_(self.decoder.leave.bias)
        with torch.no_grad():
            self.decoder.leave.weight.mul_(0.01)

    def predict_mask(self, features, valid=None):
        """Standardised magnitude ``(batch, frames, bins)`` -> mask of the same shape."""
        frames, bins = features.shape[-2], features.shape[-1]
        hidden = self.encoder(features.unsqueeze(1))

        for axis in ("frequency", "time"):
            hidden = self.resample[f"{axis}_down"](hidden)

        # The time axis is now 4 times shorter, so the padding mask must be
        # shortened the same way, otherwise the attention would ignore the
        # wrong frames.
        if valid is not None and self.time_stride > 1:
            valid = coarsen_mask(valid, self.time_stride)

        for block in self.blocks:
            hidden = block(hidden, valid)

        for axis, size in (("time", frames), ("frequency", bins)):
            hidden = self.resample[f"{axis}_up"](hidden, size)

        return torch.relu(self.decoder(hidden)).squeeze(1)

    def forward(self, waveform, lengths=None):
        unbatched = waveform.ndim == 1
        if unbatched:
            waveform = waveform.unsqueeze(0)

        spectrum = self.stft.analyze(waveform)
        magnitude, phase = self.stft.to_magnitude_phase(spectrum)

        valid = None
        if lengths is not None:
            valid = frame_mask(self.stft.frame_lengths(lengths), magnitude.shape[-2])
        ready = self.features(magnitude, None if valid is None else valid.sum(dim=-1))

        mask = self.predict_mask(ready.features, valid)
        enhanced = self.features.enhanced_magnitude(mask, ready.compressed)
        if valid is not None:
            # Zero the padded frames. Their STFT windows overlap the last real
            # samples, so anything left there would leak into the audio.
            enhanced = enhanced * valid.unsqueeze(-1)

        audio = self.stft.synthesize(
            self.stft.from_magnitude_phase(enhanced, phase),
            num_samples=waveform.shape[-1],
        )

        result = Enhanced(audio, enhanced, ready.compressed, mask)
        if unbatched:
            return Enhanced(*(tensor.squeeze(0) for tensor in result))
        return result
