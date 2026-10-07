"""STFT and inverse STFT.

Two things to know:

* Axis order. Everything here returns ``(..., frames, bins)``. ``torch.stft``
  gives ``(..., bins, frames)``, and this file is the only place we swap them.
* Always float32. The noisy phase goes around the network and is used again to
  build the output, so if it was rounded to float16 we would hear it as
  broadband noise. That is why autocast is switched off inside analyze and
  synthesize.

We use a Hann window with 75% overlap (512 window, 128 hop). The windows then
sum to a constant (the COLA condition), so the inverse STFT gives back exactly
the input.
"""

import torch
from torch import nn


class StftTransform(nn.Module):
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # A buffer so it moves to the GPU with the model, but it is not saved
        # in the checkpoint (persistent=False) because it is always the same.
        self.register_buffer(
            "window",
            torch.hann_window(n_fft, periodic=True, dtype=torch.float32),
            persistent=False,
        )

    def frame_lengths(self, sample_lengths):
        """Number of frames for each length in samples (center=True adds one)."""
        return sample_lengths // self.hop_length + 1

    def analyze(self, waveform):
        """Waveform ``(..., samples)`` -> complex spectrum ``(..., frames, bins)``."""
        leading = waveform.shape[:-1]
        flat = waveform.reshape(-1, waveform.shape[-1]).to(torch.float32)

        with torch.amp.autocast(device_type=flat.device.type, enabled=False):
            spectrum = torch.stft(
                flat,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=True,
                return_complex=True,
            )

        spectrum = spectrum.transpose(-2, -1)
        return spectrum.reshape(*leading, *spectrum.shape[-2:])

    def synthesize(self, spectrum, num_samples):
        """Complex spectrum ``(..., frames, bins)`` -> waveform ``(..., num_samples)``.

        ``num_samples`` is the length of the original audio, otherwise the
        output could be a few samples short.
        """
        leading = spectrum.shape[:-2]
        flat = spectrum.reshape(-1, *spectrum.shape[-2:]).transpose(-2, -1)
        flat = flat.to(torch.complex64)

        with torch.amp.autocast(device_type=flat.device.type, enabled=False):
            waveform = torch.istft(
                flat,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=True,
                length=num_samples,
            )

        return waveform.reshape(*leading, num_samples)

    @staticmethod
    def to_magnitude_phase(spectrum):
        """Complex spectrum -> (magnitude, phase in radians)."""
        return spectrum.abs(), spectrum.angle()

    @staticmethod
    def from_magnitude_phase(magnitude, phase):
        """(magnitude, phase) -> complex spectrum.

        Written as real + imaginary parts instead of ``torch.polar`` because
        this works on every device (MPS included).
        """
        magnitude = magnitude.to(torch.float32)
        phase = phase.to(torch.float32)
        return torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
