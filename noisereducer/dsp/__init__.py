"""Signal processing: the STFT and the magnitude compression."""

from noisereducer.dsp.features import MagnitudeFeatures, frame_mask
from noisereducer.dsp.stft import StftTransform

__all__ = ["MagnitudeFeatures", "StftTransform", "frame_mask"]
