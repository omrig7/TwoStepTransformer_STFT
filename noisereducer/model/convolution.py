"""The convolutions before and after the attention, and the 4x grouping.

ConvBridge (the encoder and the decoder) does two things:
  1. changes the number of channels (1 -> 48 before the attention, 48 -> 1 after),
  2. gives every frame some context from the frames before it.

For 2. we use 4 dilated convolutions with a kernel of 2 frames x 3 bins and
dilations 1, 2, 4, 8. Together they see 1 + 1 + 2 + 4 + 8 = 16 frames = 128 ms
(about a syllable) for the price of four small layers. In time the padding is
only on the past side, so a frame never looks at the future. In frequency the
kernel is centred.

The bridges keep the full size (all frames, all 257 bins). Making the plane
smaller is done by AxisDown / AxisUp, only around the attention, because the
attention is the only part whose cost grows with the number of positions.
"""

import torch
from torch import nn

DILATIONS = (1, 2, 4, 8)
AXIS_DIM = {"time": 2, "frequency": 3}  # tensors are (batch, channels, frames, bins)


class ChannelNorm(nn.LayerNorm):
    """LayerNorm over the channels of each (frame, bin) position separately.

    GroupNorm / InstanceNorm / BatchNorm take statistics over the whole plane,
    which would make every output depend on the entire input and break the
    128 ms receptive field of the dilated convolutions.
    """

    def forward(self, x):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


def dense_layer(in_channels, out_channels, dilation):
    """One dilated conv + norm + PReLU. The output has the same frames and bins."""
    return nn.Sequential(
        # (left, right) in frequency, then (past, future) in time
        nn.ZeroPad2d((1, 1, dilation, 0)),
        nn.Conv2d(in_channels, out_channels, (2, 3), dilation=(dilation, 1)),
        ChannelNorm(out_channels),
        nn.PReLU(out_channels),
    )


class DilatedDenseBlock(nn.Module):
    """Dilated convolutions with dense connections.

    Layer i gets the block input and the outputs of all the layers before it,
    concatenated, so it has ``channels * (i + 1)`` input channels. These short
    paths back to the input make it easy to train without residuals.
    """

    def __init__(self, channels=64):
        super().__init__()
        self.layers = nn.ModuleList(
            dense_layer(channels * (index + 1), channels, dilation)
            for index, dilation in enumerate(DILATIONS)
        )

    def forward(self, x):
        seen = [x]
        for layer in self.layers:
            seen.append(layer(torch.cat(seen, dim=1)))
        return seen[-1]


class ConvBridge(nn.Module):
    """1x1 conv -> dilated dense block -> 1x1 conv.

    The encoder is ``ConvBridge(1, width)`` and the decoder is
    ``ConvBridge(width, 1)``.
    """

    def __init__(self, in_channels, out_channels, channels=64):
        super().__init__()
        self.enter = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.dense = DilatedDenseBlock(channels)
        self.leave = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.leave(self.dense(self.enter(x)))


def coarsen_mask(valid, stride):
    """Shorten a padding mask the same way AxisDown shortens the time axis.

    A group of ``stride`` frames counts as real if at least one frame in it is
    real. Otherwise we would throw away the real audio at the end of a file.
    """
    overhang = (-valid.shape[-1]) % stride
    if overhang:
        valid = nn.functional.pad(valid, (0, overhang), value=False)
    return valid.reshape(valid.shape[0], -1, stride).any(dim=-1)


class AxisDown(nn.Module):
    """Group ``stride`` neighbouring positions along one axis into one.

    Attention cost grows with (length of its axis)^2, and it is repeated for
    every position of the other axis. At full size (257 bins x 251 frames for
    2 s) one training step needed about 15 GiB just for attention weights.

    * frequency: 257 bins -> 65 bands.
    * time: 8 ms steps -> 32 ms steps. This is what lets us train on whole
      utterances (up to 8 s) and still attend over all of it, instead of cutting
      the audio into short windows.

    The grouping is a learned convolution (kernel = stride = 4), not an
    average, so the model decides what to keep. If the size does not divide
    by the stride we pad it; AxisUp removes the extra.
    """

    def __init__(self, channels, stride, axis):
        super().__init__()
        self.stride = stride
        self.axis = axis
        self.dim = AXIS_DIM[axis]
        shape = (stride, 1) if axis == "time" else (1, stride)
        self.project = nn.Conv2d(channels, channels, shape, stride=shape)
        self.norm = ChannelNorm(channels)
        self.activation = nn.PReLU(channels)

    def forward(self, x):
        overhang = (-x.shape[self.dim]) % self.stride
        if overhang:
            # F.pad lists (left, right) starting from the LAST axis, so the
            # time axis needs four numbers.
            pad = (0, overhang) if self.axis == "frequency" else (0, 0, 0, overhang)
            x = nn.functional.pad(x, pad)
        return self.activation(self.norm(self.project(x)))


class AxisUp(nn.Module):
    """Undo AxisDown: spread every group back to ``stride`` positions.

    A transposed convolution learns how to spread a group, because its members
    are not the same: the bottom of a band can be speech and the top noise.
    ``size`` is the original length, needed to cut off the padding AxisDown added.
    """

    def __init__(self, channels, stride, axis):
        super().__init__()
        self.dim = AXIS_DIM[axis]
        shape = (stride, 1) if axis == "time" else (1, stride)
        self.project = nn.ConvTranspose2d(channels, channels, shape, stride=shape)
        self.norm = ChannelNorm(channels)
        self.activation = nn.PReLU(channels)

    def forward(self, x, size):
        spread = self.activation(self.norm(self.project(x)))
        return spread.narrow(self.dim, 0, size)
