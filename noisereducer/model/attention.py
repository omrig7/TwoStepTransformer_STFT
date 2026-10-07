"""Attention along one axis of the spectrogram, and the dual-path block.

Attention over all (frame, bin) positions at once is too expensive: every
position is compared with every other one. So, like DPT-FSNet and TSTNN, we
attend along one axis at a time and treat the other axis as a batch:

* frequency attention: inside one frame, every band looks at every other band
  (harmonics, the shape of the spectrum at that moment),
* time attention: inside one band, every frame looks at every other frame
  (how that frequency changes over the whole utterance).

One after the other, information can get from any position to any other.

There is no positional encoding. Attention by itself does not know the order
of its inputs, and the bidirectional GRU (in place of the first linear layer
of the feed-forward part) is what gives it the order. The convolutions before
the attention also give some local order. This is the same as in the papers.
"""

from einops import rearrange, repeat
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AxisAttention(nn.Module):
    """One transformer layer along one axis ("frequency" or "time").

        x -> multi-head attention -> + x -> LayerNorm
          -> bidirectional GRU -> Linear -> + -> LayerNorm

    With width 48 and 4 heads every head gets 12 numbers. The GRU has 48 units
    in each direction, so its output is 96 and the Linear brings it back to 48.
    Input and output are ``(batch, width, frames, bins)``. We use post-norm
    (LayerNorm after the residual) like the papers we compare to.
    """

    def __init__(self, axis, width=48, heads=4):
        super().__init__()
        self.axis = axis
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.project = nn.Linear(2 * width, width)
        self.ffn_norm = nn.LayerNorm(width)

    def _recurrent(self, sequences, padding):
        """Run the GRU without letting the padding affect the real steps.

        The backward direction of the GRU starts at the last step, which in a
        padded sequence is padding. Packing tells the GRU where each sequence
        really ends, so it starts from the last real frame.
        """
        if padding is None:
            output, _ = self.gru(sequences)
            return output

        packed = pack_padded_sequence(
            sequences,
            (~padding).sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=sequences.shape[1]
        )
        return output

    def forward(self, x, valid=None):
        batch, _, frames, bins = x.shape

        # Turn the plane into a batch of sequences along our axis.
        if self.axis == "frequency":
            sequences = rearrange(x, "b c t f -> (b t) f c")
        else:
            sequences = rearrange(x, "b c t f -> (b f) t c")

        # Only the time axis can have padding (frames after the end of a
        # short file). Every frame has all its bins, so frequency never does.
        padding = None
        if valid is not None and self.axis == "time":
            padding = repeat(~valid, "b t -> (b f) t", f=bins)

        attended, _ = self.attention(
            sequences,
            sequences,
            sequences,
            key_padding_mask=padding,
            need_weights=False,
        )
        sequences = self.attention_norm(sequences + attended)

        recurrent = self._recurrent(sequences, padding)
        sequences = self.ffn_norm(sequences + self.project(recurrent))

        # And back to (batch, width, frames, bins).
        if self.axis == "frequency":
            return rearrange(sequences, "(b t) f c -> b c t f", b=batch, t=frames)
        return rearrange(sequences, "(b f) t c -> b c t f", b=batch, f=bins)


class DualPathBlock(nn.Module):
    """Frequency attention, then time attention on its output.

    Because the time layer gets the output of the frequency layer, one
    position can reach the whole plane inside a single block.
    """

    def __init__(self, width=48, heads=4):
        super().__init__()
        self.components = nn.ModuleList([
            AxisAttention("frequency", width, heads),
            AxisAttention("time", width, heads),
        ])

    def forward(self, x, valid=None):
        for component in self.components:
            x = component(x, valid)
        return x
