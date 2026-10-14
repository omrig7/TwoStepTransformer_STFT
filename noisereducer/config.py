"""All the settings of the project in one place.

The default values are exactly the ones of our original run (runs/baseline),
the one that reached PESQ 2.80 on the validation speakers after 15 epochs on a
Kaggle T4. Our best model (runs/l1_expanded_data) continued it with an L1 loss
and more data, see its config.json. If you want to try something, change it
here and every other file will pick it up.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

# This folder (noisereducer/). The dataset, the trained runs and the audio
# folders are all inside it, and every path is built from here so the scripts
# work no matter which working directory PyCharm uses.
ROOT_DIR = Path(__file__).resolve().parent


@dataclass
class StftConfig:
    """How we turn audio into a spectrogram and back (Hann window)."""

    sample_rate: int = 16_000
    n_fft: int = 512  # 32 ms window -> 257 frequency bins
    hop_length: int = 128  # 8 ms hop -> 75% overlap, every sample is rebuilt from 4 frames

    def build(self):
        from noisereducer.dsp import StftTransform

        return StftTransform(n_fft=self.n_fft, hop_length=self.hop_length)


@dataclass
class FeatureConfig:
    """How the magnitude is prepared before it goes into the network."""

    exponent: float = 0.3  # power-law compression |Y|^0.3, like in the papers
    magnitude_floor: float = 1e-5  # avoids the huge gradient of x^0.3 near zero
    std_floor: float = 1e-5  # avoids dividing by zero on a silent file

    def build(self):
        from noisereducer.dsp import MagnitudeFeatures

        return MagnitudeFeatures(
            exponent=self.exponent,
            magnitude_floor=self.magnitude_floor,
            std_floor=self.std_floor,
        )


@dataclass
class ModelConfig:
    """The size of the network (877,489 parameters with these values)."""

    channels: int = 64  # channels inside the convolution encoder and decoder
    width: int = 48  # size of a feature vector inside the attention blocks
    heads: int = 4  # attention heads, width must divide by heads
    blocks: int = 4  # how many dual-path blocks are stacked
    # Before the attention we group 4 neighbouring bins and 4 neighbouring
    # frames into one position. Without this, attention over 257 bins and a few
    # hundred frames does not fit in memory.
    frequency_stride: int = 4
    time_stride: int = 4


@dataclass
class LossConfig:
    """The loss is 0.4 * waveform MSE + 0.6 * compressed magnitude MSE.

    The weights look balanced but the magnitude error is about 27 times bigger
    than the waveform error, so in practice the magnitude term is ~98% of the
    total. The waveform term still matters because it is the only part of the
    loss that "sees" the phase.
    """

    time_weight: float = 0.4
    magnitude_weight: float = 0.6
    error: str = "mse"  # "mse" (squared error) or "l1" (absolute error, runs/l1_expanded_data)


@dataclass
class DataConfig:
    """Where the audio is and how it is batched.

    ``root`` is the 16 kHz copy of VoiceBank-DEMAND (the original is 48 kHz).
    """

    root: Path = ROOT_DIR / "data" / "voicebank_demand" / "16k"
    noisy_train: str = "noisy_trainset_28spk_wav"
    clean_train: str = "clean_trainset_28spk_wav"
    noisy_test: str = "noisy_testset_wav"
    clean_test: str = "clean_testset_wav"
    # Utterances are used whole. Only the few longer than 8 s are cut, and
    # that is almost nothing (median length is 2.7 s).
    max_seconds: float = 8.0
    # A batch holds at most 12 seconds of (padded) audio, so a batch of long
    # files is small and a batch of short files is bigger. This keeps the
    # memory of every batch about the same.
    batch_seconds: float = 12.0
    # 10% of the training speakers (3 of 28) are kept aside for validation.
    # The official test set is never used during training.
    held_out_speakers: float = 0.1
    workers: int = 4  # processes that read audio in the background


@dataclass
class TrainConfig:
    """How the training runs."""

    epochs: int = 60  # an upper limit; the baseline was at its best at epoch 15
    batch_size: int = 4  # max utterances in a batch (the 12 s budget above also limits it)
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    clip_norm: float = 5.0  # attention and GRU sometimes give huge gradients
    seed: int = 0
    # After every epoch we compute PESQ on this many validation utterances and
    # keep the epoch with the best PESQ (the loss and PESQ do not always agree).
    score_utterances: int = 200
    # If PESQ did not improve for lr_patience epochs, multiply the learning
    # rate by lr_factor (but never go below lr_minimum).
    lr_patience: int = 3
    lr_factor: float = 0.5
    lr_minimum: float = 1e-5


@dataclass
class Config:
    stft: StftConfig = field(default_factory=StftConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        """Plain dict (paths as strings) so it can go into a checkpoint or a json."""
        values = asdict(self)
        values["data"]["root"] = str(values["data"]["root"])
        return values
