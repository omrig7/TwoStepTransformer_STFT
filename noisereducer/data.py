"""Reading the audio and building the batches.

VoiceBank-DEMAND keeps every pair as two files with the same name, one in a
noisy folder and one in a clean folder, for example
``noisy_trainset_28spk_wav/p226_001.wav`` and ``clean_trainset_28spk_wav/p226_001.wav``.
The part before the underscore is the speaker.
"""

import math
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset, Sampler

# Formats soundfile cannot open (phone recordings). We convert them to wav
# with afconvert, which comes with macOS.
CONVERT_FIRST = {".m4a", ".mp4", ".aac", ".mov"}


def read_audio(path, sample_rate=16_000):
    """Read one file as a mono float32 tensor at ``sample_rate``."""
    path = Path(path)
    if path.suffix.lower() in CONVERT_FIRST:
        with tempfile.TemporaryDirectory() as folder:
            wav = Path(folder) / "converted.wav"
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16", str(path), str(wav)],
                check=True,
            )
            return read_audio(wav, sample_rate)

    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)  # stereo -> mono

    # A proper resampling filter, not just taking every 3rd sample. Otherwise
    # everything above 8 kHz folds back into the band as noise.
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        data = resample_poly(data, sample_rate // divisor, rate // divisor)

    return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))


def matched_names(noisy_dir, clean_dir):
    """Sorted list of the .wav names that exist in both folders."""
    noisy = {path.name for path in Path(noisy_dir).glob("*.wav")}
    clean = {path.name for path in Path(clean_dir).glob("*.wav")}
    if not noisy & clean:
        raise FileNotFoundError(f"no matching .wav files in {noisy_dir} and {clean_dir}")
    return sorted(noisy & clean)


def speaker_of(name):
    return name.split("_")[0]


def split_by_speaker(names, held_out=0.1, seed=0):
    """Split into (train, validation) so that no speaker is in both.

    If we split by file, the same voice would be in train and validation and
    the validation score would be too optimistic. With 28 speakers and 0.1 we
    keep 3 speakers aside (p230, p250 and p259 at seed 0).
    """
    speakers = sorted({speaker_of(name) for name in names})
    random.Random(seed).shuffle(speakers)
    validation = set(speakers[: max(1, round(len(speakers) * held_out))])

    train = [name for name in names if speaker_of(name) not in validation]
    held = [name for name in names if speaker_of(name) in validation]
    return train, held


def subset_by_speaker(names, count, seed=0):
    """Pick ``count`` names, taking the speakers in turns.

    Taking the first ``count`` names would be a bad idea because the names are
    sorted by speaker, so we would get only the first speaker or two.
    """
    if count <= 0 or count >= len(names):
        return list(names)

    by_speaker = {}
    for name in names:
        by_speaker.setdefault(speaker_of(name), []).append(name)

    generator = random.Random(seed)
    for utterances in by_speaker.values():
        generator.shuffle(utterances)

    chosen = []
    for index in range(max(len(u) for u in by_speaker.values())):
        for utterances in by_speaker.values():
            if index < len(utterances):
                chosen.append(utterances[index])
                if len(chosen) == count:
                    return sorted(chosen)
    return sorted(chosen)


class PairedAudio(Dataset):
    """Gives ``(noisy, clean)`` waveforms of the same length.

    Files longer than ``max_samples`` are cut, and the same window is taken
    from both files so they still line up. For training the window is random
    (a bit of free augmentation), for validation it is the centre so the loss
    is measured on the same audio every epoch.
    """

    def __init__(self, noisy_dir, clean_dir, names, sample_rate=16_000,
                 max_samples=None, random_crop=True):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.names = list(names)
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.random_crop = random_crop

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        noisy = read_audio(self.noisy_dir / name, self.sample_rate)
        clean = read_audio(self.clean_dir / name, self.sample_rate)

        if self.max_samples and len(noisy) > self.max_samples:
            room = len(noisy) - self.max_samples
            if self.random_crop:
                start = int(torch.randint(room + 1, (1,)).item())
            else:
                start = room // 2
            noisy = noisy[start:start + self.max_samples]
            clean = clean[start:start + self.max_samples]

        return noisy, clean


def file_lengths(directory, names, sample_rate):
    """Length of each file in samples. Reads only the header, so it is fast."""
    lengths = []
    for name in names:
        info = soundfile.info(str(Path(directory) / name))
        lengths.append(int(round(info.frames * sample_rate / info.samplerate)))
    return lengths


class LengthBatches(Sampler):
    """Makes batches of files with similar length, with a limit on total audio.

    A batch is padded to its longest file, so mixing a 8 s file with 2 s files
    wastes most of the batch on zeros. So we:
      1. shuffle all the files,
      2. cut the shuffled list into pools of 200 (max_count * 50),
      3. sort each pool by length and fill batches until adding one more file
         would pass ``budget_samples`` of padded audio or ``max_count`` files,
      4. shuffle the order of the batches.
    Because of step 1 the batches are different every epoch. With
    ``shuffle=False`` nothing is random, which is what validation needs.
    """

    def __init__(self, lengths, budget_samples, max_count, shuffle=True, pool=50):
        self.lengths = list(lengths)
        self.budget = budget_samples
        self.max_count = max_count
        self.shuffle = shuffle
        self.span = max_count * pool
        self.count = len(self.group(list(range(len(self.lengths)))))

    def group(self, order):
        batches = []
        for start in range(0, len(order), self.span):
            pool = sorted(order[start:start + self.span], key=self.lengths.__getitem__)
            batch = []
            longest = 0
            for index in pool:
                # The pool is sorted, so this file is the longest one so far.
                padded = max(longest, self.lengths[index])
                full = len(batch) >= self.max_count
                over = (len(batch) + 1) * padded > self.budget
                if batch and (full or over):
                    batches.append(batch)
                    batch, padded = [], self.lengths[index]
                batch.append(index)
                longest = padded
            if batch:
                batches.append(batch)
        return batches

    def __len__(self):
        # The real number changes a little every epoch, this is just for tqdm.
        return self.count

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self.lengths)).tolist()
        else:
            order = list(range(len(self.lengths)))
        batches = self.group(order)
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        yield from batches


def collate(batch):
    """Pad the batch with zeros to its longest file and keep the real lengths.

    The lengths are passed to the model and the loss so the padding is ignored.
    """
    lengths = torch.tensor([len(noisy) for noisy, _ in batch])
    noisy = torch.zeros(len(batch), int(lengths.max()))
    clean = torch.zeros(len(batch), int(lengths.max()))
    for i, (one_noisy, one_clean) in enumerate(batch):
        noisy[i, :len(one_noisy)] = one_noisy
        clean[i, :len(one_clean)] = one_clean
    return noisy, clean, lengths


def make_loader(config, noisy_dir, clean_dir, names, shuffle, max_samples,
                batch_size):
    rate = config.stft.sample_rate
    dataset = PairedAudio(noisy_dir, clean_dir, names, rate, max_samples,
                          random_crop=shuffle)
    lengths = [min(n, max_samples or n) for n in file_lengths(clean_dir, names, rate)]
    sampler = LengthBatches(lengths, int(config.data.batch_seconds * rate),
                            batch_size, shuffle)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=config.data.workers,
        persistent_workers=config.data.workers > 0,
        pin_memory=torch.cuda.is_available(),  # faster copies to an NVIDIA GPU
    )


def make_loaders(config):
    """The three loaders the training needs.

    train      - the training speakers, shuffled, random crops
    validation - the held-out speakers, fixed order, for the validation loss
    score      - 200 of the held-out utterances, whole and one at a time, for PESQ
    """
    data = config.data
    noisy_dir = Path(data.root) / data.noisy_train
    clean_dir = Path(data.root) / data.clean_train
    max_samples = int(data.max_seconds * config.stft.sample_rate)
    seed = config.train.seed

    names = matched_names(noisy_dir, clean_dir)
    train_names, validation_names = split_by_speaker(names, data.held_out_speakers, seed)
    score_names = subset_by_speaker(validation_names, config.train.score_utterances, seed)

    batch = config.train.batch_size
    train = make_loader(config, noisy_dir, clean_dir, train_names, True, max_samples, batch)
    validation = make_loader(config, noisy_dir, clean_dir, validation_names, False,
                             max_samples, batch)
    score = make_loader(config, noisy_dir, clean_dir, score_names, False, None, 1)
    return train, validation, score
