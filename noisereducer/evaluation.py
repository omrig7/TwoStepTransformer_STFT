"""Scoring a model on a test set that has a clean version of every file.

Every test file is scored three times against its clean version:
    noisy     the input as it is, where we start from
    enhanced  the output of our model
    ceiling   the clean magnitude with the noisy phase, the best a model that
              only changes the magnitude can do (see metrics.oracle_waveform)
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from noisereducer.config import StftConfig
from noisereducer.data import read_audio
from noisereducer.enhancement import enhance, list_audio_files
from noisereducer.metrics import METRICS, oracle_waveform, score

STAGES = ("noisy", "enhanced", "ceiling")


def evaluate(model, noisy_dir, clean_dir, sample_rate=16_000):
    """Score every file in noisy_dir against the file with the same name in
    clean_dir. Returns the averages as averages[stage][metric]."""
    stft = StftConfig().build()  # a CPU copy, for the ceiling

    # values[stage][metric] is the list of scores of every file
    values = {}
    for stage in STAGES:
        values[stage] = {metric: [] for metric in METRICS}

    files = list_audio_files(noisy_dir)
    print(f"scoring {len(files)} files from {noisy_dir}")
    for path in tqdm(files, desc="scoring"):
        noisy = read_audio(path, sample_rate)
        clean = read_audio(Path(clean_dir) / path.name, sample_rate)
        outputs = {
            "noisy": noisy,
            "enhanced": enhance(model, noisy, sample_rate),
            "ceiling": oracle_waveform(noisy, clean, stft),
        }
        for stage, waveform in outputs.items():
            for metric, value in score(clean, waveform, sample_rate).items():
                values[stage][metric].append(value)

    # nanmean skips the files a score could not be computed for
    averages = {}
    for stage in STAGES:
        averages[stage] = {}
        for metric in METRICS:
            averages[stage][metric] = float(np.nanmean(values[stage][metric]))
    return averages


def print_scores(averages):
    """A table of the averages, plus how much of the possible gain we got.
    A score that could not be computed (PESQ without the pesq package) is n/a."""
    header = ""
    for metric in METRICS:
        header += f"{metric.upper():>9}"
    print(f"{'':14}{header}")

    for stage in STAGES:
        line = ""
        for metric in METRICS:
            line += cell(averages[stage][metric], "9.3f")
        print(f"{stage:14}{line}")

    # 0% means the model did nothing, 100% means it reached the ceiling.
    line = ""
    for metric in METRICS:
        noisy = averages["noisy"][metric]
        room = averages["ceiling"][metric] - noisy
        gained = (averages["enhanced"][metric] - noisy) / room if room else float("nan")
        line += cell(100 * gained, "8.1f", "%")
    print(f"{'% of ceiling':14}{line}")


def cell(value, form, suffix=""):
    """One number of the table, or n/a if it could not be computed."""
    if np.isnan(value):
        return f"{'n/a':>9}"
    return f"{value:{form}}{suffix}"
