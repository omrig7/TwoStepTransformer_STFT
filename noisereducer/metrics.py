"""The five benchmark scores, and the "ceiling" of our method.

    PESQ   perceptual quality, wideband, -0.5 to 4.5
    STOI   intelligibility, 0 to 1
    ESTOI  extended STOI, a bit stricter, 0 to 1
    SI-SDR scale-invariant signal-to-distortion ratio, dB
    SSNR   segmental SNR, dB

If a score cannot be computed (PESQ refuses silence, for example) we return
nan instead of crashing, so one bad file does not stop an evaluation of 824.

The pesq package is optional. pip has no ready-made pesq for Windows, so on
Windows it is installed separately (see README.md). Without it, cleaning
recordings and the other four scores still work, and PESQ is reported as n/a.
"""

import warnings

import numpy as np
import torch
from pystoi import stoi

try:
    from pesq import pesq
except ImportError:  # not installed, see pesq_available()
    pesq = None

FLOOR = 1e-12
METRICS = ("pesq", "stoi", "estoi", "sisdr", "ssnr")


def as_array(signal):
    """1-D float64 numpy array, which is what pesq and pystoi want."""
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu()
    return np.asarray(signal, dtype=np.float64).reshape(-1)


def si_sdr(reference, estimate):
    """SI-SDR in dB. The estimate is first scaled to best match the reference,
    so being a bit too loud or too quiet is not counted as an error."""
    reference, estimate = as_array(reference), as_array(estimate)
    energy = reference @ reference
    # Silence would get an infinite score (it is the reference times zero).
    if energy < FLOOR or estimate @ estimate < FLOOR:
        return float("nan")

    target = (reference @ estimate) / energy * reference
    residual = estimate - target
    if residual @ residual < FLOOR:
        return float("inf")
    return float(10 * np.log10((target @ target) / (residual @ residual)))


def segmental_snr(reference, estimate, frame=256, low=-10.0, high=35.0):
    """Average SNR over 16 ms frames, each one clamped to [-10, 35] dB.

    Without the clamp one silent frame gives a huge negative number and ruins
    the average.
    """
    reference, estimate = as_array(reference), as_array(estimate)
    frames = len(reference) // frame
    if frames == 0:
        return float("nan")

    clean = reference[: frames * frame].reshape(frames, frame)
    error = (estimate[: frames * frame] - reference[: frames * frame]).reshape(frames, frame)
    signal = (clean ** 2).sum(axis=1)
    noise = (error ** 2).sum(axis=1)
    active = signal > FLOOR  # skip frames of digital silence
    if not active.any():
        return float("nan")

    ratio = 10 * np.log10((signal[active] + FLOOR) / (noise[active] + FLOOR))
    return float(np.clip(ratio, low, high).mean())


PESQ_MISSING = ("the pesq package is not installed, so PESQ cannot be computed. "
                "On Windows pip cannot install it, see \"Windows\" in README.md.")


def pesq_available():
    return pesq is not None


def pesq_score(reference, estimate, sample_rate=16_000):
    if pesq is None:
        return float("nan")
    try:
        return float(pesq(sample_rate, reference.astype(np.float32),
                          estimate.astype(np.float32), "wb"))
    except Exception as error:
        warnings.warn(f"PESQ gave no score: {error}")
        return float("nan")


def stoi_score(reference, estimate, sample_rate=16_000, extended=False):
    try:
        return float(stoi(reference, estimate, sample_rate, extended=extended))
    except Exception as error:
        warnings.warn(f"STOI gave no score: {error}")
        return float("nan")


def score(reference, estimate, sample_rate=16_000):
    """All five scores for one file, as a dict."""
    reference, estimate = as_array(reference), as_array(estimate)
    return {
        "pesq": pesq_score(reference, estimate, sample_rate),
        "stoi": stoi_score(reference, estimate, sample_rate),
        "estoi": stoi_score(reference, estimate, sample_rate, extended=True),
        "sisdr": si_sdr(reference, estimate),
        "ssnr": segmental_snr(reference, estimate),
    }


def oracle_waveform(noisy, clean, stft):
    """The best our method could possibly do on this file.

    We only change the magnitude and keep the noisy phase. So the best case is
    the *clean* magnitude with the *noisy* phase (the ideal amplitude mask).
    Whatever this scores is our ceiling; a model that beats it has a bug.
    """
    noisy_magnitude, noisy_phase = stft.to_magnitude_phase(stft.analyze(noisy))
    clean_magnitude = stft.analyze(clean).abs()
    return stft.synthesize(
        stft.from_magnitude_phase(clean_magnitude, noisy_phase),
        num_samples=noisy.shape[-1],
    )
