"""Speech enhancement with a two-stage (dual-path) transformer.

The three files you run (from PyCharm, no command line arguments):
    run_training.py   train a model
    run_eval.py       score a model on the VoiceBank-DEMAND test set
    run_inference.py  clean your own recordings

The code, in the order the audio goes through it:
    config.py         every setting in one place
    data.py           reading the audio and making batches
    dsp/              STFT and the magnitude compression
    model/            the network (convolutions, attention + GRU blocks)
    loss.py           the training loss
    metrics.py        PESQ, STOI, ESTOI, SI-SDR, SSNR
    train.py          the training loop
    enhancement.py    loading a model and cleaning files
    evaluation.py     scoring a model on a test set

The folders:
    data/             VoiceBank-DEMAND at 16 kHz (data/voicebank_demand/16k/)
    runs/             one folder per training run; runs/baseline is our best model
    inference/noisy/  put your own recordings here
    inference/clean/  the cleaned recordings are saved here
"""

from noisereducer.config import Config

__all__ = ["Config"]
