# Noise Reducer

Speech enhancement with a two-stage (dual-path) transformer in the
time-frequency domain, trained on VoiceBank-DEMAND at 16 kHz.
Deep Learning course project.

Our best model (`noisereducer/runs/baseline/best.pt`, 877,489 parameters) on
the official test set (824 files):

|           | PESQ  | STOI  | ESTOI | SI-SDR | SSNR  |
|-----------|-------|-------|-------|--------|-------|
| noisy     | 1.967 | 0.921 | 0.787 | 8.45   | 1.55  |
| ours      | 3.144 | 0.951 | 0.870 | 18.61  | 9.45  |
| ceiling   | 3.953 | 0.989 | 0.957 | 20.08  | 11.81 |

The ceiling is the clean magnitude with the noisy phase, the best any
magnitude mask can do.

## How to use

Everything is inside `noisereducer/` and runs from PyCharm, there are no
command line arguments.

1. `pip install -r requirements.txt`
2. Download VoiceBank-DEMAND from
   [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791) and
   unzip the folders `noisy_trainset_28spk_wav`, `clean_trainset_28spk_wav`,
   `noisy_testset_wav` and `clean_testset_wav` into
   `noisereducer/data/voicebank_demand/16k/`. The download is at 48 kHz, which
   works as it is: every file is converted to 16 kHz when it is read, which
   gives exactly the audio our results were measured on.
3. **Train:** open `noisereducer/run_training.py` and press Run. `RUN_NAME`
   is the folder in `noisereducer/runs/`: a new name starts a new run, the
   name of an existing run continues it.
4. **Test:** open `noisereducer/run_eval.py` and press Run. It scores
   `runs/<RUN_NAME>/best.pt` (`baseline` by default) on the test set and
   prints the table above.
5. **Clean your own recordings:** put them in `noisereducer/inference/noisy/`,
   open `noisereducer/run_inference.py` and press Run. The cleaned files are
   saved in `noisereducer/inference/clean/`.

All the settings are in `noisereducer/config.py`. `noisereducer/__init__.py`
lists what every file and folder does.
