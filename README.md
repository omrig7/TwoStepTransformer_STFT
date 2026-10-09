# Noise Reducer

Speech enhancement with a two-stage (dual-path) transformer in the
time-frequency domain, trained on VoiceBank-DEMAND at 16 kHz.
Deep Learning course project.

Our best model (`noisereducer/runs/l1_expanded_data/best.pt`, 877,489
parameters) on the official test set (824 files):

|           | PESQ  | STOI  | ESTOI | SI-SDR | SSNR  |
|-----------|-------|-------|-------|--------|-------|
| noisy     | 1.967 | 0.921 | 0.787 | 8.45   | 1.55  |
| ours      | 3.146 | 0.951 | 0.871 | 18.85  | 9.65  |
| ceiling   | 3.953 | 0.989 | 0.957 | 20.08  | 11.81 |

The ceiling is the clean magnitude with the noisy phase, the best any
magnitude mask can do.

## Our runs

All the runs have the same network. The two later runs both start from epoch
16 of `baseline` and train 8 more epochs. The model of every run is the epoch
with the best PESQ on the 200 validation utterances, the test set is scored
only after that choice.

| run (`noisereducer/runs/`) | what changed                                     | epoch | validation PESQ | test PESQ | test SI-SDR |
|----------------------------|--------------------------------------------------|-------|-----------------|-----------|-------------|
| `baseline`                 | the original run, 28 speakers, MSE loss           | 15    | 2.800           | 3.144     | 18.61       |
| `baseline_continued`       | 8 more epochs, same data and loss                 | 22    | 2.811           | 3.133     | 18.72       |
| `l1_expanded_data`         | 8 more epochs, L1 loss and the 56spk files added  | 19    | 2.829           | 3.146     | 18.85       |

The differences in PESQ are small (one run per setting), the clearer gain
of `l1_expanded_data` is in SI-SDR and SSNR. The full test scores of every
run are in its `test_scores.json`.

`config.py` has the settings of `baseline`. The two later runs were trained
with an earlier version of the code (the L1 loss and the extra data are not in
this version), their settings are in their `config.json`.

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
   `runs/<RUN_NAME>/best.pt` (`l1_expanded_data` by default) on the test set and
   prints the table above.
5. **Clean your own recordings:** put them in `noisereducer/inference/noisy/`,
   open `noisereducer/run_inference.py` and press Run. The cleaned files are
   saved in `noisereducer/inference/clean/`.

All the settings are in `noisereducer/config.py`. `noisereducer/__init__.py`
lists what every file and folder does.
