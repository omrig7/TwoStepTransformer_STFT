# Noise Reducer

Removes background noise from speech recordings. A two-stage (dual-path)
transformer over time and frequency, trained on VoiceBank-DEMAND at 16 kHz.
Deep Learning course project.

## Results

Our best model, `noisereducer/runs/l1_expanded_data/best.pt` (877,489
parameters), on the official test set (824 files, speakers the model never
heard):

|         | PESQ  | STOI  | ESTOI | SI-SDR | SSNR  |
|---------|-------|-------|-------|--------|-------|
| noisy   | 1.967 | 0.921 | 0.787 | 8.45   | 1.55  |
| ours    | 3.146 | 0.951 | 0.871 | 18.85  | 9.65  |
| ceiling | 3.953 | 0.989 | 0.957 | 20.08  | 11.81 |

The ceiling is the clean magnitude with the noisy phase, the best any
magnitude mask can do.

## How to use

The three scripts are in `noisereducer/` and have no arguments: open one in
PyCharm and press Run, or run it from a terminal in the project folder. They
use an NVIDIA GPU or an Apple GPU if there is one, otherwise the CPU.

**1. Install** (Python 3.10 or newer):

    pip install -r requirements.txt

On Windows, see [Windows](#windows) below first.

**2. Clean your own recordings** (no dataset needed, the trained models are in
the repository):

    python noisereducer/run_inference.py

The first run creates `noisereducer/inference/noisy/`. Put your recordings
there (wav, flac, mp3, ogg, m4a...) and run it again. The cleaned files are
saved in `noisereducer/inference/clean/` as `<name>_clean.wav`.
On Windows and Linux, m4a/mp4/aac files need [ffmpeg](https://ffmpeg.org)
installed (macOS opens them without it).

**3. Download the dataset** (only for testing and training): get
VoiceBank-DEMAND from
[Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791) and unzip
`noisy_trainset_28spk_wav`, `clean_trainset_28spk_wav`, `noisy_testset_wav`
and `clean_testset_wav` into `noisereducer/data/voicebank_demand/16k/`.
The files are 48 kHz; they are converted to 16 kHz when read.

**4. Test** a model on the test set, which prints the table above (about 5
minutes on a laptop GPU):

    python noisereducer/run_eval.py

**5. Train:**

    python noisereducer/run_training.py

The run is saved in `noisereducer/runs/<RUN_NAME>/`. Stop it at any time and
run it again to continue.

In every script, `RUN_NAME` at the top chooses the run in
`noisereducer/runs/` (the default for testing and cleaning is
`l1_expanded_data`). All the settings are in `noisereducer/config.py`.

## Windows

On Windows, `pip install -r requirements.txt` installs everything except the
`pesq` package. pip has no ready-made `pesq` for Windows, and building it
needs the Microsoft C++ compiler, so requirements.txt skips it there. Without
`pesq`:

* `run_inference.py` (cleaning your recordings) works as usual,
* `run_eval.py` prints every score except PESQ, which shows as `n/a`,
* `run_training.py` stops with a message, because the best epoch is chosen
  by PESQ.

To get PESQ too, the easiest way is conda
([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda), which
has a ready-made `pesq` for Windows. In the Anaconda Prompt, in the project
folder:

    conda create -n noisereducer python=3.13
    conda activate noisereducer
    conda install -c conda-forge --no-deps pesq
    pip install -r requirements.txt

Keep `--no-deps`: it takes only `pesq` from conda and everything else from pip.
Without it conda also installs its own numpy, and a conda numpy next to the pip
torch crashes on Windows (two OpenMP libraries). With these four lines
inference, run_eval.py with PESQ, and training all work. In PyCharm, choose
this `noisereducer` conda environment as the project interpreter, then press
Run as usual.

Without conda, install
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
with the "Desktop development with C++" workload, and then `pip install pesq`.

On Windows, pip installs the CPU-only build of PyTorch, so even a computer
with an NVIDIA GPU would train on the CPU. If you have an NVIDIA GPU and want to
train, first install the CUDA build of PyTorch with the command that
[pytorch.org](https://pytorch.org/get-started/locally/) gives for Windows, pip
and CUDA, and only then run `pip install -r requirements.txt`. The training
prints "training on cuda" when the GPU is used.

Use `py` instead of `python` if the `python` command is not found. In PyCharm,
pick the environment you installed into as the project interpreter, then
press Run as usual.

On a computer without an NVIDIA GPU everything runs on the CPU. Cleaning a
recording takes a few seconds and `run_eval.py` about 40 minutes on a
laptop, but training is far too slow on a CPU (one epoch takes many hours).

## Our runs

All the runs have the same network. The three later runs start from epoch 16
of `baseline` and train 8 more epochs. In every run the model is the epoch
with the best PESQ on 200 validation utterances; the test set was scored only
after that choice.

| run (`noisereducer/runs/`) | what changed                                     | epoch | validation PESQ | test PESQ | test SI-SDR |
|----------------------------|--------------------------------------------------|-------|-----------------|-----------|-------------|
| `baseline`                 | the original run, 28 speakers, MSE loss           | 15    | 2.800           | 3.144     | 18.61       |
| `baseline_continued`       | 8 more epochs, same data and loss                 | 22    | 2.811           | 3.133     | 18.72       |
| `expanded_data`            | 8 more epochs, MSE loss and the 56spk files added | 19    | 2.826           | 3.149     | 18.64       |
| `l1_expanded_data`         | 8 more epochs, L1 loss and the 56spk files added  | 19    | 2.829           | 3.146     | 18.85       |

The PESQ differences are small (one run per setting). The clearer gain of
`l1_expanded_data` is in SI-SDR and SSNR. The full test scores of every run
are in its `test_scores.json`.

`config.py` has the settings of `baseline`. The three later runs were trained
with an earlier version of the code (the L1 loss and the extra data are not in
this version); their settings are in their `config.json`.

`noisereducer/__init__.py` lists what every file and folder does.
