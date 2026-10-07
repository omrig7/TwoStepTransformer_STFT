"""Clean your own recordings with a trained model: press Run in PyCharm.

The first run creates the folder inference/ with two folders in it:
    inference/noisy/   put the recordings you want to clean here
                       (wav, mp3, m4a...)
    inference/clean/   the cleaned recordings are saved here, as 16 kHz wav
                       files with the same names

RUN_NAME is the folder in runs/ whose best.pt is used. The default,
"baseline", is our best model. A real recording has no clean version to
compare with, so nothing is scored here; run_eval.py does the scoring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "import noisereducer" works

from noisereducer.config import ROOT_DIR
from noisereducer.enhancement import (best_checkpoint, enhance_folder,
                                      list_audio_files, load_model, pick_device)

RUN_NAME = "baseline"

INFERENCE_DIR = ROOT_DIR / "inference"
NOISY_DIR = INFERENCE_DIR / "noisy"
CLEAN_DIR = INFERENCE_DIR / "clean"


def main():
    NOISY_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    if not list_audio_files(NOISY_DIR):
        print(f"{NOISY_DIR} is empty, put the recordings you want to clean there "
              f"and press Run again")
        return

    model = load_model(best_checkpoint(RUN_NAME), pick_device())
    count = enhance_folder(model, NOISY_DIR, CLEAN_DIR)
    print(f"saved {count} cleaned files to {CLEAN_DIR}")


if __name__ == "__main__":
    main()
