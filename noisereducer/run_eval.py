"""Score a trained model on the VoiceBank-DEMAND test set: press Run in PyCharm.

This is the test the article reports: the 824 test files, from 2 speakers the
model never heard in training. It prints PESQ, STOI, ESTOI, SI-SDR and SSNR
for the noisy input, for our output and for the ceiling of our method.

RUN_NAME is the folder in runs/ whose best.pt is scored. The default,
"baseline", is our best model (epoch 15, PESQ 2.80 on the validation speakers).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "import noisereducer" works

from noisereducer.config import Config
from noisereducer.enhancement import best_checkpoint, load_model, pick_device
from noisereducer.evaluation import evaluate, print_scores

RUN_NAME = "baseline"


def main():
    data = Config().data
    noisy_dir = data.root / data.noisy_test
    clean_dir = data.root / data.clean_test
    if not noisy_dir.exists():
        raise FileNotFoundError(f"{noisy_dir} does not exist, download the dataset first "
                                f"(see README.md)")

    model = load_model(best_checkpoint(RUN_NAME), pick_device())
    averages = evaluate(model, noisy_dir, clean_dir)
    print_scores(averages)


if __name__ == "__main__":
    main()
