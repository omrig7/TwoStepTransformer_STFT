"""Train the model: open this file in PyCharm and press Run.

It uses the settings in config.py, which are the settings of our original run
runs/baseline (PESQ 2.80 on the validation speakers). Everything is saved in runs/<RUN_NAME>:

    best.pt       weights of the epoch with the best PESQ (inference uses this)
    last.pt       the latest state of everything, to continue a stopped run
    history.json  losses, PESQ, STOI and learning rate of every epoch
    config.json   the settings the run used

RUN_NAME is the name of the folder in runs/:
  * if runs/<RUN_NAME> already exists (for example "baseline"), the training
    continues from its last finished epoch,
  * if it does not exist, a new run starts in a new folder with that name.
So you can stop the training at any time and press Run again to continue.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "import noisereducer" works

from noisereducer.config import Config
from noisereducer.metrics import PESQ_MISSING, pesq_available
from noisereducer.train import train

# "baseline" continues our 2.8 run (from epoch 17), a new name starts a new run.
RUN_NAME = "my_run"

def main():
    if not pesq_available():
        print(f"cannot train: {PESQ_MISSING}")
        return
    config = Config()
    # To try something for one run only, change it here, for example:
    # config.train.epochs = 20
    # config.data.workers = 0   # simpler to debug, but slower
    train(config, RUN_NAME)


# The data loader starts worker processes, and on macOS they re-import this
# file, so main() must only be called inside this "if".
if __name__ == "__main__":
    main()
