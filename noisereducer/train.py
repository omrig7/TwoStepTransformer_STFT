"""The training loop.

Every epoch we:
  1. train on the training speakers,
  2. measure the loss on the validation speakers,
  3. compute PESQ and STOI on 200 validation utterances,
  4. save the model if its PESQ is the best so far (best.pt),
  5. save everything needed to continue later (last.pt).

We keep the epoch with the best PESQ and not the one with the lowest loss,
because late in training the loss keeps going down while the sound quality
does not improve any more.

On an NVIDIA GPU the network runs in float16 (about twice as fast). float16
gradients can become so small they turn into zero, so a GradScaler multiplies
the loss before backward() and divides the gradients back before the step. On
a Mac or CPU everything stays in float32 and the scaler does nothing.
"""

import json
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from noisereducer.config import ROOT_DIR
from noisereducer.data import make_loaders
from noisereducer.enhancement import enhance, pick_device
from noisereducer.loss import EnhancementLoss
from noisereducer.metrics import PESQ_MISSING, as_array, pesq_available, pesq_score, stoi_score
from noisereducer.model import NoiseReducer


def run_epoch(model, loss_fn, loader, device, optimizer=None, scaler=None,
              clip_norm=5.0, label=""):
    """One pass over ``loader``. Trains if an optimizer is given, otherwise it
    only measures. Returns the average loss."""
    training = optimizer is not None
    model.train(training)
    half = device.type == "cuda"

    total, batches = 0.0, 0
    with torch.set_grad_enabled(training):
        for noisy, clean, lengths in tqdm(loader, desc=label, leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            # If no file in the batch is padded we can skip all the masking.
            if lengths.min() == noisy.shape[-1]:
                lengths = None
            else:
                lengths = lengths.to(device)

            with torch.autocast(device.type, dtype=torch.float16, enabled=half):
                loss = loss_fn(model(noisy, lengths), clean, lengths).total

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # clip the real gradient, not the scaled one
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)  # skips the step if a gradient is inf/nan
                scaler.update()

            total += loss.item()
            batches += 1
    return total / max(batches, 1)


def score_model(model, loader, device, sample_rate=16_000):
    """Average PESQ and STOI over whole utterances (batch size 1).

    Pass ``model=None`` to score the noisy input itself, which is the
    "do nothing" reference to compare against.
    """
    if model is not None:
        model.eval()
    pesq_values, stoi_values = [], []
    for noisy, clean, lengths in tqdm(loader, desc="scoring", leave=False):
        noisy, clean = noisy[0, :lengths[0]], clean[0, :lengths[0]]
        output = noisy if model is None else enhance(model, noisy, sample_rate)
        reference, output = as_array(clean), as_array(output)
        pesq_values.append(pesq_score(reference, output, sample_rate))
        stoi_values.append(stoi_score(reference, output, sample_rate))
    return float(np.nanmean(pesq_values)), float(np.nanmean(stoi_values))


def train(config, run_name):
    """Train in runs/<run_name>. If that run already has a last.pt we continue
    it from its last finished epoch, otherwise a new run starts there."""
    # The best epoch is chosen by PESQ, so training cannot run without it.
    if not pesq_available():
        raise RuntimeError(PESQ_MISSING)
    run_dir = ROOT_DIR / "runs" / run_name
    continuing = (run_dir / "last.pt").exists()
    if not continuing:
        print(f"starting a new run in {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    device = pick_device()
    settings = config.train

    train_loader, validation_loader, score_loader = make_loaders(config)

    torch.manual_seed(settings.seed)
    model = NoiseReducer(config).to(device)
    loss_fn = EnhancementLoss(model.stft, model.features, config.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=settings.lr_factor,
        patience=settings.lr_patience, min_lr=settings.lr_minimum)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best_pesq, first_epoch = [], -1.0, 1
    if continuing:
        state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        history, best_pesq, first_epoch = state["history"], state["best"], state["epoch"] + 1
        print(f"continuing {run_dir} from epoch {first_epoch} (best PESQ so far {best_pesq:.3f})")

    parameters = sum(p.numel() for p in model.parameters())
    print(f"{parameters:,} parameters, training on {device}")
    print(f"{len(train_loader)} training batches, {len(validation_loader)} validation batches")
    noisy_pesq, noisy_stoi = score_model(None, score_loader, device)
    print(f"noisy input (no model): PESQ {noisy_pesq:.3f}  STOI {noisy_stoi:.3f}")

    for epoch in range(first_epoch, settings.epochs + 1):
        started = time.time()
        label = f"epoch {epoch}/{settings.epochs}"
        train_loss = run_epoch(model, loss_fn, train_loader, device, optimizer, scaler,
                               settings.clip_norm, f"{label} train")
        validation_loss = run_epoch(model, loss_fn, validation_loader, device,
                                    label=f"{label} validation")
        pesq, stoi = score_model(model, score_loader, device)

        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(pesq)  # lowers the learning rate if PESQ stopped improving

        improved = pesq > best_pesq
        if improved:
            best_pesq = pesq
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "config": config.to_dict(), "pesq": pesq}, run_dir / "best.pt")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss, "pesq": pesq, "stoi": stoi,
                        "learning_rate": learning_rate,
                        "minutes": (time.time() - started) / 60})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimiser": optimizer.state_dict(), "schedule": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "best": best_pesq, "history": history,
                    "config": config.to_dict()}, run_dir / "last.pt")

        print(f"{label}  train {train_loss:.3e}  validation {validation_loss:.3e}  "
              f"PESQ {pesq:.3f}  STOI {stoi:.3f}  lr {learning_rate:.1e}  "
              f"{history[-1]['minutes']:.0f} min" + ("  <- best" if improved else ""))

    print(f"done, best PESQ {best_pesq:.3f} (noisy input {noisy_pesq:.3f}), "
          f"model saved in {run_dir / 'best.pt'}")
