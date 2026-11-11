import hashlib
import json
import platform
import subprocess

import torch

from src import config


def run_name(regime, seed):
    return f"{regime}_seed{seed}"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return commit, dirty


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return "mps"
    return platform.processor() or "cpu"


def load_runs(runs_dir):
    return [json.loads((runs_dir / run_name(r, s) / "run.json").read_text())
            for r, s in config.RUNS]


def select_run(logs):
    means = {}
    for regime in config.REGIMES:
        f = [l["val_macro_f1"][-1] for l in logs if l["regime"] == regime]
        means[regime] = sum(f) / len(f)
    regime = max(means, key=means.get)
    candidates = [l for l in logs if l["regime"] == regime]
    best = max(candidates, key=lambda l: l["val_macro_f1"][-1])
    return regime, best["seed"]


def save_selected(runs_dir, regime, seed, path):
    weights = torch.load(runs_dir / run_name(regime, seed) / "model.pt", map_location="cpu")
    torch.save({
        "state_dict": weights,
        "regime": regime,
        "seed": seed,
        "classes": config.CLASSES,
        "img_size": config.IMG_SIZE,
    }, path)
