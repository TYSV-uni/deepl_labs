import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import f1_score
from tqdm import tqdm

from src import checks, config, data, net, runs


def settings(model, opt, loader, regime, seed):
    cfg = config.REGIMES[regime]
    return {
        "regime": regime,
        "seed": seed,
        "seeding": "torch.manual_seed(seed); np.random.seed(seed) before model construction",
        "weights": "ResNet18_Weights.IMAGENET1K_V1" if cfg["pretrained"] else None,
        "trainable_modules": "all" if cfg["trainable"] is None else cfg["trainable"],
        "eval_modules_during_training": cfg["eval"],
        "n_params_total": sum(p.numel() for p in model.parameters()),
        "n_params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "classes": config.CLASSES,
        "img_size": config.IMG_SIZE,
        "transforms": {"train": str(data.train_transform()),
                       "eval": str(data.eval_transform())},
        "normalize": {"mean": config.MEAN, "std": config.STD},
        "model": "torchvision.models.resnet18",
        "fc": "Linear(512, 6)",
        "bn": {"momentum": model.bn1.momentum, "eps": model.bn1.eps},
        "optimizer": {
            "name": "torch.optim.AdamW",
            "defaults": opt.defaults,
            "groups": [{"name": g["name"], "lr": g["lr"],
                        "params": sum(p.numel() for p in g["params"])}
                       for g in opt.param_groups],
        },
        "loss": "CrossEntropyLoss()",
        "epochs": config.EPOCHS,
        "steps_per_epoch": len(loader),
        "total_steps": config.EPOCHS * len(loader),
        "train_loader": {"batch_size": config.TRAIN_BATCH, "shuffle": True,
                         "drop_last": False, "num_workers": 0},
        "eval_loader": {"batch_size": config.EVAL_BATCH, "shuffle": False,
                        "num_workers": 0},
        "metric": "f1_score(average='macro')",
    }


def run_one(regime, seed, data_dir, split_path, runs_dir, device, force=False):
    run_dir = runs_dir / runs.run_name(regime, seed)
    split_hash = runs.sha256(split_path)
    log_path = run_dir / "run.json"
    if (not force and log_path.exists() and (run_dir / "model.pt").exists()
            and json.loads(log_path.read_text())["split_sha256"] == split_hash):
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    model = net.build_model(regime, seed, device)
    opt = net.build_optimizer(model, regime)
    train_loader = data.train_loader(data_dir, split_path)
    val_loader = data.eval_loader(data_dir, split_path, "validation")
    val_labels = np.array(val_loader.dataset.labels)
    n_train = len(train_loader.dataset)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss = []
    val_f1 = []
    bar = tqdm(
        total=config.EPOCHS * len(train_loader),
        desc=regime,
        bar_format="{desc} {percentage:.0f}%",
    )
    start = time.perf_counter()
    for _ in range(config.EPOCHS):
        net.set_train_mode(model, regime)
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            bar.update(1)
        train_loss.append(total / n_train)
        logits = net.predict_logits(model, val_loader, device)
        val_f1.append(float(f1_score(val_labels, logits.argmax(1), average="macro")))
    train_time = time.perf_counter() - start
    bar.close()

    commit, dirty = runs.git_info()
    log = {
        "regime": regime,
        "seed": seed,
        "settings": settings(model, opt, train_loader, regime, seed),
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "split_sha256": split_hash,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": runs.device_name(device),
        "train_loss": train_loss,
        "val_macro_f1": val_f1,
        "train_time_sec": train_time,
    }
    np.save(run_dir / "val_logits.npy", logits.astype(np.float32))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "model.pt")
    log_path.write_text(json.dumps(log, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--model", type=Path, default=Path("lab3_model.pt"))
    ap.add_argument("--force", action="store_true",
                    help="retrain all seven runs even if run.json and model.pt exist")
    args = ap.parse_args()

    split_path = args.out / "split.csv"
    rows = data.build_split(args.data)
    data.save_split(rows, split_path)
    if not checks.check_split(rows, args.out / "split_summary.csv"):
        sys.exit("split does not match the control table, see split_summary.csv")

    device = net.get_device()
    if not checks.run_all(args.data, split_path, device, args.out / "checks"):
        sys.exit("training mode checks failed, see results/checks")

    for regime, seed in config.RUNS:
        run_one(regime, seed, args.data, split_path, args.runs, device, args.force)

    regime, seed = runs.select_run(runs.load_runs(args.runs))
    runs.save_selected(args.runs, regime, seed, args.model)


if __name__ == "__main__":
    main()
