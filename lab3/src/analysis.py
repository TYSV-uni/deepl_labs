import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from src import config, data, net, runs
from src.tables import write_csv


def check_hashes(logs, split_path):
    hashes = {l["split_sha256"] for l in logs}
    if hashes != {runs.sha256(split_path)}:
        sys.exit("split.csv hash mismatch: the runs were trained on a different split, "
                 "retrain all seven runs")


def save_runs_table(logs, path):
    rows = [
        (i + 1, l["regime"], l["seed"], l["train_loss"][-1],
         l["val_macro_f1"][-1], l["train_time_sec"])
        for i, l in enumerate(logs)
    ]
    write_csv(path, ["run", "regime", "seed", "train_loss_ep8",
                     "val_macro_f1_ep8", "train_time_sec"], rows)


def save_curves(logs, path):
    colors = {"scratch": "tab:gray", "frozen": "tab:blue", "unfreeze": "tab:orange"}
    styles = {1: "-", 2: "--", 3: ":"}
    epochs = list(range(1, len(logs[0]["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for l in logs:
        kw = dict(color=colors[l["regime"]], linestyle=styles[l["seed"]], marker="o",
                  markersize=3, label=f'{l["regime"]} seed {l["seed"]}')
        axes[0].plot(epochs, l["train_loss"], **kw)
        axes[1].plot(epochs, l["val_macro_f1"], **kw)
    axes[0].set_ylabel("train loss")
    axes[1].set_ylabel("validation macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_effects(logs):
    f = {r: [l["val_macro_f1"][-1] for l in logs if l["regime"] == r]
         for r in config.REGIMES}
    scratch = f["scratch"][0]
    mean_frozen = float(np.mean(f["frozen"]))
    mean_unfreeze = float(np.mean(f["unfreeze"]))
    range_frozen = max(f["frozen"]) - min(f["frozen"])
    range_unfreeze = max(f["unfreeze"]) - min(f["unfreeze"])
    return {
        "scratch": scratch,
        "mean_frozen": mean_frozen,
        "mean_unfreeze": mean_unfreeze,
        "range_frozen": range_frozen,
        "range_unfreeze": range_unfreeze,
        "R": max(range_frozen, range_unfreeze),
        "delta_pre": mean_frozen - scratch,
        "delta_unf": mean_unfreeze - mean_frozen,
        "delta_total": mean_unfreeze - scratch,
    }


def save_effects(e, out):
    write_csv(out / "group_stats.csv", ["regime", "n_runs", "mean_macro_f1", "range"], [
        ("scratch", 1, e["scratch"], ""),
        ("frozen", 3, e["mean_frozen"], e["range_frozen"]),
        ("unfreeze", 3, e["mean_unfreeze"], e["range_unfreeze"]),
    ])
    rows = []
    for name in ["delta_pre", "delta_unf", "delta_total"]:
        v = e[name]
        if name == "delta_total":
            verdict = ""
        else:
            verdict = "within spread" if abs(v) <= e["R"] else "exceeds spread"
        ratio = abs(v) / e["R"] if e["R"] > 0 else ""
        rows.append((name, v, e["R"], ratio, verdict))
    write_csv(out / "effects.csv", ["effect", "value", "R", "abs_value_over_R", "verdict"], rows)


def check_selection(logs, model_path, e):
    regime, seed = runs.select_run(logs)
    ckpt = torch.load(model_path, map_location="cpu")
    if ckpt["regime"] != regime or ckpt["seed"] != seed:
        sys.exit(f"{model_path} is {ckpt['regime']} seed {ckpt['seed']}, but the selection "
                 f"rule over the logs gives {regime} seed {seed}")
    val = next(l for l in logs if l["regime"] == regime and l["seed"] == seed)[
        "val_macro_f1"][-1]
    return ckpt, val, abs(e["delta_unf"]) <= e["R"]


def evaluate_test(data_dir, split_path, ckpt):
    device = torch.device("cpu")
    model = net.new_resnet(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    loader = data.eval_loader(data_dir, split_path, "test")
    logits = net.predict_logits(model, loader, device)
    ds = loader.dataset
    return logits, np.array(ds.labels), ds.files, ds.paths


def save_confusion(cm, path_csv, path_png):
    write_csv(path_csv, ["true\\predicted"] + config.CLASSES,
              [[c] + [int(v) for v in row] for c, row in zip(config.CLASSES, cm)])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(config.CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(config.CLASSES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def characteristic_errors(cm, y_true, y_pred, probs, files):
    pairs = [(t, p) for t in range(6) for p in range(6) if t != p and cm[t, p] > 0]
    pairs.sort(key=lambda tp: (-cm[tp], tp[0], tp[1]))
    by_pair = {
        pair: sorted(
            (i for i in range(len(files)) if y_true[i] == pair[0] and y_pred[i] == pair[1]),
            key=lambda i: files[i])
        for pair in pairs
    }
    chosen = []
    k = 0
    max_k = max(len(v) for v in by_pair.values())
    while len(chosen) < 5 and k < max_k:
        for pair in pairs:
            if len(chosen) == 5:
                break
            if len(by_pair[pair]) > k:
                chosen.append((pair, by_pair[pair][k]))
        k += 1
    return [
        (files[i], config.CLASSES[t], config.CLASSES[p], float(probs[i, p]), int(cm[t, p]), i)
        for (t, p), i in chosen
    ]


def save_errors(rows, paths, out):
    write_csv(out / "errors.csv", [
        "file", "true_class", "predicted_class", "predicted_prob", "n_such_errors"],
        [r[:5] for r in rows])
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 5.2))
    for ax, (file, true, pred, prob, _, i) in zip(np.atleast_1d(axes), rows):
        ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.set_title(f"{file}\n{true} -> {pred}\np = {prob:.2f}", fontsize=10, pad=8)
        ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.savefig(out / "errors.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--model", type=Path, default=Path("lab3_model.pt"))
    args = ap.parse_args()

    split_path = args.out / "split.csv"
    out = args.out / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    logs = runs.load_runs(args.runs)
    check_hashes(logs, split_path)

    save_runs_table(logs, out / "runs_table.csv")
    save_curves(logs, out / "curves.png")
    e = compute_effects(logs)
    save_effects(e, out)

    ckpt, val_f1, by_noise = check_selection(logs, args.model, e)
    write_csv(out / "selection.csv",
              ["regime", "seed", "val_macro_f1", "selection_decided_by_noise"],
              [(ckpt["regime"], ckpt["seed"], val_f1, "yes" if by_noise else "no")])

    logits, y_true, files, paths = evaluate_test(args.data, split_path, ckpt)
    y_pred = logits.argmax(1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    test_f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float((y_true == y_pred).mean())
    n = len(y_true)
    write_csv(out / "test_summary.csv", [
        "test_macro_f1", "test_accuracy", "n_test", "accuracy_std_error",
        "val_macro_f1_selected", "val_minus_test_macro_f1"],
        [(test_f1, acc, n, float(np.sqrt(acc * (1 - acc) / n)), val_f1, val_f1 - test_f1)])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    save_confusion(cm, out / "confusion_matrix.csv", out / "confusion_matrix.png")
    save_errors(characteristic_errors(cm, y_true, y_pred, probs, files), paths, out)


if __name__ == "__main__":
    main()
