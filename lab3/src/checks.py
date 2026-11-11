import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src import config, data, net
from src.tables import write_csv

EXPECTED = {
    ("buildings", "train"): (200, 1996761),
    ("buildings", "validation"): (218, 4837323),
    ("buildings", "test"): (219, 4837559),
    ("forest", "train"): (200, 2032625),
    ("forest", "validation"): (237, 5276962),
    ("forest", "test"): (237, 5291326),
    ("glacier", "train"): (200, 2183923),
    ("glacier", "validation"): (276, 6135652),
    ("glacier", "test"): (277, 6157250),
    ("mountain", "train"): (200, 1950345),
    ("mountain", "validation"): (262, 5789535),
    ("mountain", "test"): (263, 5849765),
    ("sea", "train"): (200, 1987613),
    ("sea", "validation"): (255, 5655068),
    ("sea", "test"): (255, 5668011),
    ("street", "train"): (200, 1841817),
    ("street", "validation"): (250, 5541845),
    ("street", "test"): (251, 5547533),
    ("total", "train"): (1200, 11993084),
    ("total", "validation"): (1498, 33236385),
    ("total", "test"): (1502, 33351444),
}
EXPECTED_TOTAL_PARAMS = 11179590
EXPECTED_TRAINABLE = {"scratch": 11179590, "frozen": 3078, "unfreeze": 8396806}
MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def yn(v):
    return "yes" if v else "no"


def check_split(rows, path):
    cells = {}
    for file, label, part in rows:
        n = int(Path(file).stem)
        for key in ((label, part), ("total", part)):
            count, id_sum = cells.get(key, (0, 0))
            cells[key] = (count + 1, id_sum + n)
    out = []
    for key, (exp_count, exp_sum) in EXPECTED.items():
        count, id_sum = cells.get(key, (0, 0))
        out.append((*key, count, id_sum, exp_count, exp_sum,
                    yn(count == exp_count and id_sum == exp_sum)))
    write_csv(path, ["label", "part", "count", "id_sum", "expected_count",
                     "expected_id_sum", "match"], out)
    files = [r[0] for r in rows]
    return all(r[-1] == "yes" for r in out) and len(files) == len(set(files))


def save_batch_figure(data_dir, split_path, path):
    torch.manual_seed(1)
    x, y = next(iter(data.train_loader(data_dir, split_path)))
    mean = torch.tensor(config.MEAN).view(1, 3, 1, 1)
    std = torch.tensor(config.STD).view(1, 3, 1, 1)
    imgs = (x[:16] * std + mean).clamp(0, 1)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img, label in zip(axes.flat, imgs, y[:16]):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(config.CLASSES[label])
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def initial_losses(data_dir, split_path, device, path):
    criterion = torch.nn.CrossEntropyLoss()
    rows = []
    for regime in ["scratch", "frozen"]:
        model = net.build_model(regime, 1, device)
        loader = data.train_loader(data_dir, split_path)
        net.set_train_mode(model, regime)
        x, y = next(iter(loader))
        with torch.no_grad():
            loss = criterion(model(x.to(device)), y.to(device)).item()
        rows.append((regime, loss, math.log(6), loss - math.log(6)))
    write_csv(path, ["regime", "initial_loss", "ln6", "diff"], rows)


def parameter_report(device):
    rows = []
    for regime in config.REGIMES:
        model = net.build_model(regime, 1, device)
        opt = net.build_optimizer(model, regime)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        clean = all(p.requires_grad for p in model.parameters() if id(p) in in_opt)
        matches = total == EXPECTED_TOTAL_PARAMS and trainable == EXPECTED_TRAINABLE[regime]
        for g in opt.param_groups:
            rows.append((regime, total, trainable, g["name"],
                         sum(p.numel() for p in g["params"]), g["lr"],
                         yn(clean), yn(matches)))
    return rows


def run_five_steps(regime, data_dir, split_path, device, train_mode):
    model = net.build_model(regime, 1, device)
    opt = net.build_optimizer(model, regime)
    loader = data.train_loader(data_dir, split_path)
    criterion = torch.nn.CrossEntropyLoss()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if train_mode == "set_train_mode":
        net.set_train_mode(model, regime)
    else:
        model.train()
    for i, (x, y) in enumerate(loader):
        if i == 5:
            break
        opt.zero_grad()
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
    return model, before, model.state_dict()


def change_report(regime, variant, model, before, after):
    param_names = {n for n, _ in model.named_parameters()}
    frozen_modules = config.REGIMES[regime]["eval"]
    rows = []
    for m in MODULES:
        keys = [k for k in before if k.split(".")[0] == m]
        p_keys = [k for k in keys if k in param_names]
        b_keys = [k for k in keys if k.split(".")[-1] in BUFFERS]
        p_changed = any(not torch.equal(before[k], after[k]) for k in p_keys)
        has_bn = len(b_keys) > 0
        s_changed = any(not torch.equal(before[k], after[k]) for k in b_keys)
        if m in frozen_modules:
            ok = not p_changed and not s_changed
        else:
            ok = p_changed and (s_changed if has_bn else True)
        rows.append((regime, variant, m, yn(p_changed),
                     yn(s_changed) if has_bn else "no BN", yn(ok)))
    return rows


def bn_change(variant, before, after):
    mean_key = "layer1.0.bn1.running_mean"
    var_key = "layer1.0.bn1.running_var"
    return (
        variant,
        (after[mean_key] - before[mean_key]).abs().max().item(),
        (after[var_key] - before[var_key]).abs().max().item(),
    )


def mode_checks(data_dir, split_path, device, out):
    param_rows = parameter_report(device)
    write_csv(out / "params.csv", [
        "regime", "total_params", "trainable_params", "group", "group_params", "lr",
        "no_frozen_in_optimizer", "matches_table"], param_rows)

    change_rows = []
    bn_rows = []
    for regime in ["frozen", "unfreeze"]:
        model, before, after = run_five_steps(
            regime, data_dir, split_path, device, "set_train_mode")
        change_rows += change_report(regime, "set_train_mode", model, before, after)
        if regime == "frozen":
            bn_rows.append(bn_change("set_train_mode", before, after))

    model, before, after = run_five_steps(
        "frozen", data_dir, split_path, device, "model_train")
    change_rows += change_report("frozen", "model_train", model, before, after)
    bn_rows.append(bn_change("model_train", before, after))

    write_csv(out / "changes.csv", [
        "regime", "variant", "module", "params_changed", "bn_stats_changed",
        "matches_expected"], change_rows)
    write_csv(out / "bn_layer1_0_bn1.csv", [
        "variant", "max_abs_change_running_mean", "max_abs_change_running_var"], bn_rows)

    return (all(r[6] == "yes" and r[7] == "yes" for r in param_rows)
            and all(r[-1] == "yes" for r in change_rows if r[1] == "set_train_mode"))


def run_all(data_dir, split_path, device, out):
    save_batch_figure(data_dir, split_path, out / "batch.png")
    initial_losses(data_dir, split_path, device, out / "initial_loss.csv")
    return mode_checks(data_dir, split_path, device, out)
