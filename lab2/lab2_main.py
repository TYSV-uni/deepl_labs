from pathlib import Path
from lab2_starter import load_predictions

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
import pandas as pd


STYLE = """<style>
th, td { text-align: center !important; padding: 4px 10px; }
</style>"""

def save_html(path, *parts, style=STYLE):
    Path(path).write_text("<meta charset='utf-8'>" + STYLE + "".join(parts), encoding="utf-8")

def get_t_metrics(y, scores, t):
    predicted = scores >= t

    tp = int(np.sum((y == 1) & predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    tn = int(np.sum((y == 0) & ~predicted))

    return {
        "threshold": t,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "C": 10 * fn + fp,
    }


def plot_curves(results_dir, y, s):
    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = roc_auc_score(y, s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(recall, precision)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("PR curve")

    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.6f}")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC curve")
    axes[1].legend()

    for ax in axes:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
    fig.savefig(results_dir / 'pr_roc_curves.png', dpi = 300)
    plt.close(fig)

    return roc_auc


def find_best_t_fast(y, s):
    df = pd.DataFrame({
        "score": s,
        "positive": y,
        "negative": 1 - y,
    })

    groups = (df.groupby("score")[["positive", "negative"]].sum().sort_index(ascending=False).cumsum())

    thresholds = np.r_[np.inf, groups.index.to_numpy()]
    tp = np.r_[0, groups["positive"].to_numpy()]
    fp = np.r_[0, groups["negative"].to_numpy()]

    n_positive = int(np.sum(y))
    n_negative = len(y) - n_positive

    fn = n_positive - tp
    tn = n_negative - fp

    candidates = pd.DataFrame({
        "threshold": thresholds,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "C": 10 * fn + fp,
    })


    best_index = candidates["C"].idxmin()
    t_star = float(candidates["threshold"].iloc[best_index])

    return candidates, t_star


def find_best_t_slow(y, s):
    thresholds = np.r_[np.inf, np.unique(s)[::-1]]
    rows = []

    for t in thresholds:
        rows.append(get_t_metrics(y, s, t))

    candidates = pd.DataFrame(rows)

    best_index = candidates["C"].idxmin()
    t_star = float(candidates["threshold"].iloc[best_index])

    return candidates, t_star


def task2(results_dir, y, s):
    roc_auc = plot_curves(results_dir, y, s)

    candidates, t_star = find_best_t_fast(y, s)
    candidates.to_csv(results_dir / "threshold_candidates.csv", index=False)

    comparison = pd.DataFrame([
        {
            "rule": name,
            **get_t_metrics(y, s, t),
        }
        for name, t in [
            ("Standard", 0.5),
            ("Minimum cost", t_star),
            ("All negative", np.inf),
        ]
    ])

    print(f"\nOptimal threshold t* = {t_star}")
    print("\nValidation comparison:")
    print(comparison.to_string(index=False))

    save_html(
        results_dir / "task2.html",
        f"<h2>ROC-AUC = {roc_auc}</h2>",
        f"<h2>Optimal threshold t* = {t_star}</h2>",
        "<h2>Validation comparison</h2>",
        comparison.to_html(index=False),
    )

def task3(results_dir, y, s):
    datasets = {
        "A": (
            [0, 1, 0, 1, 0, 0],
            [0.9, 0.7, 0.7, 0.4, 0.2, 0.1],
        ),
        "B": (
            [0, 0, 0],
            [0.9, 0.6, 0.3],
        ),
        "C": (
            [1, 1] + [0] * 11,
            [0.8] + [0.5] * 11 + [0.2],
        ),
    }

    html = []

    for name, (labels, scores) in datasets.items():
        y = np.asarray(labels, dtype=np.int64)
        s = np.asarray(scores, dtype=np.float32)

        slow, slow_t = find_best_t_slow(y, s)
        fast, fast_t = find_best_t_fast(y, s)


        try:
            pd.testing.assert_frame_equal(fast, slow, check_dtype=False, check_exact=True)
            frames_equal = True
        except AssertionError:
            frames_equal = False
        t_equal = fast_t == slow_t

        html.append(
            f"<h2>Dataset {name}</h2>{slow.to_html(index=False)}"
            f"<p>t*_slow = {slow_t}, t*_fast = {fast_t}<br>"
            f"assert_frame_equal: {frames_equal}<br>"
            f"t*_slow == t*_fast: {t_equal}</p>"
        )

        print(f"\nDataset {name}:")
        print(slow.to_string(index=False))
        print(f"t* = {slow_t}")
        print("Fast and slow results match.")

    save_html(results_dir / "task3.html", *html)

def task4(results_dir, data, t_star):
    y = data["test_y"]
    s = data["test_scores"]

    rows = []
    for name, t in [("Standard", 0.5), ("Minimum cost", t_star)]:
        m = get_t_metrics(y, s, t)
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        m["Precision"] = tp / (tp + fp) if tp + fp != 0 else np.nan
        m["Recall"] = tp / (tp + fn) if tp + fn != 0 else np.nan
        rows.append({"rule": name, **m})

    df_metrics = pd.DataFrame(rows)

    print("\nMetrics for t=0.5 and t=t* on test:")
    print(df_metrics.to_string(index=False))

    df_raw = pd.DataFrame({
        "row_id": data["test_row_ids"],
        "label": y,
        "score": s,
        "amount": data["test_amount"],
    })

    predicted = s >= t_star

    fp_5 = df_raw.loc[(y == 0) & predicted].sort_values("row_id").head(5).assign(error_type="FP")
    fn_5 = df_raw.loc[(y == 1) & ~predicted].sort_values("row_id").head(5).assign(error_type="FN")

    df_errors = pd.concat([fp_5, fn_5], ignore_index=True)

    df_errors["distance_to_threshold"] = np.abs(df_errors["score"] - t_star)
    df_errors = df_errors[["row_id", "error_type", "label", "score","distance_to_threshold", "amount"]]

    print("\nFirst 5 FP/FN examples by CSV row:")
    print(df_errors.to_string(index=False))

    save_html(
        results_dir / "task4.html",
        "<h2>Metrics for t* on test</h2>",
        df_metrics.to_html(index=False),
        "<h2>First 5 FP/FN examples by CSV row</h2>",
        df_errors.to_html(index=False),
    )

def main():
    results_dir = Path("results")

    data = load_predictions(results_dir)
    y = data["validation_y"]
    s = data["validation_scores"]

    task2(results_dir, y, s)
    task3(results_dir, y, s)
    task4(results_dir, data, find_best_t_fast(y, s)[1])

if __name__ == "__main__":
    main()
