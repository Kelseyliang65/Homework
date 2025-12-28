import csv
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SUMMARY_PATH = "nmt_comparison_summary.csv"
RNN_TEST_SUMMARY = os.path.join("rnn_test_results", "summary.csv")
TRANS_TEST_SUMMARY = os.path.join("transformer_test_results", "summary.csv")
OUTPUT_PATH = "nmt_best_model_comparison.png"


def read_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_best(rows, prefix):
    candidates = [r for r in rows if r["model"].startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r.get("best_bleu", 0) or 0))


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_test_bleu(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scores = {}
    for row in rows:
        model = row.get("model")
        if not model:
            continue
        scores[model] = _to_float(row.get("bleu"), 0.0)
    return scores


def plot_best_comparison(rnn_row, trans_row, test_bleu, out_path):
    labels = ["RNN (best)", "Transformer (best)"]
    best_bleu = [_to_float(rnn_row["best_bleu"]), _to_float(trans_row["best_bleu"])]
    test_bleu_vals = [
        _to_float(test_bleu.get(rnn_row["model"])),
        _to_float(test_bleu.get(trans_row["model"])),
    ]
    params_m = [_to_int(rnn_row["params"]) / 1e6, _to_int(trans_row["params"]) / 1e6]
    epoch_90 = [_to_int(rnn_row["epoch_to_90pct_best"]), _to_int(trans_row["epoch_to_90pct_best"])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300, constrained_layout=True)

    axes[0, 0].bar(labels, best_bleu, color=["#457B9D", "#E63946"])
    axes[0, 0].set_title("Best BLEU (Valid)")
    axes[0, 0].set_ylabel("BLEU")
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))

    axes[0, 1].bar(labels, test_bleu_vals, color=["#457B9D", "#E63946"])
    axes[0, 1].set_title("BLEU (Test)")
    axes[0, 1].set_ylabel("BLEU")
    axes[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=5))

    axes[1, 0].bar(labels, params_m, color=["#457B9D", "#E63946"])
    axes[1, 0].set_title("Model Size")
    axes[1, 0].set_ylabel("Params (M)")
    axes[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))

    axes[1, 1].bar(labels, epoch_90, color=["#457B9D", "#E63946"])
    axes[1, 1].set_title("Convergence Speed")
    axes[1, 1].set_ylabel("Epoch to 90% Best BLEU")
    axes[1, 1].yaxis.set_major_locator(MaxNLocator(nbins=5))

    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=10)

    fig.suptitle(
        f"Best on Valid Set (Selected Models)\nRNN: {rnn_row['model']} | Transformer: {trans_row['model']}",
        fontsize=11,
    )
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


def main():
    if not os.path.exists(SUMMARY_PATH):
        print(f"Missing {SUMMARY_PATH}. Run analyze_results.py first.")
        return

    rows = read_summary(SUMMARY_PATH)
    rnn_best = pick_best(rows, "rnn_")
    trans_best = pick_best(rows, "transformer_")

    if not rnn_best or not trans_best:
        print("Missing RNN or Transformer rows in summary.")
        return

    test_bleu = {}
    test_bleu.update(_read_test_bleu(RNN_TEST_SUMMARY))
    test_bleu.update(_read_test_bleu(TRANS_TEST_SUMMARY))
    plot_best_comparison(rnn_best, trans_best, test_bleu, OUTPUT_PATH)


if __name__ == "__main__":
    main()
