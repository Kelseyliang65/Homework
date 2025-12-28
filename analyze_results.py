#!/usr/bin/env python3
"""
Parse training logs and checkpoints to summarize BLEU, params, and convergence.
Outputs:
  - nmt_comparison_summary.csv
  - nmt_comparison_summary.png
"""
import csv
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import torch


LOG_PATTERN = re.compile(r"Epoch\s+(\d+).*?Loss=([\d\.]+).*?BLEU=([\d\.]+)")


def parse_log(path):
    epochs, losses, bleus = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                bleus.append(float(match.group(3)))
    return epochs, losses, bleus


def count_params_from_ckpt(ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return sum(v.numel() for v in state.values())


def find_ckpt(model_tag):
    candidates = [os.path.join("ckpt", f"{model_tag}_best.pt")]
    if model_tag == "gru":
        candidates.append(os.path.join("ckpt", "gru_best.pt"))
    if model_tag == "transformer":
        candidates.append(os.path.join("ckpt", "transformer_best.pt"))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def summarize_log(model_tag, log_path):
    epochs, losses, bleus = parse_log(log_path)
    if not epochs:
        return None
    best_bleu = max(bleus)
    best_idx = bleus.index(best_bleu)
    best_epoch = epochs[best_idx]
    final_bleu = bleus[-1]
    target_bleu = best_bleu * 0.9
    epoch_to_90 = next((e for e, b in zip(epochs, bleus) if b >= target_bleu), None)
    ckpt = find_ckpt(model_tag)
    params = count_params_from_ckpt(ckpt)
    return {
        "model": model_tag,
        "log_path": log_path,
        "ckpt_path": ckpt or "",
        "epochs": len(epochs),
        "best_bleu": best_bleu,
        "best_epoch": best_epoch,
        "final_bleu": final_bleu,
        "epoch_to_90pct_best": epoch_to_90 if epoch_to_90 is not None else "",
        "params": params if params is not None else "",
        "training_time": "unknown",
        "epochs_list": epochs,
        "losses": losses,
        "bleus": bleus,
    }


def write_csv(rows, out_path):
    fieldnames = [
        "model",
        "log_path",
        "ckpt_path",
        "epochs",
        "best_bleu",
        "best_epoch",
        "final_bleu",
        "epoch_to_90pct_best",
        "params",
        "training_time",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_comparison(rows, out_path):
    if sns:
        sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "serif"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), dpi=300)

    # BLEU curves
    for row in rows:
        ax1.plot(row["epochs_list"], row["bleus"], marker="o", linewidth=2, label=row["model"])
    ax1.set_title("BLEU Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BLEU")
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(
            handles,
            labels,
            loc="upper left",
            fontsize=7,
            frameon=True,
        )

    # Best BLEU bars
    labels = [r["model"] for r in rows]
    bests = [r["best_bleu"] for r in rows]
    colors = ["#457B9D", "#E63946", "#2A9D8F", "#F4A261"]
    bars = ax2.bar(labels, bests, color=colors[: len(rows)])
    ax2.set_title("Best BLEU")
    ax2.set_ylabel("BLEU")
    ax2.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax2.set_xticklabels(labels, ha="right")
    if bests:
        ax2.set_ylim(0, max(bests) * 1.15)
    for bar, row in zip(bars, rows):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bests) * 0.02 if bests else bar.get_height(),
            f"@{row['best_epoch']}",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Convergence speed (epoch to 90% best BLEU)
    conv = [r["epoch_to_90pct_best"] or 0 for r in rows]
    ax3.bar(labels, conv, color=colors[: len(rows)])
    ax3.set_title("Convergence Speed")
    ax3.set_ylabel("Epoch to 90% Best BLEU")
    ax3.set_ylim(0, max(conv) + 1 if conv else 1)
    ax3.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax3.set_xticklabels(labels, ha="right")

    plt.tight_layout()
    plt.savefig(out_path)


def main():
    logs = [f for f in os.listdir(".") if f.endswith("_history.log")]
    rows = []
    for log in logs:
        model_tag = log.replace("_history.log", "")
        summary = summarize_log(model_tag, log)
        if summary:
            rows.append(summary)

    if not rows:
        print("No valid *_history.log files found.")
        return

    rows.sort(key=lambda r: r["model"])
    csv_path = "nmt_comparison_summary.csv"
    plot_path = "nmt_comparison_summary.png"
    write_csv(rows, csv_path)
    plot_comparison(rows, plot_path)

    print(f"Wrote {csv_path} and {plot_path}")
    for row in rows:
        print(
            f"{row['model']}: best BLEU {row['best_bleu']} (epoch {row['best_epoch']}), "
            f"final BLEU {row['final_bleu']}, params {row['params'] or 'unknown'}"
        )


if __name__ == "__main__":
    main()
