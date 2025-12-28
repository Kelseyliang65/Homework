#!/usr/bin/env python3
"""
Plot Transformer training BLEU curves from *_history.log files.
Output:
  - transformer_train_bleu_over_epochs.png
"""
import os
import re
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


LOG_PATTERN = re.compile(r"Epoch\s+(\d+).*?BLEU=([\d\.]+)")


def parse_bleu(path):
    epochs, bleus = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if match:
                epochs.append(int(match.group(1)))
                bleus.append(float(match.group(2)))
    return epochs, bleus


def main():
    logs = sorted(glob("transformer_*_history.log"))
    if not logs:
        print("No transformer_*_history.log files found.")
        return

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    cmap = plt.get_cmap("tab10")

    plotted = 0
    for idx, path in enumerate(logs):
        epochs, bleus = parse_bleu(path)
        if not epochs:
            continue
        label = os.path.basename(path).replace("_history.log", "")
        ax.plot(
            epochs,
            bleus,
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=label,
            color=cmap(idx % 10),
        )
        plotted += 1

    if not plotted:
        print("No BLEU entries found in Transformer logs.")
        return

    ax.set_title("Transformer Ablations: BLEU over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=7, ncol=1)
    plt.tight_layout()
    out_path = "transformer_train_bleu_over_epochs.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
