#!/usr/bin/env python3
"""
Batch-run multiple RNN configurations and summarize results.
Usage:
  python rnn_sweep.py
"""
import os
import subprocess
import sys


DEVICE = "cuda"
RESUME = True

RUNS = [
    {"cell": "gru", "attention": "dot_product", "teacher_forcing": 0.5, "decode": "greedy"},
    {"cell": "gru", "attention": "multiplicative", "teacher_forcing": 0.5, "decode": "greedy"},
    {"cell": "gru", "attention": "additive", "teacher_forcing": 0.5, "decode": "greedy"},
    {"cell": "lstm", "attention": "dot_product", "teacher_forcing": 0.5, "decode": "greedy"},
    {"cell": "lstm", "attention": "additive", "teacher_forcing": 0.5, "decode": "greedy"},
    {"cell": "gru", "attention": "dot_product", "teacher_forcing": 0.0, "decode": "greedy"},
]


def run_one(cfg, max_retries=1):
    cmd = [
        sys.executable,
        "main.py",
        "--model",
        "rnn",
        "--cell",
        cfg["cell"],
        "--attention",
        cfg["attention"],
        "--teacher-forcing",
        str(cfg["teacher_forcing"]),
        "--decode",
        cfg["decode"],
        "--device",
        DEVICE,
    ]
    if RESUME:
        cmd.append("--resume")
    print("Running:", " ".join(cmd))
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            last_err = e
            print(f"Run failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
    return False


def main():
    completed = []
    failed = []
    for cfg in RUNS:
        tag = f"rnn_{cfg['cell']}_{cfg['attention']}_tf{cfg['teacher_forcing']:.2f}_{cfg['decode']}"
        log_name = f"{tag}_history.log"
        if os.path.exists(log_name) and os.path.getsize(log_name) > 0 and not RESUME:
            print(f"Skipping completed run (log exists): {log_name}")
            completed.append(tag)
            continue
        ok = run_one(cfg, max_retries=1)
        if ok:
            completed.append(tag)
        else:
            failed.append(tag)
    subprocess.run([sys.executable, "analyze_results.py"], check=True)
    print("Sweep complete. See nmt_comparison_summary.csv and nmt_comparison_summary.png")
    if failed:
        print("Failed runs:", ", ".join(failed))


if __name__ == "__main__":
    main()
