#!/usr/bin/env python3
"""
Batch-run Transformer ablations: position encoding x normalization x hyperparams.
Usage:
  python transformer_sweep.py
"""
import os
import subprocess
import sys


DEVICE = "cuda"
RESUME = False

POS_ENCODINGS = ["relative", "absolute"]
NORMS = ["rmsnorm", "layernorm"]
EXPERIMENTS = [
    # A) Architecture ablations (4)
    {"pos_encoding": "relative", "norm": "rmsnorm", "batch": 64, "lr": 5e-4,
     "d_model": 512, "heads": 8, "enc_layers": 3, "dec_layers": 3, "d_ff": 2048},
    {"pos_encoding": "relative", "norm": "layernorm", "batch": 64, "lr": 5e-4,
     "d_model": 512, "heads": 8, "enc_layers": 3, "dec_layers": 3, "d_ff": 2048},
    {"pos_encoding": "absolute", "norm": "rmsnorm", "batch": 64, "lr": 5e-4,
     "d_model": 512, "heads": 8, "enc_layers": 3, "dec_layers": 3, "d_ff": 2048},
    {"pos_encoding": "absolute", "norm": "layernorm", "batch": 64, "lr": 5e-4,
     "d_model": 512, "heads": 8, "enc_layers": 3, "dec_layers": 3, "d_ff": 2048},
    # B) Hyperparameter sensitivity (3)
    {"pos_encoding": "relative", "norm": "rmsnorm", "batch": 32, "lr": 5e-4,
     "d_model": 512, "heads": 8, "enc_layers": 3, "dec_layers": 3, "d_ff": 2048},
    {"pos_encoding": "relative", "norm": "rmsnorm", "batch": 64, "lr": 1e-4,
     "d_model": 512, "heads": 8, "enc_layers": 3, "dec_layers": 3, "d_ff": 2048},
    {"pos_encoding": "relative", "norm": "rmsnorm", "batch": 64, "lr": 5e-4,
     "d_model": 256, "heads": 4, "enc_layers": 2, "dec_layers": 2, "d_ff": 1024},
]


def model_tag(cfg):
    return (
        f"transformer_{cfg['pos_encoding']}_{cfg['norm']}"
        f"_d{cfg['d_model']}_h{cfg['heads']}"
        f"_e{cfg['enc_layers']}{cfg['dec_layers']}"
        f"_ff{cfg['d_ff']}_bs{cfg['batch']}_lr{cfg['lr']}"
    )


def run_one(cfg, max_retries=1):
    tag = model_tag(cfg)
    log_name = f"{tag}_history.log"
    if os.path.exists(log_name) and os.path.getsize(log_name) > 0 and not RESUME:
        print(f"Skipping completed run (log exists): {log_name}")
        return True

    cmd = [
        sys.executable,
        "main.py",
        "--model",
        "transformer",
        "--t-pos-encoding",
        cfg["pos_encoding"],
        "--t-norm",
        cfg["norm"],
        "--t-d-model",
        str(cfg["d_model"]),
        "--t-heads",
        str(cfg["heads"]),
        "--t-enc-layers",
        str(cfg["enc_layers"]),
        "--t-dec-layers",
        str(cfg["dec_layers"]),
        "--t-d-ff",
        str(cfg["d_ff"]),
        "--batch",
        str(cfg["batch"]),
        "--lr",
        str(cfg["lr"]),
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
    completed = 0
    failed = []
    for cfg in EXPERIMENTS:
        ok = run_one(cfg, max_retries=1)
        if ok:
            completed += 1
        else:
            failed.append(model_tag(cfg))

    subprocess.run([sys.executable, "analyze_results.py"], check=True)
    print(f"Sweep complete. Total finished: {completed}")
    if failed:
        print("Failed runs:", ", ".join(failed))


if __name__ == "__main__":
    main()
