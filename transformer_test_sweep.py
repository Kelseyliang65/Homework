#!/usr/bin/env python3
"""
Evaluate all Transformer checkpoints on the test set and compare results.
Outputs:
  - transformer_test_results/summary.csv
  - transformer_test_results/bleu_comparison.png
  - transformer_test_results/sample_comparisons.jsonl
  - transformer_test_results/<model_tag>_test_results.jsonl
"""
import argparse
import json
import os
from typing import List, Dict

import sacrebleu
import torch
import matplotlib.pyplot as plt

from inference import load_transformer, translate_sentence
from settings import DATA_CONFIG


DEMO_SAMPLES = [
    "今天天气不错，我们去公园散步吧。",
    "这款手机的电池续航比我想象的要好。",
    "如果明天还下雨，比赛就会延期。",
    "他决定辞职去读研究生，这是个很大胆的选择。",
    "请把会议改到下周三下午三点。",
    "这家餐厅味道很好，但是价格有点高。",
    "我们需要在月底前提交项目报告。",
    "机器学习模型对数据质量非常敏感。",
]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_test_data(test_file: str, src_field: str, tgt_field: str) -> List[Dict[str, str]]:
    rows = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            src = data.get(src_field, data.get("zh", ""))
            tgt = data.get(tgt_field, data.get("en", ""))
            if not src or not tgt:
                continue
            rows.append({"src": src, "tgt": tgt})
    return rows


def read_sample_data(sample_mode: str, sample_file: str, test_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if sample_mode == "demo":
        return [{"src": s, "tgt": ""} for s in DEMO_SAMPLES]
    if sample_mode == "file":
        if not sample_file:
            raise RuntimeError("sample_mode=file requires --sample-file.")
        rows = []
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    rows.append({"src": text, "tgt": ""})
        return rows
    return test_rows


def list_transformer_checkpoints(ckpt_dir: str) -> List[str]:
    return sorted(
        [
            os.path.join(ckpt_dir, f)
            for f in os.listdir(ckpt_dir)
            if f.startswith("transformer_") and f.endswith("_best.pt")
        ]
    )


def model_tag_from_ckpt(path: str) -> str:
    base = os.path.basename(path)
    return base.replace("_best.pt", "")


def compute_scores(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    final_refs = [refs]
    bleu = sacrebleu.corpus_bleu(hyps, final_refs, lowercase=True, tokenize="13a")
    gleu = sacrebleu.corpus_bleu(
        hyps, final_refs, lowercase=True, tokenize="13a", use_effective_order=True
    )
    return {"bleu": bleu.score, "gleu": gleu.score}


def write_jsonl(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: str, rows: List[Dict[str, str]]) -> None:
    headers = [
        "model",
        "bleu",
        "gleu",
        "samples",
        "ckpt_path",
        "decode",
        "beam_width",
        "max_len",
        "repetition_penalty",
        "device",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(
                f"{row['model']},{row['bleu']:.2f},{row['gleu']:.2f},"
                f"{row['samples']},{row['ckpt_path']},{row['decode']},"
                f"{row['beam_width']},{row['max_len']},{row['repetition_penalty']},"
                f"{row['device']}\n"
            )


def plot_bleu(summary: List[Dict[str, str]], out_path: str) -> None:
    labels = [r["model"] for r in summary]
    bleu = [r["bleu"] for r in summary]

    x = range(len(labels))
    color_line = "#1D3557"
    color_bar = "#E63946"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300, constrained_layout=True)

    # Line chart: BLEU trend across methods
    ax1.plot(list(x), bleu, marker="o", linewidth=2.5, color=color_line)
    ax1.set_title("Transformer Test BLEU (Line)")
    ax1.set_ylabel("BLEU")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Bar chart: BLEU comparison
    bars = ax2.bar(list(x), bleu, color=color_bar, alpha=0.9)
    ax2.set_title("Transformer Test BLEU (Bar)")
    ax2.set_ylabel("BLEU")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.savefig(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default=DATA_CONFIG["test_file"])
    parser.add_argument("--src-field", default=DATA_CONFIG.get("src_field", "zh"))
    parser.add_argument("--tgt-field", default=DATA_CONFIG.get("tgt_field", "en"))
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--out-dir", default="transformer_test_results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--sample-mode", choices=["test", "demo", "file"], default="demo")
    parser.add_argument("--sample-file", default="")
    parser.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    test_rows = read_test_data(args.test_file, args.src_field, args.tgt_field)
    if not test_rows:
        raise RuntimeError("No valid test rows found.")

    ckpts = list_transformer_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise RuntimeError("No transformer_*_best.pt checkpoints found.")

    sample_source = read_sample_data(args.sample_mode, args.sample_file, test_rows)
    sample_rows = sample_source[: args.sample_size]
    sample_output = []

    summary = []
    device_label = str(device)
    settings_row = {
        "decode": args.decode,
        "beam_width": args.beam_width,
        "max_len": args.max_len,
        "repetition_penalty": args.repetition_penalty,
        "device": device_label,
    }
    for ckpt in ckpts:
        tag = model_tag_from_ckpt(ckpt)
        model, zh_vocab, en_id2w = load_transformer(ckpt, device)

        results = []
        hyps, refs = [], []
        for row in test_rows:
            pred = translate_sentence(
                model,
                row["src"],
                zh_vocab,
                en_id2w,
                device,
                "transformer",
                decode_method=args.decode,
                beam_width=args.beam_width,
                max_len=args.max_len,
                repetition_penalty=args.repetition_penalty,
            )
            results.append({"src": row["src"], "en_ref": row["tgt"], "en_pred": pred})
            hyps.append(pred)
            refs.append(row["tgt"])

        scores = compute_scores(hyps, refs)
        out_path = os.path.join(args.out_dir, f"{tag}_test_results.jsonl")
        write_jsonl(out_path, results)

        summary.append(
            {
                "model": tag,
                "bleu": scores["bleu"],
                "gleu": scores["gleu"],
                "samples": len(results),
                "ckpt_path": ckpt,
                **settings_row,
            }
        )

        for row in sample_rows:
            pred = translate_sentence(
                model,
                row["src"],
                zh_vocab,
                en_id2w,
                device,
                "transformer",
                decode_method=args.decode,
                beam_width=args.beam_width,
                max_len=args.max_len,
                repetition_penalty=args.repetition_penalty,
            )
            sample_output.append(
                {
                    "model": tag,
                    "src": row["src"],
                    "en_ref": row["tgt"],
                    "en_pred": pred,
                }
            )

    summary.sort(key=lambda r: r["bleu"], reverse=True)
    write_summary_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    plot_bleu(summary, os.path.join(args.out_dir, "bleu_comparison.png"))
    write_jsonl(os.path.join(args.out_dir, "sample_comparisons.jsonl"), sample_output)

    print("Done. Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
