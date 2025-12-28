#!/usr/bin/env python3
"""
Fine-tune T5 for Chinese-to-English translation.
Usage:
  python finetune_t5.py --train-file ... --valid-file ... --save-dir ./ckpt/t5_nmt
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sacrebleu

from settings import DATA_CONFIG, FINETUNE_CONFIG
from preprocess import read_jsonl


class JsonlTranslationDataset(Dataset):
    def __init__(self, path, tokenizer, max_input_len, max_output_len, prefix, src_field, tgt_field):
        self.rows = list(read_jsonl(path))
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.prefix = prefix
        self.src_field = src_field
        self.tgt_field = tgt_field

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        src = self.prefix + item.get(self.src_field, "")
        tgt = item.get(self.tgt_field, "")
        inputs = self.tokenizer(
            src,
            max_length=self.max_input_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            tgt,
            max_length=self.max_output_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0), labels.squeeze(0)


def evaluate_bleu(model, loader, tokenizer, device, max_output_len):
    model.eval()
    hyps = []
    refs = []
    with torch.no_grad():
        for input_ids, attn_mask, labels in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_length=max_output_len,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            hyps.extend(preds)
            ref_labels = labels.clone()
            ref_labels[ref_labels == -100] = tokenizer.pad_token_id
            refs.extend(tokenizer.batch_decode(ref_labels, skip_special_tokens=True))
    bleu = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True, tokenize="13a")
    return bleu.score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=DATA_CONFIG["train_file"])
    parser.add_argument("--valid-file", default=DATA_CONFIG["valid_file"])
    parser.add_argument("--test-file", default=DATA_CONFIG.get("test_file"))
    parser.add_argument("--model-name", default=FINETUNE_CONFIG["model_name"])
    parser.add_argument("--batch", type=int, default=FINETUNE_CONFIG["batch"])
    parser.add_argument("--lr", type=float, default=FINETUNE_CONFIG["lr"])
    parser.add_argument("--epochs", type=int, default=FINETUNE_CONFIG["epochs"])
    parser.add_argument("--max-input-len", type=int, default=FINETUNE_CONFIG["max_input_len"])
    parser.add_argument("--max-output-len", type=int, default=FINETUNE_CONFIG["max_output_len"])
    parser.add_argument("--save-dir", default="./ckpt/t5_nmt")
    parser.add_argument("--prefix", default="translate Chinese to English: ")
    parser.add_argument("--src-field", default=DATA_CONFIG.get("src_field", "zh"))
    parser.add_argument("--tgt-field", default=DATA_CONFIG.get("tgt_field", "en"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    train_ds = JsonlTranslationDataset(
        args.train_file, tokenizer, args.max_input_len, args.max_output_len, args.prefix,
        args.src_field, args.tgt_field
    )
    valid_ds = JsonlTranslationDataset(
        args.valid_file, tokenizer, args.max_input_len, args.max_output_len, args.prefix,
        args.src_field, args.tgt_field
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_bleu = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for input_ids, attn_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            optim.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
            loss.backward()
            optim.step()
            total_loss += loss.item()

        bleu = evaluate_bleu(model, valid_loader, tokenizer, device, args.max_output_len)
        avg_loss = total_loss / max(1, len(train_loader))
        info = f"Epoch {epoch + 1}: Loss={avg_loss:.4f} BLEU={bleu:.2f}"
        print(info)

        if bleu > best_bleu:
            best_bleu = bleu
            os.makedirs(args.save_dir, exist_ok=True)
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            with open(os.path.join(args.save_dir, "metrics.txt"), "w", encoding="utf-8") as f:
                f.write(info + "\n")

    if args.test_file:
        test_ds = JsonlTranslationDataset(
            args.test_file, tokenizer, args.max_input_len, args.max_output_len, args.prefix,
            args.src_field, args.tgt_field
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)
        test_bleu = evaluate_bleu(model, test_loader, tokenizer, device, args.max_output_len)
        print(f"Test BLEU: {test_bleu:.2f}")


if __name__ == "__main__":
    main()
