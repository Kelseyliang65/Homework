#!/usr/bin/env python3
"""
Plan 2: Inference script (Final Clean Version)
Usage: python inference.py --model transformer --input_file test.jsonl --output_file test_results.jsonl
"""
import argparse
import json
import torch
import re
import jieba
import sys
import sacrebleu  # 用于最后算分
import torch.nn.functional as F
from gru_nmt import create_rnn_model
from transformer_nmt import TransformerNMT
from settings import RNN_CONFIG, TRANSFORMER_CONFIG, DATA_CONFIG


# --- 1. 本地兼容型分词器 (关键修复) ---
def local_tokenize_zh(text):
    text = text.lower()
    tokens = jieba.lcut(text)
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens


# --- 2. 反分词与去重工具 ---
def simple_detokenize(text):
    if not text: return ""
    text = text.strip()
    # 修复标点空格
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # 修复缩写
    text = text.replace(" 's", "'s").replace(" 't", "'t").replace(" 'm", "'m")
    text = text.replace(" 're", "'re").replace(" 'd", "'d").replace(" 'll", "'ll")
    return text


def remove_repetition(text):
    words = text.split()
    if not words: return ""
    new_words = []
    repeat_count = 0
    last_word = None
    for w in words:
        if w == last_word:
            repeat_count += 1
        else:
            repeat_count = 0
        if repeat_count < 2:
            new_words.append(w)
        last_word = w
    return ' '.join(new_words)


# --- 3. Beam Search (Transformer专用) ---
def beam_search(model, src, device, beam_size=5, max_len=80, alpha=0.6):
    model.eval()
    src_mask = (src != 0).unsqueeze(-2)
    memory = model.encode(src, src_mask)
    cur_hypotheses = [(torch.tensor([2], device=device), 0.0)]

    for _ in range(max_len):
        next_hypotheses = []
        for tokens, score in cur_hypotheses:
            if tokens[-1].item() == 3:
                next_hypotheses.append((tokens, score))
                continue
            tgt = tokens.unsqueeze(0)
            tgt_mask = model.make_std_mask(tgt, 0)
            out = model.decode(tgt, memory, src_mask, tgt_mask)
            prob = model.generator(out[:, -1])
            log_prob = F.log_softmax(prob, dim=-1)
            topk_probs, topk_ids = torch.topk(log_prob, beam_size)

            for i in range(beam_size):
                next_token = topk_ids[0][i].unsqueeze(0)
                next_score = score + topk_probs[0][i].item()
                next_seq = torch.cat([tokens, next_token], dim=0)
                next_hypotheses.append((next_seq, next_score))

        cur_hypotheses = sorted(next_hypotheses, key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)[:beam_size]
        if cur_hypotheses[0][0][-1].item() == 3:
            break

    best_seq, _ = cur_hypotheses[0]
    return best_seq.unsqueeze(0)


# --- 4. 模型加载 ---
def load_rnn(ckpt_path, device):
    print(f"Loading RNN from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    zh_vocab = ckpt['zh_vocab']
    en_vocab = ckpt['en_vocab']
    en_id2w = ckpt['en_id2w']
    rnn_cfg = ckpt.get('rnn_config', RNN_CONFIG)
    model = create_rnn_model(len(zh_vocab), len(en_vocab), rnn_cfg, device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, zh_vocab, en_id2w


def load_gru(ckpt_path, device):
    return load_rnn(ckpt_path, device)


def load_transformer(ckpt_path, device):
    print(f"Loading Transformer from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    zh_vocab = ckpt['zh_vocab']
    en_vocab = ckpt['en_vocab']
    en_id2w = ckpt['en_id2w']
    cfg = ckpt.get('transformer_config', TRANSFORMER_CONFIG)
    model = TransformerNMT(
        src_vocab=len(zh_vocab),
        tgt_vocab=len(en_id2w),
        d_model=cfg['d_model'],
        nhead=cfg['heads'],
        n_enc=cfg['enc_layers'],
        n_dec=cfg['dec_layers'],
        d_ff=cfg['d_ff'],
        dropout=cfg['dropout'],
        max_len=cfg['max_pos'],
        use_rmsnorm=(cfg['norm'] == 'rmsnorm'),
        pos_encoding=cfg.get('pos_encoding', 'relative')
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, zh_vocab, en_id2w


# --- 5. 单句翻译 ---
def translate_sentence(
    model,
    text,
    zh_vocab,
    en_id2w,
    device,
    model_type="gru",
    decode_method="greedy",
    beam_width=5,
    max_len=100,
    repetition_penalty=1.2,
):
    tokens = local_tokenize_zh(text)
    ids = [zh_vocab.get(t, 1) for t in tokens]
    src = torch.tensor([ids], device=device)
    src_len = torch.tensor([len(ids)], device=device)

    with torch.no_grad():
        if model_type in ('gru', 'rnn'):
            if decode_method == "beam":
                try:
                    out = model.beam_search(src, src_len, beam_width=beam_width, max_len=max_len)
                except TypeError:
                    out = model.beam_search(src, src_len, beam_width=beam_width)
            else:
                try:
                    out = model.greedy_translate(
                        src, src_len, max_len=max_len, repetition_penalty=repetition_penalty
                    )
                except TypeError:
                    out = model.greedy_translate(src, src_len, max_len=max_len)
        else:
            try:
                out = beam_search(model, src, device, beam_size=beam_width, max_len=max_len)
            except AttributeError:
                print("Warning: Beam search failed, using greedy.")
                out = model.translate(src, bos_id=2, eos_id=3, max_len=max_len)

    pred_ids = out[0].cpu().tolist()
    raw_tokens = [en_id2w.get(i, '<unk>') for i in pred_ids if i not in [0, 2, 3]]
    raw_str = ' '.join(raw_tokens)
    norep_str = remove_repetition(raw_str)
    final_str = simple_detokenize(norep_str)
    return final_str


# --- 6. 文件处理与最终评分 ---
def translate_file(model, input_file, output_file, zh_vocab, en_id2w, device, model_type, src_field=None):
    results = []
    print(f"开始处理: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        data = json.loads(line)
        use_field = src_field or DATA_CONFIG.get('src_field', 'zh')
        src_text = data.get(use_field, data.get('zh', ''))
        pred = translate_sentence(model, src_text, zh_vocab, en_id2w, device, model_type)

        results.append({
            use_field: src_text,
            'en_ref': data.get('en', ''),
            'en_pred': pred
        })

        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(lines)} 句...")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Translation complete. Results saved to: {output_file}")

    # --- 最终计算分数 ---
    print("\n" + "=" * 40)
    print("📊 正在计算最终指标...")

    final_hyps = [r['en_pred'] for r in results]
    final_refs = [[r['en_ref'] for r in results]]  # 注意格式是 List of Lists

    # 计算 BLEU
    bleu = sacrebleu.corpus_bleu(final_hyps, final_refs, lowercase=True, tokenize='13a')
    # 计算 GLEU (Google-BLEU)
    gleu = sacrebleu.corpus_bleu(final_hyps, final_refs, lowercase=True, tokenize='13a', use_effective_order=True)

    print(f"🏆 {model_type.upper()} 最终得分:")
    print(f"   BLEU: {bleu.score:.2f}")
    print(f"   GLEU: {gleu.score:.2f}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rnn', 'gru', 'transformer'], default='rnn')
    default_rnn_ckpt = (
        f"./ckpt/rnn_{RNN_CONFIG['cell_type'].lower()}_{RNN_CONFIG['attention']}"
        f"_tf{RNN_CONFIG['teacher_forcing']:.2f}_{RNN_CONFIG['decode_method']}_best.pt"
    )
    parser.add_argument('--ckpt', default=default_rnn_ckpt)
    parser.add_argument('--text', type=str, help='Single sentence test')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', default='./output.jsonl')
    parser.add_argument('--src-field', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model in ('gru', 'rnn'):
        model, zh_vocab, en_id2w = load_rnn(args.ckpt, device)
    else:
        model, zh_vocab, en_id2w = load_transformer(args.ckpt, device)

    if args.text:
        res = translate_sentence(model, args.text, zh_vocab, en_id2w, device, args.model)
        print(f"\n原文: {args.text}")
        print(f"译文: {res}")
    elif args.input_file:
        translate_file(model, args.input_file, args.output_file, zh_vocab, en_id2w, device, args.model, args.src_field)
    else:
        print("请指定 --text 或 --input_file")


if __name__ == "__main__":
    main()
