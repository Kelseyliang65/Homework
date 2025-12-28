#!/usr/bin/env python3
"""
Plan 2: Main training script (Fixed & Optimized)
Features:
1. Correct Vocabulary Sharing between Train/Valid
2. Accurate BLEU Evaluation (Detokenized)
3. Weight Decay for Regularization
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import sacrebleu
import re  # 【新增】用于反分词

from settings import DATA_CONFIG, RNN_CONFIG, TRANSFORMER_CONFIG, TRAIN_CONFIG
from preprocess import prepare_data, MTDataset, collate_batch
from gru_nmt import create_rnn_model
from transformer_nmt import TransformerNMT


# --- 核心工具：反分词逻辑 (与 inference.py 对齐) ---
def simple_detokenize(text):
    """
    反分词：把 'hello , world .' 还原为 'hello, world.'
    确保验证集评估的 BLEU 是真实的
    """
    if not text: return ""
    text = text.strip()
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = text.replace(" 's", "'s").replace(" 't", "'t").replace(" 'm", "'m")
    return text


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, mode="loss", min_loss=None):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.min_loss = min_loss
        self.counter = 0
        self.best_bleu = None
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss, current_bleu=None):
        if self.min_loss is not None and current_loss <= self.min_loss:
            print(f"Early stop: ValLoss {current_loss:.4f} <= min_loss {self.min_loss:.4f}")
            self.early_stop = True
            return
        if self.mode == "bleu":
            return self._update_bleu(current_bleu)
        if self.mode == "combined":
            improved = self._is_loss_improved(current_loss) or self._is_bleu_improved(current_bleu)
        else:
            improved = self._is_loss_improved(current_loss)

        if improved:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_loss_improved(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return True
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            return True
        return False

    def _is_bleu_improved(self, current_bleu):
        if current_bleu is None:
            return False
        if self.best_bleu is None:
            self.best_bleu = current_bleu
            return True
        if current_bleu > self.best_bleu + self.delta:
            self.best_bleu = current_bleu
            return True
        return False

    def _update_bleu(self, current_bleu):
        if self._is_bleu_improved(current_bleu):
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_epoch(model, loader, optim, criterion, device, clip=1.0, tf_ratio=0.0):
    model.train()
    total_loss = 0
    # 使用 tqdm 显示进度
    pbar = tqdm(loader, desc="Training", leave=False)

    for src, src_len, tgt in pbar:
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        optim.zero_grad()

        if hasattr(model, 'enc'):  # RNN model
            # GRU 需要 src_len 来做 pack_padded_sequence
            out = model(src, src_len, tgt, tf_ratio=tf_ratio)
        else:  # Transformer
            # Transformer 只需要 src 和 tgt (src_len 隐含在 mask 中)
            out = model(src, tgt[:, :-1])

        # Compute loss
        # 展平输出和标签: (B * T, V) vs (B * T)
        if hasattr(model, 'enc'):
            # GRU 的 out 是包含全长度的
            loss = criterion(out[:, 1:].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        else:
            # Transformer 的 out 已经是移位过的
            loss = criterion(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


@torch.no_grad()
def validate_loss(model, loader, criterion, device, tf_ratio=0.0):
    model.eval()
    total_loss = 0
    for src, src_len, tgt in loader:
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)

        if hasattr(model, 'enc'):
            out = model(src, src_len, tgt, tf_ratio=tf_ratio)
            loss = criterion(out[:, 1:].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        else:
            out = model(src, tgt[:, :-1])
            loss = criterion(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))

        total_loss += loss.item()

    return total_loss / len(loader)


def _rnn_decode_batch(model, src, src_len, decode_method, beam_width):
    if decode_method == "beam":
        outs = []
        for i in range(src.size(0)):
            out = model.beam_search(src[i:i + 1], src_len[i:i + 1], beam_width=beam_width)
            outs.append(out)
        max_len = max(o.size(1) for o in outs)
        preds = torch.full((src.size(0), max_len), 0, device=src.device, dtype=outs[0].dtype)
        for i, out in enumerate(outs):
            preds[i, :out.size(1)] = out[0]
        return preds
    return model.greedy_translate(src, src_len)


@torch.no_grad()
def evaluate(model, loader, id2word, device, bos_id=2, eos_id=3,
             decode_method="greedy", beam_width=5):
    """
    修正后的评估函数：
    1. 使用 Greedy Search 生成句子
    2. 执行 Detokenize
    3. 格式化 Reference 为 List of Lists
    """
    model.eval()
    hyps, refs = [], []

    for src, src_len, tgt in tqdm(loader, desc="Validating", leave=False):
        src, src_len = src.to(device), src_len.to(device)

        # 统一使用 greedy_translate / translate 接口
        if hasattr(model, 'enc'):
            preds = _rnn_decode_batch(model, src, src_len, decode_method, beam_width)
        else:
            preds = model.translate(src, bos_id, eos_id)

        for i in range(src.size(0)):
            # --- 处理预测 (Hypothesis) ---
            pred_ids = preds[i].cpu().tolist()
            hyp_tokens = [id2word.get(j, '<unk>') for j in pred_ids if j not in [0, 2, 3]]
            # 先拼成字符串，再反分词
            hyps.append(simple_detokenize(' '.join(hyp_tokens)))

            # --- 处理参考 (Reference) ---
            ref_ids = tgt[i].tolist()
            ref_tokens = [id2word.get(j, '<unk>') for j in ref_ids if j not in [0, 2, 3]]
            refs.append(simple_detokenize(' '.join(ref_tokens)))

    # 【关键修正】SacreBLEU 要求 refs 是 [ref1_list, ref2_list...]
    # 这里的 ref1_list 是所有样本的第一个参考答案
    final_refs = [refs]

    # 计算 BLEU (忽略大小写，使用 standard tokenize)
    bleu = sacrebleu.corpus_bleu(hyps, final_refs, lowercase=True, tokenize='13a')
    return bleu.score


def _resolve_device(device_arg):
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return torch.device('cpu')
    return torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')


def _read_log_stats(log_path):
    if not os.path.exists(log_path):
        return 0, 0.0
    best_bleu = 0.0
    last_epoch = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"Epoch\s+(\d+).*?BLEU=([\d\.]+)", line)
            if match:
                epoch = int(match.group(1))
                bleu = float(match.group(2))
                last_epoch = max(last_epoch, epoch)
                if bleu > best_bleu:
                    best_bleu = bleu
    return last_epoch, best_bleu


def run_rnn_experiment(rnn_cfg, device_arg, resume=False):
    """Train RNN model"""
    device = _resolve_device(device_arg)
    print(f"Device: {device}")
    print(f"Configs: {rnn_cfg}")

    # 1. 准备数据
    print("Loading Training Data...")
    data = prepare_data(DATA_CONFIG['train_file'], DATA_CONFIG['max_len'], DATA_CONFIG['min_word_freq'],
                        zh_tokenizer=DATA_CONFIG.get('zh_tokenizer', 'jieba'),
                        en_tokenizer=DATA_CONFIG.get('en_tokenizer', 'nltk'),
                        truncate_long=DATA_CONFIG.get('truncate_long', False),
                        src_field=DATA_CONFIG.get('src_field', 'zh'),
                        tgt_field=DATA_CONFIG.get('tgt_field', 'en'))

    print("Loading Validation Data...")
    # 【修复】必须传入 vocab_dict=data，确保词表 ID 一致！
    valid_data = prepare_data(DATA_CONFIG['valid_file'], DATA_CONFIG['max_len'], vocab_dict=data,
                             zh_tokenizer=DATA_CONFIG.get('zh_tokenizer', 'jieba'),
                             en_tokenizer=DATA_CONFIG.get('en_tokenizer', 'nltk'),
                             truncate_long=DATA_CONFIG.get('truncate_long', False),
                        src_field=DATA_CONFIG.get('src_field', 'zh'),
                        tgt_field=DATA_CONFIG.get('tgt_field', 'en'))

    train_ds = MTDataset(data['pairs'], data['zh_vocab'], data['en_vocab'], DATA_CONFIG['max_len'])
    valid_ds = MTDataset(valid_data['pairs'], data['zh_vocab'], data['en_vocab'], DATA_CONFIG['max_len'])

    train_loader = DataLoader(train_ds, TRAIN_CONFIG['batch'], shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, TRAIN_CONFIG['batch'], shuffle=False, collate_fn=collate_batch)

    # 2. 创建模型
    model = create_rnn_model(len(data['zh_vocab']), len(data['en_vocab']), rnn_cfg, device)
    model_tag = f"rnn_{rnn_cfg['cell_type'].lower()}_{rnn_cfg['attention']}_tf{rnn_cfg['teacher_forcing']:.2f}_{rnn_cfg['decode_method']}"
    print(f"RNN Params: {sum(p.numel() for p in model.parameters()):,}")

    # 【优化】加入 Weight Decay 防止过拟合
    optim = AdamW(model.parameters(), lr=TRAIN_CONFIG['lr'], weight_decay=TRAIN_CONFIG.get('weight_decay', 0))
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    stopper = EarlyStopping(
        patience=TRAIN_CONFIG.get('early_stop_patience', 5),
        delta=TRAIN_CONFIG.get('early_stop_delta', 0.0),
        mode=TRAIN_CONFIG.get('early_stop_metric', 'loss'),
        min_loss=TRAIN_CONFIG.get('early_stop_min_loss'),
    )

    base_epoch = 0
    best_bleu = 0
    if resume:
        log_path = f"{model_tag}_history.log"
        base_epoch, best_bleu = _read_log_stats(log_path)
        ckpt_path = f"{TRAIN_CONFIG['save_path']}/{model_tag}_best.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Resumed from {ckpt_path}")

    for epoch in range(TRAIN_CONFIG['epochs']):
        loss = train_epoch(model, train_loader, optim, criterion, device, TRAIN_CONFIG['clip'],
                           tf_ratio=rnn_cfg['teacher_forcing'])
        val_loss = validate_loss(model, valid_loader, criterion, device, tf_ratio=rnn_cfg['teacher_forcing'])
        bleu = evaluate(model, valid_loader, data['en_id2w'], device,
                        decode_method=rnn_cfg['decode_method'],
                        beam_width=rnn_cfg.get('beam_width', 5))

        log_epoch = epoch + 1 + base_epoch
        info = f"Epoch {log_epoch}: Loss={loss:.4f} ValLoss={val_loss:.4f} BLEU={bleu:.2f}"
        print(info)
        with open(f"{model_tag}_history.log", "a", encoding="utf-8") as f:
            f.write(info + "\n")

        if bleu > best_bleu:
            best_bleu = bleu
            os.makedirs(TRAIN_CONFIG['save_path'], exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'zh_vocab': data['zh_vocab'],
                'en_vocab': data['en_vocab'],
                'en_id2w': data['en_id2w'],
                'rnn_config': rnn_cfg
            }, f"{TRAIN_CONFIG['save_path']}/{model_tag}_best.pt")
            print(f"🔥 New Best Model Saved! BLEU: {best_bleu:.2f}")

        stopper(val_loss, bleu)
        if stopper.early_stop:
            print(f"Early stopping triggered. Best BLEU: {stopper.best_bleu}")
            break

    return model


def run_transformer_experiment(trans_cfg, train_cfg, device_arg, resume=False):
    """Train Transformer model"""
    device = _resolve_device(device_arg)
    mode = 'transformer'
    print(f"Device: {device}")
    print(f"Configs: {trans_cfg}")
    print(f"Train Configs: {train_cfg}")

    # 1. 准备数据
    print("Loading Training Data...")
    data = prepare_data(DATA_CONFIG['train_file'], DATA_CONFIG['max_len'], DATA_CONFIG['min_word_freq'],
                        zh_tokenizer=DATA_CONFIG.get('zh_tokenizer', 'jieba'),
                        en_tokenizer=DATA_CONFIG.get('en_tokenizer', 'nltk'),
                        truncate_long=DATA_CONFIG.get('truncate_long', False),
                        src_field=DATA_CONFIG.get('src_field', 'zh'),
                        tgt_field=DATA_CONFIG.get('tgt_field', 'en'))

    print("Loading Validation Data...")
    # 【关键】词表继承
    valid_data = prepare_data(DATA_CONFIG['valid_file'], DATA_CONFIG['max_len'], vocab_dict=data,
                             zh_tokenizer=DATA_CONFIG.get('zh_tokenizer', 'jieba'),
                             en_tokenizer=DATA_CONFIG.get('en_tokenizer', 'nltk'),
                             truncate_long=DATA_CONFIG.get('truncate_long', False),
                        src_field=DATA_CONFIG.get('src_field', 'zh'),
                        tgt_field=DATA_CONFIG.get('tgt_field', 'en'))

    train_ds = MTDataset(data['pairs'], data['zh_vocab'], data['en_vocab'], DATA_CONFIG['max_len'])
    valid_ds = MTDataset(valid_data['pairs'], data['zh_vocab'], data['en_vocab'], DATA_CONFIG['max_len'])

    train_loader = DataLoader(train_ds, train_cfg['batch'], shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, train_cfg['batch'], shuffle=False, collate_fn=collate_batch)

    # 2. 实例化模型
    cfg = trans_cfg
    model = TransformerNMT(
        src_vocab=len(data['zh_vocab']),
        tgt_vocab=len(data['en_vocab']),
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
    model = model.to(device)
    print(f"Transformer Params: {sum(p.numel() for p in model.parameters()):,}")

    # 【优化】加入 Weight Decay
    optim = AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg.get('weight_decay', 1e-4))
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=train_cfg['label_smooth'])

    stopper = EarlyStopping(
        patience=train_cfg.get('early_stop_patience', 5),
        delta=train_cfg.get('early_stop_delta', 0.0),
        mode=train_cfg.get('early_stop_metric', 'loss'),
        min_loss=train_cfg.get('early_stop_min_loss'),
    )

    model_tag = (
        f"transformer_{cfg['pos_encoding']}_{cfg['norm']}"
        f"_d{cfg['d_model']}_h{cfg['heads']}"
        f"_e{cfg['enc_layers']}{cfg['dec_layers']}"
        f"_ff{cfg['d_ff']}_bs{train_cfg['batch']}_lr{train_cfg['lr']}"
    )

    base_epoch = 0
    best_bleu = 0
    if resume:
        log_path = f"{model_tag}_history.log"
        base_epoch, best_bleu = _read_log_stats(log_path)
        ckpt_path = f"{TRAIN_CONFIG['save_path']}/{model_tag}_best.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Resumed from {ckpt_path}")

    for epoch in range(train_cfg['epochs']):
        loss = train_epoch(model, train_loader, optim, criterion, device, train_cfg['clip'])
        val_loss = validate_loss(model, valid_loader, criterion, device)
        bleu = evaluate(model, valid_loader, data['en_id2w'], device)

        log_epoch = epoch + 1 + base_epoch
        info = f"Epoch {log_epoch}: Loss={loss:.4f} ValLoss={val_loss:.4f} BLEU={bleu:.2f}"
        print(info)
        with open(f"{model_tag}_history.log", "a", encoding="utf-8") as f:
            f.write(info + "\n")

        if bleu > best_bleu:
            best_bleu = bleu
            os.makedirs(TRAIN_CONFIG['save_path'], exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'zh_vocab': data['zh_vocab'],
                'en_vocab': data['en_vocab'],
                'en_id2w': data['en_id2w'],
                'transformer_config': trans_cfg,
                'train_config': train_cfg
            }, f"{TRAIN_CONFIG['save_path']}/{model_tag}_best.pt")
            print(f"🔥 New Best Model Saved! BLEU: {best_bleu:.2f}")

        stopper(val_loss, bleu)
        if stopper.early_stop:
            print(f"Early stopping triggered. Best BLEU: {stopper.best_bleu}")
            break


def _parse_args():
    legacy_mode = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        legacy_mode = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rnn', 'transformer'], default='rnn')
    parser.add_argument('--cell', choices=['gru', 'lstm'], default=RNN_CONFIG['cell_type'].lower())
    parser.add_argument('--attention', choices=['dot_product', 'multiplicative', 'additive'],
                        default=RNN_CONFIG['attention'])
    parser.add_argument('--teacher-forcing', type=float, default=RNN_CONFIG['teacher_forcing'])
    parser.add_argument('--decode', choices=['greedy', 'beam'], default=RNN_CONFIG['decode_method'])
    parser.add_argument('--beam-width', type=int, default=RNN_CONFIG.get('beam_width', 5))
    parser.add_argument('--t-pos-encoding', choices=['relative', 'absolute'], default=TRANSFORMER_CONFIG['pos_encoding'])
    parser.add_argument('--t-norm', choices=['rmsnorm', 'layernorm'], default=TRANSFORMER_CONFIG['norm'])
    parser.add_argument('--t-d-model', type=int, default=TRANSFORMER_CONFIG['d_model'])
    parser.add_argument('--t-heads', type=int, default=TRANSFORMER_CONFIG['heads'])
    parser.add_argument('--t-enc-layers', type=int, default=TRANSFORMER_CONFIG['enc_layers'])
    parser.add_argument('--t-dec-layers', type=int, default=TRANSFORMER_CONFIG['dec_layers'])
    parser.add_argument('--t-d-ff', type=int, default=TRANSFORMER_CONFIG['d_ff'])
    parser.add_argument('--t-dropout', type=float, default=TRANSFORMER_CONFIG['dropout'])
    parser.add_argument('--t-max-pos', type=int, default=TRANSFORMER_CONFIG['max_pos'])
    parser.add_argument('--batch', type=int, default=TRAIN_CONFIG['batch'])
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['lr'])
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--label-smooth', type=float, default=TRAIN_CONFIG['label_smooth'])
    parser.add_argument('--weight-decay', type=float, default=TRAIN_CONFIG.get('weight_decay', 0))
    parser.add_argument('--clip', type=float, default=TRAIN_CONFIG['clip'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if legacy_mode:
        if legacy_mode == 'gru':
            args.model = 'rnn'
        elif legacy_mode == 'transformer':
            args.model = 'transformer'
        else:
            raise ValueError(f"Unknown mode: {legacy_mode}")
    return args


if __name__ == "__main__":
    args = _parse_args()

    if args.model == 'rnn':
        print("=" * 30 + "\nStarting RNN Experiment\n" + "=" * 30)
        rnn_cfg = dict(RNN_CONFIG)
        rnn_cfg['cell_type'] = args.cell.upper()
        rnn_cfg['attention'] = args.attention
        rnn_cfg['teacher_forcing'] = args.teacher_forcing
        rnn_cfg['decode_method'] = args.decode
        rnn_cfg['beam_width'] = args.beam_width
        run_rnn_experiment(rnn_cfg, args.device, args.resume)
    elif args.model == 'transformer':
        print("=" * 30 + "\nStarting Transformer Experiment\n" + "=" * 30)
        trans_cfg = dict(TRANSFORMER_CONFIG)
        trans_cfg['pos_encoding'] = args.t_pos_encoding
        trans_cfg['norm'] = args.t_norm
        trans_cfg['d_model'] = args.t_d_model
        trans_cfg['heads'] = args.t_heads
        trans_cfg['enc_layers'] = args.t_enc_layers
        trans_cfg['dec_layers'] = args.t_dec_layers
        trans_cfg['d_ff'] = args.t_d_ff
        trans_cfg['dropout'] = args.t_dropout
        trans_cfg['max_pos'] = args.t_max_pos
        train_cfg = dict(TRAIN_CONFIG)
        train_cfg['batch'] = args.batch
        train_cfg['lr'] = args.lr
        train_cfg['epochs'] = args.epochs
        train_cfg['label_smooth'] = args.label_smooth
        train_cfg['weight_decay'] = args.weight_decay
        train_cfg['clip'] = args.clip
        run_transformer_experiment(trans_cfg, train_cfg, args.device, args.resume)
    else:
        print(f"Unknown mode: {args.model}")
