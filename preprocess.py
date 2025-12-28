#!/usr/bin/env python3
"""
Plan 2: Data preprocessing module (Fixed Regex)
Style: Functional programming, uses generators
"""
import json
import re
from collections import Counter
from typing import Iterator, List, Tuple

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import jieba
try:
    import nltk
    from nltk.tokenize import word_tokenize
except ImportError:
    nltk = None
    word_tokenize = None

# Special tokens
PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']


def read_jsonl(path: str) -> Iterator[dict]:
    """Generator to read jsonl file"""
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.strip())


def clean_zh(text: str) -> str:
    """
    Clean Chinese text - 修正版
    保留了：连字符(-), 百分号(%), 小数点(.), 以及常见的英文标点，防止 HMX-1 变成 HMX1
    """
    # 1. 统一转小写 (对应 inference 的逻辑)
    text = text.lower().strip()

    # 2. 稍微宽松的正则：允许 中文 + 英文数字 + 常见符号
    # 之前的正则把 '-' 删了，这里补上了
    text = re.sub(r"[^\u4e00-\u9fa5a-z0-9，。！？、；：“”‘’（）\s\-%\.]", "", text)

    return text.strip()


def clean_en(text: str) -> str:
    """Clean English text - 修正版"""
    text = text.lower().strip()
    # 保留字母数字和常见标点，允许 -, %, $ 等
    text = re.sub(r'[^a-z0-9\s.,!?;:\'\"\-\%\$\€]', '', text)
    return text


def tokenize_zh(text: str, tokenizer: str = "jieba") -> List[str]:
    """Tokenize Chinese using jieba."""
    cleaned = clean_zh(text)
    if tokenizer == "jieba":
        tokens = jieba.lcut(cleaned)
    else:
        raise ValueError(f"Unknown Chinese tokenizer: {tokenizer}")
    return [t for t in tokens if t.strip()]

def tokenize_en(text: str, tokenizer: str = "nltk") -> List[str]:
    """Tokenize English using NLTK or regex split."""
    cleaned = clean_en(text)
    if tokenizer == "nltk" and word_tokenize is not None:
        try:
            return [t for t in word_tokenize(cleaned) if t.strip()]
        except LookupError:
            warnings.warn("NLTK punkt not found, falling back to regex split.")
    text = re.sub(r"([.,!?;:])", r" \1 ", cleaned)
    return [t for t in text.split() if t.strip()]

def build_vocab(sentences: List[List[str]], min_freq: int = 3) -> Tuple[dict, dict]:
    """Build vocabulary from tokenized sentences"""
    counter = Counter()
    for sent in sentences:
        counter.update(sent)

    word2id = {w: i for i, w in enumerate(SPECIAL)}
    for word, freq in counter.most_common():
        if freq >= min_freq:
            word2id[word] = len(word2id)

    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word


def encode_sent(tokens: List[str], vocab: dict, max_len: int) -> List[int]:
    """Encode tokens to ids"""
    # 截断时预留 BOS 和 EOS 的位置
    ids = [vocab.get(t, UNK) for t in tokens[:max_len - 2]]
    return [BOS] + ids + [EOS]


def prepare_data(data_path: str, max_len: int = 100, min_freq: int = 3, vocab_dict=None,
                 zh_tokenizer: str = "jieba", en_tokenizer: str = "nltk",
                 truncate_long: bool = False, src_field: str = "zh", tgt_field: str = "en"):
    """
    data_path: 数据路径
    vocab_dict: 如果传入已有的词表（训练集生成的），则不再重新构建
    """
    print(f"正在处理数据: {data_path} ...")
    raw_data = list(read_jsonl(data_path))

    # 增加一点进度提示
    print(f"  - 加载原始数据: {len(raw_data)} 条")

    zh_sents = [tokenize_zh(d.get(src_field, ""), zh_tokenizer) for d in raw_data]
    en_sents = [tokenize_en(d.get(tgt_field, ""), en_tokenizer) for d in raw_data]

    # 过滤掉过长或过短的句子
    if truncate_long:
        pairs = []
        for z, e in zip(zh_sents, en_sents):
            if len(z) < 1 or len(e) < 1:
                continue
            pairs.append((z[:max_len], e[:max_len]))
        print(f"  - 截断后剩余 {len(pairs)} 条(最大长度: {max_len})")
    else:
        pairs = [(z, e) for z, e in zip(zh_sents, en_sents)
                 if 1 <= len(z) <= max_len and 1 <= len(e) <= max_len]
        print(f"  - 过滤后剩余 {len(pairs)} 条(过滤规则: 1 <= len <= {max_len})")

    if vocab_dict is None:
        print("  - [训练模式] 正在构建新词表...")
        zh_filtered, en_filtered = zip(*pairs) if pairs else ([], [])
        zh_vocab, zh_id2w = build_vocab(list(zh_filtered), min_freq)
        en_vocab, en_id2w = build_vocab(list(en_filtered), min_freq)
        print(f"  - 词表构建完成: 中文 {len(zh_vocab)}, 英文 {len(en_vocab)}")
    else:
        print("  - [验证/推理模式] 复用现有词表...")
        zh_vocab = vocab_dict['zh_vocab']
        en_vocab = vocab_dict['en_vocab']
        zh_id2w = vocab_dict['zh_id2w']  # 注意这里要对应 main.py 里的 key
        en_id2w = vocab_dict['en_id2w']

    return {
        'pairs': pairs,
        'zh_vocab': zh_vocab,
        'en_vocab': en_vocab,
        'zh_id2w': zh_id2w,
        'en_id2w': en_id2w
    }


class MTDataset:
    """Simple dataset class"""

    def __init__(self, pairs, zh_vocab, en_vocab, max_len):
        self.pairs = pairs
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        zh_toks, en_toks = self.pairs[idx]
        src = encode_sent(zh_toks, self.zh_vocab, self.max_len)
        tgt = encode_sent(en_toks, self.en_vocab, self.max_len)
        return src, tgt


def collate_batch(batch):
    """Collate function for dataloader"""
    import torch
    src_batch, tgt_batch = zip(*batch)

    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    # 初始化全 0 (PAD) 的张量
    src_tensor = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_tensor = torch.zeros(len(batch), max_tgt, dtype=torch.long)

    for i, (s, t) in enumerate(batch):
        src_tensor[i, :len(s)] = torch.tensor(s)
        tgt_tensor[i, :len(t)] = torch.tensor(t)

    return src_tensor, torch.tensor(src_lens), tgt_tensor
