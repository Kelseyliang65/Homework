#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan 2: Configuration Settings (Optimized for 100k Dataset & Generalization)
Author style: Uses simple dictionaries, minimal comments, functional approach
"""

# Data settings
DATA_CONFIG = {
    "train_file": "C:/Users/250010024/Desktop/datasets/train_mixed_v2.jsonl",
    "valid_file": "C:/Users/250010024/Desktop/datasets/valid_retranslated_hunyuan.jsonl",
    "test_file": "C:/Users/250010024/Desktop/datasets/test_retranslated_hunyuan.jsonl",
    "max_len": 100,
    "zh_tokenizer": "jieba",
    "en_tokenizer": "nltk",
    "truncate_long": False,
    "src_field": "zh_hy",
    "tgt_field": "en",
    # 增加最小词频，过滤噪声，减少 <unk> 并不是坏事，反而能让模型聚焦核心词汇
    "min_word_freq": 5
}

# RNN model settings (GRU)
RNN_CONFIG = {
    "cell_type": "GRU",
    "embed_size": 300,
    "hidden_size": 512,
    "num_layers": 2,
    # 【关键修改】提高 Dropout 防止过拟合
    "dropout": 0.3,
    "attention": "dot_product",
    # 【关键修改】Teacher Forcing 不建议为 0.0，否则很难收敛
    # 建议设为 0.5 (一半概率看正确答案，一半概率靠自己)，平衡训练速度和稳定性
    "teacher_forcing": 0.5,
    "decode_method": "greedy",
    "beam_width": 5
}

# Transformer settings
# 对于 100k 数据集，6层 Transformer 太深了，极易过拟合。
# 我们将其“轻量化”，强制模型学习通用规律而不是死记硬背。
TRANSFORMER_CONFIG = {
    "d_model": 512,
    "heads": 8,
    # 【关键修改】层数减半，减少参数量，强迫模型从有限参数中提取规律
    "enc_layers": 3,
    "dec_layers": 3,
    "d_ff": 2048,
    # 【核心修改】大幅提高 Dropout (0.1 -> 0.3)
    # 这是解决 Test BLEU 低、Valid BLEU 虚高最有效的手段
    "dropout": 0.3,
    "pos_encoding": "relative",
    "norm": "rmsnorm",
    # 稍微缩减最大位置编码，节省内存
    "max_pos": 128
}

# Training settings
TRAIN_CONFIG = {
    "batch": 64,
    "lr": 5e-4,
    # 增加 Epoch，因为 Dropout 变高了，模型学得慢但学得更扎实
    "epochs": 30,
    "warmup": 3000,
    "clip": 1.0,
    "label_smooth": 0.1,
    # 【新增】权重衰减，AdamW 的核心参数，防止权重爆炸，抑制过拟合
    "weight_decay": 1e-4,
    "device": "cuda",
    "save_path": "./ckpt",
    "early_stop_metric": "combined",
    "early_stop_patience": 5,
    "early_stop_delta": 0.0,
    "early_stop_min_loss": 2.0
}

# Fine-tune settings (T5-base)
FINETUNE_CONFIG = {
    "model_name": "google-t5/t5-base",
    "lr": 2e-5,
    "batch": 8,
    "epochs": 5,
    "max_input_len": 256,
    "max_output_len": 256
}
