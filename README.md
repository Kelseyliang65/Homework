================================================================================



                        方案2：中英机器翻译项目说明



        GRU/LSTM + 多种注意力 + RMSNorm + 相对位置编码 + Sweep/Resume



================================================================================







【项目概述】



本方案以 RNN（GRU/LSTM）+ 注意力 为主线，Transformer 为对照模型。



统一清洗/分词 + 共享词表 + 统一评估（detokenize + SacreBLEU），支持 sweep、



resume、test 全流程，强调可复现与可对比。







核心目标：



1) 统一数据预处理，保证 train/valid/test 词表一致。



2) 用日志 + checkpoint 固化实验状态（可追溯、可恢复）。



3) 用 BLEU 保存最优模型，同时记录 ValLoss 观察泛化。







================================================================================



                              全局流程图（文字版）



================================================================================







数据流：



JSONL -> clean/tokenize -> vocab -> encode -> DataLoader



    -> train_epoch -> validate_loss + BLEU -> log + ckpt



    -> analyze_results / rnn_test_sweep / inference







推理流：



text -> tokenize -> ids -> model.decode -> tokens -> detokenize







================================================================================



                              文件详细说明（含核心代码块）



================================================================================







【文件1: settings.py - 配置管理（字典方式）】







核心代码块：



```python



DATA_CONFIG = {



    "train_file": "C:/Users/250010024/Desktop/datasets/train_mixed_v2.jsonl",



    "valid_file": "C:/Users/250010024/Desktop/datasets/valid_retranslated_hunyuan.jsonl",



    "test_file": "C:/Users/250010024/Desktop/datasets/test_retranslated_hunyuan.jsonl",



    "src_field": "zh_hy",



    "tgt_field": "en",



    "max_len": 100,



    "min_word_freq": 5



}







RNN_CONFIG = {



    "cell_type": "GRU",



    "embed_size": 300,



    "hidden_size": 512,



    "num_layers": 2,



    "dropout": 0.3,



    "attention": "dot_product",



    "teacher_forcing": 0.5,



    "decode_method": "greedy",



    "beam_width": 5



}







TRAIN_CONFIG = {



    "batch": 64,



    "lr": 5e-4,



    "epochs": 30,



    "clip": 1.0,



    "weight_decay": 1e-4,



    "device": "cuda",



    "save_path": "./ckpt",



    "early_stop_metric": "combined",



    "early_stop_patience": 5,



    "early_stop_delta": 0.0,



    "early_stop_min_loss": 2.0



}



```







细致讲解：



- DATA_CONFIG 直接映射 JSONL 字段：src_field / tgt_field 必须一致。



- max_len 影响过滤范围和显存占用，越大越难学但信息保留多。



- min_word_freq 越大词表越小，UNK 越多，稳定但表达弱。



- TRAIN_CONFIG：lr/batch/weight_decay 影响训练速度与泛化。



- early_stop_min_loss=2.0：ValLoss<=2 直接停止（过拟合阈值）。







================================================================================







【文件2: preprocess.py - 数据预处理】







1) 清洗与分词



```python



def clean_zh(text: str) -> str:



    text = text.lower().strip()



    text = re.sub(r"[^\u4e00-\u9fa5a-z0-9，。！？、；：“”‘’（）\s\-%\.]", "", text)



    return text.strip()







def tokenize_zh(text: str, tokenizer: str = "jieba") -> List[str]:



    cleaned = clean_zh(text)



    tokens = jieba.lcut(cleaned)



    return [t for t in tokens if t.strip()]







def tokenize_en(text: str, tokenizer: str = "nltk") -> List[str]:



    cleaned = clean_en(text)



    if tokenizer == "nltk" and word_tokenize is not None:



        return [t for t in word_tokenize(cleaned) if t.strip()]



    text = re.sub(r"([.,!?;:])", r" \1 ", cleaned)



    return [t for t in text.split() if t.strip()]



```







输入/输出维度：



- 输入：原始字符串



- 输出：token 列表 List[str]







细致讲解：



- clean_zh/clean_en 保留连字符、百分号、小数点，避免型号/数值被破坏。



- tokenize_en 优先 NLTK（更接近标准英文分词），不可用时退化为正则分词。



- 统一小写，减少词表稀疏。







2) 词表构建



```python



def build_vocab(sentences: List[List[str]], min_freq: int = 3) -> Tuple[dict, dict]:



    counter = Counter()



    for sent in sentences:



        counter.update(sent)



    word2id = {w: i for i, w in enumerate(SPECIAL)}



    for word, freq in counter.most_common():



        if freq >= min_freq:



            word2id[word] = len(word2id)



    id2word = {i: w for w, i in word2id.items()}



    return word2id, id2word



```







输入/输出：



- 输入：List[List[str]]



- 输出：word2id(dict), id2word(dict)







3) 编码与数据集



```python



def encode_sent(tokens: List[str], vocab: dict, max_len: int) -> List[int]:



    ids = [vocab.get(t, UNK) for t in tokens[:max_len - 2]]



    return [BOS] + ids + [EOS]







class MTDataset:



    def __getitem__(self, idx):



        return src_ids, tgt_ids







def collate_batch(batch):



    return src_tensor, src_lens, tgt_tensor



```







张量形状：



- src_tensor: (B, Smax)



- src_lens: (B,)



- tgt_tensor: (B, Tmax)







4) 训练/验证共享词表



```python



def prepare_data(..., vocab_dict=None, ...):



    if vocab_dict is None:



        build vocab from train



    else:



        reuse vocab_dict for valid/test



```







细致讲解：



- 训练集构建词表，验证/测试共享，确保 ID 一致。



- 过滤长度不合格样本，避免极端长句影响训练。







================================================================================







【文件3: gru_nmt.py - RNN Seq2Seq 模型】







Attention 计算：



```python



scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2) / sqrt(d)



attn = softmax(scores)  # (B, L)



ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)  # (B, D)



```







张量形状：



- query: (B, D)



- keys/values: (B, L, D)



- scores: (B, L)



- ctx: (B, D)







Encoder：



```python



enc_out, h = self.enc(src, src_len)



```



张量形状：



- src: (B, S)



- src_len: (B,)



- enc_out: (B, S, H)



- h: (layers, B, H) 或 tuple(LSTM)







Decoder step：



```python



pred, h = self.dec.step(inp, h, enc_out, mask)



```



张量形状：



- inp: (B,)



- pred: (B, V)







Seq2Seq forward（teacher forcing）：



```python



outs: (B, T, V)



```







greedy_translate：



```python



result: (B, T')



```







beam_search（当前实现 batch=1）：



```python



output: (1, T')



```







细致讲解：



- dot/multiplicative/additive 注意力可切换，影响对齐与速度。



- tf_ratio 越大训练越稳，但推理时偏差更大。



- repetition_penalty 抑制重复译文。







================================================================================







【文件4: transformer_nmt.py - Transformer 模型】







RelativeMultiheadAttention：



```python



query = (B, nhead, Lq, d_k)



key   = (B, nhead, Lk, d_k)



scores = (B, nhead, Lq, Lk)



```







make_std_mask：



```python



tgt_mask: (B, 1, T, T)



src_mask: (B, 1, 1, S)



```







encode/decode：



- encode 输出 memory: (B, S, D)



- decode 输出 out: (B, T, D)



- generator 输出 logits: (B, T, V)







translate（greedy）：



- 输入 src: (B, S)



- 输出 ys: (B, T')







细致讲解：



- RMSNorm 替代 LayerNorm，计算更快。



- 相对位置编码通过 bias 加入注意力分数。



- Pre-Norm 结构提升训练稳定性。







================================================================================







【文件5: main.py - 训练主脚本】







训练流程：



1) prepare_data -> train_loader / valid_loader



2) train_epoch



3) validate_loss + evaluate(BLEU)



4) 日志 + 保存 best ckpt



5) EarlyStopping 检查







train_epoch 关键张量：



- RNN：out (B, T, V) -> loss on out[:,1:] vs tgt[:,1:]



- Transformer：out (B, T-1, V) -> loss on tgt[:,1:]







validate_loss 与 train_epoch 相同，但无梯度。







日志格式：



```



Epoch N: Loss=... ValLoss=... BLEU=...



```







Resume：



- 加 --resume 会读取 ckpt/<model_tag>_best.pt 并继续写日志。



- epoch 编号根据历史 log 自动接续。







================================================================================







【文件6: inference.py - 推理脚本】







translate_sentence：



- 输入：text(str)



- 输出：英文译文(str)







张量形状：



- src: (1, S)



- RNN 输出：pred_ids (1, T')



- Transformer 输出：pred_ids (1, T')







detokenize：



把 "hello , world ." -> "hello, world."







细致讲解：



- RNN 支持 greedy/beam。



- Transformer 默认 beam，失败时回退 greedy。



- repetition_penalty 减少重复。







================================================================================







【文件7: rnn_sweep.py - RNN 批量实验】







流程：



RUNS -> 构造命令 -> subprocess.run



每个配置一个 log + ckpt







关键点：



- RESUME=True 会加 --resume



- 有日志但 RESUME=True 不跳过







================================================================================







【文件7-2: transformer_sweep.py - Transformer 对比实验】







当前一晚跑的必要组合（7 组）：



1) relative + rmsnorm + base scale (d=512,h=8,enc/dec=3/3,d_ff=2048) + batch=64 + lr=5e-4



2) relative + layernorm + base scale + batch=64 + lr=5e-4



3) absolute + rmsnorm + base scale + batch=64 + lr=5e-4



4) absolute + layernorm + base scale + batch=64 + lr=5e-4



5) relative + rmsnorm + base scale + batch=32 + lr=5e-4



6) relative + rmsnorm + base scale + batch=64 + lr=1e-4



7) relative + rmsnorm + small scale (d=256,h=4,enc/dec=2/2,d_ff=1024) + batch=64 + lr=5e-4







核心逻辑：



- 固定 7 组配置，避免组合爆炸



- transformer_sweep.py 顺序执行，结束后调用 analyze_results.py



- 日志/ckpt 命名：



  transformer_{pos}_{norm}_d{d}_h{h}_e{enc}{dec}_ff{d_ff}_bs{batch}_lr{lr}







================================================================================







【文件8: rnn_test_sweep.py - RNN 测试集评估】







流程：



ckpt 列表 -> 逐个推理 -> BLEU/GLEU -> summary.csv + plot







输出：



- 每行输出 jsonl：{"src","en_ref","en_pred"}



- summary.csv 记录 decode 参数与 device







================================================================================







【文件9: analyze_results.py - 训练日志汇总】







输入/输出：



- 输入：*_history.log



- 输出：nmt_comparison_summary.csv / .png







关键逻辑：



```python



LOG_PATTERN = r"Epoch\\s+(\\d+).*?Loss=([\\d\\.]+).*?BLEU=([\\d\\.]+)"



```







================================================================================







【文件10: compute_bleu.py - 外部 BLEU 计算】







输入/输出：



- 输入：*_test_results.jsonl



- 输出：BLEU 分数







================================================================================







【文件11: plot_final_comparision.py / visualize_results.py】







- plot_final_comparision.py：画训练 Loss/BLEU 曲线



- visualize_results.py：画 latency/throughput 对比（数据需手填）







================================================================================







【文件12: finetune_t5.py】







流程：



tokenizer -> DataLoader -> T5ForConditionalGeneration -> BLEU







================================================================================



                              实验结果对比分析（RNN）



================================================================================







【训练集/验证集（来自 nmt_comparison_summary.csv）】



1) 最优验证 BLEU（Best BLEU）



   - rnn_lstm_dot_product_tf0.50_greedy: 10.47 (epoch 9)



   - rnn_lstm_additive_tf0.50_greedy: 10.46 (epoch 1)



   - rnn_gru_multiplicative_tf0.50_greedy: 10.29 (epoch 7)



   - rnn_gru_additive_tf0.50_greedy: 9.25 (epoch 1)



   - rnn_gru_dot_product_tf0.50_greedy: 9.23 (epoch 2)



   - rnn_gru_dot_product_tf0.00_greedy: 8.35 (epoch 8)







2) 最终验证 BLEU（Final BLEU）



   - rnn_lstm_additive_tf0.50_greedy: 9.69



   - rnn_lstm_dot_product_tf0.50_greedy: 9.39



   - rnn_gru_additive_tf0.50_greedy: 8.64



   - rnn_gru_dot_product_tf0.50_greedy: 8.00



   - rnn_gru_multiplicative_tf0.50_greedy: 7.92



   - rnn_gru_dot_product_tf0.00_greedy: 7.14







3) 参数量（Params）



   - GRU 系列约 26.1M~26.7M



   - LSTM 系列约 28.3M~28.9M



   - 基线 GRU (gru_history.log) 约 34.9M（旧模型对照）







结论（验证集）：



- LSTM 系列整体优于 GRU（Best BLEU 与 Final BLEU 都更高）。



- dot/additive 注意力在 LSTM 上表现接近，multiplicative 在 GRU 上更稳。



- teacher_forcing=0.0 明显偏弱（Best BLEU 仅 8.35）。







【测试集（来自 rnn_test_results/summary.csv）】



测试集样本数：200，解码策略：greedy（beam_width=5 仅作为参数记录，未启用 beam）







- rnn_gru_additive_tf0.50_greedy: BLEU 1.72 / GLEU 1.72



- rnn_gru_multiplicative_tf0.50_greedy: BLEU 1.08 / GLEU 1.08



- rnn_lstm_dot_product_tf0.50_greedy: BLEU 0.74 / GLEU 0.74



- rnn_gru_dot_product_tf0.50_greedy: BLEU 0.70 / GLEU 0.70



- rnn_lstm_additive_tf0.50_greedy: BLEU 0.27 / GLEU 0.27



- rnn_gru_dot_product_tf0.00_greedy: BLEU 0.22 / GLEU 0.22







结论（测试集）：



- 测试 BLEU 明显低于验证 BLEU，可能存在：



  1) 数据分布差异（valid/test domain shift）



  2) 分词/字段不匹配导致 UNK 增多



  3) 仅使用 greedy 解码导致指标偏低



- 当前最优测试结果是 rnn_gru_additive_tf0.50_greedy，但整体仍很低。







【生成样例观察（sample_comparisons.jsonl）】



- 多模型输出中 <unk> 较多，句子不流畅，说明词表覆盖或泛化不足。



- 说明训练集与测试集词分布差异明显，需进一步清洗或提升词表覆盖率。







【图像产物】



- 训练对比图：nmt_comparison_summary.png



- RNN 测试对比图：rnn_test_results/bleu_comparison.png



- 最优模型对比图：nmt_best_model_comparison.png



- 训练日志 CSV：nmt_comparison_summary.csv



- 测试评估 CSV：rnn_test_results/summary.csv







================================================================================



                              RNN vs Transformer 对比表（Markdown）



================================================================================







【测试集评测汇总（BLEU/GLEU）】



| model | bleu | gleu | samples | ckpt_path | decode | beam_width | max_len | repetition_penalty | device |



|---|---|---|---|---|---|---|---|---|---|



| rnn_gru_additive_tf0.50_greedy | 1.72 | 1.72 | 200 | ckpt\rnn_gru_additive_tf0.50_greedy_best.pt | greedy | 5 | 80 | 1.2 | cuda |



| rnn_gru_multiplicative_tf0.50_greedy | 1.08 | 1.08 | 200 | ckpt\rnn_gru_multiplicative_tf0.50_greedy_best.pt | greedy | 5 | 80 | 1.2 | cuda |



| rnn_lstm_dot_product_tf0.50_greedy | 0.74 | 0.74 | 200 | ckpt\rnn_lstm_dot_product_tf0.50_greedy_best.pt | greedy | 5 | 80 | 1.2 | cuda |



| rnn_gru_dot_product_tf0.50_greedy | 0.70 | 0.70 | 200 | ckpt\rnn_gru_dot_product_tf0.50_greedy_best.pt | greedy | 5 | 80 | 1.2 | cuda |



| rnn_lstm_additive_tf0.50_greedy | 0.27 | 0.27 | 200 | ckpt\rnn_lstm_additive_tf0.50_greedy_best.pt | greedy | 5 | 80 | 1.2 | cuda |



| rnn_gru_dot_product_tf0.00_greedy | 0.22 | 0.22 | 200 | ckpt\rnn_gru_dot_product_tf0.00_greedy_best.pt | greedy | 5 | 80 | 1.2 | cuda |







【训练/验证汇总（Best BLEU 排序）】



| model | epochs | best_bleu | best_epoch | final_bleu | epoch_to_90pct_best | params | log_path | ckpt_path |



|---|---|---|---|---|---|---|---|---|



| transformer_relative_rmsnorm_d256_h4_e22_ff1024_bs64_lr0.0005 | 30 | 19.47 | 27 | 19.42 | 15 | 12932884 | transformer_relative_rmsnorm_d256_h4_e22_ff1024_bs64_lr0.0005_history.log | ckpt\transformer_relative_rmsnorm_d256_h4_e22_ff1024_bs64_lr0.0005_best.pt |



| transformer_absolute_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 23 | 19.0 | 18 | 17.67 | 6 | 40545296 | transformer_absolute_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_history.log | ckpt\transformer_absolute_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt |



| transformer_relative_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 19 | 18.71 | 14 | 18.33 | 9 | 40555032 | transformer_relative_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_history.log | ckpt\transformer_relative_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt |



| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 26 | 18.46 | 21 | 17.7 | 10 | 40547352 | transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_history.log | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt |



| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs32_lr0.0005 | 17 | 18.45 | 12 | 16.9 | 6 | 40547352 | transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs32_lr0.0005_history.log | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs32_lr0.0005_best.pt |



| transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 21 | 18.4 | 11 | 17.03 | 5 | 40552976 | transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_history.log | ckpt\transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt |



| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0001 | 30 | 10.69 | 29 | 10.57 | 25 | 40547352 | transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0001_history.log | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0001_best.pt |



| rnn_lstm_dot_product_tf0.50_greedy | 28 | 10.47 | 9 | 9.39 | 1 | 28328716 | rnn_lstm_dot_product_tf0.50_greedy_history.log | ckpt\rnn_lstm_dot_product_tf0.50_greedy_best.pt |



| rnn_lstm_additive_tf0.50_greedy | 22 | 10.46 | 1 | 9.69 | 1 | 28853516 | rnn_lstm_additive_tf0.50_greedy_history.log | ckpt\rnn_lstm_additive_tf0.50_greedy_best.pt |



| rnn_gru_multiplicative_tf0.50_greedy | 17 | 10.29 | 7 | 7.92 | 1 | 26444556 | rnn_gru_multiplicative_tf0.50_greedy_history.log | ckpt\rnn_gru_multiplicative_tf0.50_greedy_best.pt |



| rnn_gru_additive_tf0.50_greedy | 19 | 9.25 | 1 | 8.64 | 1 | 26707212 | rnn_gru_additive_tf0.50_greedy_history.log | ckpt\rnn_gru_additive_tf0.50_greedy_best.pt |



| rnn_gru_dot_product_tf0.50_greedy | 18 | 9.23 | 2 | 8.0 | 1 | 26182412 | rnn_gru_dot_product_tf0.50_greedy_history.log | ckpt\rnn_gru_dot_product_tf0.50_greedy_best.pt |



| rnn_gru_dot_product_tf0.00_greedy | 17 | 8.35 | 8 | 7.14 | 2 | 26182412 | rnn_gru_dot_product_tf0.00_greedy_history.log | ckpt\rnn_gru_dot_product_tf0.00_greedy_best.pt |







================================================================================



                              使用方法



================================================================================







1) 训练 RNN（单次）



   python main.py --model rnn --cell gru --attention dot_product --teacher-forcing 0.5 --decode greedy --device cuda







2) 训练 Transformer



   python main.py --model transformer --device cuda







3) 批量 sweep



   python rnn_sweep.py







4) 推理（单句）



   python inference.py --model rnn --ckpt .\ckpt\rnn_gru_dot_product_tf0.50_greedy_best.pt --text "今天天气不错，我们去公园散步吧。"







5) Test 评估（全部 RNN）



   python rnn_test_sweep.py --test-file "C:\Users\250010024\Desktop\datasets\test_retranslated_hunyuan.jsonl" --device cuda







================================================================================



                              依赖库



================================================================================



torch, jieba, sacrebleu, tqdm, matplotlib, seaborn(可选), transformers(T5)











================================================================================






================================================================================






- rnn_*_history.log



- transformer_*_history.log













- rnn_train_bleu_over_epochs.png



- transformer_train_bleu_over_epochs.png










- rnn_test_results/bleu_comparison.png



- transformer_test_results/bleu_comparison.png










- rnn_test_results/best_rnn_test_results.jsonl



  best: rnn_gru_additive_tf0.50_greedy



- transformer_test_results/best_transformer_test_results.jsonl



  best: transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005










- nmt_best_model_comparison.png



- nmt_final_comparison_log.png









================================================================================




================================================================================







| model | bleu | gleu | samples | ckpt_path | decode | beam_width | max_len | repetition_penalty | device |


|---|---|---|---|---|---|---|---|---|---|


| transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 7.18 | 7.18 | 200 | ckpt\transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |


| transformer_absolute_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 6.65 | 6.65 | 200 | ckpt\transformer_absolute_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |


| transformer | 4.65 | 4.65 | 200 | ckpt\transformer_best.pt | beam | 5 | 80 | 1.2 | cuda |


| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 3.91 | 3.91 | 200 | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |


| transformer_relative_rmsnorm_d256_h4_e22_ff1024_bs64_lr0.0005 | 3.62 | 3.62 | 200 | ckpt\transformer_relative_rmsnorm_d256_h4_e22_ff1024_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |


| transformer_relative_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 3.34 | 3.34 | 200 | ckpt\transformer_relative_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |


| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs32_lr0.0005 | 2.70 | 2.70 | 200 | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs32_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |


| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0001 | 1.65 | 1.65 | 200 | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0001_best.pt | beam | 5 | 80 | 1.2 | cuda |












================================================================================
================================================================================

- rnn_*_history.log
- transformer_*_history.log

- rnn_train_bleu_over_epochs.png
- transformer_train_bleu_over_epochs.png

- rnn_test_results/bleu_comparison.png
- transformer_test_results/bleu_comparison.png

- rnn_test_results/best_rnn_test_results.jsonl
  best: rnn_gru_additive_tf0.50_greedy
- transformer_test_results/best_transformer_test_results.jsonl
  best: transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005

- nmt_best_model_comparison.png
- nmt_final_comparison_log.png


```python
LOG_PATTERN = re.compile(r"Epoch\s+(\d+).*?BLEU=([\d\.]+)")
for path in logs:
    epochs, bleus = parse_bleu(path)
    ax.plot(epochs, bleus, marker="o", linewidth=1.8, markersize=3, label=label)
```

```python
LOG_PATTERN = re.compile(r"Epoch\s+(\d+).*?BLEU=([\d\.]+)")
for path in logs:
    epochs, bleus = parse_bleu(path)
    ax.plot(epochs, bleus, marker="o", linewidth=1.8, markersize=3, label=label)
```

```python
ax1.plot(x, bleu, marker="o", linewidth=2.5)
ax2.bar(x, bleu, alpha=0.9)
```

RNN?
```powershell
$env:PYTHONIOENCODING='utf-8'; python inference.py --model rnn --ckpt .\ckpt\rnn_gru_additive_tf0.50_greedy_best.pt --input_file C:\Users\250010024\Desktop\datasets\test_retranslated_hunyuan.jsonl --output_file rnn_test_results\best_rnn_test_results.jsonl
```

Transformer?
```powershell
$env:PYTHONIOENCODING='utf-8'; python inference.py --model transformer --ckpt .\ckpt\transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt --input_file C:\Users\250010024\Desktop\datasets\test_retranslated_hunyuan.jsonl --output_file transformer_test_results\best_transformer_test_results.jsonl
```

================================================================================
================================================================================

| model | bleu | gleu | samples | ckpt_path | decode | beam_width | max_len | repetition_penalty | device |
|---|---|---|---|---|---|---|---|---|---|
| transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 7.18 | 7.18 | 200 | ckpt\transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_absolute_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 6.65 | 6.65 | 200 | ckpt\transformer_absolute_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 3.91 | 3.91 | 200 | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_rmsnorm_d256_h4_e22_ff1024_bs64_lr0.0005 | 3.62 | 3.62 | 200 | ckpt\transformer_relative_rmsnorm_d256_h4_e22_ff1024_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 3.34 | 3.34 | 200 | ckpt\transformer_relative_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs32_lr0.0005 | 2.70 | 2.70 | 200 | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs32_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0001 | 1.65 | 1.65 | 200 | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0001_best.pt | beam | 5 | 80 | 1.2 | cuda |


================================================================================
                              Latest Results (Updated)
================================================================================

Updated artifacts:
- nmt_comparison_summary.csv / nmt_comparison_summary.png (regenerated with improved layout)
- rnn_train_bleu_over_epochs.png (RNN train BLEU curves)
- transformer_train_bleu_over_epochs.png (Transformer train BLEU curves)
- rnn_test_results/bleu_comparison.png (test BLEU line + bar)
- transformer_test_results/bleu_comparison.png (test BLEU line + bar)
- rnn_test_results/best_rnn_test_results.jsonl
- transformer_test_results/best_transformer_test_results.jsonl (generated after running best test inference)

Transformer test summary (BLEU/GLEU):
| model | bleu | gleu | samples | ckpt_path | decode | beam_width | max_len | repetition_penalty | device |
|---|---|---|---|---|---|---|---|---|---|
| transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 7.18 | 7.18 | 200 | ckpt\transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_absolute_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 6.65 | 6.65 | 200 | ckpt\transformer_absolute_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 3.91 | 3.91 | 200 | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_rmsnorm_d256_h4_e22_ff1024_bs64_lr0.0005 | 3.62 | 3.62 | 200 | ckpt\transformer_relative_rmsnorm_d256_h4_e22_ff1024_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005 | 3.34 | 3.34 | 200 | ckpt\transformer_relative_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs32_lr0.0005 | 2.70 | 2.70 | 200 | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs32_lr0.0005_best.pt | beam | 5 | 80 | 1.2 | cuda |
| transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0001 | 1.65 | 1.65 | 200 | ckpt\transformer_relative_rmsnorm_d512_h8_e33_ff2048_bs64_lr0.0001_best.pt | beam | 5 | 80 | 1.2 | cuda |

Best transformer on test: transformer_absolute_layernorm_d512_h8_e33_ff2048_bs64_lr0.0005
