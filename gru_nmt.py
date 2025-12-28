#!/usr/bin/env python3
"""
Plan 2: RNN-based NMT with configurable attention and decoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Luong/Bahdanau attention with multiple alignment functions."""

    def __init__(self, dim, attn_type="dot_product"):
        super().__init__()
        self.attn_type = attn_type
        self.scale = dim ** 0.5
        if attn_type == "multiplicative":
            self.linear_q = nn.Linear(dim, dim, bias=False)
        elif attn_type == "additive":
            self.linear_q = nn.Linear(dim, dim, bias=False)
            self.linear_k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, 1, bias=False)

    def _score(self, query, keys):
        if self.attn_type == "dot_product":
            scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2) / self.scale
        elif self.attn_type == "multiplicative":
            q = self.linear_q(query)
            scores = torch.bmm(keys, q.unsqueeze(2)).squeeze(2) / self.scale
        elif self.attn_type == "additive":
            q = self.linear_q(query).unsqueeze(1)
            k = self.linear_k(keys)
            scores = self.v(torch.tanh(q + k)).squeeze(2)
        else:
            raise ValueError(f"Unknown attention type: {self.attn_type}")
        return scores

    def forward(self, query, keys, values, mask=None):
        # query: (B, D), keys: (B, L, D)
        scores = self._score(query, keys)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
        return ctx, attn


class RNNEncoder(nn.Module):
    def __init__(self, vocab_sz, emb_sz, hid_sz, n_layers=2, drop=0.2, pad_id=0, cell_type="GRU"):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_id)
        rnn_cls = nn.GRU if cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(emb_sz, hid_sz, n_layers, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x, lens):
        emb = self.drop(self.embed(x))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, h


class RNNDecoder(nn.Module):
    def __init__(self, vocab_sz, emb_sz, hid_sz, n_layers=2, drop=0.2, pad_id=0,
                 cell_type="GRU", attn_type="dot_product"):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_id)
        self.attn = Attention(hid_sz, attn_type=attn_type)
        rnn_cls = nn.GRU if cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(emb_sz + hid_sz, hid_sz, n_layers, dropout=drop, batch_first=True)
        self.out = nn.Linear(hid_sz * 2, vocab_sz)
        self.drop = nn.Dropout(drop)
        self.vocab_sz = vocab_sz

    def _query(self, h):
        if isinstance(h, tuple):
            return h[0][-1]
        return h[-1]

    def step(self, inp, h, enc_out, mask):
        emb = self.drop(self.embed(inp)).unsqueeze(1)
        ctx, _ = self.attn(self._query(h), enc_out, enc_out, mask)
        gru_in = torch.cat([emb, ctx.unsqueeze(1)], dim=2)
        out, h = self.rnn(gru_in, h)
        pred = self.out(torch.cat([out.squeeze(1), ctx], dim=1))
        return pred, h


class Seq2SeqRNN(nn.Module):
    def __init__(self, enc, dec, device, bos_id, eos_id):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.device = device
        self.bos = bos_id
        self.eos = eos_id

    def forward(self, src, src_len, tgt, tf_ratio=0.0):
        B, T = tgt.shape
        outs = torch.zeros(B, T, self.dec.vocab_sz, device=self.device)
        enc_out, h = self.enc(src, src_len)
        mask = (src == 0)

        inp = tgt[:, 0]
        for t in range(1, T):
            pred, h = self.dec.step(inp, h, enc_out, mask)
            outs[:, t] = pred
            use_teacher = torch.rand(1).item() < tf_ratio
            inp = tgt[:, t] if use_teacher else pred.argmax(-1)
        return outs

    @torch.no_grad()
    def greedy_translate(self, src, src_len, max_len=50, repetition_penalty=1.2):
        """
        增强版贪心搜索：加入了 Repetition Penalty (重复惩罚)
        repetition_penalty > 1.0 时会抑制已经生成的词
        """
        self.eval()
        batch_size = src.size(0)
        enc_out, h = self.enc(src, src_len)
        mask = (src == 0)

        inp = torch.full((batch_size,), self.bos, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # 记录每个句子已经生成的词，用于惩罚
        generated_tokens = [[] for _ in range(batch_size)]
        result = [inp]

        for _ in range(max_len):
            pred, h = self.dec.step(inp, h, enc_out, mask)

            # --- 核心改进：重复惩罚 ---
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in generated_tokens[i]:
                        # 如果该词已经出现过，将其 logits 除以惩罚系数 (如果是负数则乘以)
                        if pred[i, token_id] < 0:
                            pred[i, token_id] *= repetition_penalty
                        else:
                            pred[i, token_id] /= repetition_penalty

            next_tok = pred.argmax(-1)
            next_tok = next_tok.masked_fill(finished, 0)

            # 记录生成的词
            for i, t in enumerate(next_tok.tolist()):
                if not finished[i]:
                    generated_tokens[i].append(t)

            result.append(next_tok)
            finished |= (next_tok == self.eos)
            if finished.all(): break
            inp = next_tok

        return torch.stack(result, dim=1)

    @torch.no_grad()
    def beam_search(self, src, src_len, beam_width=5, max_len=50):
        """
        实现标准的 Beam Search (目前仅支持 Batch Size = 1)
        """
        self.eval()
        # 1. Encode
        enc_out, h = self.enc(src, src_len)  # (1, L, D), (Layers, 1, D)
        mask = (src == 0)

        # 2. Prepare Beam
        # 每个候选: (accumulated_log_prob, last_token, hidden_state, sequence)
        # hidden_state 需要是 tuple (h) 对于 GRU
        hypotheses = [(0.0, torch.tensor([self.bos], device=self.device), h, [])]

        completed_hypotheses = []

        for _ in range(max_len):
            new_hypotheses = []

            for score, inp_seq, curr_h, seq_list in hypotheses:
                # 如果这个候选已经结束，或者是 PAD，跳过
                last_tok = inp_seq[-1].unsqueeze(0)  # (1)

                if last_tok.item() == self.eos:
                    completed_hypotheses.append((score, seq_list))
                    continue

                # Decode step
                pred, next_h = self.dec.step(last_tok, curr_h, enc_out, mask)
                log_probs = F.log_softmax(pred, dim=-1)  # (1, V)

                # Get top k
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    next_score = score + topk_probs[0, i].item()
                    next_id = topk_ids[0, i].item()
                    new_seq_list = seq_list + [next_id]

                    # 这里的 inp_seq 只是为了保持格式，实际上我们只需要 last_token
                    # 但为了通用性，我们还是传 tensor
                    new_input = torch.cat([inp_seq, topk_ids[0, i].unsqueeze(0)])

                    new_hypotheses.append((next_score, new_input, next_h, new_seq_list))

            # 排序并保留最好的 K 个
            hypotheses = sorted(new_hypotheses, key=lambda x: x[0], reverse=True)[:beam_width]

            # 如果所有的候选都结束了，退出
            if len(completed_hypotheses) >= beam_width:
                break

        # 如果没有完成的句子（比如超长），就取当前最好的
        if not completed_hypotheses:
            completed_hypotheses = [(score, seq) for score, _, _, seq in hypotheses]

        # 选得分最高的
        best_hyp = sorted(completed_hypotheses, key=lambda x: x[0], reverse=True)[0]
        best_seq = best_hyp[1]

        # 构造成 (1, L) 的 tensor 返回，兼容 batch 格式
        output = torch.tensor([self.bos] + best_seq, device=self.device).unsqueeze(0)
        return output


def create_rnn_model(src_vocab_sz, tgt_vocab_sz, cfg, device):
    cell_type = cfg.get("cell_type", "GRU").upper()
    attn_type = cfg.get("attention", "dot_product")
    enc = RNNEncoder(src_vocab_sz, cfg['embed_size'], cfg['hidden_size'],
                     cfg['num_layers'], cfg['dropout'], cell_type=cell_type)
    dec = RNNDecoder(tgt_vocab_sz, cfg['embed_size'], cfg['hidden_size'],
                     cfg['num_layers'], cfg['dropout'], cell_type=cell_type,
                     attn_type=attn_type)
    model = Seq2SeqRNN(enc, dec, device, bos_id=2, eos_id=3)
    return model.to(device)


def create_gru_model(src_vocab_sz, tgt_vocab_sz, cfg, device):
    return create_rnn_model(src_vocab_sz, tgt_vocab_sz, cfg, device)
