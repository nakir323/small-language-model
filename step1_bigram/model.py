"""
Step1: Bigram 言語モデル

「今の1文字」だけを見て「次の1文字」を当てる、いちばん単純な言語モデル。
中身は vocab_size × vocab_size の数値の表が1枚あるだけ。

    表[今の文字][次の文字候補] = その候補の "スコア"（logit）

このスコアを softmax で確率に変換して、次の文字を選ぶ。
学習とは、この表の数値を「データに出てくる並びの確率が高くなる方向」に
少しずつ動かしていくこと。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # これがモデルの全パラメータ。
        # nn.Embedding は「行番号で引ける表」で、
        # 行 = 今の文字、列 = 次の文字候補ごとのスコア。
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) 文字番号の列。B=バッチ数, T=長さ
        targets: (B, T) 各位置の「正解の次の文字」。学習時だけ渡す

        返り値: logits (B, T, vocab_size) と loss
        """
        # 各位置の文字番号で表を引く → 次の文字候補のスコアが並ぶ
        logits = self.table(idx)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        # cross_entropy は「正解の文字のスコアが高いほど小さくなる値」。
        # (B*T, vocab_size) と (B*T,) の形に潰して渡す
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        idx: (B, T) 生成の出だし。ここから max_new_tokens 文字を継ぎ足す
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)              # (B, T, vocab_size)
            logits = logits[:, -1, :]          # 最後の文字の分だけ使う (B, vocab_size)
            probs = F.softmax(logits, dim=-1)  # スコア → 確率
            next_idx = torch.multinomial(probs, num_samples=1)  # 確率に従って1文字サンプリング (B, 1)
            idx = torch.cat([idx, next_idx], dim=1)  # 末尾に継ぎ足す (B, T+1)
        return idx
