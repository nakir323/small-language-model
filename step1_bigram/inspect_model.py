"""
Step1: model.pt の中身を見る

  uv run step1_bigram/inspect_model.py        # 表の形と、全体の様子
  uv run step1_bigram/inspect_model.py む     # 「む」の行（次に来る文字の確率）を見る
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ckpt = torch.load(Path(__file__).parent / "model.pt")
chars = ckpt["chars"]
table = ckpt["state_dict"]["table.weight"]  # (vocab_size, vocab_size)

print(f"keys       : {list(ckpt.keys())}")
print(f"vocab_size : {len(chars)}")
print(f"table.shape: {tuple(table.shape)}  ({table.numel()} マス, float32 で {table.numel() * 4 / 1024:.1f} KB)")
print(f"値の範囲   : {table.min():.2f} 〜 {table.max():.2f}")

if len(sys.argv) > 1:
    c = sys.argv[1]
    row = table[chars.index(c)]
    probs = F.softmax(row, dim=0)
    print(f"\n'{c}' の行（スコア → 確率, 上位10件）")
    for v, i in zip(*torch.topk(probs, 10)):
        print(f"  {chars[i]!r:6} スコア {row[i]:6.2f}  確率 {v:.3f}")
else:
    print("\n各文字の「次に来る文字」1位だけ並べたもの:")
    for i, c in enumerate(chars):
        j = table[i].argmax().item()
        print(f"  {c!r} → {chars[j]!r}", end="   " if (i + 1) % 4 else "\n")
    print()
