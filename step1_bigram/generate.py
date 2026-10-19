"""
Step1: 生成スクリプト

model.pt を読み込んで、出だしの文字列に続きを生成する。

実行:
  uv run step1_bigram/generate.py              # 改行1文字から始める
  uv run step1_bigram/generate.py むかしむかし   # 出だしを指定
"""

import sys
from pathlib import Path

import torch
from model import BigramLanguageModel

LOAD_PATH = Path(__file__).parent / "model.pt"
MAX_NEW_TOKENS = 300

# 保存した表と文字の対応表を復元
ckpt = torch.load(LOAD_PATH)
chars = ckpt["chars"]
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

model = BigramLanguageModel(len(chars))
model.load_state_dict(ckpt["state_dict"])

# 出だし: 引数があればそれ、なければ改行1文字
prompt = sys.argv[1] if len(sys.argv) > 1 else "\n"
unknown = [ch for ch in prompt if ch not in stoi]
if unknown:
    sys.exit(f"学習データに無い文字が含まれています: {unknown}")

# 文字列 → 番号の列。形は (B=1, T=len(prompt))
start = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long)
out = model.generate(start, MAX_NEW_TOKENS)  # (1, len(prompt) + MAX_NEW_TOKENS)

# 番号の列 → 文字列（出だしも含めて表示）
print("".join(itos[i] for i in out[0].tolist()))
