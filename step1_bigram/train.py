"""
Step1: 学習スクリプト

ゴール: 「今の1文字の次に、どの文字が来やすいか」の表を、シェイクスピアの
テキストから作る。

具体的にやること（テキストが "hello world" だったとして）:

  1. 文字 ↔ 番号 の対応表を作る
       テキストに出てくる文字を集めて番号を振る。
       ' '→0, 'd'→1, 'e'→2, 'h'→3, 'l'→4, 'o'→5, 'r'→6, 'w'→7
       モデルは文字そのものを扱えないので、以後は全部この番号で扱う。

  2. テキスト全体を番号の列に変換する
       "hello world" → [3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1]
       これが学習データ。実際のデータは 111万文字なので長さ 111万の列になる。

  3. ランダムな位置から短い断片を切り出す
       長さ 111万の列を一気に扱えないので、適当な位置から BLOCK_SIZE 文字だけ取る。
       例: 位置 1 から 4 文字取ると
         x = [2, 4, 4, 5]   "ello"   ← 入力
         y = [4, 4, 5, 0]   "llo "   ← x を 1 文字右にずらしたもの
       "ello" → "llo " という文字列の対応ではなく、同じ位置を縦に見たペアが 4 個:
         位置:   0    1    2    3
         x:     'e'  'l'  'l'  'o'
         y:     'l'  'l'  'o'  ' '
                 ↑    ↑    ↑    ↑
                eの次 lの次 lの次 oの次
       Bigram は 1 文字しか見ないので、この 4 ペアはお互い無関係な独立した問題。
       4 文字まとめて切り出すのは、1 ペアずつ取るより効率がいいからというだけ。
       これを BATCH_SIZE 組まとめて 1 回分の学習に使う。

  4. モデルに食わせて loss を出し、backward で表を更新する
       モデル（= 8×8 の表）に x を渡すと、各文字について「次の文字候補 8 個の
       スコア」が返ってくる。最初は表がランダムなのでスコアもデタラメ。
       loss = 「正解 y の文字に、どれだけ低いスコアを付けてしまったか」の数値。
       backward() が「各マスの数値を増やすべきか減らすべきか（と、どのくらい）」を
       全マス分計算し、optimizer.step() がその通りに各マスの数値を少しだけ足し引きする。
       マスの位置は変わらない。中の数値だけが変わる。
       結果、'e' の行の 'l' の列の数値が 0.60 → 1.03 のように少し大きくなり、
       他の列は少し小さくなる、みたいなことが起きる。

  5. 3〜4 を STEPS 回繰り返し、最後に表を model.pt に保存する
       繰り返すうちに loss が下がる（= 次の文字を当てられるようになる）。
       保存した表を generate.py が読んで文章を作る。

実行: uv run step1_bigram/train.py
"""

from pathlib import Path

import torch
from model import BigramLanguageModel

# ---- 設定 ----
DATA_PATH = Path(__file__).parent.parent / "data" / "mukashibanashi.txt"
SAVE_PATH = Path(__file__).parent / "model.pt"
BATCH_SIZE = 32      # 一度に処理する断片の数
BLOCK_SIZE = 8       # 断片1つの長さ（文字数）
STEPS = 3000         # 学習の繰り返し回数
LEARNING_RATE = 1e-2 # 1回の更新で表をどれだけ動かすか
torch.manual_seed(1337)

# ---- 1. 文字 ↔ 番号 の対応表 ----
text = DATA_PATH.read_text()
chars = sorted(set(text))              # テキストに出てくる文字を全部集める
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # 文字 → 番号
itos = {i: ch for i, ch in enumerate(chars)}  # 番号 → 文字
print(f"vocab_size = {vocab_size}")
print(f"chars = {''.join(chars)!r}")

# ---- 2. テキスト全体を番号の列に ----
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
print(f"data.shape = {tuple(data.shape)}  (文字数)")


# ---- 3. バッチの切り出し ----
def get_batch():
    """ランダムな位置から BLOCK_SIZE 文字の断片を BATCH_SIZE 個取る。
    x = 入力、y = x を1文字右にずらしたもの（= 各位置での「次の文字」）"""
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y


# ---- 4〜5. 学習ループ ----
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for step in range(STEPS):
    x, y = get_batch()
    _, loss = model(x, y)           # 順伝播: 今の表でどれくらい外してるか
    optimizer.zero_grad()           # 前回の勾配を消す
    loss.backward()                 # 逆伝播: 表の各数値をどっちに動かすべきか計算
    optimizer.step()                # その方向に少し動かす
    if step % 300 == 0 or step == STEPS - 1:
        print(f"step {step:5d}  loss {loss.item():.4f}")

# 表（重み）と、文字の対応表をセットで保存する
torch.save({"state_dict": model.state_dict(), "chars": chars}, SAVE_PATH)
print(f"saved -> {SAVE_PATH}")
