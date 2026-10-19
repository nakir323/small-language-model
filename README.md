# small-language-model

言語モデルを最小実装から段階的に育てて理解するためのリポジトリ。

## セットアップ

```bash
brew install uv
uv sync
uv run data/prepare_mukashibanashi.py   # 青空文庫の昔話をひらがな化 → data/mukashibanashi.txt
uv run step1_bigram/train.py             # 学習 → step1_bigram/model.pt（数秒）
uv run step1_bigram/generate.py むかしむかし
```

学習済みの `model.pt` は commit していない（`.gitignore`）。数秒で再現できるため。

## 構成

```
data/
  prepare_mukashibanashi.py   楠山正雄の日本昔話45本を青空文庫から取得してひらがな化する
  mukashibanashi.txt          約22万文字、語彙87種類（gitignore）
step1_bigram/         Step1: Bigram 言語モデル
  model.py            モデル定義。vocab_size × vocab_size の表1枚
  train.py            学習 → model.pt
  generate.py         model.pt → 文章生成
```

各 step は単体で読めるように独立している。step 間で `model.py` だけが大きく変わり、
`train.py` と `generate.py` はほぼ同じ、というのを見るのが狙い。

## データ

全部ひらがなにしているのは、語彙を 87 種類に抑えるため（漢字混じりだと 2000〜3000 種類に
なり、Bigram の表がスカスカになる）。漢字の読みは形態素解析器 fugashi で取っている。

## Step1: Bigram

「今の1文字」だけを見て「次の1文字」を当てるモデル。

```bash
uv run step1_bigram/train.py                   # 数秒で終わる
uv run step1_bigram/generate.py むかしむかし     # 出だしを指定して生成
uv run step1_bigram/inspect_model.py む         # model.pt の中身（「む」の次に来る文字）を見る
```

文脈を直前1文字しか見ないので出力はほぼデタラメだが、

- 文字 ↔ 番号 の変換
- バッチの切り出し（x と、1文字ずらした y）
- loss の計算と backward による更新
- 学習済みの重みからのサンプリング

という学習と生成の骨組みは全部入っている。
