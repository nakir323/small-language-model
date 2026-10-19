"""
青空文庫の日本昔話（楠山正雄）を集めて、全部ひらがなにした学習データを作る。

やること:
  1. 作品ごとの zip を青空文庫からダウンロード（data/raw/ にキャッシュ）
     zip の URL は分かっているので直接取る。1リクエストごとに 0.5 秒あける
  2. ヘッダ・フッタ・ルビ・注記を取り除く
  3. fugashi（形態素解析器）で読みを取り、漢字・カタカナをひらがなに変換
  4. 全作品をつなげて data/mukashibanashi.txt に書き出す

実行: uv run data/prepare_mukashibanashi.py
"""

import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import fugashi

# (作品名, zip ファイル名)。ファイル名の先頭の数字は青空文庫の作品カード番号
BASE_URL = "https://www.aozora.gr.jp/cards/000329/files"
WORKS = [
    ("桃太郎", "18376_ruby_12074"),
    ("花咲かじじい", "3391_ruby"),
    ("かちかち山", "18377_ruby_11923"),
    ("猿かに合戦", "18334_ruby_11929"),
    ("舌切りすずめ", "18378_ruby_12072"),
    ("一寸法師", "43457_ruby_23696"),
    ("浦島太郎", "3390_ruby_6090"),
    ("文福茶がま", "18336_ruby_11922"),
    ("ねずみの嫁入り", "18335_ruby_11926"),
    ("金太郎", "18337_ruby_11924"),
    ("瓜子姫子", "43459_ruby_24171"),
    ("瘤とり", "43461_ruby_23698"),
    ("姨捨山", "43460_ruby_24172"),
    ("一本のわら", "43458_ruby_23697"),
    ("鉢かつぎ", "50756_ruby_76500"),
    ("しっぺい太郎", "50755_ruby_76474"),
    ("田原藤太", "18338_ruby_13237"),
    ("牛若と弁慶", "18384_ruby_14095"),
    ("大江山", "18339_ruby_13234"),
    ("安達が原", "33208_ruby_13233"),
    ("鬼六", "18389_ruby_11927"),
    ("和尚さんと小僧", "18387_ruby_11928"),
    ("長い名", "18388_ruby_11925"),
    ("くらげのお使い", "18379_ruby_12073"),
    ("たにしの出世", "3389_ruby_5672"),
    ("物のいわれ", "18390_ruby_13242"),
    ("猫の草紙", "18380_ruby_12075"),
    ("白い鳥", "33210_ruby_13236"),
    ("松山鏡", "33214_ruby_14592"),
    ("三輪の麻糸", "33215_ruby_13241"),
    ("雷のさずけもの", "33209_ruby_14591"),
    ("赤い玉", "33207_ruby_13479"),
    ("春山秋山", "33212_ruby_14590"),
    ("人馬", "33213_ruby_24173"),
    ("夢占", "33216_ruby_13243"),
    ("忠義な犬", "33211_ruby_13238"),
    ("山姥の話", "43462_ruby_24174"),
    ("葛の葉狐", "18386_ruby_13235"),
    ("殺生石", "18385_ruby_11076"),
    ("鵺", "18342_ruby_13239"),
    ("羅生門", "18340_ruby_13244"),
    ("八幡太郎", "18341_ruby_13240"),
    ("田村将軍", "18382_ruby_14097"),
    ("鎮西八郎", "18383_ruby_13484"),
    ("夢殿", "18381_ruby_14098"),
]

RAW_DIR = Path(__file__).parent / "raw"
OUT_PATH = Path(__file__).parent / "mukashibanashi.txt"


def download(name: str) -> str:
    """青空文庫から zip を取ってきて、中の .txt を Shift_JIS として読む。
    一度取った zip は data/raw/ に置いて再利用する"""
    dest = RAW_DIR / f"{name}.zip"
    if not dest.exists():
        url = f"{BASE_URL}/{name}.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "small-language-model study script"})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        time.sleep(0.5)  # 青空文庫に優しく
    with zipfile.ZipFile(io.BytesIO(dest.read_bytes())) as z:
        txt = next(n for n in z.namelist() if n.endswith(".txt"))
        return z.read(txt).decode("cp932", errors="replace")


def clean(text: str) -> str:
    """青空文庫のヘッダ・フッタ・ルビ・注記を取り除く"""
    lines = text.splitlines()
    # ヘッダ: 「-----」だけの行が2回出てくるまで
    dash = [i for i, l in enumerate(lines) if re.fullmatch(r"-{10,}", l.strip())]
    if len(dash) >= 2:
        lines = lines[dash[1] + 1 :]
    else:
        lines = lines[1:]  # タイトル行だけ飛ばす
    # フッタ: 「底本：」以降
    for i, l in enumerate(lines):
        if l.startswith("底本"):
            lines = lines[:i]
            break
    body = "\n".join(lines)
    body = re.sub(r"《.*?》", "", body)      # ルビ
    body = body.replace("｜", "")             # ルビの開始記号
    body = re.sub(r"［＃.*?］", "", body)     # 注記
    body = re.sub(r"^[\u3000 ]*[一二三四五六七八九十]+[\u3000 ]*$", "", body, flags=re.M)  # 章番号だけの行（字下げ付きも）
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


_tagger = fugashi.Tagger()  # pyright: ignore[reportAttributeAccessIssue]  Tagger は .so の中で型情報がない


def _kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def to_hiragana(text: str) -> str:
    """1行ずつ形態素解析して読み（カタカナ）を取り、ひらがなに直す。改行はそのまま残す"""
    out = []
    for line in text.split("\n"):
        words = []
        for w in _tagger(line):
            reading = w.feature.kana or w.surface  # 読みが取れない記号などは表層のまま
            words.append(_kata_to_hira(reading))
        out.append("".join(words))
    return "\n".join(out)


def main():
    RAW_DIR.mkdir(exist_ok=True)
    parts = []
    for title, name in WORKS:
        try:
            body = to_hiragana(clean(download(name)))
            parts.append(body)
            print(f"ok   {title:12s} {len(body):6d} 文字")
        except Exception as e:
            print(f"skip {title:12s} {e}")
    text = "\n\n".join(parts) + "\n"
    OUT_PATH.write_text(text, encoding="utf-8")
    chars = sorted(set(text))
    print(f"\n合計 {len(text)} 文字, 語彙 {len(chars)} 種類 -> {OUT_PATH}")
    print("語彙:", "".join(chars))


if __name__ == "__main__":
    main()
