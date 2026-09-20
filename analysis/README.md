# analysis — アプリ層

[core/](../core/)（計測）と [domains/](../domains/)（競技）をつないで、
返す形を決めるだけの薄い層。

```python
import analysis

metrics, feedback = analysis.analyze(joints, fps=60)                # テニス
metrics, feedback = analysis.analyze(joints, 60, "golf_swing")      # ゴルフ
print(analysis.format_report(metrics, feedback))

res = analysis.analyze_json(joints, 60)   # Web/ビューア用（numpy を残さない）
```

ドメインを省略するとテニスのサーブになる（`domains.DEFAULT`）。

## CLI

```bash
python -m analysis --list
python -m analysis --joints gv_joints.npy --fps 120
python -m analysis --joints out/joints.npy --fps 60 --domain golf_swing --save out
```

## 入力

`backend/` が出力する関節ファイル 1つ。

| ファイル | 形状 | 内容 |
|---|---|---|
| `gv_joints.npy` | (F, 24, 3) | SMPL 24関節の world座標 [m] |

他の骨格（GEM-X の SOMA 77関節など）は `core.convert` で並べ替えてから渡す。
重心・上軸・利き側はすべて関節から導出するので、入力はこの1ファイルでよい。

## テスト

```bash
uv pip install -e ".[dev]"
pytest
```

`tests/synth.py` が合成サーブを生成する。タイミングは秒で定義してあり、
fps を変えても同じ実時間の動作になる。このテストは実際に2つのバグを検出した。
閾値やルールを触ったら必ず走らせること。
