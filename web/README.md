# web — フロントエンド

issue #3 フェーズB。まず**単一HTMLのプロトタイプ**で動きを固め、後で Next.js/Vercel に載せ替える方針（B→A）。

## `index.html`（プロトタイプ）

1画面で完結する。依存は Three.js（CDN）のみ。

- 動画アップロード ＋ fps 入力
- Modal の Web API へ base64 で投入 → 5秒間隔でポーリング（「解析中… N秒」）
- 結果表示:
  - **コーチング文**（フィードバックを最大2件）
  - **主要数値**（沈み込み・打点・伸び上がり・膝角・打点タイミング・肘角）
  - **3Dビューア**（Three.js で骨格を自由視点。オレンジ=沈み込み、赤=打点）
  - **重ね合わせ動画**（左:元動画+3Dメッシュ / 右:別角度の3D）
  - **描き込み**（回転/描き込みモード切替。静止フレームに指・ペンで注釈。色4色+消去。
    コーチがiPad等で生徒に説明する用途。pointer eventsでタッチ対応）
  - **アバター再生**（VRMに動きを移して表示。🧍/🦴 で切替。[issue 005](../docs/issues/005-avatar-retargeting.md)）
  - fps<60 のとき「キネティックチェーンは判定不可」を明示

バックエンドは `https://<workspace>--serve-api.modal.run`（[../backend/](../backend/)）。

## ローカルで動かす

```bash
# サンプル結果を生成（自分のサーブの保存済み関節から）
python -c "import json,numpy as np,analysis; \
  json.dump(analysis.analyze_json(np.load('output/gv_joints.npy'),30.0), \
  open('web/result_sample.json','w'), ensure_ascii=False)"

# 配信して開く（file:// では fetch が制限されるため）
python web/devserver.py
# → http://127.0.0.1:8123/index.html
#   「(サンプル結果を読み込む)」で 3Dビューア等をオフライン確認できる
```

`result_sample.json` は生成物（自分のサーブ由来）なので gitignore 済み。

`devserver.py` は `Cache-Control: no-store` を返す。標準の `http.server` だと
ブラウザが古い `index.html` / `avatar.js` を掴んだままになり、
「直したのに変わらない」という誤解を生むため。

## `compare.html` — 二画面で見比べる（issue #7）

2つのサーブを左右に並べ、**骨格**で見比べる。アバターは切替で重ねられる。

```bash
python tools/make_compare.py \
    --mine output          --mine-fps 60 --mine-label "あなた" \
    --ref  output_zverev2  --ref-fps 240 --ref-label  "お手本" \
    --out  web/data
python web/devserver.py   # → http://127.0.0.1:8123/compare.html
```

- **既定は骨格**（復元された実寸そのまま）。`🧍 アバター` で同じ体に切替
  ＝体格差が消え、残る差が技術の差になる
- **視点の同期**: 片方を回すともう片方も同じ角度になる。個別操作にも切替可
- スライダーは**動作の局面**で揃う。沈み込みを 0、打点を 1 として各クリップの
  自分のフレームに写すので、**撮影fpsやスイングの速さが違っても同じ瞬間が並ぶ**
- オレンジ=沈み込み、赤=打点
- 復元が壊れている素材は警告を出す（接地していない・フェーズが検出できない）。
  素材の問題を解析の結果と取り違えないため

## `models.html` — 復元手法を見比べる（issue #9）

**同じ動画**を複数の手法（GVHMR / GEM-X / TRAM）で復元した結果を三画面で並べる。
`compare.html` と用途が違うので別ページにしてある:

|  | `compare.html` | `models.html` |
|---|---|---|
| 比べる対象 | 2本の動画 | 1本の動画 × 複数の手法 |
| 時間軸 | 別々（fpsも尺も違う） | 共通（同じ動画なので） |
| スライダー | 画面ごとに独立 | 1本で全部動く |
| 体格差 | あるのでアバターで消す | 無い（同一人物） |

指標表には**ラケットドロップ**を載せて強調している。手首→手の向きが鉛直から
倒れる角度で、3手法での順位が目視評価の順位と一致した唯一の指標だった
（GVHMR 116° > TRAM 72° ≒ GEM-X 72°）。腕の高さや肘角はむしろ逆の順位を示す。

```bash
python tools/make_models.py \
  --clip GVHMR=output --clip GEM-X=output_gemx_mine --clip TRAM=output_tram_mine \
  --fps 60 --out web/data/models.json
```

## `skeleton.js` — 3D骨格の表示（共有）

`compare.html` と `models.html` が共有する。復元結果を見られる形に直す部分だけを
持ち、ページ固有のUI（再生・同期・アバター・描き込み）は各ページに置いてある。

`toDisplay()` が揃える3つが要点:

1. up軸を +Y に
2. **体の向きを揃える** — 揃えないと2本並べたとき片方が背中向きになる（実測187°差）
3. **足元を床に。基準は最低値ではなく中央値** — 最低値だと一度の沈み込みが床になり、
   さらに世界座標の原点の置き方は手法ごとに違う（GVHMR は床を y≈0 に置くが TRAM は置かない）

`createPane()` が `renderer.setSize(w, h, false)` を使うのも重要で、`true` だと
three がキャンバスに px 幅を書き込み、それがグリッド列を押し広げ、次のフレームで
さらに広がる増幅ループになる。表示サイズはCSSに任せる。

## `session.html` — 1回の練習をまとめて見る（issue #10）

多数本のサーブを集計する。**1本では原理的に分からないこと**が見える:

1. **ばらつき** — 中央値と範囲。あなたの典型値と振れ幅
2. **測定の安定度** — CV（変動係数）で指標どうしを比べる。実測42本では
   打点の高さ 1%（9cmの幅）に対し、跳躍は 42%。
   **CVの大きい指標は1本から語ってはいけない**
3. **入った / それ以外** — 中央値の差と、Holm 補正後の p
4. **時系列** — 打った順の推移。疲労で変わるか

```bash
python tools/make_session.py \
  --recon ~/.../recon --labels ~/.../clips/labels.csv \
  --fps 60 --out web/data/session.json
```

骨格は1本ずつ `web/data/serves/serve_NN.json` に分けてある。42本ぶんを1つに
まとめると40MB超になり、集計を見るだけでも全部読むことになるため。

**差が出なかったときの文言**をページに埋め込んである。「差が無いことの証明では
なく、この本数では言えない」という意味だと明示しないと、過剰に読まれるため。
差が出たときも「探索的な結果であり、直せば入るようになるとは言えない」と出る。

## アバター（`avatar.js`）

SMPL の**回転**を VRM に移す。位置ではなく回転を使うのは、体格が違っても成立するため。

肝は**レストポーズ差の吸収**で、VRM と SMPL は基準姿勢が約180°ヨーで食い違う。
これをボーンごとの最短回転に任せると軸が不定になりねじれるので、
先に体全体の向き `W` を揃えてから各ボーンの残差を補正する（補正角: 平均11.8°）。

VRM は `web/avatar/avatar.vrm` に置く（gitignore 済み。ライセンスは各自のモデルに従う）。
ラケットは**追跡していない飾り**で、前腕の延長方向に固定しているだけ。

## 既知の限界・次（→ フェーズ B の Vercel 化 = ユーザーの言う "A"）

- **動画は base64 で投入**している。将来は multipart / 署名付きアップロードに
- Modal の URL を直書き。Vercel 化時は環境変数 or API プロキシに
- 3Dビューアは**骨格のみ**（フルメッシュは reconstruct が頂点を返す拡張が必要）
- 元動画への骨格オーバーレイ、履歴、認証はスコープ外（別issue）

## clip.html — 1本を見る（競技を問わない）

元動画と3D骨格を並べ、フレームを同期させる。局面のボタンで飛べる。
指標のラベルと向きの説明は各ドメインの `metric_labels` から来るので、
競技を足しても HTML は触らない。

```bash
python tools/make_clip.py --joints output_pitch_s3/s3_joints.npy --fps 24 \
    --domain baseball_pitch --video videos/baseball/x.mp4 --label "投球フォーム解析"
python web/devserver.py   # → http://127.0.0.1:8123/clip.html
```

測れなかった指標は **「測定できません」** と出る（NaN を 0 にしない）。

複数のデモを並存させるときは `--name`。`web/data/<name>/` に出て、
`clip.html?clip=<name>` で開く。

**スロー映像**は撮影fpsと再生fpsが違う。指標は撮影fps（`--fps`）で、動画の
同期は再生fps（`--playback-fps`）で扱う。切り出して復元した関節なら
`--start` で元動画の開始秒を渡す。単一画像モデルの出力をスロー映像で使うときは
`--smooth-to-fps 30` 前後で平滑化しないと、ジッタが速度を埋め尽くす
（[core/filter.py](../core/filter.py)）。

```bash
python tools/make_clip.py --name batting --domain baseball_swing \
    --joints output_bat_s3/s3_joints_3-9s.npy --fps 240 --playback-fps 24 --start 3.0 \
    --smooth-to-fps 30 --video videos/baseball/batting_form.mp4 --label "バッティングフォーム解析"
# → http://127.0.0.1:8123/clip.html?clip=batting

python tools/make_clip.py --name opera --domain opera_posture \
    --joints output_opera_s3/s3_joints.npy --fps 30 --smooth-to-fps 10 \
    --video videos/opera/opera_form.mp4 --label "発声姿勢の解析"
# → http://127.0.0.1:8123/clip.html?clip=opera

python tools/make_clip.py --name golf --domain golf_swing \
    --joints output_golf_s3/s3_joints.npy --fps 30 \
    --video videos/golf/golf_form1.mp4 --label "ゴルフスイング解析"
# → http://127.0.0.1:8123/clip.html?clip=golf
# スマホ撮影はカメラが 10〜15° 傾く。録画の最初に直立する1秒があれば
# --level-from 0:1 で水平を取れる（無いと前傾角がそのぶんずれる）
```

オペラは局面が無いので、局面ボタンは「開始 / 終了」だけになる。指標はすべて
区間の平均とばらつき。**ばらつき系（SD・揺れ）は単一画像モデルのジッタを含む**ので、
GVHMR より 1.5〜3倍大きく出る。人と比べるなら同じ手法どうしで。

**視点の既定は「横から撮った映像」を前提**にしている。ビューアは撮影カメラの
位置を知らないので、体の向き（腰の左→右）から横向きを作る。投手の背中側から
撮った映像では「映像と同じ視点」ボタンは一致しない。計測値はカメラ位置に依存しない。

見せる結果は **SAM 3D Body**（`s3_joints.npy`）で作ること。GVHMR は非商用なので
社内の検証に留める（[backend/README.md](../backend/README.md)）。
