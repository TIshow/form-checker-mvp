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
