# backend — GVHMR を Modal のサーバーレスGPUで動かす

3D復元 (GVHMR) を Modal 上で実行し、**24関節** (`gv_joints.npy`) と
レンダリング動画を返す。GPUを使うのはこの復元段だけで、重心・角度・フィードバックの
導出は後段の [analysis/](../analysis/)（純numpy、CPU/ローカル）の責務。
重心も上軸も関節の純関数なので、GPU側は関節までを担う。

計画と背景は [docs/issues/002](../docs/issues/002-modal-gvhmr-backend.md)。

## なぜ Modal か（Colab との違い）

Colab は毎セッション環境を作り直していた（Python 3.10 venv、chumpy の
`--no-build-isolation`、`from turtle import` バグ修正、5GBチェックポイントの再取得）。
Modal は**この環境をイメージに一度だけ焼き、二度と再構築しない**。
`reconstruct.py` の `image` 定義がその確定レシピそのもの。

## セットアップ（初回のみ）

```bash
# 依存
uv pip install --only-binary :all: modal   # cbor2 が Rust を要求するため wheel 強制
modal token new                             # GitHub連携アカウントで認証

# チェックポイントを Volume に取得（Modal上で実行、HuggingFace から）
modal run backend/reconstruct.py::fetch_checkpoints

# body models を Volume にアップロード（ライセンス制のためローカルから）
modal volume put gvhmr-assets \
    ~/Desktop/gvhmr_body_models/SMPL_NEUTRAL.pkl  /checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
modal volume put gvhmr-assets \
    ~/Desktop/gvhmr_body_models/SMPLX_NEUTRAL.npz /checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
```

## 実行

```bash
modal run backend/reconstruct.py --video temp_my_serve.mp4
# → gv_joints.npy とレンダ動画 render_*.mp4 がカレントに返る

python -m analysis --joints gv_joints.npy --fps 30 --save output
```

ブラウザもセッション管理も不要。将来この関数に HTTP エンドポイントを足せば
そのまま Web バックエンドになる。

## Web エンドポイント（issue 003 フェーズA）

ブラウザから使うための非同期HTTP。復元は約10分かかるため投入と取得を分ける。

```bash
modal deploy backend/reconstruct.py
# POST https://<workspace>--serve-submit.modal.run   {video_b64, name?, fps?} → {job_id}
# GET  https://<workspace>--serve-result.modal.run?job_id=...  → {status, result?}
#   status: pending / done(+result) / error
```

`result` の JSON は `analysis.analyze_json` の出力（metrics・feedback・
3Dビューア用の joints (F,24,3)・up_axis）。GPUを使うのは `run_job` の復元段だけで、
指標導出は同コンテナ内の analysis（純numpy）が行う。

## 構成

| 要素 | 役割 |
|---|---|
| `image` | GVHMR環境（Python 3.10 / torch 2.3+cu121 / chumpy / turtleバグ除去）＋ analysis |
| Volume `gvhmr-assets` | チェックポイント + body models。一度置けば毎回マウント |
| `fetch_checkpoints()` | HuggingFace `ryanrudes/gvhmr` から4つのckptを取得（初回のみ） |
| `_gvhmr_joints()` | 共通ヘルパー。`demo.py -s` で復元し 24関節とレンダ動画を返す |
| `reconstruct()` | CLI用。関節を .npy にし動画と共に返す（`modal run`） |
| `run_job()` | Web用。復元 → `analysis.analyze_json` で指標JSONを返す |
| `submit` / `result` | 非同期Webエンドポイント（投入 / 状態確認） |
| `main()` | ローカルの動画を送り結果を引き戻す `modal run` の入口 |

## 注意

- GPUは `T4`（P0で実績。14GB で足りた）。速くしたければ `gpu="L4"` に変更可
- 静止カメラ前提で `-s`（SLAM回避）。三脚固定の撮影が前提
- body models が Volume に無いと `reconstruct` は明確に失敗する（準備手順を実行）
- GVHMR は `git clone` で最新を取得している。再現性を厳密にするなら将来コミットを固定する
