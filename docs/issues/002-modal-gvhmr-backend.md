# Issue 002: GVHMR を Modal のサーバーレスGPUで動かす

Status: Open
Created: 2026-07-22
GitHub: [#2](https://github.com/TIshow/form-checker-mvp/issues/2)
関連: [notebooks/](../../notebooks/) の Colab 手順を置き換える / [001](001-contact-and-phase-detection.md) の前提基盤

---

## 背景 / 問題

3D復元 (GVHMR) は現在 Colab で動かしているが、実用的でない。

- **毎セッション環境を作り直す。** Python 3.10 の venv 構築、chumpy の
  `--no-build-isolation`、`from turtle import` バグ修正、5GBチェックポイントの
  再取得を毎回やり直している
- 対話的でスクリプト化しにくく、ブラウザとセッション管理に縛られる
- Web アプリのバックエンドに発展させられない

**サーバー化の本質的な利点はコストではなく「環境を一度だけイメージに焼き、
二度と再構築しない」こと。**

## なぜ Modal か

候補比較（[REDESIGN.md](../../REDESIGN.md) の議論より）:

| 候補 | 判断 |
|---|---|
| **Modal** | イメージを完全定義でき、5GBの重みはVolumeに一度置けば毎回マウント。ゼロスケール。Python native。**採用** |
| GCP Cloud Run + GPU | 同等で移植性も高いが、今は開発速度を優先 |
| AWS (SageMaker/EC2) | できるが設定が重い |
| Vercel | GPU不可。将来フロント＋APIで使う |

GVHMR は pip 一発では入らない壊れやすい環境のため、**イメージを自分で定義できる**
ことが決め手。量が増えてGPU単価が効いてきたら Cloud Run 等へ移せる
（`analysis/` は純numpyで移植自由、変わるのはGVHMRを包む起動部分だけ）。

## 設計

### 責務の分離（既存のコード分割を踏襲）

```
動画
 └─▶ [Modal GPU関数] GVHMR で 3D復元        ← GPUが要るのはここだけ
        出力: gv_joints.npy / gv_com.npy / gv_upaxis.npy
     └─▶ [CPU / ローカル] analysis/ で指標とフィードバック  ← GPU不要
```

GPUを使うのは復元の一段だけ。解析層は CPU 関数かローカルで動かし、コストを抑える。

### イメージ定義（P0で確定したレシピを焼く）

`memory` に記録済みの手順をそのまま Modal Image に落とす:

- ベース: CUDA 12.1 + Python 3.10
- torch 2.3.0+cu121, numpy 1.23.5, pytorch3d(cp310), chumpy(`--no-build-isolation`)
- GVHMR を clone し `from turtle import` 行を除去
- ライセンス制の body models と 5GB チェックポイントは**イメージに焼かず Volume** へ
  （公開イメージに入れられないため）

### チェックポイント / body models の配置

- Modal Volume を1つ用意し、初回に一度だけアップロード
  - チェックポイント: HuggingFace `ryanrudes/gvhmr` から取得（Colabで実績あり）
  - body models: 手元の登録済みファイル（SMPL_NEUTRAL.pkl / SMPLX_NEUTRAL.npz）
- 以降は毎回マウントするだけ。再DL不要

### 呼び出し方（開発ループ）

```bash
modal run reconstruct.py --video temp_my_serve.mp4
# → クラウドの温めた環境で GVHMR 実行
# → gv_*.npy をローカルに引き戻す
# → python -m analysis で解析
```

ブラウザもセッション管理も不要。将来この関数に HTTP エンドポイントを足せば
そのまま Web バックエンドになる。

## 作業計画

### フェーズ A: 動く最小構成（新機能なし、Colabの再現）
- [ ] Modal アカウント設定 / `modal` CLI 認証
- [ ] GVHMR イメージ定義（P0レシピを Modal Image に移植）
- [ ] Volume 作成、チェックポイントと body models を一度アップロード
- [ ] GPU 関数で `demo.py --video X -s` を実行し `.pt` を生成
- [ ] 付属 tennis.mp4 で完走を確認（Colabのスモークテスト相当）

### フェーズ B: 出力を解析層につなぐ
- [ ] `.pt` から `gv_joints.npy / gv_com.npy / gv_upaxis.npy` を生成（P0のCOM算出を移植）
- [ ] ローカルへ結果を返す `modal run` ラッパー
- [ ] `analysis/` に流して自分のサーブでレポートが出るまでを1コマンド化

### フェーズ C: 仕上げ
- [ ] コールドスタート対策の検討（min-instances / 重みのVolumeマウント確認）
- [ ] 1本あたりの実コスト・所要時間を計測して記録
- [ ] `notebooks/` を「参考（P0の記録）」に格下げし、正規の手順を Modal に切替

## 受け入れ条件

- `modal run reconstruct.py --video <clip>` の1コマンドで `gv_*.npy` が返る
- 2回目以降、環境の再構築も5GBの再DXも発生しない
- 付属 tennis.mp4 と自分のサーブの両方で完走する
- 1本あたりの所要時間とコストが記録されている
- GPUを使うのは復元段のみ（解析はCPU/ローカル）

## リスク / 未解決

- **イメージ構築は初期の一手間**（Dockerfile相当のImage定義）。ただし一度作れば
  バージョン管理され、Colabのように毎回作り直さない
- **コールドスタート 30〜60秒**（ゼロスケール復帰時）。2〜3分のバッチ処理なので許容
- **body models のライセンス**。公開イメージに含めず非公開Volumeに置く
- Modal の Python API / Image 定義が P0 当時の版と差異がある可能性 → フェーズAで確認

## スコープ外

- Web フロント / API（Vercel 側。別issue）
- 打点検出・フェーズ検出のロジック改善（[001](001-contact-and-phase-detection.md)）
- 他プラットフォーム（Cloud Run 等）への移植（必要になれば別issue）
