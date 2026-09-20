# 🎾 Tennis Form Analyzer

動画から自分のテニスフォームを**3Dで正確に定量化**し、
プロと比較して**「どこをどう直すか」を学びとして返す**ことを目指すプロジェクト。

> 動画を撮って「なんか崩れてたね」で終わらせるのではなく、
> データに裏付けられた、行動に移せるフィードバックを返す。

**現在の状態: 2D実装を破棄し、3Dベースでゼロから再設計中。**

---

## なぜ作り直しているか

初代MVPは MediaPipe の **2D姿勢推定**で作られていたが、目標に対して原理的な限界があった。

| やりたいこと | 2Dでの問題 |
|---|---|
| 重心・体重移動 | 奥行きが無いため**測定不可能**。2Dピクセルの加重平均は単位も床基準も無い物理的に無意味な値 |
| 関節角度 | カメラアングルで見かけの角度が歪む |
| キネティックチェーン | 3次元の回転が取れず評価できない |
| 打点(ヒットポイント) | 旧実装のボール検出はモック（偽データ） |
| プロ比較 | 時間同期・体格正規化なしのフレーム単位差分は、同じ局面を比べていない |

→ **土台を world-grounded な3D人体復元に置き換える。**

旧2D実装は `legacy-2d` ブランチに保存してある（参照専用）。

---

## アプローチ

```
動画
 └─▶ ① 3D人体復元 (GVHMR / world-grounded SMPL)
     └─▶ ② バイオメカ算出 (3D重心・関節角・体重移動)
         └─▶ ③ 動作の意味づけ (フェーズ分割・キネティックチェーン・打点検出)
             └─▶ ④ プロとDTWで整列比較 → 行動可能なフィードバック
```

詳細な設計方針・技術選定・ロードマップは **[REDESIGN.md](REDESIGN.md)** を参照。

---

## 現在の進捗

### ✅ P0: 技術検証クリア（2026-07-19）

自分のサーブ動画で GVHMR を実行し、3D重心が妥当かを検証した。

| 指標 | 結果 |
|---|---|
| 重心の高さ範囲 | 0.890 – 1.120 m（身長の約55%＝解剖学的に妥当） |
| 沈み込み（トロフィーポーズ） | frame 87 / 0.953 m |
| **打点（動画上の実際の接球と一致）** | **frame 103 / 1.120 m** |
| 伸び上がり | +0.167 m / **0.28 秒** |

**重心の最高点が実際の打点と一致した。**
動画のピクセルを見ずに、3D重心の波形だけからサーブの打点を特定できたことになり、
重心が物理的に正しく計算できている証拠となった。

→ 当時の Colab 手順は Modal へ移行後に削除（[issue 002](docs/issues/002-modal-gvhmr-backend.md)）。レシピは `backend/reconstruct.py` の image 定義に焼いてある。

### ✅ フィードバック生成

計測値から改善点を言葉で返す層を実装した（[analysis/](analysis/)）。

**プロとの比較はしない。** プロ同士でもフォームは大きく異なり、差は必ずしも
欠点ではないし、体格が違えば真似できない。代わりに誰にでも当てはまる
力学的原理（キネティックチェーンの順序、打点タイミング）で判定する。
参照動画も権利処理も不要で、かつ最も価値の高い指摘が得られる。

提示は**一度に1〜2件まで**。人は複数の修正キューを同時に処理できず、
15件の指摘は0件と同じになる。

### ⚠️ 撮影要件: キネティックチェーンには 120fps 以上

実データで、システムが**1フレーム差を根拠に力の伝達順序の逆転を指摘した**。
30fps では1フレーム=33ms だが、連鎖の隣接体節の時間差は 20〜40ms しかなく、
**そもそも分解できない量について主張していた**。

現在は 60fps 未満では順序を判定せず、レポートにその旨を明示する。
評価したい場合は iPhone のスローモード等で 120/240fps 撮影のこと。

### ⚠️ 撮影要件: カメラは固定（三脚）

3D復元は**静止カメラ前提**で呼んでいる（GVHMR の `-s`）。カメラが動くと、その
動きが人物の動きとして世界座標に足し込まれる。実測でお手本用の放送クリップは
背景が毎秒6.3%流れており、その結果:

| | 影響 |
|---|---|
| 重心の伸び上がり・跳躍・水平移動 | **使えない**（世界座標の並進そのもの） |
| 打点タイミング | **使えない**（重心ピークとの差なので） |
| 膝・肘・体幹の角度 | 使える（`body_pose` 由来で影響を受けにくい） |
| 沈み込み・打点のフレーム | 使える（座標に依存しない） |

数値は計算できてしまうので、**もっともらしく間違った値**が出る。比較ビューアは
該当する指標を「— 使えません」と表示し、値を出さない。

投げる前に確認する:

```bash
python tools/camera_motion.py X.mp4        # 静止なら 0.0%/秒 前後
```

**お手本の素材も三脚固定で撮ったものに統一する方針**（2026-08-09 決定）。
カメラ運動に対応する道（SLAM）は [issue 008](docs/issues/008-moving-camera-slam.md)
に残してあるが、素材側を揃えれば不要になる。

### 次の課題

1. **120fpsで再撮影** — キネティックチェーンを評価可能にする
2. **複数サーブでの再現性確認** — 1本では偶然の可能性が残る
3. **サーブ区間の自動切り出し** — 1クリップに他の動作が混在する
4. **人物追跡の頑健化** — 背景に別プレイヤーがいると追跡が移る可能性
5. **コート座標の定義** — 水平軸がコートのどちら向きか未定義で踏み込みを評価できない
6. **P1: パイプライン化** — 現在は手作業。サーバーレスGPUバックエンドの土台

---

## リポジトリ構成

```
REDESIGN.md    設計方針・技術選定・ロードマップ
backend/       3D復元を Modal のサーバーレスGPUで実行。4系統が並行して動く
  reconstruct_sam3d.py SAM 3D Body + MHR — 出荷用。商用可。人に見せる結果はこれ
  reconstruct.py       GVHMR  — 基準。非商用。検証・突き合わせにだけ使う
  reconstruct_gemx.py  GEM-X  — 評価記録。SAM 3D Body のイメージ・重みの供給元
  reconstruct_tram.py  TRAM   — 評価記録
videos/        元動画の置き場（中身は git に入らない）。解析前の確認手順はここの README
core/          競技に依存しない計測（純numpy / GPU不要）
  skeleton.py    SMPL 24関節の定義・体節質量比
  geometry.py    幾何ユーティリティ
  kinematics.py  重心・床・関節角・捻転・手の向き・連鎖
  convert.py     他の骨格 → SMPL24（GEM-X の SOMA 77/78。次は MHR 127）
domains/       競技ごとの局面・指標・判定
  base.py           ドメインの型。Tier A/B/C と連鎖の判定（全競技共通）
  tennis_serve.py   テニス サーブ — 実装済み
  golf_swing.py     ゴルフ       — 指標のみ（判定は出典待ち）
  baseball_pitch.py 野球 投球    — 指標のみ。実映像で局面検出を確認済み（連鎖判定は240fps以上）
  baseball_swing.py 野球 打撃    — 指標のみ。接地・インパクト（手の最速で代用）
  opera_posture.py  オペラ 姿勢  — 指標のみ（音声側が未実装）
analysis/      アプリ層。core と domains をつなぐ薄い層 + CLI
web/           ブラウザで見る（配信は web/devserver.py）
  clip.html      1本を見る — 元動画と3D骨格を同期表示。競技を問わない。人に見せる画面
  index.html     単体解析 — 動画を投げて結果を見る（GVHMR の Web API）
  compare.html   二画面   — 2本の動画を見比べる（自分 vs お手本）
  models.html    三画面   — 1本を複数の復元手法で見比べる（報告用）
  session.html   多数本   — 1回の練習をまとめて見る（ばらつき・時系列）
  skeleton.js    3D骨格の表示。compare/models が共有
  avatar.js      VRMへのリターゲット
tests/         合成サーブデータによる検証
tools/         補助スクリプト
  videoinfo.py     動画の実fps（コンテナ上の再生レート）を読む
  estimate_fps.py  空中の重心の落ち方から実fpsを推定（スロー動画用）
  camera_motion.py カメラが動いていないか（動くと世界座標が壊れる）
  find_serves.py   長い動画からサーブ区間を見つけ、1本ずつ切り出す
  make_clip.py     clip.html 用のデータ生成（元動画も縮小して同梱）
  make_compare.py  compare.html 用のデータ生成
  make_models.py   models.html 用のデータ生成
  make_session.py  多数本をまとめて集計（ばらつき・結果との関係）
  compare_backends.py 復元手法を同じ物差しで比べる
```

復元手法が4系統あるのは、**GVHMR が非商用ライセンス**で製品化できないため。
商用可能な代替を3つ評価し、**SAM 3D Body + MHR が GVHMR とほぼ一致した**
（[issue 009](docs/issues/009-licensing-for-productization.md)）。

```
出荷   SAM 3D Body + MHR   商用可。デモ・納品・比較画面はすべてこれ
基準   GVHMR               非商用。検証にだけ使い、成果物に混ぜない
```

3D復元は GPU が要るため Modal 上で実行して**24関節**を返し、その関節から
`core/` が重心・角度を、`domains/` が局面と判定を導出する、という分業。
重心も上軸も関節の純関数なので、GPU側は関節までを担い、以降はローカルで動く。

**1リポジトリで複数の競技**を扱う。共有するのは `core/`（計測）と
保存・比較・表示で、競技ごとに違うのは局面・指標・判定だけ
（[issue 011](docs/issues/011-commercial-architecture.md)）。

```bash
uv venv && uv pip install -e ".[dev]"
pytest

# 動画の実fpsを確認（時間の指標はすべてこの値に依存する）
python tools/videoinfo.py temp_my_serve.mp4

# 3D復元（Modal GPU）→ 解析。詳細は backend/README.md
modal run backend/reconstruct_sam3d.py --video videos/baseball/x.mp4 --out output_x
python -m analysis --joints output_x/s3_joints.npy --fps 24 --domain baseball_pitch

# 人に見せる（元動画＋3D骨格＋計測値）
python tools/make_clip.py --joints output_x/s3_joints.npy --fps 24 \
    --domain baseball_pitch --video videos/baseball/x.mp4 --label "投球フォーム解析"
python web/devserver.py   # → http://127.0.0.1:8123/clip.html

# 競技を指定する（既定はテニスのサーブ）
python -m analysis --list
python -m analysis --joints out/joints.npy --fps 60 --domain golf_swing
```

## 実行環境について

3Dモデルの推論には **NVIDIA GPU が必要**（Apple Silicon では動かない）。
**Modal のサーバーレスGPU（T4）**で実行する。`modal run` の1コマンドで、
環境の再構築なしにクラウドGPUで復元し、結果をローカルへ返す。

- 計算コスト: 約 $0.03〜0.04 / 本（約5円）
- 待ち時間: 約10分（大half はコールドスタート。非同期利用のため許容）

セットアップ手順は [backend/README.md](backend/README.md) を参照。

---

## 参考（2024–2026）

- **GVHMR**: World-Grounded Human Motion Recovery via Gravity-View Coordinates (SIGGRAPH Asia 2024 / TPAMI 2026)
- **WHAM**: Reconstructing World-Grounded Humans with Accurate 3D Motion (CVPR 2024)
- **TRAM**: Global Trajectory and Motion of 3D Humans from In-the-Wild Videos (ECCV 2024)
- **OpenCap**: 低コスト・マーカーレス動作解析（妥当性検証論文多数）
