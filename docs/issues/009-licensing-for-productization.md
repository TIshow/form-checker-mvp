# Issue 009: 製品化のライセンス制約（GVHMR と SMPL が非商用）

Status: Open
作成: 2026-08-08

## なぜ今これを書くか

「日本の部活動で使えるように製品化したい」という目標が出た。現在のパイプラインは
**2箇所が非商用ライセンス**で、そのままでは製品化できない。技術的にどれだけ
良くなっても、ここが解けないと出口がない。設計判断（どの手法に乗るか）にも
効き続けるので、先に整理しておく。

## 依存部品のライセンス（2026-08-08 実地確認）

| 部品 | 役割 | ライセンス | 商用 |
|---|---|---|---|
| **GVHMR** | 世界座標の3D復元 | 独自（非商用） | ✗ 個別許諾 |
| **SMPL / SMPL-X** | 人体モデル | MPI 学術ライセンス | ✗ 別途商用契約 |
| **YOLOv8** (Ultralytics) | 人物検出 | AGPL-3.0 | △ 公開 or 有償 |
| HMR2 (4D-Humans) | 体の回帰 | MIT | ○ |
| ViTPose | 2D関節 | Apache-2.0 | ○ |
| DPVO | SLAM | MIT | ○ |

GVHMR の LICENSE 原文（該当部）:

> Permission to use, copy, modify and distribute this software and its
> documentation for educational, research and non-profit purposes only.
> Any modification based on this work must be open-source and prohibited
> for commercial use.
> For commercial uses of this software, please send email to xwzhou@zju.edu.cn

SMPL は商用製品・サービスへの組み込みを明示的に禁止し、加えて**再配布・
サブライセンスも禁止**している。商用窓口は Meshcapade / smpl@max-planck-innovation.de。

## 無償で配る場合はどうか

どちらも**非商用の教育目的は明示的に許可**している。

- SMPL: "non-commercial scientific research, **non-commercial education**, or
  non-commercial artistic projects"
- GVHMR: "**educational**, research and non-profit purposes only"

完全無償・非収益なら該当する可能性がある。ただし:

- **SMPL のモデルファイルは配布できない。** 学校側にインストールさせる形は取れず、
  サーバ側で完結させる必要がある（それでも「サービスへの組み込み」に触れうる）
- GVHMR は「改変物はオープンソースでなければならない」— 自前の改造を
  クローズドにできない

## ★ 有力な回避策: NVIDIA GEM-X + SOMA（2026-08-08 判明）

[NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) は、**GVHMR・SMPL・YOLOv8 の3つを
まとめて置き換えられる**可能性がある。全部が商用可能なライセンスで揃っている。

| | 現行 | GEM-X |
|---|---|---|
| 3D復元 | GVHMR（非商用） | GEM-X コード **Apache-2.0** |
| 重み | — | **NVIDIA Open Model License**（"Models are commercially usable"） |
| 人体モデル | SMPL（非商用） | **SOMA**（NVIDIA独自・77関節） |
| 人物検出 | YOLOv8（AGPL） | **YOLOX**（Apache-2.0）+ ByteTrack（MIT） |
| 2D関節 | ViTPose | **同梱**（SOMA 77点用。自己完結） |
| 学習データ | — | **"trained on NVIDIA-owned data only"** |

依存の第三者ライセンスを全部見たが、**非商用・GPL・Max Planck 由来のものは無い**:
guided-diffusion (MIT) / PyTorch3D・ACTOR (BSD-3・MIT) / YOLOX (Apache-2.0) /
ByteTrack (MIT) / SAM 3D Body (Meta SAM License — royalty-free で商用可、
ただし再配布時の条項と輸出規制の遵守義務あり)。

SOMA 本体も Apache-2.0。README は SMPL モデルファイルについて
「別ライセンスであり同梱できない」と明記しているが、これは **SMPL 相互変換を
使う場合だけ**の話。SOMA 単体で使うなら SMPL は要らない。

### さらに: issue 008 にも効くかもしれない

GEM-X は "handles **dynamic cameras** and recovers **global motion trajectories**"
と明記している。[008](008-moving-camera-slam.md) のカメラ運動問題が、
手法の乗り換えで一緒に解ける可能性がある。

### 精度はどうなのか（GENMO 論文の実測値）

GEM-X は GVHMR との比較を公開していない（評価は自社の MetroSim 検証分割）。
研究版 [GENMO (ICCV 2025)](https://arxiv.org/abs/2505.01425) の Table 1 が
一番近い材料になる。EMDB (24) / RICH (24) の世界座標評価:

| 指標 | GVHMR | GENMO | |
|---|---|---|---|
| WA-MPJPE₁₀₀ (EMDB) | 109.1 | **69.5** | GENMO 36%改善 |
| W-MPJPE₁₀₀ (EMDB) | 274.9 | **185.9** | GENMO 32%改善 |
| RTE % (EMDB) | 1.9 | **0.9** | GENMO |
| Jitter (EMDB) | **16.5** | 17.7 | GVHMR |
| **Foot-Sliding (EMDB)** | **3.5** | 8.8 | **GVHMR 2.5倍良い** |

RICH も同傾向（W-MPJPE 126.3 → 118.6、Foot-Sliding 3.0 vs 6.7）。
**カメラ空間はほぼ互角**（EMDB PA-MPJPE 42.7 vs 42.5, Table 2）。
差がつくのは世界座標化の段だけで、[008](008-moving-camera-slam.md) で
問題を切り分けた場所と一致する。

**Foot-Sliding の差はトレードオフの表と裏。** GVHMR が足の滑りで勝つのは
接地の事前分布が強いからで、それが跳躍を潰している当の仕組み。

- GVHMR: 足を地面に留める → 滑らない → **跳ばない**
- GENMO: 拘束が緩い → 軌跡が正確 → **足が滑る**

跳ぶサーブには GENMO 側が向く可能性があるが、**体重移動の解析には
足の滑りが効く**。指標だけでは決まらない。

**この数字は GEM-X のものではない。** GENMO は SMPL・公開データ込みで
学習しており、GEM-X は SOMA・NVIDIA所有データのみで再学習している。
ライセンスが綺麗なのはその代償を払った結果で、同じ精度が出る保証はない。
論文の限界にも「off-the-shelf の SLAM に依存している」とある。

### 確かめていないこと（ここが本番）

ライセンスは確認済みだが、**使えるかどうかは別**。評価が要る。

1. **跳躍を再現するか。** 008 の核心。GVHMR は潰していた。同じ2本
   （自分のサーブ / Zverev）で比べれば一発で分かる
2. **足が滑らないか。** 上の表のとおり GENMO はここが弱い。滑ると
   接地と体重移動の判定が濁る。`tools/make_compare.py` の
   `quality()` が既に立位の足の高さを見ているので、指標は流用できる
3. **精度が足りるか。** 膝・肘・体幹の角度が GVHMR と同等以上か
4. **手が使えるか。** 77関節に手が入っている。テニスならグリップや
   手首の使い方に直結する（[006](006-racket-tracking.md) のラケット追跡にも効く）
5. **移行コスト。** `analysis/serve.py` も `web/avatar.js` も **SMPL の24関節前提**。
   SOMA 77関節への対応表を作る作業が要る
6. **計算資源。** Modal のイメージを作り直すことになる

### 評価の進め方

Modal に GEM-X のイメージを立て、既存の2クリップを流す。判断材料は
「跳躍が出るか」と「関節角が妥当か」の2つで足りる。ここが通れば、
下の交渉（GVHMR / SMPL）はどちらも不要になる。

## 進め方（順番が大事）

**GEM-X の評価を最優先にする。** 通れば以下の交渉は全部不要になり、
通らなければ以下に戻る。

### 1. GVHMR の商用可否を先に当たる（最優先・費用ゼロ）
`xwzhou@zju.edu.cn` に照会する。**ここが一番読めない**（大学研究室の裁量で、
可否も条件も事前に分からない）ため、他に投資する前に確かめる。
NG なら設計の前提が変わる。

### 2. YOLOv8 を差し替える（費用ゼロ・交渉不要）
人物検出なので代替が効く。**YOLOX / RT-DETR / RTMDet はいずれも Apache-2.0**。
差し替えれば AGPL の問題は消える。これは可否を待たずに着手してよい。

### 3. SMPL の商用ライセンスを取る（費用の問題）
Meshcapade が扱っており、経路は確立している。**可否ではなく費用の交渉**。
WHAM も TRAM も SMPL ベースなので、手法を変えても回避できない。事実上の前提条件。

## GVHMR が NG だった場合

残りは全部許諾型なので、**世界座標化の段だけが問題**になる。

- ViTPose + HMR2 でカメラ空間の姿勢は取れる（MIT / Apache-2.0）
- 足りないのは「重力に対する向き」と「世界座標での軌跡」
- ここを自前で作るのは研究プロジェクトになる。安易に見積もらないこと

なお [008](008-moving-camera-slam.md) で分かったとおり、**跳躍が失われているのは
まさにこの世界座標化の段**。自前で作るなら、その弱点を直せる可能性はある。

## 注意

ここに書いたのはライセンス文書を読んだ結果であって、法的助言ではない。
実際に進めるときは権利者からの書面確認と弁護士のレビューを取ること。
ただし**確認すべき相手と論点は上の3つに絞れている**。

## 関連

- [008](008-moving-camera-slam.md) — 世界座標化の段の技術的な弱点
- [002](002-modal-gvhmr-backend.md) — 差し替え対象のイメージ定義
