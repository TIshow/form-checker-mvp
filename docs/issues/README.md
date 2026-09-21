# Issues

作業単位の計画書。設計・理由の**正本はここ（リポジトリ）**で管理し、
GitHub issue は追跡用にミラーする。

## 運用ルール

1. 方針・複数セッションにまたがる作業は `docs/issues/NNN-*.md` に書く（正本）
2. 追跡したくなったら `gh issue create` で GitHub にミラーし、相互にリンクする
   （issue本文は正本へのリンク＋要約に留め、二重管理を避ける）
3. コミットメッセージで issue 番号を参照する（`Closes #N` で自動クローズ）

### 完了時（削除しない）

- コミットで `Closes #N` → GitHub issue が自動クローズ
- 正本の `Status:` を `Done` にし、完了日を記入（**ファイルは残す**）
- 残すべき結論を「生きたdoc」（[REDESIGN.md](../../REDESIGN.md) や各 README）に移植
- 下の表の状態を更新

docは消さない。完了後の価値は手順ではなく「なぜそうしたか」にあり、
再検討を防ぐ。ただし未来形の記述が現状と食い違うと誤誘導するため、
**結論は生きたdocへ移植**し、issue doc は記録として残す。

## 一覧

| # | タイトル | GitHub | Status |
|---|---|---|---|
| [001](001-contact-and-phase-detection.md) | フェーズ検出を「自動検出→人が微調整」にし、打点を打球音で精密化 | [#1](https://github.com/TIshow/form-checker-mvp/issues/1) | Open |
| [002](002-modal-gvhmr-backend.md) | GVHMR を Modal のサーバーレスGPUで動かす | [#2](https://github.com/TIshow/form-checker-mvp/issues/2) | ✅ Done |
| [003](003-web-app.md) | Web アプリ化（アップロード → フィードバック + 3Dビューア） | [#3](https://github.com/TIshow/form-checker-mvp/issues/3) | Open |
| [004](004-3d-annotations.md) | 3D空間に貼り付く注釈（回しても追従） | [#4](https://github.com/TIshow/form-checker-mvp/issues/4) | Open |
| [005](005-avatar-retargeting.md) | 自分のサーブをアバター（VRM）で再生する | [#5](https://github.com/TIshow/form-checker-mvp/issues/5) | ✅ Done |
| [006](006-racket-tracking.md) | ラケットを追跡する | [#6](https://github.com/TIshow/form-checker-mvp/issues/6) | Open |
| [007](007-motion-transfer-comparison.md) | 同じ体に揃えて比較する（お手本の動きを自分の体で） | [#7](https://github.com/TIshow/form-checker-mvp/issues/7) | Open |
| [008](008-moving-camera-slam.md) | カメラが動く素材で世界座標が壊れる（跳躍が再現されない） | — | Open（優先度低） |
| [009](009-licensing-for-productization.md) | 製品化のライセンス制約（GVHMR と SMPL が非商用） | — | Open |
| [010](010-outcome-measurement.md) | 結果を測る（ボール速度・コース・ポイント） | — | Open |
| [011](011-commercial-architecture.md) | 商用化に向けた構成の転換（三脚前提・蒸留・結果計測） | — | Open |
| [012](012-commercial-multidomain-direction.md) | 商用化と複数分野への展開方針・現行コードからの移行 | — | Proposal |
| [013](013-posture-and-voice.md) | 姿勢 × 響き（オペラの結果を音声で測る） | — | Open |
| [014](014-golf-sam3d-tilt-diagnosis.md) | ゴルフのSAM前傾差：座標・推定・後処理の切り分けと修正順 | — | Diagnosis / Proposal |
| [015](015-golf-gemx-pose-refinement.md) | ゴルフのGEM-X精度：同一フレーム診断と改善実験 | — | Closed（不採用・018 でリセット） |
| [016](016-golf-refinement-rejection.md) | 補正不採用：2D色順・奥行き・床表示の追加診断と修正順 | — | Closed（記録。018 でリセット） |
| [017](017-gemx-temporal-accuracy-plan.md) | GEM-Xの時系列精度：DDIM経路の検証と比較計画 | — | Closed（記録。018 でリセット） |
| [018](018-golf-gemx-from-scratch.md) | ゴルフ × GEM-X を 0 から積み上げる（上流に手を入れない） | — | Step 1 |
