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
