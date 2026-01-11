# Python Graphs

## 概要
- Python と Matplotlib を利用した代表的なグラフのサンプル集とドキュメントです。
- `graph.md` を中心に各種プロットの書き方と出力例を整理し、HTML 版 (`graph.html`) やランディングページ (`index.html`) から閲覧できます。
- 同梱のスクリプトでサンプル画像を再生成できるため、学習や資料作成のテンプレートとして活用できます。

## 主な構成
- `generate_graphs.py` — Matplotlib を使って PNG 形式のサンプル画像を `images/` 以下に再生成するスクリプト。
- `images/` — スクリプトで生成される各種グラフのサンプル画像。
- `graph.md` — Matplotlib で描ける代表的なプロットの一覧とサンプルコードをまとめた Markdown ドキュメント。
- `graph.html` — `graph.md` を Pandoc などで HTML 化した版。`style.css` を読み込み済み。
- `learning_guide.pdf` — 学習ガイド（LuaLaTeX で生成した PDF）。
- `learning_guide.tex` — 学習ガイドのソース。
- `index.html` — サイトのトップページ。ドキュメントやサンプルへの導線を提供。
- `style.css` — HTML ドキュメントのスタイル定義。
- `package-lock.json` — Web アセットを管理するための npm 依存関係ロックファイル (ビルド環境で利用)。

## 使い方
### グラフ画像の再生成
1. Python 3.9 以上を用意し、必要なら仮想環境を作成します。
2. 依存ライブラリをインストールします。
   ```bash
   pip install matplotlib numpy
   ```
3. スクリプトを実行して `images/` ディレクトリ内の PNG を再生成します。
   ```bash
   python generate_graphs.py
   ```

### ドキュメントの更新と公開
1. `graph.md` を編集して内容を更新します。
2. 必要に応じて以下のコマンドなどで HTML を再生成します (Pandoc を例示)。
   ```bash
   pandoc graph.md -s -c style.css -o graph.html
   ```
3. `index.html` やスタイルシートに変更がある場合は、ブラウザで `index.html` を開き表示を確認します。

### 学習ガイドの閲覧
- `learning_guide.pdf` を PDF ビューアで開いて確認します。

## 更新履歴
- 2025-09-27: `learning_guide.tex` と `learning_guide.pdf` を追加し、README に学習ガイドの案内を追記。
- 2025-09-27: `index.html` をヒーローレイアウトとカード型セクションに刷新し、`style.css` に新コンポーネント用のスタイルを追加、注釈付きサンプルの導線を整備。
- 2025-09-27: `AGENTS.md` を追加し、ドキュメント向けに `style.css` をモダン化。
- 2025-09-27: `README.md` を追加し、リポジトリ概要と更新履歴を整備。
- 2025-05-24: サンプル画像生成スクリプトと `images/` を追加。`graph.md` から `graph.html` を生成し、`style.css` と `index.html` を整備。
