# AI Music Coach - YouTube Stem Extractor

YouTubeの動画URLから音声をダウンロードし、高品質なAIモデルで楽器パートを分離するCLIツールです。

## 機能

- YouTubeから音声をダウンロード（WAV形式）
- 6種類の楽器パートを分離
  - **vocals** - ボーカル（BS-RoFormer, SDR 12.9）
  - **drums** - ドラム
  - **bass** - ベース
  - **guitar** - ギター
  - **piano** - ピアノ
  - **other** - その他の楽器

## 必要な環境

- Python 3.10
- Poetry
- ffmpeg（音声変換に必要）
- GPU推奨（CUDA対応、なくても動作可）

### ffmpegのインストール

**Windows:**
```bash
winget install ffmpeg
```

**Ubuntu/WSL:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/YOUR_USERNAME/ai-music-coach.git
cd ai-music-coach

# 依存関係をインストール
poetry install

# （初回のみ）モデルは自動でダウンロードされます
```

## 使い方

### 単一のパートを抽出

```bash
# ドラムを抽出
poetry run python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem drums

# ベースを抽出
poetry run python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem bass

# ボーカルを抽出（BS-RoFormerで最高品質）
poetry run python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem vocals

# ギターを抽出
poetry run python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem guitar

# ピアノを抽出
poetry run python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem piano
```

### 全6パートを一括抽出

```bash
poetry run python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --all
```

## 出力先

- ダウンロードした音声: `data/downloads/`
- 分離されたパート: `separated/`

## 使用モデル

| パート | モデル | 品質 |
|--------|--------|------|
| vocals | BS-RoFormer | SDR 12.9（最高品質） |
| drums | htdemucs_6s | 6stem同時分離 |
| bass | htdemucs_6s | 6stem同時分離 |
| guitar | htdemucs_6s | 6stem同時分離 |
| piano | htdemucs_6s | 6stem同時分離 |
| other | htdemucs_6s | 6stem同時分離 |

## その他のツール

### ドラム採譜（WAV → MIDI/タブ譜）

```bash
poetry run python drum_transcribe.py "separated/song_drums.wav"
```

出力:
- `*_drums.mid` - MIDI形式
- `*_drums.txt` - テキスト形式
- `*_drums_tab.txt` - ASCIIドラムタブ譜

## トラブルシューティング

### ffmpegが見つからない

```
ERROR: ffmpeg not found
```

→ ffmpegをインストールしてPATHに追加してください。

### GPU関連のエラー

GPU版を使用する場合は、CUDAがインストールされている必要があります。
CPUのみでも動作しますが、処理速度は遅くなります。

### メモリ不足

長い曲（10分以上）を処理する場合、メモリ不足になることがあります。
その場合は曲を分割するか、より多くのRAMを搭載したマシンで実行してください。

## ライセンス

このツールは個人的な研究・学習目的でのみ使用してください。
ダウンロードしたコンテンツを商用利用したり、再配布したりしないでください。

## 謝辞

- [Demucs](https://github.com/facebookresearch/demucs) - Facebook Research
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - nomadkaraoke
- [BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) - ByteDance AI Labs
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - yt-dlp team
