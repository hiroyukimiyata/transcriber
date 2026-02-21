# Audio to Score Converter

ライブ演奏のようなテンポ揺れを含む音源から、分離済みパートを拍基準で量子化して MusicXML を生成するツールです。

## 特徴

- `full_mix.wav` / `full_mix.mp3` から **動的ビート列**（拍タイムスタンプ）を抽出
  - `madmom` を優先利用（downbeat 対応）
  - 未導入時は `librosa` にフォールバック
- `isolated_part.wav` / `isolated_part.mp3` から `basic-pitch` でノートイベント抽出
- 秒ベースのノートを拍空間へ写像し、16分等の分割で量子化
- `music21` で MusicXML を生成
- 小節ごと推定テンポ（`MetronomeMark`）を埋め込み
- オプションでピアノ大譜表（RH/LH）に分割

## インストール

```bash
pip install numpy music21 basic-pitch madmom librosa soundfile
```

> `madmom` の導入が難しい環境では `--no-madmom` で `librosa` 強制にできます。

## 使い方

```bash
python audio_to_score_converter.py \
  --full-mix full_mix.mp3 \
  --isolated-part isolated_part.mp3 \
  --output output.musicxml
```

### 主なオプション

- `--subdivision` : 1拍あたりの量子化分割（既定: `4` = 16分）
- `--beats-per-measure` : 小節拍数のフォールバック値（既定: `4`）
- `--piano` : ピアノ大譜表（RH/LH）で出力
- `--no-madmom` : `madmom` を使わず `librosa` を使用
- 入力形式: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aiff`, `.aif`

## 処理フロー

1. **Beat tracking** (`full_mix.wav` / `full_mix.mp3`)
   - 拍タイムスタンプ列を取得
   - 可能なら downbeat を検出して小節境界基準を強化
2. **Pitch extraction** (`isolated_part.wav` / `isolated_part.mp3`)
   - MIDI note / start / end / velocity を秒単位で取得
3. **Dynamic quantization**
   - 秒→拍へ補間変換
   - 拍分割グリッドへ吸着（テンポ揺れに追従）
4. **MusicXML build**
   - 拍基準オフセットでノート配置
   - 小節ごとのテンポ記号を挿入して再生同期性を改善

## 出力

- MuseScore 等で読み込める `.musicxml`

## 開発者向けチェック

```bash
python -m py_compile audio_to_score_converter.py
python -m unittest -v tests/test_audio_to_score_converter.py
```

