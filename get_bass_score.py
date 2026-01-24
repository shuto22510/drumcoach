#!/usr/bin/env python
"""
Bass Score Generator CLI

Converts bass audio (WAV) to MIDI and MusicXML sheet music.
Optimized for electric bass guitar transcription.

Usage:
    python get_bass_score.py <bass_wav_file>
    python get_bass_score.py <bass_wav_file> --output <output_dir>
    python get_bass_score.py <bass_wav_file> --threshold 0.6

Examples:
    python get_bass_score.py separated/song_bass.wav
    python get_bass_score.py separated/song_bass.wav --threshold 0.6 --title "My Bass Line"
"""

import argparse
import sys
from pathlib import Path

from src.bass_transcriber import transcribe_bass


def main():
    parser = argparse.ArgumentParser(
        description="Convert bass WAV to MIDI and MusicXML sheet music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python get_bass_score.py separated/song_bass.wav

  # Specify output directory
  python get_bass_score.py separated/song_bass.wav --output output/

  # Higher threshold for cleaner transcription (less noise)
  python get_bass_score.py separated/song_bass.wav --threshold 0.6

  # With custom title
  python get_bass_score.py separated/song_bass.wav --title "Funk Bass Line"

Output:
  - [name]_bass.mid      : MIDI file
  - [name]_bass.musicxml : MusicXML sheet music (bass clef)
        """
    )

    parser.add_argument(
        "input_wav",
        type=str,
        help="Path to the bass WAV file"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: input_dir/bass_score/)"
    )

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Detection threshold (0.0-1.0, higher = less noise, default: 0.5)"
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the sheet music"
    )

    args = parser.parse_args()

    wav_path = Path(args.input_wav)
    if not wav_path.exists():
        print(f"Error: File not found: {wav_path}")
        sys.exit(1)

    print("=" * 60)
    print("Bass Score Generator")
    print("=" * 60)
    print(f"Input: {wav_path}")
    print(f"Threshold: {args.threshold}")
    if args.title:
        print(f"Title: {args.title}")
    print("=" * 60)

    try:
        midi_path, musicxml_path = transcribe_bass(
            wav_path=str(wav_path),
            output_dir=args.output,
            onset_threshold=args.threshold,
            frame_threshold=args.threshold,
            title=args.title,
        )

        print("\n" + "=" * 60)
        print("完了: 楽譜ファイルを出力しました")
        print("=" * 60)
        print(f"MIDI:     {Path(midi_path).absolute()}")
        print(f"MusicXML: {Path(musicxml_path).absolute()}")
        print("=" * 60)
        print("\nMusicXMLはMuseScore, Finale, Sibelius等で開けます。")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
