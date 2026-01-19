#!/usr/bin/env python
"""
YouTube Stem Extractor CLI (High-Quality Edition)

Downloads audio from YouTube and extracts instrument stems using best-in-class models:
- Vocals: BS-RoFormer (SDR 12.9) - highest quality vocal separation
- Instruments: htdemucs_6s - 6-stem separation (drums, bass, guitar, piano, other)

Usage:
    python get_stem.py <youtube_url> --stem <stem_name>
    python get_stem.py <youtube_url> --all

Examples:
    python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem vocals
    python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem drums
    python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --all

NOTE: This is for personal research and learning only.
Do not use for commercial purposes or redistribute downloaded content.
"""

import argparse
import sys
from pathlib import Path

from src.youtube_handler import (
    download_audio,
    extract_stem,
    extract_all_stems,
    ALL_STEMS,
    BEST_MODELS,
)


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Stem Extractor - High-quality stem separation using BS-RoFormer and htdemucs_6s",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models used:
  vocals  -> BS-RoFormer (SDR 12.9) - best quality vocal separation
  drums   -> htdemucs_6s (6-stem model)
  bass    -> htdemucs_6s (6-stem model)
  guitar  -> htdemucs_6s (6-stem model)
  piano   -> htdemucs_6s (6-stem model)
  other   -> htdemucs_6s (6-stem model)

Examples:
  # Extract vocals (highest quality with BS-RoFormer)
  python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem vocals

  # Extract drums
  python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem drums

  # Extract bass line
  python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem bass

  # Extract guitar
  python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem guitar

  # Extract piano
  python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --stem piano

  # Extract all 6 stems at once (vocals with BS-RoFormer, others with htdemucs_6s)
  python get_stem.py "https://www.youtube.com/watch?v=xxxxx" --all

NOTE: This tool is for personal research and learning only.
Do not use for commercial purposes or redistribute downloaded content.
        """
    )

    parser.add_argument(
        "url",
        type=str,
        help="YouTube video URL"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stem",
        type=str,
        choices=ALL_STEMS,
        help="Single stem to extract: vocals, drums, bass, guitar, piano, or other"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Extract all 6 stems (vocals with BS-RoFormer, instruments with htdemucs_6s)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("YouTube Stem Extractor (High-Quality Edition)")
    print("=" * 60)
    print(f"URL: {args.url}")
    if args.all:
        print("Mode: Extract ALL 6 stems")
        print("  - vocals: BS-RoFormer (SDR 12.9)")
        print("  - drums, bass, guitar, piano, other: htdemucs_6s")
    else:
        model = BEST_MODELS[args.stem]
        print(f"Target stem: {args.stem}")
        print(f"Model: {model}")
    print("=" * 60)

    try:
        # Step 1: Download audio
        print("\n[Step 1/2] Downloading audio from YouTube...")
        print("-" * 60)
        wav_path = download_audio(args.url, output_dir="data/downloads")

        # Step 2: Extract stem(s)
        if args.all:
            print(f"\n[Step 2/2] Extracting all 6 stems...")
            print("-" * 60)
            results = extract_all_stems(wav_path)

            # Done!
            print("\n" + "=" * 60)
            print("SUCCESS! All stems extracted.")
            print("=" * 60)
            print("\nExtracted stems:")
            for stem_name, stem_path in sorted(results.items()):
                if not stem_name.startswith('_'):  # Skip internal keys
                    print(f"  {stem_name}: {Path(stem_path).absolute()}")
            print("=" * 60)
        else:
            print(f"\n[Step 2/2] Extracting {args.stem} stem...")
            print("-" * 60)
            stem_path = extract_stem(wav_path, args.stem)

            # Done!
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"{args.stem}音源はここに保存されました: {Path(stem_path).absolute()}")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
