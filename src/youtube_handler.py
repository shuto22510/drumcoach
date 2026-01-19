"""
YouTube Audio Handler for AI Music Coach.
Downloads audio from YouTube and extracts specific stems using best-in-class models.

Uses audio-separator library with optimal models for each stem:
- Vocals: BS-RoFormer (SDR 12.9) - best quality
- Drums/Bass: Specialized MDX models or htdemucs_6s
- Guitar/Piano: htdemucs_6s

NOTE: This is for personal research and learning only.
Do not use for commercial purposes or redistribute downloaded content.
"""

import re
from pathlib import Path
from typing import Optional

import yt_dlp

# Lazy import for audio_separator to avoid startup delay
_separator_instance = None


def sanitize_filename(title: str) -> str:
    """
    Convert video title to a safe filename (alphanumeric and underscores only).

    Args:
        title: Original video title

    Returns:
        Safe filename string
    """
    # Replace spaces with underscores
    safe = title.replace(" ", "_")
    # Keep only alphanumeric characters and underscores
    safe = re.sub(r'[^a-zA-Z0-9_]', '', safe)
    # Remove consecutive underscores
    safe = re.sub(r'_+', '_', safe)
    # Remove leading/trailing underscores
    safe = safe.strip('_')
    # Ensure non-empty
    if not safe:
        safe = "audio"
    return safe


def download_audio(url: str, output_dir: str = "data/downloads") -> str:
    """
    Download audio from YouTube and convert to high-quality WAV format.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the downloaded audio

    Returns:
        Path to the downloaded WAV file

    Raises:
        RuntimeError: If download fails
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # First, get video info to determine filename
    print("Fetching video information...")

    info_opts = {
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(info_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', 'Unknown')
        duration = info.get('duration', 0)

    safe_title = sanitize_filename(title)
    output_file = output_path / f"{safe_title}.wav"

    print(f"Title: {title}")
    print(f"Duration: {duration // 60}:{duration % 60:02d}")
    print(f"Output: {output_file}")
    print("-" * 60)

    # Configure yt-dlp options for high-quality WAV output
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',  # Best quality
        }],
        'outtmpl': str(output_path / safe_title) + '.%(ext)s',
        'quiet': False,
        'no_warnings': False,
    }

    print("Downloading and converting to WAV...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    if not output_file.exists():
        raise RuntimeError(f"Download completed but output file not found: {output_file}")

    print(f"Download complete: {output_file}")
    return str(output_file)


# =============================================================================
# High-Quality Stem Separation using audio-separator
# =============================================================================

# Best models for each stem type (based on SDR benchmarks)
# These are automatically downloaded by audio-separator on first use
BEST_MODELS = {
    # Vocals: BS-RoFormer has the best SDR (12.9)
    'vocals': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    # For other stems, use htdemucs_6s which outputs all 6 at once
    'drums': 'htdemucs_6s.yaml',
    'bass': 'htdemucs_6s.yaml',
    'guitar': 'htdemucs_6s.yaml',
    'piano': 'htdemucs_6s.yaml',
    'other': 'htdemucs_6s.yaml',
}

ALL_STEMS = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']


def _get_separator():
    """Get or create the Separator instance (lazy initialization)."""
    global _separator_instance
    if _separator_instance is None:
        from audio_separator.separator import Separator
        _separator_instance = Separator(
            output_dir="separated",
            output_format="wav",
        )
    return _separator_instance


def extract_stem(wav_path: str, stem_name: str) -> str:
    """
    Extract a specific stem using the best available model.

    For vocals: Uses BS-RoFormer (SDR 12.9) - highest quality
    For instruments: Uses htdemucs_6s for 6-stem separation

    Args:
        wav_path: Path to the input WAV file
        stem_name: Name of the stem to extract
                   ('vocals', 'drums', 'bass', 'guitar', 'piano', 'other')

    Returns:
        Path to the extracted stem WAV file

    Raises:
        ValueError: If stem_name is not valid
        RuntimeError: If separation fails
    """
    if stem_name not in ALL_STEMS:
        raise ValueError(f"Invalid stem name: {stem_name}. Must be one of {ALL_STEMS}")

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise RuntimeError(f"Input file not found: {wav_path}")

    model_name = BEST_MODELS[stem_name]

    print(f"Extracting '{stem_name}' from: {wav_path.name}")
    print(f"Using model: {model_name}")
    print("-" * 60)

    separator = _get_separator()

    # Load the appropriate model
    print(f"Loading model (this may download on first use)...")
    separator.load_model(model_filename=model_name)

    # Perform separation
    print("Separating audio...")
    output_files = separator.separate(str(wav_path))

    print(f"Separation complete. Output files: {output_files}")

    # Find the correct output file based on stem name
    stem_file = _find_stem_file(output_files, stem_name, wav_path.stem)

    if stem_file is None:
        raise RuntimeError(
            f"Could not find {stem_name} in output files: {output_files}"
        )

    print("-" * 60)
    print(f"Extraction complete!")
    return stem_file


def _find_stem_file(output_files: list, stem_name: str, track_name: str) -> Optional[str]:
    """
    Find the output file matching the requested stem.

    Args:
        output_files: List of output file paths from separator
        stem_name: Name of the stem to find
        track_name: Original track name (for matching)

    Returns:
        Path to the matching file, or None if not found
    """
    # Normalize stem name for matching
    stem_patterns = {
        'vocals': ['vocal', 'vocals'],
        'drums': ['drum', 'drums'],
        'bass': ['bass'],
        'guitar': ['guitar'],
        'piano': ['piano'],
        'other': ['other', 'no_vocal', 'instrumental', 'instrum'],
    }

    patterns = stem_patterns.get(stem_name, [stem_name])

    for filepath in output_files:
        filepath_lower = filepath.lower()
        for pattern in patterns:
            if pattern in filepath_lower:
                return filepath

    return None


def extract_all_stems(wav_path: str) -> dict:
    """
    Extract all 6 stems from the audio file.

    This uses htdemucs_6s for drums, bass, guitar, piano, other
    and BS-RoFormer for vocals (highest quality).

    Args:
        wav_path: Path to the input WAV file

    Returns:
        Dictionary mapping stem names to output file paths

    Raises:
        RuntimeError: If separation fails
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise RuntimeError(f"Input file not found: {wav_path}")

    results = {}
    separator = _get_separator()

    # Step 1: Extract vocals with BS-RoFormer (best quality)
    print("=" * 60)
    print("[1/2] Extracting vocals with BS-RoFormer (highest quality)")
    print("=" * 60)

    separator.load_model(model_filename=BEST_MODELS['vocals'])
    vocal_outputs = separator.separate(str(wav_path))

    for filepath in vocal_outputs:
        filepath_lower = filepath.lower()
        if 'vocal' in filepath_lower:
            results['vocals'] = filepath
        elif 'instrumental' in filepath_lower or 'no_vocal' in filepath_lower:
            # Keep instrumental for reference
            results['_instrumental'] = filepath

    # Step 2: Extract instrument stems with htdemucs_6s
    print("\n" + "=" * 60)
    print("[2/2] Extracting instruments with htdemucs_6s (6-stem model)")
    print("=" * 60)

    separator.load_model(model_filename='htdemucs_6s.yaml')
    instrument_outputs = separator.separate(str(wav_path))

    for filepath in instrument_outputs:
        filepath_lower = filepath.lower()
        if 'drum' in filepath_lower:
            results['drums'] = filepath
        elif 'bass' in filepath_lower:
            results['bass'] = filepath
        elif 'guitar' in filepath_lower:
            results['guitar'] = filepath
        elif 'piano' in filepath_lower:
            results['piano'] = filepath
        elif 'other' in filepath_lower:
            results['other'] = filepath

    return results
