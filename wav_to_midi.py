#!/usr/bin/env python
"""
WAV to MIDI Converter

Converts audio WAV files to MIDI using basic-pitch with ONNX backend.
Useful for transcribing extracted stems to sheet music.

Usage:
    python wav_to_midi.py <wav_file>
    python wav_to_midi.py <wav_file> --output <output_dir>
"""

import argparse
import sys
from pathlib import Path


def get_onnx_model_path():
    """Get the path to the ONNX model file."""
    import basic_pitch
    model_dir = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022"
    onnx_path = model_dir / "nmp.onnx"
    if onnx_path.exists():
        return str(onnx_path)
    raise FileNotFoundError(f"ONNX model not found at {onnx_path}")


def convert_wav_to_midi(
    wav_path: str,
    output_dir: str = None,
    save_midi: bool = True,
    save_csv: bool = False,
    save_model_outputs: bool = False,
) -> str:
    """
    Convert a WAV file to MIDI using basic-pitch with ONNX backend.

    Args:
        wav_path: Path to the input WAV file
        output_dir: Directory to save output files (default: same as input)
        save_midi: Whether to save MIDI file
        save_csv: Whether to save note events as CSV
        save_model_outputs: Whether to save raw model outputs

    Returns:
        Path to the generated MIDI file
    """
    from basic_pitch.inference import predict_and_save

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    if output_dir is None:
        output_dir = wav_path.parent / "midi"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting: {wav_path.name}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Use ONNX model explicitly for NumPy 2 compatibility
    onnx_model_path = get_onnx_model_path()
    print(f"Using ONNX model: {onnx_model_path}")

    # Run basic-pitch prediction with ONNX model
    predict_and_save(
        audio_path_list=[str(wav_path)],
        output_directory=str(output_dir),
        save_midi=save_midi,
        sonify_midi=False,
        save_model_outputs=save_model_outputs,
        save_notes=save_csv,
        model_or_model_path=onnx_model_path,
    )

    # Find the generated MIDI file
    midi_path = output_dir / f"{wav_path.stem}_basic_pitch.mid"

    if midi_path.exists():
        print(f"MIDI file created: {midi_path}")
        return str(midi_path)
    else:
        raise RuntimeError(f"MIDI file was not created: {midi_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert WAV files to MIDI using basic-pitch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single WAV file
  python wav_to_midi.py separated/song_drums.wav

  # Specify output directory
  python wav_to_midi.py separated/song_bass.wav --output midi_output

  # Also save CSV with note events
  python wav_to_midi.py separated/song_bass.wav --csv
        """
    )

    parser.add_argument(
        "wav_file",
        type=str,
        help="Path to the WAV file to convert"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for MIDI files"
    )

    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also save note events as CSV"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WAV to MIDI Converter (basic-pitch)")
    print("=" * 60)

    try:
        midi_path = convert_wav_to_midi(
            wav_path=args.wav_file,
            output_dir=args.output,
            save_csv=args.csv,
        )

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"MIDIファイルはここに保存されました: {Path(midi_path).absolute()}")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
