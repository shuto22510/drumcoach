"""
Bass Transcriber Module

Converts bass audio (WAV) to MIDI and MusicXML (sheet music) format.
Uses basic-pitch for audio-to-MIDI conversion and music21 for notation.

Optimized for electric bass guitar (monophonic/low polyphony).
"""

from pathlib import Path
from typing import Optional, Tuple


def get_onnx_model_path() -> str:
    """Get the path to the ONNX model file for basic-pitch."""
    import basic_pitch
    model_dir = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022"
    onnx_path = model_dir / "nmp.onnx"
    if onnx_path.exists():
        return str(onnx_path)
    raise FileNotFoundError(f"ONNX model not found at {onnx_path}")


def wav_to_midi(
    wav_path: str,
    output_dir: Optional[str] = None,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    minimum_note_length: float = 58.0,
    minimum_frequency: Optional[float] = 30.0,  # Low E on bass ~ 41Hz, allow some margin
    maximum_frequency: Optional[float] = 400.0,  # High notes on bass rarely exceed this
) -> str:
    """
    Convert bass WAV file to MIDI using basic-pitch.

    Parameters are tuned for bass guitar:
    - Higher thresholds to reduce noise
    - Frequency range limited to bass register

    Args:
        wav_path: Path to input WAV file
        output_dir: Output directory (default: same as input with /midi subfolder)
        onset_threshold: Note onset detection threshold (0.5-0.6 for bass)
        frame_threshold: Frame activation threshold (0.5-0.6 for bass)
        minimum_note_length: Minimum note duration in ms
        minimum_frequency: Minimum frequency in Hz (bass low E ~ 41Hz)
        maximum_frequency: Maximum frequency in Hz (bass high range)

    Returns:
        Path to generated MIDI file
    """
    from basic_pitch.inference import predict_and_save

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    if output_dir is None:
        output_dir = wav_path.parent / "bass_score"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting WAV to MIDI: {wav_path.name}")
    print(f"Parameters: onset_threshold={onset_threshold}, frame_threshold={frame_threshold}")
    print(f"Frequency range: {minimum_frequency}Hz - {maximum_frequency}Hz")
    print("-" * 60)

    # Use ONNX model for NumPy 2 compatibility
    onnx_model_path = get_onnx_model_path()

    # Run basic-pitch with bass-optimized parameters
    predict_and_save(
        audio_path_list=[str(wav_path)],
        output_directory=str(output_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=onnx_model_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    # Find the generated MIDI file (basic-pitch naming convention)
    midi_path = output_dir / f"{wav_path.stem}_basic_pitch.mid"

    if not midi_path.exists():
        raise RuntimeError(f"MIDI file was not created: {midi_path}")

    # Rename to _bass.mid for clarity
    final_midi_path = output_dir / f"{wav_path.stem}_bass.mid"
    midi_path.rename(final_midi_path)

    print(f"MIDI created: {final_midi_path}")
    return str(final_midi_path)


def midi_to_musicxml(
    midi_path: str,
    output_path: Optional[str] = None,
    title: str = "Bass Transcription",
) -> str:
    """
    Convert MIDI file to MusicXML with bass clef notation.

    Args:
        midi_path: Path to input MIDI file
        output_path: Output MusicXML path (default: same name with .musicxml)
        title: Title for the score

    Returns:
        Path to generated MusicXML file
    """
    from music21 import converter, clef, instrument, metadata

    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    if output_path is None:
        output_path = midi_path.with_suffix('.musicxml')
    else:
        output_path = Path(output_path)

    print(f"Converting MIDI to MusicXML: {midi_path.name}")
    print("-" * 60)

    # Parse MIDI file
    score = converter.parse(str(midi_path))

    # Set metadata
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = "AI Transcription (basic-pitch)"

    # Ensure bass clef is used for all parts
    for part in score.parts:
        # Set instrument to Electric Bass
        bass_instrument = instrument.ElectricBass()
        part.insert(0, bass_instrument)

        # Find all measures and ensure bass clef
        measures = part.getElementsByClass('Measure')
        if measures:
            first_measure = measures[0]
            # Check if clef exists, if not add bass clef
            existing_clefs = first_measure.getElementsByClass('Clef')
            if not existing_clefs:
                first_measure.insert(0, clef.BassClef())
            else:
                # Replace with bass clef if not already
                for c in existing_clefs:
                    if not isinstance(c, clef.BassClef):
                        first_measure.remove(c)
                        first_measure.insert(0, clef.BassClef())

        print(f"Part: {part.partName or 'Bass'} - Set to Bass Clef")

    # Write MusicXML
    score.write('musicxml', fp=str(output_path))

    print(f"MusicXML created: {output_path}")
    return str(output_path)


def transcribe_bass(
    wav_path: str,
    output_dir: Optional[str] = None,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    title: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Full pipeline: WAV -> MIDI -> MusicXML for bass.

    Args:
        wav_path: Path to bass WAV file
        output_dir: Output directory
        onset_threshold: Note onset threshold (higher = less noise)
        frame_threshold: Frame threshold (higher = less noise)
        title: Score title (default: derived from filename)

    Returns:
        Tuple of (midi_path, musicxml_path)
    """
    wav_path = Path(wav_path)

    if title is None:
        title = f"Bass: {wav_path.stem}"

    if output_dir is None:
        output_dir = wav_path.parent / "bass_score"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: WAV to MIDI
    print("=" * 60)
    print("[Step 1/2] Converting WAV to MIDI (basic-pitch)")
    print("=" * 60)
    midi_path = wav_to_midi(
        wav_path=str(wav_path),
        output_dir=str(output_dir),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

    # Step 2: MIDI to MusicXML
    print("\n" + "=" * 60)
    print("[Step 2/2] Converting MIDI to MusicXML (music21)")
    print("=" * 60)
    musicxml_path = output_dir / f"{wav_path.stem}_bass.musicxml"
    musicxml_path = midi_to_musicxml(
        midi_path=midi_path,
        output_path=str(musicxml_path),
        title=title,
    )

    return midi_path, musicxml_path
