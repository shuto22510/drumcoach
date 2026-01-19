"""
Drum Rhythm Extractor using librosa HPSS.
Separates percussive (drum) components from audio.
"""

import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


def extract_drums(
    audio_path: str,
    output_path: str = None,
    margin: float = 3.0
) -> tuple:
    """
    Extract drum/percussive components from audio using HPSS.

    Args:
        audio_path: Path to input audio file
        output_path: Optional path to save percussive audio
        margin: Separation margin (higher = cleaner separation, default 3.0)

    Returns:
        Tuple of (y_percussive, y_harmonic, sr)
    """
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")

    print(f"\nApplying HPSS (margin={margin})...")
    # HPSS: Harmonic-Percussive Source Separation
    # margin controls the separation strength
    # Higher margin = more aggressive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=margin)

    print("  Separation complete!")

    # Calculate energy ratios
    harmonic_energy = np.sum(y_harmonic ** 2)
    percussive_energy = np.sum(y_percussive ** 2)
    total_energy = harmonic_energy + percussive_energy

    print(f"\nEnergy distribution:")
    print(f"  Harmonic (melody/chords): {100 * harmonic_energy / total_energy:.1f}%")
    print(f"  Percussive (drums/rhythm): {100 * percussive_energy / total_energy:.1f}%")

    # Save if output path provided
    if output_path:
        print(f"\nSaving percussive track to: {output_path}")
        sf.write(output_path, y_percussive, sr)
        print("  Saved successfully!")

    return y_percussive, y_harmonic, sr


def extract_and_save_both(
    audio_path: str,
    output_dir: str = "output",
    margin: float = 3.0
) -> dict:
    """
    Extract and save both harmonic and percussive components.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save output files
        margin: Separation margin

    Returns:
        Dictionary with paths to saved files
    """
    input_path = Path(audio_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    stem = input_path.stem

    y_percussive, y_harmonic, sr = extract_drums(audio_path, margin=margin)

    # Save both components
    perc_path = output_path / f"{stem}_drums.wav"
    harm_path = output_path / f"{stem}_melody.wav"

    print(f"\nSaving separated tracks:")

    sf.write(str(perc_path), y_percussive, sr)
    print(f"  Drums/Percussion: {perc_path}")

    sf.write(str(harm_path), y_harmonic, sr)
    print(f"  Melody/Harmony: {harm_path}")

    return {
        'percussive': str(perc_path),
        'harmonic': str(harm_path),
        'sample_rate': sr
    }


def analyze_rhythm(audio_path: str, output_dir: str = "output") -> dict:
    """
    Extract drums and analyze rhythm patterns.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory for output files

    Returns:
        Dictionary with rhythm analysis results
    """
    # First extract drums
    y_percussive, _, sr = extract_drums(audio_path)

    print("\n" + "=" * 50)
    print("RHYTHM ANALYSIS")
    print("=" * 50)

    # Compute onset strength (attack detection)
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

    # Estimate tempo
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # tempo may be an array in newer librosa versions
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    print(f"\nEstimated tempo: {tempo:.1f} BPM")
    print(f"Detected beats: {len(beat_times)}")

    if len(beat_times) > 1:
        # Calculate beat intervals
        beat_intervals = np.diff(beat_times)
        avg_interval = np.mean(beat_intervals)
        std_interval = np.std(beat_intervals)

        print(f"Average beat interval: {avg_interval:.3f} seconds")
        print(f"Beat consistency (std): {std_interval:.3f} seconds")

        if std_interval < 0.05:
            print("Rhythm: Very consistent (machine-like)")
        elif std_interval < 0.1:
            print("Rhythm: Consistent (tight performance)")
        elif std_interval < 0.2:
            print("Rhythm: Moderate variation (human feel)")
        else:
            print("Rhythm: High variation (expressive/rubato)")

    # Detect onsets (individual hits)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units='time'
    )
    print(f"\nDetected drum hits: {len(onsets)}")

    if len(onsets) > 0:
        # Calculate hit density (hits per second)
        duration = len(y_percussive) / sr
        hit_density = len(onsets) / duration
        print(f"Hit density: {hit_density:.1f} hits/second")

    # Save onset times to file
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    onset_file = output_path / "drum_onsets.txt"
    with open(onset_file, 'w') as f:
        f.write(f"Tempo: {tempo:.1f} BPM\n")
        f.write(f"Total beats: {len(beat_times)}\n")
        f.write(f"Total hits: {len(onsets)}\n\n")
        f.write("Beat times (seconds):\n")
        for i, t in enumerate(beat_times):
            f.write(f"  Beat {i+1}: {t:.3f}\n")

    print(f"\nOnset data saved to: {onset_file}")

    return {
        'tempo': float(tempo),
        'beat_times': beat_times.tolist(),
        'onset_times': onsets.tolist(),
        'beat_count': len(beat_times),
        'onset_count': len(onsets)
    }


def main():
    if len(sys.argv) < 2:
        print("Drum Rhythm Extractor using librosa HPSS")
        print("=" * 60)
        print("\nUsage:")
        print("  python drum_extractor.py <audio_file>")
        print("  python drum_extractor.py <audio_file> --both")
        print("  python drum_extractor.py <audio_file> --analyze")
        print("  python drum_extractor.py <audio_file> --margin <value>")
        print()
        print("Options:")
        print("  --both      Save both drums and melody tracks")
        print("  --analyze   Extract drums and analyze rhythm")
        print("  --margin    Separation strength (default: 3.0)")
        print("              Higher = cleaner but may lose some content")
        print()
        print("Examples:")
        print("  # Extract drums only")
        print("  python drum_extractor.py data/lemon.wav")
        print()
        print("  # Extract both drums and melody")
        print("  python drum_extractor.py data/lemon.wav --both")
        print()
        print("  # Extract and analyze rhythm")
        print("  python drum_extractor.py data/lemon.wav --analyze")
        return

    audio_path = sys.argv[1]

    # Check if file exists
    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        return

    # Parse margin option
    margin = 3.0
    if "--margin" in sys.argv:
        idx = sys.argv.index("--margin")
        if idx + 1 < len(sys.argv):
            try:
                margin = float(sys.argv[idx + 1])
            except ValueError:
                print("Error: --margin requires a numeric value")
                return

    print("=" * 60)
    print("DRUM RHYTHM EXTRACTOR")
    print("=" * 60)

    if "--analyze" in sys.argv:
        analyze_rhythm(audio_path)
    elif "--both" in sys.argv:
        extract_and_save_both(audio_path, margin=margin)
    else:
        # Default: extract drums only
        input_path = Path(audio_path)
        output_path = Path("output") / f"{input_path.stem}_drums.wav"
        Path("output").mkdir(exist_ok=True)
        extract_drums(audio_path, output_path=str(output_path), margin=margin)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
