#!/usr/bin/env python
"""
Drum Transcription Tool

Transcribes drum audio to notation using librosa for onset detection
and frequency analysis to classify kick, snare, and hi-hat.

Outputs:
- Text file with onset times and drum types
- MIDI file with drum notation (GM Drum Map)
- Optional: Simple ASCII drum tab

Usage:
    python drum_transcribe.py <drum_wav_file>
    python drum_transcribe.py <drum_wav_file> --output <output_dir>
    python drum_transcribe.py <drum_wav_file> --bpm 120
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
from midiutil import MIDIFile


# General MIDI Drum Map (Channel 10)
GM_DRUM_MAP = {
    'kick': 36,      # Bass Drum 1
    'snare': 38,     # Acoustic Snare
    'hihat': 42,     # Closed Hi-Hat
    'hihat_open': 46,  # Open Hi-Hat
    'tom_low': 45,   # Low Tom
    'tom_mid': 47,   # Low-Mid Tom
    'tom_high': 50,  # High Tom
    'crash': 49,     # Crash Cymbal 1
    'ride': 51,      # Ride Cymbal 1
}

# Drum notation positions (for display)
DRUM_NOTATION = {
    'crash': ('x', 5),      # Above staff
    'hihat': ('x', 5),      # Top line
    'ride': ('x', 5),
    'tom_high': ('o', 4),
    'snare': ('o', 3),      # Middle
    'tom_mid': ('o', 2),
    'tom_low': ('o', 1),
    'kick': ('o', 0),       # Bottom (bass drum)
}


def detect_onsets(y, sr):
    """Detect onset times in audio."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        wait=1,
        pre_avg=1,
        post_avg=1,
        pre_max=1,
        post_max=1,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times, onset_frames, onset_env


def classify_drum_hit(y, sr, onset_time, window_ms=50):
    """
    Classify a drum hit based on spectral characteristics.

    - Kick: Low frequency dominant (< 150 Hz)
    - Snare: Mid frequency with noise (150-500 Hz + high freq noise)
    - Hi-hat: High frequency dominant (> 5000 Hz)
    """
    # Extract a short window around the onset
    start_sample = int(onset_time * sr)
    window_samples = int(window_ms * sr / 1000)
    end_sample = min(start_sample + window_samples, len(y))

    if start_sample >= len(y):
        return 'unknown'

    segment = y[start_sample:end_sample]

    if len(segment) < 256:
        return 'unknown'

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1/sr)

    # Frequency bands
    low_mask = freqs < 150
    mid_mask = (freqs >= 150) & (freqs < 1000)
    high_mask = freqs >= 5000

    low_energy = np.sum(spectrum[low_mask] ** 2) if np.any(low_mask) else 0
    mid_energy = np.sum(spectrum[mid_mask] ** 2) if np.any(mid_mask) else 0
    high_energy = np.sum(spectrum[high_mask] ** 2) if np.any(high_mask) else 0

    total_energy = low_energy + mid_energy + high_energy
    if total_energy == 0:
        return 'unknown'

    low_ratio = low_energy / total_energy
    mid_ratio = mid_energy / total_energy
    high_ratio = high_energy / total_energy

    # Classification logic
    if high_ratio > 0.4:
        return 'hihat'
    elif low_ratio > 0.5:
        return 'kick'
    elif mid_ratio > 0.3:
        return 'snare'
    elif low_ratio > 0.3:
        return 'kick'
    else:
        return 'snare'  # Default to snare for mid-range hits


def transcribe_drums(wav_path, bpm=None):
    """
    Transcribe drum audio to notation.

    Args:
        wav_path: Path to drum audio file
        bpm: Tempo in BPM (auto-detected if None)

    Returns:
        List of (time, drum_type) tuples and detected BPM
    """
    print(f"Loading audio: {wav_path}")
    y, sr = librosa.load(wav_path, sr=22050)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f} seconds")

    # Detect tempo if not specified
    if bpm is None:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            bpm = float(tempo[0])
        else:
            bpm = float(tempo)
        print(f"Detected tempo: {bpm:.1f} BPM")
    else:
        print(f"Using specified tempo: {bpm} BPM")

    # Detect onsets
    print("Detecting drum hits...")
    onset_times, onset_frames, onset_env = detect_onsets(y, sr)
    print(f"Found {len(onset_times)} hits")

    # Classify each hit
    print("Classifying drum sounds...")
    hits = []
    for onset_time in onset_times:
        drum_type = classify_drum_hit(y, sr, onset_time)
        hits.append((onset_time, drum_type))

    # Count by type
    type_counts = {}
    for _, dtype in hits:
        type_counts[dtype] = type_counts.get(dtype, 0) + 1

    print("\nDrum hit summary:")
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count}")

    return hits, bpm


def hits_to_midi(hits, bpm, output_path):
    """Convert drum hits to MIDI file."""
    midi = MIDIFile(1)
    track = 0
    channel = 9  # Drum channel (0-indexed, so 9 = channel 10)
    time = 0
    volume = 100

    midi.addTempo(track, time, bpm)
    midi.addTrackName(track, time, "Drums")

    for onset_time, drum_type in hits:
        if drum_type in GM_DRUM_MAP:
            pitch = GM_DRUM_MAP[drum_type]
            # Convert time to beats
            beat_time = onset_time * bpm / 60
            midi.addNote(track, channel, pitch, beat_time, 0.25, volume)

    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    print(f"MIDI saved: {output_path}")


def hits_to_text(hits, bpm, output_path):
    """Save hits to text file."""
    with open(output_path, 'w') as f:
        f.write(f"# Drum Transcription\n")
        f.write(f"# Tempo: {bpm:.1f} BPM\n")
        f.write(f"# Total hits: {len(hits)}\n")
        f.write(f"# Format: time(seconds), beat, drum_type\n\n")

        for onset_time, drum_type in hits:
            beat = onset_time * bpm / 60
            f.write(f"{onset_time:.3f}, {beat:.2f}, {drum_type}\n")

    print(f"Text saved: {output_path}")


def hits_to_ascii_tab(hits, bpm, output_path, measures=4):
    """Generate simple ASCII drum tab."""
    # Quantize to 16th notes
    beat_duration = 60 / bpm
    sixteenth = beat_duration / 4

    # Group hits by measure
    beats_per_measure = 4
    measure_duration = beats_per_measure * beat_duration

    lines = {
        'HH': [],  # Hi-hat
        'SD': [],  # Snare
        'BD': [],  # Bass drum (kick)
    }

    # Find total measures
    if hits:
        max_time = max(h[0] for h in hits)
        total_measures = int(max_time / measure_duration) + 1
    else:
        total_measures = 1

    # Create grid (16 positions per measure)
    grid_size = 16

    for drum in lines:
        lines[drum] = ['-'] * (total_measures * grid_size)

    # Place hits on grid
    for onset_time, drum_type in hits:
        grid_pos = int(onset_time / sixteenth)
        if grid_pos < len(lines['HH']):
            if drum_type == 'hihat':
                lines['HH'][grid_pos] = 'x'
            elif drum_type == 'snare':
                lines['SD'][grid_pos] = 'o'
            elif drum_type == 'kick':
                lines['BD'][grid_pos] = 'o'

    # Format output
    with open(output_path, 'w') as f:
        f.write(f"Drum Tab - {bpm:.0f} BPM\n")
        f.write("=" * 60 + "\n\n")

        # Print in chunks of 1 measure (16 positions)
        for m in range(total_measures):
            start = m * grid_size
            end = start + grid_size

            f.write(f"Measure {m + 1}\n")
            f.write(f"HH |{''.join(lines['HH'][start:end])}|\n")
            f.write(f"SD |{''.join(lines['SD'][start:end])}|\n")
            f.write(f"BD |{''.join(lines['BD'][start:end])}|\n")
            f.write(f"    1 e + a 2 e + a 3 e + a 4 e + a\n\n")

    print(f"ASCII tab saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe drum audio to notation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python drum_transcribe.py separated/drums.wav
  python drum_transcribe.py separated/drums.wav --bpm 120
  python drum_transcribe.py separated/drums.wav --output output/
        """
    )

    parser.add_argument(
        "wav_file",
        type=str,
        help="Path to the drum WAV file"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: same as input)"
    )

    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Tempo in BPM (auto-detected if not specified)"
    )

    args = parser.parse_args()

    wav_path = Path(args.wav_file)
    if not wav_path.exists():
        print(f"Error: File not found: {wav_path}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = wav_path.parent / "drum_notation"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Drum Transcription Tool")
    print("=" * 60)

    try:
        # Transcribe
        hits, bpm = transcribe_drums(str(wav_path), args.bpm)

        # Generate outputs
        stem = wav_path.stem

        # MIDI
        midi_path = output_dir / f"{stem}_drums.mid"
        hits_to_midi(hits, bpm, str(midi_path))

        # Text
        text_path = output_dir / f"{stem}_drums.txt"
        hits_to_text(hits, bpm, str(text_path))

        # ASCII Tab
        tab_path = output_dir / f"{stem}_drums_tab.txt"
        hits_to_ascii_tab(hits, bpm, str(tab_path))

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"\nOutput files:")
        print(f"  MIDI: {midi_path.absolute()}")
        print(f"  Text: {text_path.absolute()}")
        print(f"  Tab:  {tab_path.absolute()}")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
