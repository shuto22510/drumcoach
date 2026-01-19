"""
Convert drum audio to MIDI using onset detection and drum classification.
"""

import sys
from pathlib import Path

import numpy as np
import librosa
import midiutil


def detect_drum_onsets(audio_path: str, sr: int = 22050) -> dict:
    """
    Detect drum onsets and classify them by frequency range.

    Args:
        audio_path: Path to drum audio file
        sr: Sample rate

    Returns:
        Dictionary with onset times for each drum type
    """
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr
    print(f"  Duration: {duration:.2f} seconds")

    # Separate into frequency bands for drum classification
    # Kick: ~20-100 Hz, Snare: ~100-400 Hz, Hi-hat: ~3000+ Hz

    print("\nAnalyzing frequency bands...")

    # Low frequencies (Kick drum)
    y_low = librosa.effects.harmonic(y)  # Remove transients first
    S_low = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Create frequency band masks
    kick_mask = (freqs >= 20) & (freqs <= 150)
    snare_mask = (freqs >= 150) & (freqs <= 500)
    hihat_mask = (freqs >= 5000)

    # Apply masks and get onset envelope for each
    S_kick = S_low.copy()
    S_kick[~kick_mask, :] = 0
    S_snare = S_low.copy()
    S_snare[~snare_mask, :] = 0
    S_hihat = S_low.copy()
    S_hihat[~hihat_mask, :] = 0

    # Compute onset envelopes
    onset_env_kick = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_kick))
    onset_env_snare = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_snare))
    onset_env_hihat = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_hihat))

    # Detect onsets
    kick_frames = librosa.onset.onset_detect(onset_envelope=onset_env_kick, sr=sr)
    snare_frames = librosa.onset.onset_detect(onset_envelope=onset_env_snare, sr=sr)
    hihat_frames = librosa.onset.onset_detect(onset_envelope=onset_env_hihat, sr=sr)

    # Convert to times
    kick_times = librosa.frames_to_time(kick_frames, sr=sr)
    snare_times = librosa.frames_to_time(snare_frames, sr=sr)
    hihat_times = librosa.frames_to_time(hihat_frames, sr=sr)

    print(f"  Kick onsets: {len(kick_times)}")
    print(f"  Snare onsets: {len(snare_times)}")
    print(f"  Hi-hat onsets: {len(hihat_times)}")

    return {
        'kick': kick_times,
        'snare': snare_times,
        'hihat': hihat_times,
        'duration': duration,
        'sr': sr
    }


def detect_all_onsets(audio_path: str, sr: int = 22050) -> dict:
    """
    Simple onset detection without drum type classification.

    Args:
        audio_path: Path to drum audio file
        sr: Sample rate

    Returns:
        Dictionary with all onset times
    """
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr
    print(f"  Duration: {duration:.2f} seconds")

    # Get tempo
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    print(f"  Estimated tempo: {tempo:.1f} BPM")

    # Detect onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Get onset strengths for velocity
    onset_strengths = onset_env[onset_frames]
    # Normalize to MIDI velocity range (1-127)
    if len(onset_strengths) > 0:
        min_str = onset_strengths.min()
        max_str = onset_strengths.max()
        if max_str > min_str:
            velocities = 40 + 87 * (onset_strengths - min_str) / (max_str - min_str)
        else:
            velocities = np.full_like(onset_strengths, 80)
        velocities = velocities.astype(int)
    else:
        velocities = np.array([])

    print(f"  Total onsets: {len(onset_times)}")

    return {
        'onset_times': onset_times,
        'velocities': velocities,
        'tempo': tempo,
        'duration': duration,
        'sr': sr
    }


def create_drum_midi(onsets: dict, output_path: str, use_classification: bool = False):
    """
    Create MIDI file from drum onsets.

    Args:
        onsets: Dictionary with onset data
        output_path: Path for output MIDI file
        use_classification: If True, use kick/snare/hihat classification
    """
    print(f"\nCreating MIDI file: {output_path}")

    # General MIDI drum notes
    # Channel 10 (index 9) is for drums
    GM_KICK = 36      # Bass Drum 1
    GM_SNARE = 38     # Acoustic Snare
    GM_HIHAT_CLOSED = 42  # Closed Hi-Hat
    GM_HIHAT_OPEN = 46    # Open Hi-Hat

    tempo = onsets.get('tempo', 120.0)

    # Create MIDI file with 1 track
    midi = midiutil.MIDIFile(1)
    track = 0
    channel = 9  # Drum channel

    midi.addTempo(track, 0, tempo)
    midi.addTrackName(track, 0, "Drums")

    if use_classification:
        # Add classified drum hits
        for t in onsets.get('kick', []):
            beat = t * tempo / 60.0  # Convert seconds to beats
            midi.addNote(track, channel, GM_KICK, beat, 0.25, 100)

        for t in onsets.get('snare', []):
            beat = t * tempo / 60.0
            midi.addNote(track, channel, GM_SNARE, beat, 0.25, 100)

        for t in onsets.get('hihat', []):
            beat = t * tempo / 60.0
            midi.addNote(track, channel, GM_HIHAT_CLOSED, beat, 0.25, 80)

        total_notes = len(onsets.get('kick', [])) + len(onsets.get('snare', [])) + len(onsets.get('hihat', []))
    else:
        # Add all onsets as snare hits (most audible)
        onset_times = onsets.get('onset_times', [])
        velocities = onsets.get('velocities', [])

        for i, t in enumerate(onset_times):
            beat = t * tempo / 60.0
            vel = int(velocities[i]) if i < len(velocities) else 80
            midi.addNote(track, channel, GM_SNARE, beat, 0.25, vel)

        total_notes = len(onset_times)

    # Write MIDI file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    print(f"  Total notes: {total_notes}")
    print(f"  Tempo: {tempo:.1f} BPM")
    print(f"  MIDI saved successfully!")


def main():
    if len(sys.argv) < 2:
        print("Drum Audio to MIDI Converter")
        print("=" * 60)
        print("\nUsage:")
        print("  python drums_to_midi.py <drum_audio_file>")
        print("  python drums_to_midi.py <drum_audio_file> --classify")
        print()
        print("Options:")
        print("  --classify   Attempt to classify kick/snare/hihat")
        print()
        print("Examples:")
        print("  python drums_to_midi.py output/demucs/htdemucs/soramotoberuhazu/drums.wav")
        return

    audio_path = sys.argv[1]
    use_classification = "--classify" in sys.argv

    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        return

    print("=" * 60)
    print("DRUM AUDIO TO MIDI CONVERTER")
    print("=" * 60)

    # Detect onsets
    if use_classification:
        print("\nUsing frequency-based drum classification...")
        onsets = detect_drum_onsets(audio_path)
    else:
        print("\nUsing simple onset detection...")
        onsets = detect_all_onsets(audio_path)

    # Create output path
    input_path = Path(audio_path)
    output_path = input_path.parent / f"{input_path.stem}.mid"

    # Create MIDI
    create_drum_midi(onsets, str(output_path), use_classification)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
