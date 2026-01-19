"""
Data loading and preprocessing module for AI Music Coach.
Handles MIDI and audio file loading, and converts them to chromagram features.
"""

import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from typing import Tuple
from scipy import interpolate


class DataLoader:
    """Loads and preprocesses MIDI and audio files."""

    def __init__(self, sr: int = 22050, hop_length: int = 512):
        """
        Initialize DataLoader.

        Args:
            sr: Sampling rate for audio processing
            hop_length: Hop length for feature extraction
        """
        self.sr = sr
        self.hop_length = hop_length

    def load_midi_chromagram(self, midi_path: str) -> np.ndarray:
        """
        Load MIDI file and convert to chromagram using pretty_midi's built-in method.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Chromagram array of shape (12, n_frames)

        Raises:
            FileNotFoundError: If MIDI file doesn't exist
            Exception: If MIDI file is invalid or empty
        """
        midi_path = Path(midi_path)
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))

            # Check if MIDI has any notes
            total_notes = sum(len(instrument.notes) for instrument in midi_data.instruments)
            if total_notes == 0:
                raise Exception(f"MIDI file has no notes: {midi_path}")

            # Get the total duration
            end_time = midi_data.get_end_time()
            if end_time == 0:
                raise Exception(f"MIDI file has zero duration: {midi_path}")

            # Calculate frame rate that matches librosa's hop_length
            # frame_rate = sr / hop_length (frames per second)
            frame_rate = self.sr / self.hop_length

            # Use pretty_midi's get_chroma() method
            # NOTE: pretty_midi.get_chroma() returns (n_frames, 12) by default
            # but we need (12, n_frames) to match librosa's convention
            chromagram = midi_data.get_chroma(fs=frame_rate)

            # Transpose to match librosa convention: (12, n_frames)
            if chromagram.shape[0] != 12:
                chromagram = chromagram.T

            # Ensure chromagram is not empty
            if chromagram.size == 0 or chromagram.shape[1] == 0:
                raise Exception(f"Generated chromagram is empty: {midi_path}")

            return chromagram

        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise Exception(f"Failed to process MIDI file {midi_path}: {str(e)}")

    def load_audio_chromagram(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and extract chromagram.

        Args:
            audio_path: Path to audio file

        Returns:
            Chromagram array of shape (12, n_frames)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If audio file is invalid or empty
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio file
            y, sr = librosa.load(str(audio_path), sr=self.sr)

            # Check if audio is empty
            if len(y) == 0:
                raise Exception(f"Audio file is empty: {audio_path}")

            # Extract chromagram using CENS (more robust to noise)
            chromagram = librosa.feature.chroma_cens(
                y=y,
                sr=self.sr,
                hop_length=self.hop_length
            )

            return chromagram

        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise Exception(f"Failed to process audio file {audio_path}: {str(e)}")

    def load_both(self, midi_path: str, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load both MIDI and audio files and return their chromagrams.

        Args:
            midi_path: Path to MIDI file
            audio_path: Path to audio file

        Returns:
            Tuple of (midi_chromagram, audio_chromagram)
        """
        midi_chroma = self.load_midi_chromagram(midi_path)
        audio_chroma = self.load_audio_chromagram(audio_path)

        return midi_chroma, audio_chroma
