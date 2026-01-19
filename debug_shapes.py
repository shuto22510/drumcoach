"""Debug script to check chromagram shapes."""

import numpy as np
import librosa
import pretty_midi

# Load MIDI and check shape
print("="*60)
print("Checking MIDI chromagram shape...")
pm = pretty_midi.PrettyMIDI('data/teacher.mid')
midi_chroma = pm.get_chroma(fs=43.066406)
print(f"pretty_midi chroma shape: {midi_chroma.shape}")
print(f"Dimensions: rows={midi_chroma.shape[0]}, cols={midi_chroma.shape[1]}")
print()

# Load audio and check shape
print("Checking audio chromagram shape...")
y, sr = librosa.load('data/student.wav', sr=22050)
audio_chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=512)
print(f"librosa chroma shape: {audio_chroma.shape}")
print(f"Dimensions: rows={audio_chroma.shape[0]}, cols={audio_chroma.shape[1]}")
print()

# Check if they match
print("="*60)
print("Shape comparison:")
print(f"MIDI:  {midi_chroma.shape}")
print(f"Audio: {audio_chroma.shape}")
print(f"First dimension match (should both be 12): {midi_chroma.shape[0] == audio_chroma.shape[0] == 12}")
print("="*60)
