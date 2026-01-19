"""Test script to verify chromagram shapes."""

import numpy as np
from src.loader import DataLoader

# Initialize loader
loader = DataLoader(sr=22050, hop_length=512)

# Load chromagrams
print("Loading MIDI chromagram...")
midi_chroma = loader.load_midi_chromagram('data/teacher.mid')
print(f"MIDI chromagram shape: {midi_chroma.shape}")
print(f"MIDI chromagram dtype: {midi_chroma.dtype}")
print(f"MIDI first few values: {midi_chroma[:, 0]}")
print()

print("Loading audio chromagram...")
audio_chroma = loader.load_audio_chromagram('data/student.wav')
print(f"Audio chromagram shape: {audio_chroma.shape}")
print(f"Audio chromagram dtype: {audio_chroma.dtype}")
print(f"Audio first few values: {audio_chroma[:, 0]}")
print()

# Test transpose
print("Testing transpose for DTW...")
midi_t = midi_chroma.T
audio_t = audio_chroma.T
print(f"MIDI transposed: {midi_t.shape}")
print(f"Audio transposed: {audio_t.shape}")
print(f"Feature dimension (should both be 12): MIDI={midi_t.shape[1]}, Audio={audio_t.shape[1]}")
print()

# Try DTW with correct format
print("Testing librosa DTW...")
import librosa

# DON'T transpose! librosa.sequence.dtw expects (K, N) format where K=12 features
print(f"Using MIDI shape: {midi_chroma.shape}")
print(f"Using Audio shape: {audio_chroma.shape}")

# Fix dtype and memory layout
midi_fixed = np.ascontiguousarray(midi_chroma, dtype=np.float64)
audio_fixed = np.ascontiguousarray(audio_chroma, dtype=np.float64)
print(f"After fix - MIDI: {midi_fixed.shape}, dtype: {midi_fixed.dtype}, C-cont: {midi_fixed.flags['C_CONTIGUOUS']}")
print(f"After fix - Audio: {audio_fixed.shape}, dtype: {audio_fixed.dtype}, C-cont: {audio_fixed.flags['C_CONTIGUOUS']}")

try:
    D, wp = librosa.sequence.dtw(X=midi_fixed, Y=audio_fixed, metric='cosine')
    print("SUCCESS! DTW computed without errors")
    print(f"Cost matrix shape: {D.shape}")
    print(f"Path length: {len(wp)}")
    print(f"Total cost: {D[-1, -1]:.4f}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
