"""Test loading webm file with librosa."""
import librosa

print("Loading soramotoberuhazu.webm...")
try:
    y, sr = librosa.load('data/soramotoberuhazu.webm', sr=22050, duration=10)
    print(f'Successfully loaded!')
    print(f'Sample rate: {sr}')
    print(f'Duration: {len(y)/sr:.2f} seconds (first 10 sec)')
    print(f'Samples: {len(y)}')
except Exception as e:
    print(f'Error: {e}')
    print("\nFFmpeg is required to load webm files.")
    print("Install with: winget install ffmpeg")
