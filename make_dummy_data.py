import numpy as np
import pretty_midi
from scipy.io import wavfile
import os

# 保存先
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def create_teacher_midi(filename):
    """お手本MIDIを作る (ドレミファソラシド)"""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0) # 0=Piano
    
    # ドレミファソラシド (C4〜C5)
    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    
    start_time = 0.0
    duration = 0.5 # 0.5秒間隔
    
    for note_number in notes:
        note = pretty_midi.Note(
            velocity=100, pitch=note_number, start=start_time, end=start_time + duration
        )
        inst.notes.append(note)
        start_time += duration

    pm.instruments.append(inst)
    pm.write(filename)
    print(f"作成完了: {filename}")
    return pm

def synthesize_student_wav(pm, filename):
    """
    MIDIからWAVを作る
    ただし、人間っぽくするために「テンポを遅く」し「ノイズ」を入れる
    """
    # 1. MIDIを純粋なサイン波で合成する関数
    fs = 22050
    audio_data = np.array([])
    
    # 各ノートをサイン波に変換
    for note in pm.instruments[0].notes:
        # 周波数変換
        freq = 440.0 * (2.0 ** ((note.pitch - 69.0) / 12.0))
        
        # 本来の長さよりわざと長くする（テンポが遅い生徒をシミュレート）
        duration = (note.end - note.start) * 1.2 # 1.2倍遅い
        
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # ちょっとノイズを足す（録音環境の悪さをシミュレート）
        noise = np.random.normal(0, 0.01, len(wave))
        wave = wave + noise
        
        audio_data = np.concatenate([audio_data, wave])

    # 2. WAVファイルとして書き出し
    # float32 (-1.0 ~ 1.0) を int16 に変換
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(filename, fs, audio_data_int16)
    print(f"作成完了: {filename} (わざとテンポを遅くしてあります)")

if __name__ == "__main__":
    midi_path = os.path.join(DATA_DIR, "teacher.mid")
    wav_path = os.path.join(DATA_DIR, "student.wav")
    
    # MIDI作成
    pm = create_teacher_midi(midi_path)
    
    # WAV作成（MIDIを元に劣化させる）
    synthesize_student_wav(pm, wav_path)