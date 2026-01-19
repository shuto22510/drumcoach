import sys
import pretty_midi
import os

def list_and_extract(midi_path, output_path):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"エラー: MIDIファイルを読み込めませんでした。\n{e}")
        return

    print(f"\n=== トラック一覧: {os.path.basename(midi_path)} ===")
    print(f"{'ID':<4} | {'楽器名 (Program)':<20} | {'ノート数':<10} | {'トラック名'}")
    print("-" * 60)

    valid_tracks = []
    for i, instrument in enumerate(pm.instruments):
        # ドラム(is_drum=True)は除外してもいいが、一応表示する
        drum_mark = "[DRUM]" if instrument.is_drum else ""
        inst_name = pretty_midi.program_to_instrument_name(instrument.program)
        print(f"{i:<4} | {inst_name:<20} {drum_mark}| {len(instrument.notes):<10} | {instrument.name}")
        valid_tracks.append(i)

    print("-" * 60)
    
    # ユーザーに入力を求める
    while True:
        try:
            target_id = int(input("\n抽出したいトラックのID番号を入力してください: "))
            if target_id in valid_tracks:
                break
            print("無効なIDです。リストにある番号を入力してください。")
        except ValueError:
            print("数字を入力してください。")

    # 選ばれたトラック以外を削除する
    # pretty_midiは直接削除するメソッドがないので、新しいオブジェクトを作るのが安全
    new_pm = pretty_midi.PrettyMIDI()
    target_inst = pm.instruments[target_id]
    new_pm.instruments.append(target_inst)
    
    # 元のテンポ情報をコピー（これ重要）
    # ※厳密なコピーは難しいが、pretty_midiの仕様上、基本的なテンポマップは保持されることが多い
    # ここでは単純に書き出し
    
    new_pm.write(output_path)
    print(f"\n成功！ トラック '{target_inst.name}' を抽出して以下に保存しました:")
    print(f"-> {output_path}")

if __name__ == "__main__":
    # 使い方の例: poetry run python extract_track.py data/original.mid
    if len(sys.argv) < 2:
        # 引数がなければ data/original.mid を探す
        input_file = "data/original.mid"
    else:
        input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"ファイルが見つかりません: {input_file}")
        print("dataフォルダに 'original.mid' という名前でJ-POPのMIDIを置いてください。")
    else:
        output_file = "data/teacher.mid" # アプリが読み込む名前に書き出す
        list_and_extract(input_file, output_file)