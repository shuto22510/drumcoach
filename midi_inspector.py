"""
MIDI Inspector - Analyze MIDI file structure and extract specific tracks.
"""

import pretty_midi
import sys
from pathlib import Path


def inspect_midi(midi_path: str) -> None:
    """
    Inspect MIDI file and display track information.

    Args:
        midi_path: Path to MIDI file
    """
    path = Path(midi_path)
    if not path.exists():
        print(f"Error: File not found: {midi_path}")
        return

    print(f"\n{'='*60}")
    print(f"MIDI File: {path.name}")
    print(f"{'='*60}\n")

    try:
        midi = pretty_midi.PrettyMIDI(str(path))

        # Basic info
        print(f"Duration: {midi.get_end_time():.2f} seconds")
        print(f"Estimated tempo: {midi.estimate_tempo():.1f} BPM")
        print(f"Time signature changes: {len(midi.time_signature_changes)}")
        print(f"Key signature changes: {len(midi.key_signature_changes)}")
        print()

        # Instrument/Track info
        print(f"Number of instruments/tracks: {len(midi.instruments)}")
        print("-" * 60)
        print(f"{'#':<4} {'Name':<30} {'Program':<8} {'Notes':<8} {'Drum?'}")
        print("-" * 60)

        for i, inst in enumerate(midi.instruments):
            name = inst.name if inst.name else f"Track {i}"
            program = inst.program
            note_count = len(inst.notes)
            is_drum = "Yes" if inst.is_drum else "No"

            # Get program name
            if inst.is_drum:
                prog_name = "Drums"
            else:
                prog_name = pretty_midi.program_to_instrument_name(program)

            print(f"{i:<4} {name:<30} {program:<8} {note_count:<8} {is_drum}")
            print(f"     -> {prog_name}")

        print("-" * 60)
        print()

        # Detailed note range for each track
        print("Note ranges per track:")
        print("-" * 60)
        for i, inst in enumerate(midi.instruments):
            if len(inst.notes) > 0:
                pitches = [n.pitch for n in inst.notes]
                min_pitch = min(pitches)
                max_pitch = max(pitches)
                min_note = pretty_midi.note_number_to_name(min_pitch)
                max_note = pretty_midi.note_number_to_name(max_pitch)
                avg_pitch = sum(pitches) / len(pitches)

                name = inst.name if inst.name else f"Track {i}"
                print(f"Track {i} ({name[:20]}): {min_note} - {max_note} (avg: {avg_pitch:.1f})")

        print()
        print("Melody track candidates (based on typical vocal range C3-C6):")
        print("-" * 60)
        for i, inst in enumerate(midi.instruments):
            if inst.is_drum:
                continue
            if len(inst.notes) == 0:
                continue

            pitches = [n.pitch for n in inst.notes]
            avg_pitch = sum(pitches) / len(pitches)

            # Vocal range is typically C3 (48) to C6 (84)
            # Check if average pitch is in vocal range
            if 48 <= avg_pitch <= 84:
                name = inst.name if inst.name else f"Track {i}"
                print(f"  -> Track {i}: {name} (avg pitch: {avg_pitch:.1f})")

    except Exception as e:
        print(f"Error reading MIDI file: {e}")


def extract_track(midi_path: str, track_index: int, output_path: str) -> None:
    """
    Extract a specific track from MIDI file.

    Args:
        midi_path: Path to input MIDI file
        track_index: Index of track to extract
        output_path: Path for output MIDI file
    """
    path = Path(midi_path)
    if not path.exists():
        print(f"Error: File not found: {midi_path}")
        return

    try:
        midi = pretty_midi.PrettyMIDI(str(path))

        if track_index < 0 or track_index >= len(midi.instruments):
            print(f"Error: Track index {track_index} out of range (0-{len(midi.instruments)-1})")
            return

        # Create new MIDI with only the selected track
        new_midi = pretty_midi.PrettyMIDI()

        # Copy the selected instrument
        source_inst = midi.instruments[track_index]
        new_inst = pretty_midi.Instrument(
            program=source_inst.program,
            is_drum=source_inst.is_drum,
            name=source_inst.name
        )

        # Copy all notes
        for note in source_inst.notes:
            new_inst.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end
            ))

        new_midi.instruments.append(new_inst)

        # Save
        new_midi.write(output_path)
        print(f"Extracted track {track_index} to: {output_path}")
        print(f"Notes: {len(new_inst.notes)}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Inspect MIDI:  python midi_inspector.py <midi_file>")
        print("  Extract track: python midi_inspector.py <midi_file> --extract <track_index> <output_file>")
        print()
        print("Example:")
        print("  python midi_inspector.py data/song.mid")
        print("  python midi_inspector.py data/song.mid --extract 0 data/melody.mid")
        return

    midi_path = sys.argv[1]

    if len(sys.argv) >= 5 and sys.argv[2] == "--extract":
        track_index = int(sys.argv[3])
        output_path = sys.argv[4]
        extract_track(midi_path, track_index, output_path)
    else:
        inspect_midi(midi_path)


if __name__ == "__main__":
    main()
