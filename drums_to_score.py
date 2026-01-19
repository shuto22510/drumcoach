"""
Convert drum MIDI to text-based score visualization.
"""

import sys
from pathlib import Path

import pretty_midi


def midi_to_text_score(midi_path: str, bars_per_line: int = 4) -> str:
    """
    Convert drum MIDI to text-based drum score.

    Args:
        midi_path: Path to drum MIDI file
        bars_per_line: Number of bars to display per line

    Returns:
        Text representation of drum score
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    if len(midi.instruments) == 0:
        return "No instruments found in MIDI file"

    drum_track = midi.instruments[0]

    # Get tempo
    tempo_changes = midi.get_tempo_changes()
    if len(tempo_changes[1]) > 0:
        tempo = tempo_changes[1][0]
    else:
        tempo = 120.0

    # Calculate timing
    seconds_per_beat = 60.0 / tempo
    beats_per_bar = 4
    subdivisions = 4  # 16th notes

    # GM Drum map (simplified)
    drum_names = {
        35: 'K',   # Acoustic Bass Drum
        36: 'K',   # Bass Drum 1
        38: 'S',   # Acoustic Snare
        40: 'S',   # Electric Snare
        42: 'H',   # Closed Hi-Hat
        44: 'H',   # Pedal Hi-Hat
        46: 'O',   # Open Hi-Hat
        41: 'T',   # Low Floor Tom
        43: 'T',   # High Floor Tom
        45: 'T',   # Low Tom
        47: 'T',   # Low-Mid Tom
        48: 'T',   # Hi-Mid Tom
        49: 'C',   # Crash Cymbal 1
        51: 'R',   # Ride Cymbal 1
        52: 'C',   # Chinese Cymbal
        53: 'R',   # Ride Bell
        55: 'C',   # Splash Cymbal
        57: 'C',   # Crash Cymbal 2
    }

    # Get song duration
    duration = midi.get_end_time()
    total_beats = int(duration / seconds_per_beat) + 1
    total_bars = total_beats // beats_per_bar + 1

    # Create grid
    grid_size = total_bars * beats_per_bar * subdivisions
    grid = {
        'K': ['.'] * grid_size,  # Kick
        'S': ['.'] * grid_size,  # Snare
        'H': ['.'] * grid_size,  # Hi-hat
        'T': ['.'] * grid_size,  # Tom
        'C': ['.'] * grid_size,  # Crash
        'R': ['.'] * grid_size,  # Ride
    }

    # Place notes on grid
    for note in drum_track.notes:
        beat_pos = note.start / seconds_per_beat
        grid_pos = int(beat_pos * subdivisions)

        if grid_pos >= grid_size:
            continue

        drum_type = drum_names.get(note.pitch, 'X')
        if drum_type in grid:
            # Use velocity to show dynamics
            if note.velocity > 100:
                grid[drum_type][grid_pos] = drum_type.upper()
            elif note.velocity > 60:
                grid[drum_type][grid_pos] = drum_type.lower()
            else:
                grid[drum_type][grid_pos] = '.'

    # Build output
    output = []
    output.append(f"Drum Score - Tempo: {tempo:.0f} BPM")
    output.append("=" * 60)
    output.append("")
    output.append("Legend: K=Kick, S=Snare, H=Hi-hat, T=Tom, C=Crash, R=Ride")
    output.append("        CAPS=loud, lower=soft, .=rest")
    output.append("")

    # Print bars
    cells_per_bar = beats_per_bar * subdivisions
    cells_per_line = bars_per_line * cells_per_bar

    for start_bar in range(0, total_bars, bars_per_line):
        end_bar = min(start_bar + bars_per_line, total_bars)
        start_cell = start_bar * cells_per_bar
        end_cell = min(end_bar * cells_per_bar, grid_size)

        # Bar numbers
        bar_nums = "Bar: "
        for b in range(start_bar, end_bar):
            bar_nums += f"{b+1:<16}"
        output.append(bar_nums)

        # Beat markers
        beat_line = "     "
        for b in range(start_bar, end_bar):
            beat_line += "|1  2  3  4  "
        output.append(beat_line)

        # Drum lines
        for drum_type in ['C', 'R', 'H', 'S', 'T', 'K']:
            line = f"{drum_type}:   "
            for i in range(start_cell, end_cell):
                if i % cells_per_bar == 0 and i > start_cell:
                    line += "|"
                line += grid[drum_type][i] if i < len(grid[drum_type]) else "."
            output.append(line)

        output.append("")

    return "\n".join(output)


def main():
    if len(sys.argv) < 2:
        print("Drum MIDI to Text Score Converter")
        print("=" * 60)
        print("\nUsage:")
        print("  python drums_to_score.py <drum_midi_file>")
        print()
        print("Example:")
        print("  python drums_to_score.py output/demucs/htdemucs/soramotoberuhazu/drums.mid")
        return

    midi_path = sys.argv[1]

    if not Path(midi_path).exists():
        print(f"Error: File not found: {midi_path}")
        return

    print("Converting MIDI to text score...")
    print()

    score = midi_to_text_score(midi_path)
    print(score)

    # Save to file
    output_path = Path(midi_path).with_suffix('.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(score)
    print(f"\nScore saved to: {output_path}")


if __name__ == "__main__":
    main()
