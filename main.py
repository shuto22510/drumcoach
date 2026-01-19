"""
AI Music Coach - Main entry point
Evaluates student performance against teacher reference using DTW.
"""

import sys
from pathlib import Path

from src.loader import DataLoader
from src.processor import PerformanceProcessor
from src.visualizer import PerformanceVisualizer


def main():
    """Main execution function."""

    # Configuration
    TEACHER_MIDI = "data/teacher.mid"
    STUDENT_AUDIO = "data/student.wav"
    OUTPUT_DIR = "output"

    # Audio parameters
    SR = 22050
    HOP_LENGTH = 512

    print("="*60)
    print("AI MUSIC COACH - Performance Evaluation System")
    print("="*60)
    print()

    try:
        # Initialize components
        print("Initializing system components...")
        loader = DataLoader(sr=SR, hop_length=HOP_LENGTH)
        processor = PerformanceProcessor()
        visualizer = PerformanceVisualizer()
        print("Components initialized successfully.\n")

        # Check if input files exist
        teacher_path = Path(TEACHER_MIDI)
        student_path = Path(STUDENT_AUDIO)

        if not teacher_path.exists():
            print(f"Error: Teacher MIDI file not found at '{TEACHER_MIDI}'")
            print(f"Please place your reference MIDI file at: {teacher_path.absolute()}")
            sys.exit(1)

        if not student_path.exists():
            print(f"Error: Student audio file not found at '{STUDENT_AUDIO}'")
            print(f"Please place your performance audio file at: {student_path.absolute()}")
            sys.exit(1)

        # Load data
        print(f"Loading teacher reference: {TEACHER_MIDI}")
        teacher_chroma = loader.load_midi_chromagram(TEACHER_MIDI)
        print(f"  Shape: {teacher_chroma.shape} (12 pitch classes x {teacher_chroma.shape[1]} frames)")

        print(f"\nLoading student performance: {STUDENT_AUDIO}")
        student_chroma = loader.load_audio_chromagram(STUDENT_AUDIO)
        print(f"  Shape: {student_chroma.shape} (12 pitch classes x {student_chroma.shape[1]} frames)")
        print()

        # Compute DTW and score
        print("Computing DTW alignment and performance score...")
        score_info = processor.compute_score(teacher_chroma, student_chroma)
        alignment_path, _ = processor.compute_dtw(teacher_chroma, student_chroma)
        print("DTW computation complete.\n")

        # Analyze timing
        print("Analyzing timing deviations...")
        timing_analysis = processor.analyze_timing(
            alignment_path,
            hop_length=HOP_LENGTH,
            sr=SR
        )
        print("Timing analysis complete.\n")

        # Analyze sections
        print("Analyzing sections...")
        section_analysis = processor.analyze_sections(
            teacher_chroma,
            student_chroma,
            alignment_path,
            num_sections=8,
            hop_length=HOP_LENGTH,
            sr=SR
        )
        print("Section analysis complete.\n")

        # Generate feedback
        print("Generating feedback...")
        feedback = processor.generate_feedback(
            score_info,
            section_analysis,
            timing_analysis
        )
        print("Feedback generated.\n")

        # Display results
        visualizer.print_score_summary(score_info)
        visualizer.print_section_analysis(section_analysis)
        visualizer.print_feedback(feedback)

        # Generate visualizations
        print("Generating visualizations...")

        # Create output directory if it doesn't exist
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)

        # Full analysis plot
        full_plot_path = output_path / "full_analysis.png"
        visualizer.plot_full_analysis(
            reference_chroma=teacher_chroma,
            performance_chroma=student_chroma,
            alignment_path=alignment_path,
            timing_analysis=timing_analysis,
            score_info=score_info,
            sr=SR,
            hop_length=HOP_LENGTH,
            save_path=str(full_plot_path)
        )
        print(f"  Saved full analysis plot to: {full_plot_path}")

        # Individual plots (optional)
        chromagram_path = output_path / "chromagrams.png"
        visualizer.plot_chromagrams(
            teacher_chroma,
            student_chroma,
            sr=SR,
            hop_length=HOP_LENGTH,
            save_path=str(chromagram_path)
        )
        print(f"  Saved chromagrams plot to: {chromagram_path}")

        dtw_path = output_path / "dtw_alignment.png"
        visualizer.plot_dtw_alignment(
            teacher_chroma,
            student_chroma,
            alignment_path,
            save_path=str(dtw_path)
        )
        print(f"  Saved DTW alignment plot to: {dtw_path}")

        timing_path = output_path / "timing_deviations.png"
        visualizer.plot_timing_deviations(
            timing_analysis,
            save_path=str(timing_path)
        )
        print(f"  Saved timing deviations plot to: {timing_path}")

        print("\nAll visualizations saved successfully!")
        print("\nEvaluation complete. Check the 'output/' directory for detailed visualizations.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure both teacher.mid and student.wav are in the 'data/' directory.")
        sys.exit(1)

    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
