"""
Visualization module for AI Music Coach.
Handles plotting of chromagrams, DTW alignment, and performance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from typing import Optional, Dict


class PerformanceVisualizer:
    """Visualizes performance analysis results."""

    def __init__(self, figsize: tuple = (15, 10)):
        """
        Initialize PerformanceVisualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize

    def plot_chromagrams(
        self,
        reference_chroma: np.ndarray,
        performance_chroma: np.ndarray,
        sr: int = 22050,
        hop_length: int = 512,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot reference and performance chromagrams side by side.

        Args:
            reference_chroma: Reference chromagram (teacher MIDI)
            performance_chroma: Performance chromagram (student audio)
            sr: Sampling rate
            hop_length: Hop length used for feature extraction
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)

        # Plot reference chromagram
        img1 = librosa.display.specshow(
            reference_chroma,
            y_axis='chroma',
            x_axis='time',
            sr=sr,
            hop_length=hop_length,
            ax=axes[0]
        )
        axes[0].set_title('Reference (Teacher MIDI) Chromagram')
        axes[0].label_outer()
        fig.colorbar(img1, ax=axes[0])

        # Plot performance chromagram
        img2 = librosa.display.specshow(
            performance_chroma,
            y_axis='chroma',
            x_axis='time',
            sr=sr,
            hop_length=hop_length,
            ax=axes[1]
        )
        axes[1].set_title('Performance (Student Audio) Chromagram')
        fig.colorbar(img2, ax=axes[1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    def plot_dtw_alignment(
        self,
        reference_chroma: np.ndarray,
        performance_chroma: np.ndarray,
        alignment_path: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot DTW alignment path over the cost matrix.

        Args:
            reference_chroma: Reference chromagram
            performance_chroma: Performance chromagram
            alignment_path: DTW alignment path
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Compute distance matrix
        # librosa.sequence.dtw expects (K, N) format where K=12 features
        ref = np.ascontiguousarray(reference_chroma, dtype=np.float64)
        perf = np.ascontiguousarray(performance_chroma, dtype=np.float64)

        D, _ = librosa.sequence.dtw(X=ref, Y=perf, metric='cosine')

        # Plot cost matrix
        img = ax.imshow(D.T, origin='lower', cmap='gray_r', interpolation='nearest',
                       aspect='auto')
        ax.plot(alignment_path[0], alignment_path[1], 'r-', linewidth=2,
               label='Alignment Path')

        ax.set_xlabel('Reference (Teacher) Frames')
        ax.set_ylabel('Performance (Student) Frames')
        ax.set_title('DTW Alignment Path')
        ax.legend()

        plt.colorbar(img, ax=ax, label='Cost')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    def plot_timing_deviations(
        self,
        timing_analysis: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot timing deviations over time.

        Args:
            timing_analysis: Dictionary from processor.analyze_timing()
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        time_deviations = timing_analysis['time_deviations']
        ref_times = timing_analysis['reference_times']

        # Plot deviations
        ax.plot(ref_times, time_deviations, 'b-', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Perfect Timing')
        ax.fill_between(ref_times, time_deviations, 0, alpha=0.3)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Timing Deviation (seconds)')
        ax.set_title('Timing Deviations: Positive = Ahead, Negative = Behind')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    def plot_full_analysis(
        self,
        reference_chroma: np.ndarray,
        performance_chroma: np.ndarray,
        alignment_path: np.ndarray,
        timing_analysis: Dict[str, np.ndarray],
        score_info: Dict[str, float],
        sr: int = 22050,
        hop_length: int = 512,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a comprehensive analysis plot with all visualizations.

        Args:
            reference_chroma: Reference chromagram
            performance_chroma: Performance chromagram
            alignment_path: DTW alignment path
            timing_analysis: Timing analysis results
            score_info: Score information dictionary
            sr: Sampling rate
            hop_length: Hop length
            save_path: Optional path to save the figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Chromagrams
        ax1 = fig.add_subplot(gs[0, 0])
        img1 = librosa.display.specshow(
            reference_chroma,
            y_axis='chroma',
            x_axis='time',
            sr=sr,
            hop_length=hop_length,
            ax=ax1
        )
        ax1.set_title('Reference Chromagram')
        plt.colorbar(img1, ax=ax1, format='%+2.0f')

        ax2 = fig.add_subplot(gs[0, 1])
        img2 = librosa.display.specshow(
            performance_chroma,
            y_axis='chroma',
            x_axis='time',
            sr=sr,
            hop_length=hop_length,
            ax=ax2
        )
        ax2.set_title('Performance Chromagram')
        plt.colorbar(img2, ax=ax2, format='%+2.0f')

        # DTW alignment
        ax3 = fig.add_subplot(gs[1, :])
        # librosa.sequence.dtw expects (K, N) format where K=12 features
        ref = np.ascontiguousarray(reference_chroma, dtype=np.float64)
        perf = np.ascontiguousarray(performance_chroma, dtype=np.float64)
        D, _ = librosa.sequence.dtw(X=ref, Y=perf, metric='cosine')
        img3 = ax3.imshow(D.T, origin='lower', cmap='gray_r', interpolation='nearest',
                         aspect='auto')
        ax3.plot(alignment_path[0], alignment_path[1], 'r-', linewidth=2)
        ax3.set_xlabel('Reference Frames')
        ax3.set_ylabel('Performance Frames')
        ax3.set_title('DTW Alignment Path')
        plt.colorbar(img3, ax=ax3, label='Cost')

        # Timing deviations
        ax4 = fig.add_subplot(gs[2, :])
        time_deviations = timing_analysis['time_deviations']
        ref_times = timing_analysis['reference_times']
        ax4.plot(ref_times, time_deviations, 'b-', linewidth=1.5, alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax4.fill_between(ref_times, time_deviations, 0, alpha=0.3)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Timing Deviation (s)')
        ax4.set_title(
            f'Performance Score: {score_info["score"]:.2f}/100 | '
            f'Normalized Distance: {score_info["distance"]:.4f}'
        )
        ax4.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    def print_score_summary(self, score_info: Dict[str, float]) -> None:
        """
        Print a formatted score summary.

        Args:
            score_info: Score information dictionary
        """
        print("\n" + "="*50)
        print("PERFORMANCE EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Score:       {score_info['score']:.2f} / 100")
        print(f"Normalized Distance: {score_info['distance']:.4f}")
        print(f"Raw DTW Cost:        {score_info['raw_cost']:.2f}")
        print(f"Alignment Length:    {score_info['path_length']} frames")
        print("="*50)

        # Interpret the score (STRICT grading)
        score = score_info['score']
        if score >= 95:
            grade = "S - Perfect!"
        elif score >= 85:
            grade = "A - Excellent"
        elif score >= 75:
            grade = "B - Very Good"
        elif score >= 65:
            grade = "C - Good"
        elif score >= 50:
            grade = "D - Fair"
        elif score >= 35:
            grade = "E - Needs Practice"
        else:
            grade = "F - Keep Trying!"

        print(f"Grade: {grade}")
        print("="*50 + "\n")

    def print_section_analysis(self, section_analysis: Dict) -> None:
        """
        Print section-by-section analysis.

        Args:
            section_analysis: Section analysis from processor.analyze_sections()
        """
        sections = section_analysis.get('sections', [])
        if not sections:
            return

        print("\n" + "="*50)
        print("SECTION-BY-SECTION ANALYSIS")
        print("="*50)

        for section in sections:
            section_num = section['section_num']
            start_time = section['start_time']
            end_time = section['end_time']
            score = section['score']

            # Create score bar
            bar_length = 20
            filled = int(score / 100 * bar_length)
            bar = '#' * filled + '-' * (bar_length - filled)

            # Grade indicator
            if score >= 85:
                grade = 'A'
            elif score >= 70:
                grade = 'B'
            elif score >= 50:
                grade = 'C'
            else:
                grade = 'D'

            print(f"Section {section_num:2d} ({start_time:5.1f}s-{end_time:5.1f}s): "
                  f"[{bar}] {score:5.1f} ({grade})")

        print("="*50 + "\n")

    def print_feedback(self, feedback: Dict) -> None:
        """
        Print detailed feedback and practice tips.

        Args:
            feedback: Feedback dictionary from processor.generate_feedback()
        """
        print("\n" + "="*50)
        print("FEEDBACK & PRACTICE TIPS")
        print("="*50)

        # Overall comment
        print(f"\n{feedback.get('overall_comment', '')}")

        # Problem sections
        problem_sections = feedback.get('problem_sections', [])
        if problem_sections:
            print("\n--- Areas Needing Practice ---")
            for prob in problem_sections:
                print(f"  ! {prob['message']}")

        # Good sections
        good_sections = feedback.get('good_sections', [])
        if good_sections:
            print("\n--- Well Performed Sections ---")
            for good in good_sections:
                print(f"  * {good['message']}")

        # Timing feedback
        timing_feedback = feedback.get('timing_feedback', '')
        if timing_feedback:
            print(f"\n--- Timing ---")
            print(f"  {timing_feedback}")

        # Practice tips
        practice_tips = feedback.get('practice_tips', [])
        if practice_tips:
            print("\n--- Practice Tips ---")
            for tip in practice_tips:
                print(f"  > {tip}")

        print("\n" + "="*50 + "\n")
