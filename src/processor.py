"""
Processing module for AI Music Coach.
Handles DTW computation and performance scoring.
"""

import numpy as np
import librosa
from typing import Tuple, Dict, List


class PerformanceProcessor:
    """Processes performance data using DTW and computes scores."""

    def __init__(self):
        """Initialize PerformanceProcessor."""
        pass

    def compute_dtw(
        self,
        reference_chroma: np.ndarray,
        performance_chroma: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute DTW between reference and performance chromagrams.

        Args:
            reference_chroma: Reference chromagram (teacher MIDI)
            performance_chroma: Performance chromagram (student audio)

        Returns:
            Tuple of (alignment_path, total_cost)
            - alignment_path: DTW alignment path as array of shape (2, n_steps)
            - total_cost: Total DTW cost (distance)
        """
        # Ensure both chromagrams are (12, n_frames)
        # Some sources may produce transposed output
        if reference_chroma.shape[0] != 12:
            reference_chroma = reference_chroma.T
        if performance_chroma.shape[0] != 12:
            performance_chroma = performance_chroma.T

        # librosa.sequence.dtw expects input shape (..., K, N) where K is feature dimension
        # So we keep (12, n_frames) format - do NOT transpose
        ref = reference_chroma
        perf = performance_chroma

        # Ensure same dtype and C-contiguous memory layout
        ref = np.ascontiguousarray(ref, dtype=np.float64)
        perf = np.ascontiguousarray(perf, dtype=np.float64)

        # Verify shapes are correct
        if ref.shape[0] != perf.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: ref.shape={ref.shape}, perf.shape={perf.shape}. "
                f"Both should have 12 features (chroma bins) as first dimension."
            )

        # Compute DTW using cosine distance
        # librosa.sequence.dtw uses (1 - cosine_similarity) as distance
        D, wp = librosa.sequence.dtw(
            X=ref,
            Y=perf,
            metric='cosine'
        )

        # Get the total cost (bottom-right corner of accumulated cost matrix)
        total_cost = D[-1, -1]

        # wp is a list of tuples, convert to array
        wp_array = np.array(wp).T

        return wp_array, total_cost

    def compute_score(
        self,
        reference_chroma: np.ndarray,
        performance_chroma: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute performance score based on DTW distance.

        Args:
            reference_chroma: Reference chromagram (teacher MIDI)
            performance_chroma: Performance chromagram (student audio)

        Returns:
            Dictionary containing:
            - score: Performance score (0-100)
            - distance: Normalized DTW distance
            - raw_cost: Raw DTW cost
        """
        # Compute DTW
        wp, raw_cost = self.compute_dtw(reference_chroma, performance_chroma)

        # Normalize the cost by the path length
        # This accounts for different performance lengths
        path_length = len(wp[0])
        normalized_distance = raw_cost / path_length

        # Convert distance to score (0-100) - STRICT SCORING
        # Cosine distance ranges from 0 (identical) to 2 (opposite)
        # Using stricter scoring curve:
        # distance of 0.0  -> score 100 (perfect)
        # distance of 0.1  -> score ~82
        # distance of 0.2  -> score ~67
        # distance of 0.3  -> score ~55
        # distance of 0.5  -> score ~37
        # distance of 1.0  -> score ~14
        # distance >= 1.5  -> score ~5 or below
        #
        # Formula: score = 100 * exp(-2 * distance)
        # The multiplier (2) makes the curve steeper for stricter scoring
        score = 100 * np.exp(-2.0 * normalized_distance)

        # Apply floor to avoid very low scores for reasonable attempts
        score = max(score, 0.0)

        return {
            'score': float(score),
            'distance': float(normalized_distance),
            'raw_cost': float(raw_cost),
            'path_length': int(path_length)
        }

    def analyze_timing(
        self,
        alignment_path: np.ndarray,
        hop_length: int = 512,
        sr: int = 22050
    ) -> Dict[str, np.ndarray]:
        """
        Analyze timing deviations from the alignment path.

        Args:
            alignment_path: DTW alignment path of shape (2, n_steps)
            hop_length: Hop length used for feature extraction
            sr: Sampling rate

        Returns:
            Dictionary containing timing analysis:
            - time_deviations: Time deviations in seconds at each point
            - reference_times: Reference time points
            - performance_times: Performance time points
        """
        ref_frames = alignment_path[0]
        perf_frames = alignment_path[1]

        # Convert frames to time
        frame_to_time = lambda frames: librosa.frames_to_time(
            frames, sr=sr, hop_length=hop_length
        )

        ref_times = frame_to_time(ref_frames)
        perf_times = frame_to_time(perf_frames)

        # Calculate deviations
        # Positive means student is ahead, negative means behind
        time_deviations = perf_times - ref_times

        return {
            'time_deviations': time_deviations,
            'reference_times': ref_times,
            'performance_times': perf_times
        }

    def analyze_sections(
        self,
        reference_chroma: np.ndarray,
        performance_chroma: np.ndarray,
        alignment_path: np.ndarray,
        num_sections: int = 8,
        hop_length: int = 512,
        sr: int = 22050
    ) -> Dict:
        """
        Analyze performance by sections to identify problem areas.

        Args:
            reference_chroma: Reference chromagram (12, n_frames)
            performance_chroma: Performance chromagram (12, n_frames)
            alignment_path: DTW alignment path
            num_sections: Number of sections to divide the song into
            hop_length: Hop length for time conversion
            sr: Sampling rate

        Returns:
            Dictionary with section-by-section analysis
        """
        ref_frames = alignment_path[0]
        perf_frames = alignment_path[1]

        # Divide into sections based on reference frames
        ref_length = reference_chroma.shape[1]
        section_size = ref_length // num_sections

        sections = []
        for i in range(num_sections):
            start_frame = i * section_size
            end_frame = (i + 1) * section_size if i < num_sections - 1 else ref_length

            # Find alignment indices for this section
            mask = (ref_frames >= start_frame) & (ref_frames < end_frame)
            section_ref_frames = ref_frames[mask]
            section_perf_frames = perf_frames[mask]

            if len(section_ref_frames) == 0:
                continue

            # Calculate section score using cosine similarity
            section_ref = reference_chroma[:, start_frame:end_frame]
            perf_start = int(section_perf_frames.min()) if len(section_perf_frames) > 0 else 0
            perf_end = int(section_perf_frames.max()) + 1 if len(section_perf_frames) > 0 else 1
            perf_end = min(perf_end, performance_chroma.shape[1])
            section_perf = performance_chroma[:, perf_start:perf_end]

            # Compute local distance
            if section_ref.shape[1] > 0 and section_perf.shape[1] > 0:
                # Use mean cosine similarity for the section
                ref_norm = section_ref / (np.linalg.norm(section_ref, axis=0, keepdims=True) + 1e-8)
                perf_norm = section_perf / (np.linalg.norm(section_perf, axis=0, keepdims=True) + 1e-8)

                # Average similarity
                min_len = min(ref_norm.shape[1], perf_norm.shape[1])
                if min_len > 0:
                    similarity = np.mean(np.sum(ref_norm[:, :min_len] * perf_norm[:, :min_len], axis=0))
                    distance = 1 - similarity
                else:
                    distance = 1.0
            else:
                distance = 1.0

            # Convert to score
            section_score = 100 * np.exp(-2.0 * distance)

            # Convert frames to time
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)

            sections.append({
                'section_num': i + 1,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'score': float(section_score),
                'distance': float(distance)
            })

        return {
            'sections': sections,
            'num_sections': len(sections)
        }

    def generate_feedback(
        self,
        score_info: Dict,
        section_analysis: Dict,
        timing_analysis: Dict
    ) -> Dict:
        """
        Generate detailed feedback based on analysis.

        Args:
            score_info: Overall score information
            section_analysis: Section-by-section analysis
            timing_analysis: Timing deviation analysis

        Returns:
            Dictionary containing feedback messages
        """
        feedback = {
            'overall_comment': '',
            'problem_sections': [],
            'good_sections': [],
            'timing_feedback': '',
            'practice_tips': []
        }

        overall_score = score_info['score']

        # Overall comment
        if overall_score >= 95:
            feedback['overall_comment'] = "Amazing performance! You've mastered this piece."
        elif overall_score >= 85:
            feedback['overall_comment'] = "Excellent work! Just a few minor areas to polish."
        elif overall_score >= 75:
            feedback['overall_comment'] = "Great job! Some sections need a bit more practice."
        elif overall_score >= 65:
            feedback['overall_comment'] = "Good effort! Focus on the problem areas identified below."
        elif overall_score >= 50:
            feedback['overall_comment'] = "Keep practicing! Pay attention to the sections that need work."
        elif overall_score >= 35:
            feedback['overall_comment'] = "You're making progress. Focus on accuracy before speed."
        else:
            feedback['overall_comment'] = "Don't give up! Start by practicing smaller sections slowly."

        # Analyze sections
        sections = section_analysis.get('sections', [])
        for section in sections:
            section_num = section['section_num']
            section_score = section['score']
            start_time = section['start_time']
            end_time = section['end_time']

            time_str = f"{start_time:.1f}s - {end_time:.1f}s"

            if section_score < 50:
                feedback['problem_sections'].append({
                    'section': section_num,
                    'time': time_str,
                    'score': section_score,
                    'message': f"Section {section_num} ({time_str}): Score {section_score:.1f} - Needs significant practice"
                })
            elif section_score < 70:
                feedback['problem_sections'].append({
                    'section': section_num,
                    'time': time_str,
                    'score': section_score,
                    'message': f"Section {section_num} ({time_str}): Score {section_score:.1f} - Could use more work"
                })
            elif section_score >= 85:
                feedback['good_sections'].append({
                    'section': section_num,
                    'time': time_str,
                    'score': section_score,
                    'message': f"Section {section_num} ({time_str}): Score {section_score:.1f} - Well done!"
                })

        # Timing feedback
        time_deviations = timing_analysis.get('time_deviations', np.array([]))
        if len(time_deviations) > 0:
            avg_deviation = np.mean(np.abs(time_deviations))
            max_deviation = np.max(np.abs(time_deviations))

            if avg_deviation < 0.1:
                feedback['timing_feedback'] = "Excellent timing! You're staying well synchronized."
            elif avg_deviation < 0.3:
                feedback['timing_feedback'] = "Good timing overall. Minor fluctuations detected."
            elif avg_deviation < 0.5:
                feedback['timing_feedback'] = "Timing needs work. Try practicing with a metronome."
            else:
                feedback['timing_feedback'] = "Significant timing issues. Slow down and focus on rhythm."

            # Check if consistently ahead or behind
            mean_deviation = np.mean(time_deviations)
            if mean_deviation > 0.2:
                feedback['timing_feedback'] += " You tend to rush ahead of the beat."
            elif mean_deviation < -0.2:
                feedback['timing_feedback'] += " You tend to drag behind the beat."

        # Practice tips
        if len(feedback['problem_sections']) > 0:
            worst_section = min(feedback['problem_sections'], key=lambda x: x['score'])
            feedback['practice_tips'].append(
                f"Priority: Practice Section {worst_section['section']} ({worst_section['time']}) - "
                f"This is your weakest area."
            )

        if overall_score < 70:
            feedback['practice_tips'].append(
                "Tip: Try practicing at 50-75% speed first, then gradually increase."
            )

        if len(feedback['problem_sections']) > len(sections) / 2:
            feedback['practice_tips'].append(
                "Tip: Consider breaking the piece into smaller parts and mastering each one."
            )

        if feedback.get('timing_feedback', '').find('metronome') != -1:
            feedback['practice_tips'].append(
                "Tip: Use a metronome app to develop consistent timing."
            )

        return feedback
