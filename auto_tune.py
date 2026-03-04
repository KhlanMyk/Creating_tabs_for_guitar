"""Auto-parameter tuning for note extraction.

Searches a parameter grid and picks the combination that yields the best
note extraction quality.  Quality is measured by a composite score:
  • Note count (enough notes to represent real melody)
  • Density penalty (discourage picking up too many noise notes)
  • Pitch stability bonus (notes should not jump wildly)
  • Duration regularity bonus (prefer uniform note lengths)
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Optional

import numpy as np

from pitch_detector import PitchDetector


@dataclass
class TuneResult:
    notes: List[dict]
    min_duration: float
    min_voiced_prob: float
    segment_seconds: float
    use_harmonic: bool
    score: float


def _score_notes(notes: List[dict], audio_duration: float) -> float:
    """Composite quality score (higher = better)."""
    count = len(notes)
    if count == 0:
        return -100.0

    density = count / max(audio_duration, 1e-6)

    # Penalty for unrealistic note density (> 8 notes/s is very fast)
    density_penalty = max(0.0, density - 4.0) * 50.0

    # Pitch stability: penalise large pitch jumps between consecutive notes
    midis = [n.get("midi", 0) for n in notes]
    jumps = [abs(midis[i+1] - midis[i]) for i in range(len(midis)-1)] if len(midis) > 1 else [0]
    avg_jump = sum(jumps) / max(len(jumps), 1)
    big_jumps = sum(1 for j in jumps if j > 12)  # octave+ jumps
    stability_penalty = avg_jump * 2.0 + big_jumps * 8.0

    # Duration regularity bonus: prefer notes with reasonable durations
    durs = [n.get("duration", 0) for n in notes]
    too_short = sum(1 for d in durs if d < 0.04)
    too_long = sum(1 for d in durs if d > 4.0)
    dur_penalty = too_short * 5.0 + too_long * 3.0

    # Coverage: notes should span most of the audio
    if notes:
        first_onset = notes[0].get("start_time", 0)
        last_end = notes[-1].get("end_time", notes[-1].get("start_time", 0) + notes[-1].get("duration", 0))
        coverage = (last_end - first_onset) / max(audio_duration, 1e-6)
    else:
        coverage = 0
    coverage_bonus = coverage * 30.0

    return count + coverage_bonus - density_penalty - stability_penalty - dur_penalty


def find_best_extraction(
    audio: np.ndarray,
    detector: PitchDetector,
    durations: Optional[List[float]] = None,
    probs: Optional[List[float]] = None,
    segments: Optional[List[float]] = None,
    use_harmonic: bool = True,
    preview_seconds: float = 30.0,
    progress_callback: object = None,
) -> TuneResult:
    """Search a parameter grid and return best extraction result."""
    durations = durations or [0.02, 0.035, 0.05, 0.08]
    probs = probs or [0.12, 0.20, 0.30]
    segments = segments or [8.0, 12.0]
    onset_deltas = [0.05, 0.07, 0.10]  # onset sensitivity

    duration_sec = len(audio) / detector.sample_rate if detector.sample_rate else 1.0
    preview_len = int(min(duration_sec, preview_seconds) * detector.sample_rate)
    preview_audio = audio[:preview_len] if preview_len > 0 else audio
    preview_dur = len(preview_audio) / detector.sample_rate
    best: Optional[TuneResult] = None
    best_delta: float = 0.07

    combos = list(product(durations, probs, segments, onset_deltas))
    total = len(combos) + 1  # +1 for final full run

    for i, (md, mp, seg, od) in enumerate(combos):
        if progress_callback:
            progress_callback(i / total, f"Testing combo {i+1}/{len(combos)}")

        # Temporarily set onset sensitivity
        old_delta = detector.onset_delta
        detector.onset_delta = od
        try:
            notes = detector.extract_notes_from_audio(
                preview_audio,
                min_duration=md,
                min_voiced_prob=mp,
                merge_gap=0.03,
                use_harmonic=use_harmonic,
                segment_seconds=seg,
                use_onset_alignment=True,
            )
        finally:
            detector.onset_delta = old_delta

        score = _score_notes(notes, preview_dur)

        candidate = TuneResult(
            notes=notes,
            min_duration=md,
            min_voiced_prob=mp,
            segment_seconds=seg,
            use_harmonic=use_harmonic,
            score=score,
        )

        if best is None or candidate.score > best.score:
            best = candidate
            best_delta = od

    assert best is not None

    # Run once on full audio with best params
    if progress_callback:
        progress_callback((total - 1) / total, "Final pass with best params…")

    old_delta = detector.onset_delta
    detector.onset_delta = best_delta
    try:
        final_notes = detector.extract_notes_from_audio(
            audio,
            min_duration=best.min_duration,
            min_voiced_prob=best.min_voiced_prob,
            merge_gap=0.03,
            use_harmonic=best.use_harmonic,
            segment_seconds=best.segment_seconds,
            use_onset_alignment=True,
        )
    finally:
        detector.onset_delta = old_delta

    return TuneResult(
        notes=final_notes,
        min_duration=best.min_duration,
        min_voiced_prob=best.min_voiced_prob,
        segment_seconds=best.segment_seconds,
        use_harmonic=best.use_harmonic,
        score=best.score,
    )
