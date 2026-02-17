"""Auto-parameter tuning for note extraction."""
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


def find_best_extraction(
    audio: np.ndarray,
    detector: PitchDetector,
    durations: Optional[List[float]] = None,
    probs: Optional[List[float]] = None,
    segments: Optional[List[float]] = None,
    use_harmonic: bool = True,
) -> TuneResult:
    """Search a small parameter grid and return best extraction result."""
    durations = durations or [0.015, 0.02, 0.03, 0.04]
    probs = probs or [0.10, 0.15, 0.20, 0.25]
    segments = segments or [5.0, 6.0, 8.0, 10.0]

    duration_sec = len(audio) / detector.sample_rate if detector.sample_rate else 1.0
    best: Optional[TuneResult] = None

    for md, mp, seg in product(durations, probs, segments):
        notes = detector.extract_notes_from_audio(
            audio,
            min_duration=md,
            min_voiced_prob=mp,
            merge_gap=0.03,
            use_harmonic=use_harmonic,
            segment_seconds=seg,
        )
        count = len(notes)
        density = count / max(duration_sec, 1e-6)
        penalty = max(0.0, density - 2.0) * 120.0
        score = count - penalty

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

    assert best is not None
    return best
