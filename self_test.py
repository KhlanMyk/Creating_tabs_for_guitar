"""Self-test helpers for pitch detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pitch_detector import PitchDetector


@dataclass
class TestResult:
    expected_note: str
    detected_note: Optional[str]
    detected_count: int
    success: bool


def run_sine_test(
    detector: PitchDetector,
    freq: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 44100,
) -> TestResult:
    """Generate a sine wave and verify detected note."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)

    notes = detector.extract_notes_from_audio(audio)
    detected_note = notes[0]["note"] if notes else None
    expected_note = "A4"

    success = detected_note == expected_note
    return TestResult(
        expected_note=expected_note,
        detected_note=detected_note,
        detected_count=len(notes),
        success=success,
    )
