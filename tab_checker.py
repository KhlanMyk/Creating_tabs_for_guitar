"""Tab quality checker against original audio."""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from tab_synth import parse_tabs_text, synthesize_audio_array_from_tabs_text


@dataclass
class TabCheckResult:
    overall_score: float
    chroma_score: float
    onset_score: float
    estimated_step_seconds: float
    estimated_note_seconds: float
    analyzed_seconds: float


def _cosine_mean(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return 0.0
    n = min(A.shape[1], B.shape[1])
    if n <= 0:
        return 0.0
    A = A[:, :n]
    B = B[:, :n]
    num = np.sum(A * B, axis=0)
    den = (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0)) + 1e-8
    return float(np.mean(num / den))


def _calc_metrics(original: np.ndarray, synth: np.ndarray, sr: int) -> tuple[float, float, float]:
    n = min(len(original), len(synth))
    if n <= 2048:
        return 0.0, 0.0, 0.0

    original = original[:n]
    synth = synth[:n]

    original = original / (np.max(np.abs(original)) + 1e-8)
    synth = synth / (np.max(np.abs(synth)) + 1e-8)

    c1 = librosa.feature.chroma_cqt(y=original, sr=sr)
    c2 = librosa.feature.chroma_cqt(y=synth, sr=sr)
    chroma_sim = _cosine_mean(c1, c2)

    o1 = librosa.onset.onset_strength(y=original, sr=sr)
    o2 = librosa.onset.onset_strength(y=synth, sr=sr)
    m = min(len(o1), len(o2))
    if m > 0 and np.std(o1[:m]) > 1e-6 and np.std(o2[:m]) > 1e-6:
        onset_sim = float(np.corrcoef(o1[:m], o2[:m])[0, 1])
        if np.isnan(onset_sim):
            onset_sim = 0.0
    else:
        onset_sim = 0.0

    onset_sim = max(-1.0, min(1.0, onset_sim))
    overall = float(0.75 * chroma_sim + 0.25 * onset_sim)
    return overall, float(chroma_sim), float(onset_sim)


def _estimate_timing_from_original(
    tabs_text: str,
    original_audio_path: str,
    default_step_seconds: float,
    default_note_seconds: float,
    sr: int,
) -> tuple[float, float, float]:
    grid, cols = parse_tabs_text(tabs_text)
    del grid

    original, _ = librosa.load(original_audio_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=original, sr=sr)

    if cols <= 0 or duration <= 0.5:
        return default_step_seconds, default_note_seconds, min(24.0, max(0.0, duration))

    est_step = (duration - default_note_seconds) / max(cols, 1)
    est_step = float(np.clip(est_step, 0.08, 0.18))

    onset_frames = librosa.onset.onset_detect(y=original, sr=sr, hop_length=512, units="frames")
    if len(onset_frames) > 4:
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        intervals = np.diff(onset_times)
        intervals = intervals[intervals > 1e-3]
        if len(intervals) > 0:
            median_int = float(np.median(intervals))
            est_step = float(np.clip(0.5 * est_step + 0.5 * median_int, 0.08, 0.18))

    est_note = float(np.clip(est_step * 1.65, 0.12, 0.36))
    analyzed_seconds = min(24.0, duration)
    return est_step, est_note, analyzed_seconds


def check_tabs_against_original(
    tabs_text: str,
    original_audio_path: str,
    sample_rate: int = 22050,
    step_seconds: float = 0.125,
    note_seconds: float = 0.18,
) -> TabCheckResult:
    """Synthesize tabs and compare them with the original song using objective metrics."""
    est_step, est_note, analyzed_seconds = _estimate_timing_from_original(
        tabs_text=tabs_text,
        original_audio_path=original_audio_path,
        default_step_seconds=step_seconds,
        default_note_seconds=note_seconds,
        sr=sample_rate,
    )

    synth, _, _ = synthesize_audio_array_from_tabs_text(
        tabs_text,
        sample_rate=sample_rate,
        step_seconds=est_step,
        note_seconds=est_note,
        max_duration_seconds=analyzed_seconds,
    )

    original, _ = librosa.load(
        original_audio_path,
        sr=sample_rate,
        mono=True,
        duration=analyzed_seconds,
    )

    overall, chroma, onset = _calc_metrics(original, synth, sample_rate)

    return TabCheckResult(
        overall_score=overall,
        chroma_score=chroma,
        onset_score=onset,
        estimated_step_seconds=est_step,
        estimated_note_seconds=est_note,
        analyzed_seconds=analyzed_seconds,
    )
