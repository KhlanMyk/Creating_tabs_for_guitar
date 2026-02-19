"""Compare synthesized tabs audio with original and find better synth parameters."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List

import librosa
import numpy as np
import soundfile as sf

from tab_synth import synthesize_audio_array_from_tabs_text


@dataclass
class MatchResult:
    score: float
    params: Dict[str, float | int]
    output_path: str


def _cosine_mean(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return 0.0
    n = min(A.shape[1], B.shape[1])
    if n == 0:
        return 0.0
    A = A[:, :n]
    B = B[:, :n]
    num = np.sum(A * B, axis=0)
    den = (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0)) + 1e-8
    return float(np.mean(num / den))


def compare_audio_similarity(original: np.ndarray, synth: np.ndarray, sr: int) -> float:
    """Compute a compact similarity score in [0..1-ish] using chroma and onset."""
    n = min(len(original), len(synth))
    if n <= 2048:
        return 0.0

    original = original[:n]
    synth = synth[:n]

    # normalize
    original = original / (np.max(np.abs(original)) + 1e-8)
    synth = synth / (np.max(np.abs(synth)) + 1e-8)

    c1 = librosa.feature.chroma_cqt(y=original, sr=sr)
    c2 = librosa.feature.chroma_cqt(y=synth, sr=sr)
    chroma_sim = _cosine_mean(c1, c2)

    o1 = librosa.onset.onset_strength(y=original, sr=sr)
    o2 = librosa.onset.onset_strength(y=synth, sr=sr)
    m = min(len(o1), len(o2))
    if m > 0:
        o1 = o1[:m]
        o2 = o2[:m]
        onset_sim = float(np.corrcoef(o1, o2)[0, 1]) if np.std(o1) > 1e-6 and np.std(o2) > 1e-6 else 0.0
    else:
        onset_sim = 0.0

    onset_sim = 0.0 if np.isnan(onset_sim) else onset_sim
    score = 0.75 * chroma_sim + 0.25 * max(-1.0, min(1.0, onset_sim))
    return float(score)


def optimize_synth_against_original(
    tabs_text: str,
    original_audio_path: str,
    output_path: str = "tabs_synth_matched.wav",
    sr: int = 44100,
) -> MatchResult:
    """Find synth parameters that better match the original track.

    Uses a fast preview search at lower sample rate, then renders the final
    best result at target sample rate.
    """
    search_sr = 22050
    preview_seconds = 24.0
    original_search, _ = librosa.load(
        original_audio_path,
        sr=search_sr,
        mono=True,
        duration=preview_seconds,
    )

    def score_params(
        step: float,
        note: float,
        decay: float,
        gain: float,
        trans: int,
    ) -> float:
        synth, _, _ = synthesize_audio_array_from_tabs_text(
            tabs_text,
            sample_rate=search_sr,
            step_seconds=step,
            note_seconds=note,
            decay=decay,
            gain=gain,
            transpose_semitones=trans,
            max_duration_seconds=preview_seconds,
        )
        return compare_audio_similarity(original_search, synth, search_sr)

    coarse_grid: Dict[str, List[float | int]] = {
        "step_seconds": [0.11, 0.14, 0.17],
        "note_seconds": [0.14, 0.19, 0.24],
        "decay": [0.993, 0.996, 0.998],
        "gain": [0.28, 0.38, 0.48],
        "transpose_semitones": [-2, -1, 0, 1, 2],
    }

    best_score = -1e9
    best_params: Dict[str, float | int] = {}

    for step, note, decay, gain, trans in product(
        coarse_grid["step_seconds"],
        coarse_grid["note_seconds"],
        coarse_grid["decay"],
        coarse_grid["gain"],
        coarse_grid["transpose_semitones"],
    ):
        score = score_params(float(step), float(note), float(decay), float(gain), int(trans))
        if score > best_score:
            best_score = score
            best_params = {
                "step_seconds": float(step),
                "note_seconds": float(note),
                "decay": float(decay),
                "gain": float(gain),
                "transpose_semitones": int(trans),
            }

    step0 = float(best_params["step_seconds"])
    note0 = float(best_params["note_seconds"])
    decay0 = float(best_params["decay"])
    gain0 = float(best_params["gain"])
    trans0 = int(best_params["transpose_semitones"])

    refine_steps = [max(0.08, step0 - 0.015), step0, min(0.22, step0 + 0.015)]
    refine_notes = [max(0.10, note0 - 0.02), note0, min(0.30, note0 + 0.02)]
    refine_decays = [max(0.990, decay0 - 0.001), decay0, min(0.999, decay0 + 0.001)]
    refine_gains = [max(0.15, gain0 - 0.06), gain0, min(0.65, gain0 + 0.06)]
    refine_trans = sorted(set([trans0 - 1, trans0, trans0 + 1]))

    for step, note, decay, gain, trans in product(
        refine_steps,
        refine_notes,
        refine_decays,
        refine_gains,
        refine_trans,
    ):
        score = score_params(float(step), float(note), float(decay), float(gain), int(trans))
        if score > best_score:
            best_score = score
            best_params = {
                "step_seconds": float(step),
                "note_seconds": float(note),
                "decay": float(decay),
                "gain": float(gain),
                "transpose_semitones": int(trans),
            }

    best_audio, _, _ = synthesize_audio_array_from_tabs_text(
        tabs_text,
        sample_rate=sr,
        step_seconds=float(best_params["step_seconds"]),
        note_seconds=float(best_params["note_seconds"]),
        decay=float(best_params["decay"]),
        gain=float(best_params["gain"]),
        transpose_semitones=int(best_params["transpose_semitones"]),
    )
    sf.write(output_path, best_audio, sr)

    return MatchResult(score=float(best_score), params=best_params, output_path=output_path)
