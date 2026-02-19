"""Refine generated tabs against original audio by correcting fret pitch choices."""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from tab_synth import OPEN_STRING_MIDI, parse_tabs_text


@dataclass
class RefineTabsResult:
    refined_tabs_text: str
    changes_count: int
    estimated_step_seconds: float


def _estimate_step_seconds(cols: int, step_seconds: float | None) -> float:
    if step_seconds is not None and step_seconds > 0:
        return float(step_seconds)
    # default timing used by synth in this project
    if cols <= 0:
        return 0.14
    return 0.14


def _format_tabs_from_grid(grid: list[list[str]], cols: int) -> str:
    string_names = ["e", "B", "G", "D", "A", "E"]

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("GUITAR TABS")
    lines.append("=" * 60)

    for row in range(6):
        parts = []
        for c in range(cols):
            token = grid[row][c]
            if token == "--":
                token = "-"
            parts.append(f"{token:>2}")

        line = f"{string_names[row]}|" + "-".join(parts) + "|"
        lines.append(line)

    lines.append("=" * 60)
    return "\n".join(lines)


def _closest_refined_fret(
    current_fret: int,
    string_num: int,
    target_midi: float,
    max_fret: int,
) -> int:
    open_midi = OPEN_STRING_MIDI[string_num]
    current_midi = open_midi + current_fret

    best_fret = current_fret
    best_score = abs(current_midi - target_midi)

    # Try semitone moves in reasonable range, but discourage large jumps.
    for delta in range(-12, 13):
        cand_fret = current_fret + delta
        if not (0 <= cand_fret <= max_fret):
            continue

        cand_midi = open_midi + cand_fret
        score = abs(cand_midi - target_midi) + 0.04 * abs(delta)

        if score < best_score:
            best_score = score
            best_fret = cand_fret

    return best_fret


def refine_tabs_with_original(
    tabs_text: str,
    original_audio_path: str,
    step_seconds: float | None = None,
    max_fret: int = 24,
    sr: int = 22050,
) -> RefineTabsResult:
    """Refine tab fret numbers using dominant pitch from original audio.

    This focuses on pitch correction (including octave correction where possible)
    while preserving tab rhythm/structure (column count/order).
    """
    grid, cols = parse_tabs_text(tabs_text)
    step = _estimate_step_seconds(cols, step_seconds)

    preview_duration = max(2.0, cols * step + 0.5)
    audio, _ = librosa.load(original_audio_path, sr=sr, mono=True, duration=preview_duration)

    hop_length = 512
    f0 = librosa.yin(
        y=audio,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("E6"),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    midi_track = librosa.hz_to_midi(f0)

    changes = 0

    for c in range(cols):
        t = c * step
        frame_idx = int(round(t * sr / hop_length))
        frame_idx = max(0, min(frame_idx, len(midi_track) - 1))

        target_midi = float(midi_track[frame_idx])
        if not np.isfinite(target_midi):
            continue

        for row in range(6):
            token = grid[row][c]
            if token == "--":
                continue

            fret = int(token)
            string_num = row + 1
            new_fret = _closest_refined_fret(fret, string_num, target_midi, max_fret=max_fret)

            if new_fret != fret:
                # apply only if improvement is meaningful
                open_midi = OPEN_STRING_MIDI[string_num]
                old_err = abs((open_midi + fret) - target_midi)
                new_err = abs((open_midi + new_fret) - target_midi)
                if (old_err - new_err) >= 0.8:
                    grid[row][c] = str(new_fret)
                    changes += 1

    refined_text = _format_tabs_from_grid(grid, cols)
    return RefineTabsResult(
        refined_tabs_text=refined_text,
        changes_count=changes,
        estimated_step_seconds=step,
    )
