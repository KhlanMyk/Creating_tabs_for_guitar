"""Refine generated tabs against original audio by correcting fret pitch choices.

Uses pyin (probabilistic YIN) for pitch tracking, which is more robust
than plain YIN for polyphonic and noisy audio.
"""
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
    # default: each column = one 16th note at ~120 BPM
    return 0.125


def _format_tabs_from_grid(grid: list[list[str]], cols: int) -> str:
    """Format a 6-row token grid into the timing-proportional tab format.

    Each column = one 16th-note slot.
    16 slots per measure (4 beats × 4 sixteenths).
    2 measures per display line.
    """
    string_names = ["e", "B", "G", "D", "A", "E"]
    slots_per_measure = 16
    beats_per_measure = 4
    slots_per_beat = 4
    measures_per_line = 2

    num_measures = max(1, (cols + slots_per_measure - 1) // slots_per_measure)

    output: list[str] = []

    for line_start in range(0, num_measures, measures_per_line):
        line_end = min(line_start + measures_per_line, num_measures)

        # Beat markers
        beat_line = "  "
        for m_idx in range(line_start, line_end):
            for beat in range(beats_per_measure):
                beat_line += str(beat + 1)
                beat_line += " " * (slots_per_beat - 1)
            beat_line += " "
        beat_line = beat_line.rstrip()
        if beat_line.strip():
            output.append(beat_line)

        # String lines
        for row in range(6):
            line = f"{string_names[row]}|"
            for m_idx in range(line_start, line_end):
                m_start = m_idx * slots_per_measure
                for slot_offset in range(slots_per_measure):
                    abs_slot = m_start + slot_offset
                    if abs_slot < cols:
                        token = grid[row][abs_slot]
                        fret_str = "-" if token == "--" else token
                    else:
                        fret_str = "-"
                    line += fret_str
                line += "|"
            output.append(line)
        output.append("")

    return "\n".join(output).rstrip()


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


def _detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=512,
        backtrack=True,
        units="frames",
    )
    if onset_frames.size == 0:
        return np.array([], dtype=np.float32)
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=512).astype(np.float32)


def _expand_grid_by_units(grid: list[list[str]], cols: int, units: np.ndarray) -> tuple[list[list[str]], int]:
    expanded = [[] for _ in range(6)]
    for c in range(cols):
        u = int(max(1, units[c]))
        for row in range(6):
            token = grid[row][c]
            expanded[row].append(token)
            if u > 1:
                expanded[row].extend(["--"] * (u - 1))
    return expanded, len(expanded[0]) if expanded else 0


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
    # Use pyin (probabilistic) for more robust pitch tracking
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=audio,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("E6"),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    if f0 is None:
        f0 = np.zeros(1)
    if voiced_probs is None:
        voiced_probs = np.ones_like(f0)

    # Replace NaN with 0 for hz_to_midi
    f0_clean = np.where(np.isfinite(f0), f0, 0.0)
    midi_track = np.where(f0_clean > 0, librosa.hz_to_midi(np.maximum(f0_clean, 1e-6)), 0.0)

    changes = 0

    for c in range(cols):
        t = c * step
        frame_idx = int(round(t * sr / hop_length))
        frame_idx = max(0, min(frame_idx, len(midi_track) - 1))

        target_midi = float(midi_track[frame_idx])
        if not np.isfinite(target_midi) or target_midi <= 0:
            continue

        # Also check voiced probability — skip if pitch is unreliable
        vp = float(voiced_probs[frame_idx]) if frame_idx < len(voiced_probs) else 0.0
        if vp < 0.4:
            continue

        for row in range(6):
            token = grid[row][c]
            if token == "--":
                continue

            fret = int(token)
            string_num = row + 1
            new_fret = _closest_refined_fret(fret, string_num, target_midi, max_fret=max_fret)

            if new_fret != fret:
                open_midi = OPEN_STRING_MIDI[string_num]
                old_err = abs((open_midi + fret) - target_midi)
                new_err = abs((open_midi + new_fret) - target_midi)
                # Apply if clearly better (>0.5 semitone improvement)
                if (old_err - new_err) >= 0.5:
                    grid[row][c] = str(new_fret)
                    changes += 1

    onsets = _detect_onsets(audio, sr)

    estimated_step = step
    if len(onsets) >= 4:
        timeline = np.interp(
            np.linspace(0, len(onsets) - 1, cols + 1),
            np.arange(len(onsets)),
            onsets,
        )
        timeline = np.maximum.accumulate(timeline)
        intervals = np.diff(timeline)
        positive = intervals[intervals > 1e-4]
        fallback_interval = float(np.median(positive)) if positive.size else step
        intervals = np.where(intervals > 1e-4, intervals, fallback_interval)

        median_step = float(np.median(intervals))
        estimated_step = float(np.clip(median_step, 0.08, 0.22))

        units = np.clip(np.rint(intervals / max(estimated_step, 1e-6)).astype(int), 1, 6)
        grid, cols = _expand_grid_by_units(grid, cols, units)

    refined_text = _format_tabs_from_grid(grid, cols)
    return RefineTabsResult(
        refined_tabs_text=refined_text,
        changes_count=changes,
        estimated_step_seconds=estimated_step,
    )
