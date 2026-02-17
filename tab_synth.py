"""Tab-to-audio guitar synthesizer using a simple Karplus-Strong model."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf


OPEN_STRING_MIDI: Dict[int, int] = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}


@dataclass
class SynthResult:
    sample_rate: int
    duration: float
    notes_count: int


def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _karplus_strong(freq: float, duration: float, sample_rate: int, decay: float = 0.996) -> np.ndarray:
    """Generate plucked-string audio for one note."""
    n_samples = max(1, int(duration * sample_rate))
    period = max(2, int(sample_rate / max(freq, 1e-6)))

    buf = np.random.uniform(-1.0, 1.0, period).astype(np.float32)
    out = np.zeros(n_samples, dtype=np.float32)

    idx = 0
    for i in range(n_samples):
        out[i] = buf[idx]
        nxt = (idx + 1) % period
        buf[idx] = decay * 0.5 * (buf[idx] + buf[nxt])
        idx = nxt

    # quick envelope for cleaner transients
    attack = int(0.003 * sample_rate)
    release = int(0.03 * sample_rate)
    if attack > 1 and attack < n_samples:
        out[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if release > 1 and release < n_samples:
        out[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)

    return out


def parse_tabs_text(tab_text: str) -> Tuple[List[List[str]], int]:
    """Parse generated tabs text into 6-string token grid.

    Returns:
        (grid_by_string, columns_count) where string index 0 is high-e (string 1).
    """
    lines = [ln.strip() for ln in tab_text.splitlines() if ln.strip()]
    string_lines = []
    for ln in lines:
        if re.match(r'^[eBGDAE]\|', ln):
            string_lines.append(ln)

    if len(string_lines) < 6:
        raise ValueError("Tabs text does not contain 6 string lines.")

    # Keep order as e B G D A E
    parsed: List[List[str]] = []
    max_cols = 0
    for ln in string_lines[:6]:
        # Extract only numbers and rests from the body
        body = ln.split('|', 1)[1].rsplit('|', 1)[0] if '|' in ln else ln
        tokens = re.findall(r'--|\d+', body)
        parsed.append(tokens)
        max_cols = max(max_cols, len(tokens))

    # pad to same number of columns
    for row in parsed:
        if len(row) < max_cols:
            row.extend(['--'] * (max_cols - len(row)))

    return parsed, max_cols


def synthesize_from_tabs_text(
    tab_text: str,
    output_path: str,
    sample_rate: int = 44100,
    step_seconds: float = 0.14,
    note_seconds: float = 0.18,
    play: bool = False,
) -> SynthResult:
    """Render tabs text to a wav file and optionally play it."""
    grid, cols = parse_tabs_text(tab_text)

    total_dur = max(note_seconds + cols * step_seconds, 0.1)
    audio = np.zeros(int(total_dur * sample_rate), dtype=np.float32)

    notes_count = 0

    # grid rows: [e, B, G, D, A, E] => string numbers [1..6]
    for c in range(cols):
        start_idx = int(c * step_seconds * sample_rate)
        for row_idx in range(6):
            token = grid[row_idx][c]
            if token == '--':
                continue
            fret = int(token)
            string_num = row_idx + 1
            open_midi = OPEN_STRING_MIDI[string_num]
            midi = open_midi + fret
            freq = _midi_to_hz(midi)

            note = _karplus_strong(freq, note_seconds, sample_rate)
            end_idx = min(len(audio), start_idx + len(note))
            audio[start_idx:end_idx] += note[: end_idx - start_idx] * 0.35
            notes_count += 1

    peak = float(np.max(np.abs(audio))) if len(audio) else 1.0
    if peak > 0:
        audio = audio / max(peak, 1.0)

    sf.write(output_path, audio, sample_rate)

    if play:
        sd.play(audio, sample_rate)
        sd.wait()

    return SynthResult(sample_rate=sample_rate, duration=total_dur, notes_count=notes_count)


def synthesize_from_tabs_file(
    tabs_path: str,
    output_path: str,
    sample_rate: int = 44100,
    step_seconds: float = 0.14,
    note_seconds: float = 0.18,
    play: bool = False,
) -> SynthResult:
    with open(tabs_path, 'r', encoding='utf-8') as f:
        tab_text = f.read()
    return synthesize_from_tabs_text(
        tab_text,
        output_path=output_path,
        sample_rate=sample_rate,
        step_seconds=step_seconds,
        note_seconds=note_seconds,
        play=play,
    )
