"""Tab-to-audio guitar synthesiser using Karplus-Strong.

Supports TWO modes:
1. **Timed events** – a list of dicts with real start_time / duration
   from the tab generator.  This preserves the original rhythm.
2. **Text tabs** – legacy fixed-step playback from tab text (still useful
   for old files / compatibility).
"""
from __future__ import annotations

import json
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


#  helpers
def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _karplus_strong(
    freq: float,
    duration: float,
    sample_rate: int,
    decay: float = 0.996,
    velocity: float = 1.0,
) -> np.ndarray:
    """Generate plucked-string audio for one note (vectorised)."""
    n_samples = max(1, int(duration * sample_rate))
    period = max(2, int(sample_rate / max(freq, 1e-6)))

    buf = np.random.uniform(-1.0, 1.0, period).astype(np.float64)
    out = np.empty(n_samples, dtype=np.float64)

    pos = 0
    while pos < n_samples:
        chunk = min(period, n_samples - pos)
        out[pos:pos + chunk] = buf[:chunk]
        shifted = np.roll(buf, -1)
        buf[:] = decay * 0.5 * (buf + shifted)
        pos += chunk

    out = out.astype(np.float32)

    # envelope
    attack = int(0.003 * sample_rate)
    release = int(0.05 * sample_rate)
    if 1 < attack < n_samples:
        out[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if 1 < release < n_samples:
        out[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)

    # Apply velocity
    out *= np.float32(max(0.1, min(1.0, velocity)))
    return out


# 
#  MODE 1 — Timed events (real rhythm)
# 

def synthesize_from_timed_events(
    events: List[dict],
    output_path: str | None = None,
    sample_rate: int = 44100,
    decay: float = 0.996,
    gain: float = 0.40,
    transpose_semitones: int = 0,
    play: bool = False,
) -> Tuple[np.ndarray, SynthResult]:
    """Render a list of timed tab events to audio.

    Each event dict must have:
        string, fret, start_time, duration
    Optional: velocity (0-1), midi
    """
    if not events:
        empty = np.zeros(int(0.1 * sample_rate), dtype=np.float32)
        res = SynthResult(sample_rate=sample_rate, duration=0.1, notes_count=0)
        return empty, res

    # Determine total duration
    max_end = max(e["start_time"] + e["duration"] for e in events)
    total_dur = max_end + 0.5  # small tail
    audio = np.zeros(int(total_dur * sample_rate), dtype=np.float32)
    notes_count = 0

    for ev in events:
        string_num = ev["string"]
        fret = ev["fret"]
        start_t = ev["start_time"]
        dur = max(ev["duration"], 0.08)  # minimum audible length
        velocity = ev.get("velocity", 0.7)

        open_midi = OPEN_STRING_MIDI.get(string_num, 64)
        midi = open_midi + fret + transpose_semitones
        freq = _midi_to_hz(midi)

        note_audio = _karplus_strong(freq, dur, sample_rate, decay=decay, velocity=velocity)
        start_idx = int(start_t * sample_rate)
        end_idx = min(len(audio), start_idx + len(note_audio))
        actual_len = end_idx - start_idx
        if actual_len > 0:
            audio[start_idx:end_idx] += note_audio[:actual_len] * gain
            notes_count += 1

    # Normalise
    peak = float(np.max(np.abs(audio))) if len(audio) else 1.0
    if peak > 0:
        audio = audio / max(peak, 1.0)

    if output_path:
        sf.write(output_path, audio, sample_rate)

    if play:
        sd.play(audio, sample_rate)
        sd.wait()

    return audio, SynthResult(sample_rate=sample_rate, duration=total_dur, notes_count=notes_count)


def synthesize_from_timed_json(
    json_str: str,
    output_path: str | None = None,
    **kwargs,
) -> Tuple[np.ndarray, SynthResult]:
    """Load timed events from JSON and synthesise."""
    events = json.loads(json_str)
    return synthesize_from_timed_events(events, output_path=output_path, **kwargs)



#  MODE 2 — Text tabs (legacy fixed-step)

def parse_tabs_text(tab_text: str) -> Tuple[List[List[str]], int]:
    """Parse generated tabs text into 6-string token grid.

    Supports the timing-proportional format where each character is one
    16th-note slot.  A digit (or pair of digits) = fret number,
    '-' = rest/silence.

    Returns (grid, cols) where grid is 6 rows (one per string, e→E)
    and each row has `cols` tokens ('--' for rest, or fret number string).
    """
    lines = tab_text.splitlines()

    groups: List[List[str]] = []
    buffer: List[str] = []
    for ln in lines:
        stripped = ln.strip()
        if re.match(r'^[eBGDAE]\|', stripped):
            buffer.append(stripped)
            if len(buffer) == 6:
                groups.append(buffer)
                buffer = []
        else:
            if len(buffer) == 6:
                groups.append(buffer)
            buffer = []
    if len(buffer) == 6:
        groups.append(buffer)

    if not groups:
        raise ValueError("Tabs text does not contain 6 string lines.")

    parsed: List[List[str]] = [[] for _ in range(6)]

    for group in groups:
        for row_idx, ln in enumerate(group):
            # Extract body between first | and last |
            body = ln.split('|', 1)[1] if '|' in ln else ln
            # Strip trailing pipe
            body = body.rstrip('|')
            # Remove internal measure separators (|)
            body = body.replace('|', '')

            # Parse character by character:
            # Each '-' is one rest slot.
            # Digits form fret numbers (1 or 2 digits).
            i = 0
            while i < len(body):
                ch = body[i]
                if ch == '-':
                    parsed[row_idx].append('--')
                    i += 1
                elif ch.isdigit():
                    # Check for 2-digit fret
                    if i + 1 < len(body) and body[i + 1].isdigit():
                        parsed[row_idx].append(body[i:i + 2])
                        i += 2
                    else:
                        parsed[row_idx].append(ch)
                        i += 1
                else:
                    # Skip unexpected characters
                    i += 1

    max_cols = max(len(row) for row in parsed) if parsed else 0
    for row in parsed:
        while len(row) < max_cols:
            row.append('--')

    return parsed, max_cols


def synthesize_from_tabs_text(
    tab_text: str,
    output_path: str,
    sample_rate: int = 44100,
    step_seconds: float = 0.125,
    note_seconds: float = 0.18,
    decay: float = 0.996,
    gain: float = 0.35,
    transpose_semitones: int = 0,
    play: bool = False,
) -> SynthResult:
    """Render tabs text to a wav file (each column = one 16th-note slot)."""
    audio, notes_count, total_dur = synthesize_audio_array_from_tabs_text(
        tab_text,
        sample_rate=sample_rate,
        step_seconds=step_seconds,
        note_seconds=note_seconds,
        decay=decay,
        gain=gain,
        transpose_semitones=transpose_semitones,
    )

    peak = float(np.max(np.abs(audio))) if len(audio) else 1.0
    if peak > 0:
        audio = audio / max(peak, 1.0)

    sf.write(output_path, audio, sample_rate)

    if play:
        sd.play(audio, sample_rate)
        sd.wait()

    return SynthResult(sample_rate=sample_rate, duration=total_dur, notes_count=notes_count)


def synthesize_audio_array_from_tabs_text(
    tab_text: str,
    sample_rate: int = 44100,
    step_seconds: float = 0.125,
    note_seconds: float = 0.18,
    decay: float = 0.996,
    gain: float = 0.35,
    transpose_semitones: int = 0,
    max_duration_seconds: float | None = None,
) -> Tuple[np.ndarray, int, float]:
    """Render tabs text to an in-memory audio array.

    In the timing-proportional format each column is one 16th-note slot.
    step_seconds defaults to 0.125s (~120 BPM with 4 slots per beat).
    """
    grid, cols = parse_tabs_text(tab_text)

    if max_duration_seconds is not None and max_duration_seconds > 0:
        approx_cols = int(max(1, (max_duration_seconds - note_seconds) / max(step_seconds, 1e-6)))
        cols = max(1, min(cols, approx_cols))

    total_dur = max(note_seconds + cols * step_seconds, 0.1)
    audio = np.zeros(int(total_dur * sample_rate), dtype=np.float32)
    notes_count = 0

    for c in range(cols):
        start_idx = int(c * step_seconds * sample_rate)
        for row_idx in range(6):
            token = grid[row_idx][c]
            if token == '--':
                continue
            fret = int(token)
            string_num = row_idx + 1
            open_midi = OPEN_STRING_MIDI[string_num]
            midi = open_midi + fret + transpose_semitones
            freq = _midi_to_hz(midi)

            note = _karplus_strong(freq, note_seconds, sample_rate, decay=decay)
            end_idx = min(len(audio), start_idx + len(note))
            audio[start_idx:end_idx] += note[: end_idx - start_idx] * gain
            notes_count += 1

    peak = float(np.max(np.abs(audio))) if len(audio) else 1.0
    if peak > 0:
        audio = audio / max(peak, 1.0)

    return audio, notes_count, total_dur


def synthesize_from_tabs_file(
    tabs_path: str,
    output_path: str,
    sample_rate: int = 44100,
    step_seconds: float = 0.125,
    note_seconds: float = 0.18,
    decay: float = 0.996,
    gain: float = 0.35,
    transpose_semitones: int = 0,
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
        decay=decay,
        gain=gain,
        transpose_semitones=transpose_semitones,
        play=play,
    )

