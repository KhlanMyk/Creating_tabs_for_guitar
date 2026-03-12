"""
Guitar tab generator module.

Converts detected notes (with real timing) into guitar fretboard positions
and formats them as professional tablature text.  Also provides a
timing-aware data export for the synthesiser so that the original rhythm
and melody are preserved in playback.
"""
from __future__ import annotations

import json
from typing import List, Dict, Tuple, Optional

import librosa
import numpy as np


class GuitarTabGenerator:
    """Class for generating guitar tabs"""

    OPEN_STRING_MIDI = {
        6: 40,  # E2
        5: 45,  # A2
        4: 50,  # D3
        3: 55,  # G3
        2: 59,  # B3
        1: 64,  # E4
    }

    def __init__(self, max_fret: int = 15):
        self.tabs: Dict[int, List] = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self.max_fret = max_fret
        self._last_position: Optional[Tuple[int, int]] = None
        self._hand_center: float = 3.0  # current average fret position
        # Flat list of timed tab events for the synthesiser
        self._timed_events: List[dict] = []

    def set_max_fret(self, max_fret: int):
        self.max_fret = max_fret

    def note_to_tab_positions(self, note_name: str) -> List[Tuple[int, int]]:
        try:
            note_midi = int(librosa.note_to_midi(note_name))
        except Exception:
            return []
        positions = []
        for string, open_midi in self.OPEN_STRING_MIDI.items():
            fret = note_midi - open_midi
            if 0 <= fret <= self.max_fret:
                positions.append((string, fret))
        return positions

    def _choose_position(self, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Pick the best (string, fret) using hand-position-aware scoring.

        Scoring considers:
          • Distance from current hand centre (lower is better)
          • Open-string bonus (fret 0 is easy to play)
          • Stretch penalty (frets far from hand centre)
          • String-change cost (prefer staying on same/adjacent string)
          • Low-fret preference (lower positions are generally easier)
          • String comfort (middle strings slightly preferred)
        """
        if not positions:
            return (1, 0)

        if self._last_position is None:
            # First note: prefer lowest fret, then lowest string number
            return sorted(positions, key=lambda x: (x[1], x[0]))[0]

        last_string, last_fret = self._last_position
        hc = self._hand_center

        def _score(pos: Tuple[int, int]) -> float:
            s, f = pos
            # Fret distance from hand centre
            fret_dist = abs(f - hc)
            # String change cost
            string_dist = abs(s - last_string)
            # Open-string bonus: open strings are easy
            open_bonus = -2.0 if f == 0 else 0.0
            # Low-fret preference: positions 0-5 are most comfortable
            low_fret_bonus = -0.3 * max(0, 5 - f)
            # Stretch penalty: big reach from current hand position
            stretch = max(0.0, fret_dist - 4) * 2.5
            # Prefer staying near last fret
            move_cost = abs(f - last_fret) * 0.5
            # Slight preference for middle strings (easier ergonomics)
            string_comfort = abs(s - 3.5) * 0.15
            return (fret_dist + string_dist * 0.8 + open_bonus
                    + low_fret_bonus + stretch + move_cost + string_comfort)

        best = min(positions, key=_score)
        return best

    def generate_tabs(self, notes: List[Dict]) -> Dict[int, List]:
        """Generate tab positions from note list, preserving timing."""
        self.tabs = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self._last_position = None
        self._hand_center = 3.0
        self._timed_events = []

        for note_info in notes:
            note_name = note_info.get("note")
            if not note_name:
                continue
            positions = self.note_to_tab_positions(note_name)
            string, fret = self._choose_position(positions)
            self._last_position = (string, fret)
            # Smoothly track hand centre for position-aware scoring
            if fret > 0:
                self._hand_center = self._hand_center * 0.7 + fret * 0.3

            start_time = note_info.get("start_time", 0.0)
            duration = note_info.get("duration", 0.0)
            velocity = note_info.get("velocity", 0.7)

            entry = {
                "fret": fret,
                "time": start_time,
                "note": note_name,
                "duration": duration,
                "velocity": velocity,
            }
            self.tabs[string].append(entry)

            # Also store as flat event for synthesis
            self._timed_events.append({
                "string": string,
                "fret": fret,
                "note": note_name,
                "start_time": start_time,
                "duration": duration,
                "velocity": velocity,
                "midi": note_info.get("midi", 0),
            })

        return self.tabs

    #  timing-aware exports 
    def get_timed_events(self) -> List[dict]:
        """Return flat list of timed tab events for synthesis.

        Each dict: {string, fret, note, start_time, duration, velocity, midi}
        """
        return list(self._timed_events)

    def export_timed_json(self) -> str:
        """Serialise timed events as JSON string (for file storage)."""
        return json.dumps(self._timed_events, indent=2)

    @staticmethod
    def load_timed_json(json_str: str) -> List[dict]:
        """Deserialise timed events from JSON."""
        return json.loads(json_str)

    def _guess_chord(self, notes_at_time: List[str]) -> Optional[str]:
        if not notes_at_time:
            return None
        pcs: set = set()
        for n in notes_at_time:
            pc = n[:-1] if n[-1].isdigit() else n
            pcs.add(pc)
        chord_map = {
            # Major
            frozenset({"C", "E", "G"}): "C",
            frozenset({"D", "F#", "A"}): "D",
            frozenset({"E", "G#", "B"}): "E",
            frozenset({"F", "A", "C"}): "F",
            frozenset({"G", "B", "D"}): "G",
            frozenset({"A", "C#", "E"}): "A",
            frozenset({"B", "D#", "F#"}): "B",
            # Minor
            frozenset({"A", "C", "E"}): "Am",
            frozenset({"B", "D", "F#"}): "Bm",
            frozenset({"C", "Eb", "G"}): "Cm",
            frozenset({"D", "F", "A"}): "Dm",
            frozenset({"E", "G", "B"}): "Em",
            frozenset({"F", "Ab", "C"}): "Fm",
            frozenset({"F#", "A", "C#"}): "F#m",
            frozenset({"G", "Bb", "D"}): "Gm",
            frozenset({"G#", "B", "D#"}): "G#m",
            frozenset({"C#", "E", "G#"}): "C#m",
            # Seventh
            frozenset({"G", "B", "D", "F"}): "G7",
            frozenset({"C", "E", "G", "Bb"}): "C7",
            frozenset({"D", "F#", "A", "C"}): "D7",
            frozenset({"A", "C#", "E", "G"}): "A7",
            frozenset({"E", "G#", "B", "D"}): "E7",
            frozenset({"B", "D#", "F#", "A"}): "B7",
            # Maj7
            frozenset({"C", "E", "G", "B"}): "Cmaj7",
            frozenset({"F", "A", "C", "E"}): "Fmaj7",
            frozenset({"G", "B", "D", "F#"}): "Gmaj7",
            # Minor 7
            frozenset({"A", "C", "E", "G"}): "Am7",
            frozenset({"D", "F", "A", "C"}): "Dm7",
            frozenset({"E", "G", "B", "D"}): "Em7",
            # Power chords
            frozenset({"D", "A"}): "D5",
            frozenset({"E", "B"}): "E5",
            frozenset({"A", "E"}): "A5",
            frozenset({"G", "D"}): "G5",
            frozenset({"C", "G"}): "C5",
            frozenset({"F", "C"}): "F5",
            # Sus
            frozenset({"A", "D", "E"}): "Asus4",
            frozenset({"D", "G", "A"}): "Dsus4",
            frozenset({"E", "A", "B"}): "Esus4",
        }
        fs = frozenset(pcs)
        if fs in chord_map:
            return chord_map[fs]
        # check subsets (largest first)
        best_match: Optional[str] = None
        best_size = 0
        for k, v in chord_map.items():
            if k.issubset(fs) and len(k) > best_size and len(k) >= 3:
                best_match = v
                best_size = len(k)
        return best_match

    #  text formatting (professional, timing-aware)

    def _estimate_bpm(self) -> float:
        """Estimate BPM from note onset intervals."""
        if len(self._timed_events) < 2:
            return 120.0
        onsets = sorted(e["start_time"] for e in self._timed_events)
        intervals = np.diff(onsets)
        intervals = intervals[(intervals > 0.1) & (intervals < 2.0)]
        if len(intervals) == 0:
            return 120.0
        median_interval = float(np.median(intervals))
        bpm = 60.0 / max(median_interval, 0.15)
        # Snap to reasonable range
        bpm = max(40.0, min(240.0, bpm))
        return bpm

    def _quantize_to_grid(self, time_val: float, beat_dur: float) -> float:
        """Quantize a time value to the nearest 16th-note grid position."""
        sixteenth = beat_dur / 4.0
        return round(time_val / sixteenth) * sixteenth

    def format_tabs_as_text(self) -> str:
        """Format tabs as professional guitar tablature with timing-proportional spacing.

        - Each dash '-' represents a 16th-note time slot
        - Rests (silence) are properly shown as dashes
        - Measures are aligned to actual beats (4 beats per measure)
        - Note durations are shown by the gap until the next note on that string
        """
        all_events: List[Tuple[float, int, dict]] = []
        for string_num, tab_list in self.tabs.items():
            for tab in tab_list:
                all_events.append((tab["time"], string_num, tab))

        if not all_events:
            return "No tabs generated"

        # Estimate BPM and beat duration
        bpm = self._estimate_bpm()
        beat_dur = 60.0 / bpm
        sixteenth = beat_dur / 4.0

        # Get total time span
        min_time = min(e[0] for e in all_events)
        max_time = max(e[0] + e[2].get("duration", 0.1) for e in all_events)

        # Quantize start so first note aligns to grid
        grid_start = self._quantize_to_grid(min_time, beat_dur)

        # Total number of 16th-note slots needed
        total_sixteenths = max(1, int(np.ceil((max_time - grid_start) / sixteenth)) + 1)

        # Cap at reasonable length
        total_sixteenths = min(total_sixteenths, 512)

        # Build a grid: for each string, a list of slot contents
        # Each slot is either None (rest) or a fret number string
        string_grid: Dict[int, List[Optional[str]]] = {}
        for s in range(1, 7):
            string_grid[s] = [None] * total_sixteenths

        # Place notes on the grid
        for ev_time, string_num, tab in all_events:
            slot = int(round((ev_time - grid_start) / sixteenth))
            slot = max(0, min(slot, total_sixteenths - 1))
            string_grid[string_num][slot] = str(tab["fret"])

        # Build chord labels per slot
        chord_at_slot: Dict[int, Optional[str]] = {}
        for slot_idx in range(total_sixteenths):
            notes_here = []
            for ev_t, s, tab in all_events:
                ev_slot = int(round((ev_t - grid_start) / sixteenth))
                if ev_slot == slot_idx:
                    notes_here.append(tab["note"])
            if notes_here:
                chord_at_slot[slot_idx] = self._guess_chord(notes_here)
            else:
                chord_at_slot[slot_idx] = None

        # Format into measures (4 beats = 16 sixteenth-note slots per measure)
        slots_per_measure = 16
        beats_per_measure = 4
        slots_per_beat = 4

        num_measures = max(1, (total_sixteenths + slots_per_measure - 1) // slots_per_measure)
        measures_per_line = 2  # 2 measures per text line for readability

        string_names = ["e", "B", "G", "D", "A", "E"]
        output: List[str] = []

        for line_start in range(0, num_measures, measures_per_line):
            line_end = min(line_start + measures_per_line, num_measures)

            # Build beat markers line (shows beat positions)
            beat_line = "  "
            for m_idx in range(line_start, line_end):
                for beat in range(beats_per_measure):
                    beat_num = beat + 1
                    beat_line += str(beat_num)
                    # Fill remaining slots in this beat with spaces
                    beat_line += " " * (slots_per_beat - 1)
                beat_line += " "  # measure separator space
            beat_line = beat_line.rstrip()
            if beat_line.strip():
                output.append(beat_line)

            # Build chord line
            chord_line = "  "
            for m_idx in range(line_start, line_end):
                m_start_slot = m_idx * slots_per_measure
                last_chord = None
                for slot_offset in range(slots_per_measure):
                    abs_slot = m_start_slot + slot_offset
                    if abs_slot < total_sixteenths:
                        ch = chord_at_slot.get(abs_slot)
                        if ch and ch != last_chord:
                            chord_line += ch
                            pad = max(0, 1 - len(ch))
                            chord_line += " " * pad
                            last_chord = ch
                        else:
                            chord_line += " "
                    else:
                        chord_line += " "
                chord_line += " "
            chord_line = chord_line.rstrip()
            if chord_line.strip():
                output.append(chord_line)

            # Build string lines
            for row, string_num in enumerate(range(1, 7)):
                line = f"{string_names[row]}|"
                for m_idx in range(line_start, line_end):
                    m_start_slot = m_idx * slots_per_measure
                    for slot_offset in range(slots_per_measure):
                        abs_slot = m_start_slot + slot_offset
                        if abs_slot < total_sixteenths:
                            fret_val = string_grid[string_num][abs_slot]
                            if fret_val is not None:
                                line += fret_val
                                # Double-digit frets take 2 chars, so skip
                                # adding a dash after them
                                if len(fret_val) == 1:
                                    pass  # single char, normal
                            else:
                                line += "-"
                        else:
                            line += "-"
                    line += "|"
                output.append(line)
            output.append("")

        # Add tempo info at the bottom
        output.append(f"  Tempo: ~{int(round(bpm))} BPM | "
                      f"Each '-' = 1/16 note | "
                      f"{len(self._timed_events)} notes")

        return "\n".join(output).rstrip()

    def save_tabs_to_file(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.format_tabs_as_text())
        print(f"Tabs saved to: {filename}")

    def save_timed_tabs_to_file(self, filename: str):
        """Save timing-aware JSON alongside text tabs."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.export_timed_json())
        print(f"Timed tabs saved to: {filename}")
