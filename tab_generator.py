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
        if not positions:
            return (1, 0)
        if self._last_position is None:
            return sorted(positions, key=lambda x: (x[1], x[0]))[0]
        last_string, last_fret = self._last_position
        return sorted(
            positions,
            key=lambda x: (abs(x[1] - last_fret), abs(x[0] - last_string), x[1]),
        )[0]

    def generate_tabs(self, notes: List[Dict]) -> Dict[int, List]:
        """Generate tab positions from note list, preserving timing."""
        self.tabs = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self._last_position = None
        self._timed_events = []

        for note_info in notes:
            note_name = note_info.get("note")
            if not note_name:
                continue
            positions = self.note_to_tab_positions(note_name)
            string, fret = self._choose_position(positions)
            self._last_position = (string, fret)

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

    #  chord guessing
    def _guess_chord(self, notes_at_time: List[str]) -> Optional[str]:
        if not notes_at_time:
            return None
        pcs: set = set()
        for n in notes_at_time:
            pc = n[:-1] if n[-1].isdigit() else n
            pcs.add(pc)
        chord_map = {
            frozenset({"C", "E", "G"}): "C",
            frozenset({"A", "C", "E"}): "Am",
            frozenset({"D", "F", "A"}): "Dm",
            frozenset({"D", "F#", "A"}): "D",
            frozenset({"E", "G#", "B"}): "E",
            frozenset({"E", "G", "B"}): "Em",
            frozenset({"G", "B", "D"}): "G",
            frozenset({"G", "B", "D", "F"}): "G7",
            frozenset({"A", "C#", "E"}): "A",
            frozenset({"F", "A", "C"}): "F",
            frozenset({"B", "D", "F#"}): "Bm",
            frozenset({"C", "E", "G", "B"}): "Cmaj7",
            frozenset({"D", "A"}): "D5",
            frozenset({"E", "B"}): "E5",
            frozenset({"A", "E"}): "A5",
        }
        fs = frozenset(pcs)
        if fs in chord_map:
            return chord_map[fs]
        for k, v in chord_map.items():
            if k.issubset(fs) and len(k) >= 3:
                return v
        return None

    #  text formatting (professional)
    def format_tabs_as_text(self) -> str:
        """Format tabs as professional guitar tablature with measures."""
        all_events: List[Tuple[float, int, dict]] = []
        for string_num, tab_list in self.tabs.items():
            for tab in tab_list:
                all_events.append((tab["time"], string_num, tab))

        if not all_events:
            return "No tabs generated"

        all_times = sorted({e[0] for e in all_events})

        columns: List[Dict[int, str]] = []
        chord_labels: List[Optional[str]] = []
        for t in all_times:
            col: Dict[int, str] = {}
            notes_here: List[str] = []
            for ev_t, s, tab in all_events:
                if abs(ev_t - t) < 0.01:
                    col[s] = str(tab["fret"])
                    notes_here.append(tab["note"])
            columns.append(col)
            chord_labels.append(self._guess_chord(notes_here))

        beats_per_measure = 4
        measures: List[List[int]] = []
        for i in range(0, len(columns), beats_per_measure):
            measures.append(list(range(i, min(i + beats_per_measure, len(columns)))))

        measures_per_line = 4
        lines_of_measures: List[List[List[int]]] = []
        for i in range(0, len(measures), measures_per_line):
            lines_of_measures.append(measures[i : i + measures_per_line])

        output: List[str] = []
        string_names = ["e", "B", "G", "D", "A", "E"]

        for line_measures in lines_of_measures:
            chord_line = "    "
            for mi, measure_cols in enumerate(line_measures):
                for ci in measure_cols:
                    lbl = chord_labels[ci] if ci < len(chord_labels) else None
                    if lbl:
                        chord_line += f"{lbl:<4}"
                    else:
                        chord_line += "    "
                if mi < len(line_measures) - 1:
                    chord_line += " "
            chord_line = chord_line.rstrip()
            if chord_line.strip():
                output.append(chord_line)

            for row, string_num in enumerate(range(1, 7)):
                line = f"{string_names[row]}|"
                for mi, measure_cols in enumerate(line_measures):
                    for ci in measure_cols:
                        col = columns[ci] if ci < len(columns) else {}
                        fret_str = col.get(string_num, "-")
                        if len(fret_str) == 1:
                            line += f"-{fret_str}-"
                        else:
                            line += f"{fret_str}-"
                    line += "|"
                output.append(line)
            output.append("")

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
