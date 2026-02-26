"""
Guitar tab generator module
"""
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
        """Initialize tab generator"""
        self.tabs = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self.max_fret = max_fret
        self._last_position: Optional[Tuple[int, int]] = None

    def set_max_fret(self, max_fret: int):
        """Set maximum fret for tab generation."""
        self.max_fret = max_fret

    def note_to_tab_positions(self, note_name: str) -> List[Tuple[int, int]]:
        """Convert note name to possible fretboard positions."""
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
        """Generate tab positions from note list"""
        self.tabs = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self._last_position = None

        for note_info in notes:
            note_name = note_info.get("note")
            if not note_name:
                continue
            positions = self.note_to_tab_positions(note_name)
            string, fret = self._choose_position(positions)
            self._last_position = (string, fret)
            self.tabs[string].append(
                {
                    "fret": fret,
                    "time": note_info.get("start_time", 0.0),
                    "note": note_name,
                    "duration": note_info.get("duration", 0.0),
                }
            )

        return self.tabs

    def _guess_chord(self, notes_at_time: List[str]) -> Optional[str]:
        """Try to guess a chord name from notes sounding at a time slot."""
        if not notes_at_time:
            return None
        # strip octave
        pcs = set()
        for n in notes_at_time:
            pc = n[:-1] if n[-1].isdigit() else n
            pcs.add(pc)
        # simple lookup of common open chords
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
        # try subsets
        for k, v in chord_map.items():
            if k.issubset(fs) and len(k) >= 3:
                return v
        return None

    def format_tabs_as_text(self) -> str:
        """Format tabs as professional-looking guitar tablature with measures."""
        all_events: List[Tuple[float, int, dict]] = []
        for string_num, tab_list in self.tabs.items():
            for tab in tab_list:
                all_events.append((tab["time"], string_num, tab))

        if not all_events:
            return "No tabs generated"

        all_times = sorted({e[0] for e in all_events})

        # build columns: each column is one time-step
        columns: List[Dict[int, str]] = []  # list of {string_num: fret_str}
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

        # split into measures (beats_per_measure columns each)
        beats_per_measure = 4
        measures: List[List[int]] = []  # list of column-index lists
        for i in range(0, len(columns), beats_per_measure):
            measures.append(list(range(i, min(i + beats_per_measure, len(columns)))))

        # split measures into lines (measures_per_line measures per line)
        measures_per_line = 4
        lines_of_measures: List[List[List[int]]] = []
        for i in range(0, len(measures), measures_per_line):
            lines_of_measures.append(measures[i : i + measures_per_line])

        output: List[str] = []
        string_names = ["e", "B", "G", "D", "A", "E"]

        for line_measures in lines_of_measures:
            # chord line
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

            # tab lines per string
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

            output.append("")  # blank line between systems

        return "\n".join(output).rstrip()

    def save_tabs_to_file(self, filename: str):
        """Save tabs to a file"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.format_tabs_as_text())
        print(f"Tabs saved to: {filename}")
