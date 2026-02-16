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

    def format_tabs_as_text(self) -> str:
        """Format tabs as text"""
        output = []
        output.append("=" * 60)
        output.append("GUITAR TABS")
        output.append("=" * 60)

        all_times = sorted({tab["time"] for string in self.tabs.values() for tab in string})
        if not all_times:
            return "No tabs generated"

        strings_display = {i: [] for i in range(1, 7)}
        for time in all_times:
            for string in range(1, 7):
                match = next((tab for tab in self.tabs[string] if abs(tab["time"] - time) < 0.01), None)
                strings_display[string].append(str(match["fret"]) if match else "-")

        string_names = ["e", "B", "G", "D", "A", "E"]
        for i, string in enumerate(range(1, 7)):
            line = f"{string_names[i]}|"
            line += "-".join(f"{tab:>2}" for tab in strings_display[string])
            line += "|"
            output.append(line)

        output.append("=" * 60)
        return "\n".join(output)

    def save_tabs_to_file(self, filename: str):
        """Save tabs to a file"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.format_tabs_as_text())
        print(f"Tabs saved to: {filename}")
