"""
Guitar tab generator module
"""
from typing import List, Dict, Tuple


class GuitarTabGenerator:
    """Class for generating guitar tabs"""

    STANDARD_TUNING = {
        "E2": (6, 0),
        "F2": (6, 1),
        "F#2": (6, 2),
        "G2": (6, 3),
        "A2": (5, 0),
        "B2": (5, 2),
        "C3": (5, 3),
        "D3": (4, 0),
        "E3": (4, 2),
        "F3": (4, 3),
        "G3": (3, 0),
        "A3": (3, 2),
        "B3": (2, 0),
        "C4": (2, 1),
        "D4": (2, 3),
        "E4": (1, 0),
        "F4": (1, 1),
        "G4": (1, 3),
        "A4": (1, 5),
    }

    def __init__(self):
        """Initialize tab generator"""
        self.tabs = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

    def note_to_tab_position(self, note_name: str) -> List[Tuple[int, int]]:
        """Convert note name to fretboard position"""
        if note_name in self.STANDARD_TUNING:
            return [self.STANDARD_TUNING[note_name]]
        return [(1, 0)]

    def generate_tabs(self, notes: List[Dict]) -> Dict[int, List]:
        """Generate tab positions from note list"""
        self.tabs = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

        for note_info in notes:
            note_name = note_info.get("note")
            if not note_name:
                continue
            positions = self.note_to_tab_position(note_name)
            string, fret = positions[0]
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
