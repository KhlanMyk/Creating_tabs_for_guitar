"""Main file for Guitar Tab Generator."""
from audio_processor import AudioProcessor
from pitch_detector import PitchDetector
from tab_generator import GuitarTabGenerator


def main():
    """Run the CLI application."""
    print("=" * 60)
    print("GUITAR TAB GENERATOR")
    print("Audio to Guitar Tabs Converter")
    print("=" * 60)

    audio_proc = AudioProcessor()
    pitch_det = PitchDetector()
    tab_gen = GuitarTabGenerator()

    print("\nSelect mode:")
    print("1. Record from microphone")
    print("2. Load audio file")

    choice = input("\nYour choice (1 or 2): ").strip()
    audio = None

    if choice == "1":
        try:
            duration = float(input("Recording duration in seconds: "))
        except ValueError:
            print("Invalid duration.")
            return
        audio = audio_proc.record_from_microphone(duration)
    elif choice == "2":
        file_path = input("Audio file path: ").strip()
        audio, _ = audio_proc.load_audio_file(file_path)
    else:
        print("Invalid choice.")
        return

    if audio is None:
        print("Audio loading error.")
        return

    print("\nAnalyzing audio...")
    notes = pitch_det.extract_notes_from_audio(audio)
    print(f"Notes found: {len(notes)}")

    print("\nGenerating tabs...")
    tab_gen.generate_tabs(notes)
    print("\n" + tab_gen.format_tabs_as_text())

    save = input("\nSave tabs to file? (y/n): ").strip().lower()
    if save == "y":
        filename = input("Filename (default: tabs.txt): ").strip() or "tabs.txt"
        tab_gen.save_tabs_to_file(filename)

    print("\nDone!")


if __name__ == "__main__":
    main()
