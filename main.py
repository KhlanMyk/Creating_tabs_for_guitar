"""Main file for Guitar Tab Generator."""
import argparse
import sys

from audio_processor import AudioProcessor
from auto_tune import find_best_extraction
from pitch_detector import PitchDetector
from tab_generator import GuitarTabGenerator
from self_test import run_sine_test
from tab_synth import synthesize_from_tabs_file


def main():
    """Run the CLI application."""
    parser = argparse.ArgumentParser(description="Guitar Tab Generator")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI")
    parser.add_argument("--input", type=str, help="Path to audio file")
    parser.add_argument("--duration", type=float, help="Record duration (seconds)")
    parser.add_argument("--output", type=str, help="Output tabs filename")
    parser.add_argument("--min-duration", type=float, default=0.1, help="Minimum note duration")
    parser.add_argument(
        "--min-voiced-prob",
        type=float,
        default=0.75,
        help="Minimum voiced probability",
    )
    parser.add_argument("--max-fret", type=int, default=15, help="Max fret for tabs")
    parser.add_argument("--use-harmonic", action="store_true", help="Use harmonic component")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="Process audio in segments (seconds)",
    )
    parser.add_argument("--test", action="store_true", help="Run sine test")
    parser.add_argument("--auto-tune", action="store_true", help="Auto-tune extraction parameters")
    parser.add_argument("--synth-tabs", type=str, help="Path to tabs txt for audio synthesis")
    parser.add_argument("--synth-output", type=str, default="tabs_synth.wav", help="Output wav path")
    parser.add_argument("--play-synth", action="store_true", help="Play synthesized tabs audio")
    args = parser.parse_args()

    if args.gui:
        try:
            from gui_app import GuitarTabApp
        except ModuleNotFoundError as exc:
            if str(exc).find("_tkinter") != -1:
                print("Tkinter is not available in this Python build.")
                print("Please install Python from https://www.python.org/downloads/")
                print("Then recreate the venv and run: python main.py --gui")
                return
            raise

        app = GuitarTabApp()
        app.mainloop()
        return

    pitch_det = PitchDetector()
    if args.test:
        result = run_sine_test(pitch_det)
        print(
            f"Test expected {result.expected_note}, detected {result.detected_note}, "
            f"notes={result.detected_count}, success={result.success}"
        )
        return

    if args.synth_tabs:
        result = synthesize_from_tabs_file(
            tabs_path=args.synth_tabs,
            output_path=args.synth_output,
            play=args.play_synth,
        )
        print(
            f"Synth complete: notes={result.notes_count}, "
            f"duration={result.duration:.2f}s, output={args.synth_output}"
        )
        return

    print("=" * 60)
    print("GUITAR TAB GENERATOR")
    print("Audio to Guitar Tabs Converter")
    print("=" * 60)

    audio_proc = AudioProcessor()
    tab_gen = GuitarTabGenerator(max_fret=args.max_fret)
    audio = None

    if args.input:
        audio, _ = audio_proc.load_audio_file(args.input)
    elif args.duration:
        audio = audio_proc.record_from_microphone(args.duration)
    else:
        print("\nSelect mode:")
        print("1. Record from microphone")
        print("2. Load audio file")

        choice = input("\nYour choice (1 or 2): ").strip()
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
    if args.auto_tune:
        tune = find_best_extraction(audio, pitch_det, use_harmonic=True)
        notes = tune.notes
        print(
            "Auto-tune: "
            f"min_duration={tune.min_duration}, "
            f"min_voiced_prob={tune.min_voiced_prob}, "
            f"segment_seconds={tune.segment_seconds}"
        )
    else:
        notes = pitch_det.extract_notes_from_audio(
            audio,
            min_duration=args.min_duration,
            min_voiced_prob=args.min_voiced_prob,
            use_harmonic=args.use_harmonic,
            segment_seconds=args.segment_seconds,
        )
    print(f"Notes found: {len(notes)}")

    print("\nGenerating tabs...")
    tab_gen.generate_tabs(notes)
    print("\n" + tab_gen.format_tabs_as_text())

    if args.output:
        tab_gen.save_tabs_to_file(args.output)
    else:
        save = input("\nSave tabs to file? (y/n): ").strip().lower()
        if save == "y":
            filename = input("Filename (default: tabs.txt): ").strip() or "tabs.txt"
            tab_gen.save_tabs_to_file(filename)

    print("\nDone!")


if __name__ == "__main__":
    main()
