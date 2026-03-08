"""Quick quality test for the pitch detection + tab generation pipeline."""
import numpy as np
from pitch_detector import PitchDetector
from tab_generator import GuitarTabGenerator

def make_guitar_tone(freq, duration, sr=22050):
    """Generate a guitar-like tone with harmonics and exponential decay."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * freq * t)
            + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.1 * np.sin(2 * np.pi * 3 * freq * t))
    envelope = np.exp(-2.5 * t)
    return tone * envelope

def test_melody():
    det = PitchDetector(sample_rate=22050)
    sr = 22050

    # 5-note melody: E4 G4 A4 B4 E5
    melody = [("E4", 329.63), ("G4", 392.0), ("A4", 440.0), ("B4", 493.88), ("E5", 659.26)]
    audio = np.zeros(0)
    note_dur = 0.4
    gap_dur = 0.02

    for name, freq in melody:
        tone = make_guitar_tone(freq, note_dur, sr)
        audio = np.concatenate([audio, tone])
        audio = np.concatenate([audio, np.zeros(int(sr * gap_dur))])

    # Test onset-aligned extraction
    notes = det.extract_notes_from_audio(
        audio, use_onset_alignment=True, min_voiced_prob=0.2, min_duration=0.05,
    )
    print(f"Detected {len(notes)} notes (expected {len(melody)}):")
    matched = 0
    for i, n in enumerate(notes):
        expected = melody[i][0] if i < len(melody) else "?"
        ok = n["note"] == expected
        if ok:
            matched += 1
        sym = "✓" if ok else "✗"
        print(f"  {n['note']:>4} (expected {expected:>4})  t={n['start_time']:.3f}s  dur={n['duration']:.3f}s  {sym}")
    print(f"\n{matched}/{len(melody)} notes matched correctly")

    # Generate tabs
    gen = GuitarTabGenerator()
    gen.generate_tabs(notes)
    tabs = gen.format_tabs_as_text()
    print(f"\nGenerated tabs:\n{tabs}")

    # Verify fret positions make sense
    events = gen.get_timed_events()
    print("\nTimed events:")
    for e in events:
        print(f"  s{e['string']} f{e['fret']:>2} {e['note']:>4} t={e['start_time']:.3f}s dur={e['duration']:.3f}s")


def test_scale():
    """Test a full octave scale: E3 F3 G3 A3 B3 C4 D4 E4."""
    det = PitchDetector(sample_rate=22050)
    sr = 22050
    import librosa as _lr

    scale_notes = ["E3", "F3", "G3", "A3", "B3", "C4", "D4", "E4"]
    audio = np.zeros(0)
    note_dur = 0.35

    for name in scale_notes:
        freq = float(_lr.note_to_hz(name))
        tone = make_guitar_tone(freq, note_dur, sr)
        audio = np.concatenate([audio, tone, np.zeros(int(sr * 0.015))])

    notes = det.extract_notes_from_audio(
        audio, use_onset_alignment=True, min_voiced_prob=0.2, min_duration=0.04,
    )
    print(f"Detected {len(notes)} notes (expected {len(scale_notes)}):")
    matched = 0
    for i, n in enumerate(notes):
        expected = scale_notes[i] if i < len(scale_notes) else "?"
        ok = n["note"] == expected
        if ok:
            matched += 1
        sym = "✓" if ok else "✗"
        print(f"  {n['note']:>4} (expected {expected:>4})  {sym}")
    print(f"\n{matched}/{len(scale_notes)} scale notes matched")

    gen = GuitarTabGenerator()
    gen.generate_tabs(notes)
    print(f"\nScale tabs:\n{gen.format_tabs_as_text()}")

def test_single_notes():
    """Test each guitar string open note."""
    det = PitchDetector(sample_rate=22050)
    sr = 22050
    test_freqs = {
        "E2": 82.41, "A2": 110.0, "D3": 146.83,
        "G3": 196.0, "B3": 246.94, "E4": 329.63,
        "A4": 440.0, "E5": 659.26,
    }
    passed = 0
    total = 0
    for expected, freq in test_freqs.items():
        t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        notes = det.extract_notes_from_audio(
            audio, use_onset_alignment=False, min_voiced_prob=0.2,
        )
        detected = notes[0]["note"] if notes else "None"
        ok = detected == expected
        total += 1
        if ok:
            passed += 1
        sym = "✓" if ok else "✗"
        print(f"  {expected:>4}: detected={detected:<6} {sym}")
    print(f"\n{passed}/{total} single-note tests passed")

if __name__ == "__main__":
    print("=== Single Note Tests ===")
    test_single_notes()
    print("\n=== Melody Test ===")
    test_melody()
    print("\n=== Scale Test ===")
    test_scale()
