# Guitar Tab Generator

Minimal Python project that records or loads audio, detects pitch, and converts it into guitar tabs.

## Features

- Record from microphone
- Load audio files (MP3, WAV, FLAC, etc.)
- Detect notes from audio
- Generate basic guitar tabs

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

### CLI examples

```bash
python main.py --input song.mp3 --output tabs.txt
python main.py --duration 5 --output tabs.txt
python main.py --min-duration 0.08 --min-voiced-prob 0.8 --max-fret 12
python main.py --input song.mp3 --use-harmonic --segment-seconds 15
python main.py --test
```

### GUI

```bash
python main.py --gui
```

The GUI includes a **Run Test** button to verify detection on a 440 Hz sine wave.

Follow the on-screen menu to record from microphone or load an audio file, then optionally save the generated tabs.

## Troubleshooting

**GUI does not start (ModuleNotFoundError: _tkinter)**

Your Python build does not include Tk. Install Python from https://www.python.org/downloads/, then recreate the venv and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --gui
```

## Project Structure

- `main.py` - entry point
- `audio_processor.py` - microphone recording and file loading
- `pitch_detector.py` - pitch detection and note extraction
- `tab_generator.py` - tab generation
- `gui_app.py` - Tkinter GUI
- `self_test.py` - sine wave self-test
- `requirements.txt` - dependencies

## License

MIT
