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

Follow the on-screen menu to record from microphone or load an audio file, then optionally save the generated tabs.

## Project Structure

- `main.py` - entry point
- `audio_processor.py` - microphone recording and file loading
- `pitch_detector.py` - pitch detection and note extraction
- `tab_generator.py` - tab generation
- `requirements.txt` - dependencies

## License

MIT
