"""Main file"""
from audio_processor import AudioProcessor
from pitch_detector import PitchDetector
from tab_generator import GuitarTabGenerator

def main():
    print("Guitar Tab Generator")
    AudioProcessor()

if __name__ == "__main__":
    main()
