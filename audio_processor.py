"""
Audio processing module for microphone recording and file loading
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
from typing import Tuple, Optional


class AudioProcessor:
    """Class for processing audio data"""
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize audio processor
        
        Args:
            sample_rate: Sample rate in Hz (default 44100)
        """
        self.sample_rate = sample_rate
        self.recording = None
        
    def record_from_microphone(self, duration: float) -> np.ndarray:
        """Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Numpy array with audio data
        """
        print(f"Recording {duration} seconds...")
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("Recording complete!")
        self.recording = recording.flatten()
        return self.recording
    
    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple (audio data, sample rate)
        """
        print(f"Loading file: {file_path}")
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        self.recording = audio
        print(f"File loaded. Duration: {len(audio)/sr:.2f} seconds")
        return audio, sr
    
    def save_audio(self, file_path: str, audio_data: Optional[np.ndarray] = None):
        """Save audio to file
        
        Args:
            file_path: Path to save
            audio_data: Audio data to save
        """
        if audio_data is None:
            audio_data = self.recording
        if audio_data is None:
            raise ValueError("No data to save")
        sf.write(file_path, audio_data, self.sample_rate)
        print(f"Audio saved: {file_path}")