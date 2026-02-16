"""
Pitch detection module for note extraction
"""
import numpy as np
import librosa
from typing import List, Tuple


class PitchDetector:
    """Class for detecting pitch in audio"""
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize pitch detector
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        
    def detect_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect pitch in audio signal
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple (frequencies, times)
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('E6'),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        return f0, times
    
    def frequency_to_note(self, frequency: float) -> Tuple[str, int]:
        """Convert frequency to note
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Tuple (note name, MIDI number)
        """
        if frequency is None or np.isnan(frequency) or frequency <= 0:
            return None, None
        midi_number = librosa.hz_to_midi(frequency)
        note_name = librosa.midi_to_note(int(round(midi_number)))
        return note_name, int(midi_number)
    
    def extract_notes_from_audio(self, audio: np.ndarray, min_duration: float = 0.1) -> List[dict]:
        """Extract notes from audio signal
        
        Args:
            audio: Audio data
            min_duration: Minimum note duration in seconds
            
        Returns:
            List of note dictionaries
        """
        f0, times = self.detect_pitch(audio)
        notes = []
        current_note = None
        note_start_time = None
        
        for i, (freq, time) in enumerate(zip(f0, times)):
            note_name, midi = self.frequency_to_note(freq)
            
            if note_name is None:
                if current_note is not None:
                    duration = time - note_start_time
                    if duration >= min_duration:
                        notes.append({
                            "note": current_note,
                            "midi": current_midi,
                            "frequency": current_freq,
                            "start_time": note_start_time,
                            "end_time": time,
                            "duration": duration
                        })
                    current_note = None
            else:
                if current_note != note_name:
                    if current_note is not None:
                        duration = time - note_start_time
                        if duration >= min_duration:
                            notes.append({
                                "note": current_note,
                                "midi": current_midi,
                                "frequency": current_freq,
                                "start_time": note_start_time,
                                "end_time": time,
                                "duration": duration
                            })
                    current_note = note_name
                    current_midi = midi
                    current_freq = freq
                    note_start_time = time
        
        if current_note is not None:
            duration = times[-1] - note_start_time
            if duration >= min_duration:
                notes.append({
                    "note": current_note,
                    "midi": current_midi,
                    "frequency": current_freq,
                    "start_time": note_start_time,
                    "end_time": times[-1],
                    "duration": duration
                })
        
        return notes