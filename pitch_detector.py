"""
Pitch detection module for note extraction
"""
import numpy as np
import librosa
from scipy.signal import medfilt
from typing import List, Tuple, Optional


class PitchDetector:
    """Class for detecting pitch in audio"""
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize pitch detector
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        
    def detect_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            hop_length=self.hop_length,
            frame_length=2048
        )
        if f0 is None:
            f0 = np.array([], dtype=float)
        if voiced_probs is None:
            voiced_probs = np.ones_like(f0, dtype=float)

        f0 = self._smooth_f0(f0)
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        return f0, times, voiced_probs

    def _smooth_f0(self, f0: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median filter to smooth pitch track."""
        if f0.size == 0:
            return f0
        clean = np.where(np.isfinite(f0), f0, np.nan)
        filled = np.copy(clean)
        if np.all(np.isnan(filled)):
            return f0
        median_val = np.nanmedian(filled)
        filled = np.where(np.isnan(filled), median_val, filled)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = medfilt(filled, kernel_size=kernel_size)
        smoothed = np.where(np.isfinite(clean), smoothed, np.nan)
        return smoothed
    
    def frequency_to_note(self, frequency: Optional[float]) -> Tuple[Optional[str], Optional[int]]:
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
    
    def extract_notes_from_audio(
        self,
        audio: np.ndarray,
        min_duration: float = 0.1,
        min_voiced_prob: float = 0.75,
        merge_gap: float = 0.05,
    ) -> List[dict]:
        """Extract notes from audio signal
        
        Args:
            audio: Audio data
            min_duration: Minimum note duration in seconds
            
        Returns:
            List of note dictionaries
        """
        f0, times, voiced_probs = self.detect_pitch(audio)
        notes = []
        current_note = None
        note_start_time = None
        
        for i, (freq, time) in enumerate(zip(f0, times)):
            voiced_ok = i < len(voiced_probs) and voiced_probs[i] >= min_voiced_prob
            note_name, midi = self.frequency_to_note(freq if voiced_ok else None)
            
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
        
        if current_note is not None and len(times) > 0:
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
        
        return self._merge_adjacent_notes(notes, merge_gap=merge_gap)

    def _merge_adjacent_notes(self, notes: List[dict], merge_gap: float = 0.05) -> List[dict]:
        """Merge adjacent notes of the same pitch separated by small gaps."""
        if not notes:
            return notes

        merged = [notes[0].copy()]
        for note in notes[1:]:
            last = merged[-1]
            gap = note["start_time"] - last["end_time"]
            if note["note"] == last["note"] and gap <= merge_gap:
                last["end_time"] = note["end_time"]
                last["duration"] = last["end_time"] - last["start_time"]
            else:
                merged.append(note.copy())
        return merged