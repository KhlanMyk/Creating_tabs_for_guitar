"""Pitch detection module for note extraction.

Uses onset detection to find real note attack times, then assigns pitch
to each onset window using pyin.  This preserves the original rhythm and
melody pattern so that synthesised playback sounds close to the source.

Quality features:
  • Percussive signal used for onset detection (sharper transients)
  • Probability-weighted pitch estimation per onset window
  • Octave-error correction across consecutive notes
  • RMS energy gate – filters out very quiet false detections
  • Configurable onset sensitivity
"""
import numpy as np
import librosa
from scipy.signal import medfilt
from typing import List, Tuple, Optional

# Guitar frequency range (MIDI)
_GUITAR_MIDI_LO = 40   # E2
_GUITAR_MIDI_HI = 88   # E6


class PitchDetector:
    """Class for detecting pitch in audio"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self._detect_sr = 22050
        # Minimum RMS energy for a note to be valid (relative to track peak)
        self.min_rms_ratio = 0.012
        # Onset detection sensitivity (lower → more onsets)
        self.onset_delta = 0.06

    # low-level pitch track 
    def detect_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.sample_rate > self._detect_sr:
            audio_ds = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=self._detect_sr)
            sr_used = self._detect_sr
        else:
            audio_ds = audio
            sr_used = self.sample_rate

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_ds,
            fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('E6'),
            sr=sr_used,
            hop_length=self.hop_length,
            frame_length=2048,
        )
        if f0 is None:
            f0 = np.array([], dtype=float)
        if voiced_probs is None:
            voiced_probs = np.ones_like(f0, dtype=float)

        f0 = self._smooth_f0(f0)
        times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr_used, hop_length=self.hop_length
        )
        return f0, times, voiced_probs

    def _smooth_f0(self, f0: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Light median filter to remove single-frame pitch glitches.

        Kernel=3 removes isolated outliers without smearing adjacent notes.
        """
        if f0.size < 3:
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
        if frequency is None or np.isnan(frequency) or frequency <= 0:
            return None, None
        midi_number = librosa.hz_to_midi(frequency)
        note_name = librosa.midi_to_note(int(round(midi_number)))
        return note_name, int(round(midi_number))

    def _detect_onsets(
        self,
        audio: np.ndarray,
        sr: int,
        percussive: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return onset times (seconds) using librosa onset detection.

        If *percussive* is given it is used for the onset strength
        envelope (sharper transients → more accurate onsets).
        """
        odf_audio = percussive if percussive is not None else audio
        onset_env = librosa.onset.onset_strength(
            y=odf_audio, sr=sr, hop_length=self.hop_length,
        )
        # Use backtrack + reasonable wait to avoid double-triggers
        min_wait = max(1, int(sr * 0.04 / self.hop_length))
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, hop_length=self.hop_length,
            onset_envelope=onset_env,
            backtrack=True, units="frames",
            delta=self.onset_delta,
            wait=min_wait,
            pre_max=3, post_max=3,
            pre_avg=3, post_avg=5,
        )
        if onset_frames.size == 0:
            return np.array([0.0], dtype=np.float64)
        times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        return times.astype(np.float64)

    # main extraction entry point
    def extract_notes_from_audio(
        self,
        audio: np.ndarray,
        min_duration: float = 0.05,
        min_voiced_prob: float = 0.25,
        merge_gap: float = 0.04,
        use_harmonic: bool = True,
        segment_seconds: float | None = None,
        progress_callback: object = None,
        use_onset_alignment: bool = True,
    ) -> List[dict]:
        """Extract notes preserving real onset timing.

        When *use_onset_alignment* is True (default) the method first
        detects onsets in the audio and then determines pitch for each
        onset window.  This produces notes whose start_time / duration
        reflect the **actual rhythm** of the source, not uniform spacing.
        """
        if use_onset_alignment:
            return self._extract_onset_aligned(
                audio,
                min_duration=min_duration,
                min_voiced_prob=min_voiced_prob,
                merge_gap=merge_gap,
                use_harmonic=use_harmonic,
                progress_callback=progress_callback,
            )

        # Legacy path (segment-based without onset alignment)
        if segment_seconds and segment_seconds > 0:
            hop = int(self.sample_rate * segment_seconds)
            starts = list(range(0, len(audio), hop))
            total = len(starts)
            all_notes: List[dict] = []
            for idx, start in enumerate(starts):
                segment = audio[start:start + hop]
                if segment.size < 2048:
                    continue
                if progress_callback:
                    progress_callback(idx / max(total, 1), f"Segment {idx+1}/{total}")
                offset = start / self.sample_rate
                segment_notes = self._extract_notes_from_segment(
                    segment, min_duration=min_duration,
                    min_voiced_prob=min_voiced_prob,
                    time_offset=offset, use_harmonic=use_harmonic,
                )
                all_notes.extend(segment_notes)
            if progress_callback:
                progress_callback(1.0, "Merging notes...")
            return self._merge_adjacent_notes(all_notes, merge_gap=merge_gap)

        notes = self._extract_notes_from_segment(
            audio, min_duration=min_duration,
            min_voiced_prob=min_voiced_prob,
            time_offset=0.0, use_harmonic=use_harmonic,
        )
        return self._merge_adjacent_notes(notes, merge_gap=merge_gap)

    def _extract_onset_aligned(
        self,
        audio: np.ndarray,
        min_duration: float,
        min_voiced_prob: float,
        merge_gap: float,
        use_harmonic: bool,
        progress_callback,
    ) -> List[dict]:
        """Detect onsets first, then assign pitch to each onset window.

        Quality improvements over the basic approach:
          1. Percussive component drives onset detection (sharper)
          2. Probability-weighted pitch per window (not plain median)
          3. RMS energy gate discards very quiet false notes
          4. Post-hoc octave-error correction
        """
        sr = self.sample_rate
        if sr > self._detect_sr:
            audio_ds = librosa.resample(audio, orig_sr=sr, target_sr=self._detect_sr)
            sr_used = self._detect_sr
        else:
            audio_ds = audio
            sr_used = self.sample_rate

        if progress_callback:
            progress_callback(0.05, "Separating harmonics…")

        # Harmonic/percussive separation
        harmonic, percussive = librosa.effects.hpss(audio_ds)
        analysis_audio = harmonic if use_harmonic else audio_ds

        if progress_callback:
            progress_callback(0.10, "Detecting onsets…")

        # 1. Detect onsets – use percussive component for sharper transients
        onset_times = self._detect_onsets(analysis_audio, sr_used,
                                          percussive=percussive)

        # Ensure t=0 is an onset if the audio starts with audible content
        if len(onset_times) == 0 or onset_times[0] > 0.05:
            early_rms = float(np.sqrt(np.mean(audio_ds[:int(0.05 * sr_used)] ** 2))) \
                if len(audio_ds) > int(0.02 * sr_used) else 0.0
            if early_rms > 1e-5:
                onset_times = np.concatenate([[0.0], onset_times])

        if progress_callback:
            progress_callback(0.20, "Tracking pitch…")

        # 2. Run full pitch track on harmonic content
        f0, voiced_flag, voiced_probs = librosa.pyin(
            analysis_audio,
            fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('E6'),
            sr=sr_used,
            hop_length=self.hop_length,
            frame_length=2048,
        )
        if f0 is None:
            f0 = np.array([], dtype=float)
        if voiced_probs is None:
            voiced_probs = np.ones_like(f0, dtype=float)

        f0 = self._smooth_f0(f0)
        pitch_times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr_used, hop_length=self.hop_length
        )

        if progress_callback:
            progress_callback(0.50, "Aligning notes to onsets…")

        # Compute global RMS for energy gating
        global_rms = float(np.sqrt(np.mean(audio_ds ** 2))) if len(audio_ds) > 0 else 1e-6
        rms_threshold = global_rms * self.min_rms_ratio

        # 3. For each onset window, find the dominant pitch
        audio_duration = len(audio_ds) / sr_used
        notes: List[dict] = []
        total_onsets = len(onset_times)

        for i, onset_t in enumerate(onset_times):
            if progress_callback and i % max(1, total_onsets // 20) == 0:
                frac = 0.50 + 0.40 * (i / max(total_onsets, 1))
                progress_callback(frac, f"Note {i+1}/{total_onsets}")

            next_t = onset_times[i + 1] if i + 1 < total_onsets else audio_duration
            duration = next_t - onset_t

            if duration < min_duration:
                continue

            onset_sample = int(onset_t * sr_used)
            end_sample = min(int(next_t * sr_used), len(audio_ds))
            window_audio = audio_ds[onset_sample:end_sample]
            rms = float(np.sqrt(np.mean(window_audio ** 2))) if len(window_audio) > 0 else 0.0
            if rms < rms_threshold:
                continue  # too quiet, likely noise

            # Focus pitch analysis on the early part of the window
            # (first 70% or max 0.3s) where the current note is strongest,
            # before it decays and the next note's onset encroaches.
            pitch_end_t = onset_t + min(duration * 0.7, 0.3)
            pitch_end_t = max(pitch_end_t, onset_t + min(duration, 0.08))  # at least 80ms

            mask = (pitch_times >= onset_t) & (pitch_times < pitch_end_t)
            window_f0 = f0[mask]
            window_probs = voiced_probs[mask] if len(voiced_probs) > 0 else np.array([])

            if len(window_f0) == 0:
                # Fall back to full window if early window has no data
                mask = (pitch_times >= onset_t) & (pitch_times < next_t)
                window_f0 = f0[mask]
                window_probs = voiced_probs[mask] if len(voiced_probs) > 0 else np.array([])

            if len(window_f0) == 0:
                continue

            valid = np.isfinite(window_f0)
            if len(window_probs) == len(window_f0):
                valid = valid & (window_probs >= min_voiced_prob)

            valid_freqs = window_f0[valid]
            valid_weights = window_probs[valid] if len(window_probs) == len(window_f0) else None
            if len(valid_freqs) == 0:
                continue

            # Use pitch MODE (most common semitone) rather than weighted
            # average, which blurs notes when a window spans two pitches.
            valid_midi_raw = librosa.hz_to_midi(valid_freqs)
            rounded_midis = np.round(valid_midi_raw).astype(int)

            # Find the most frequent MIDI note in the window, weighted by
            # voiced probability if available
            unique_midis, counts = np.unique(rounded_midis, return_counts=True)
            if valid_weights is not None and len(valid_weights) == len(rounded_midis):
                # Sum probability weights for each unique MIDI note
                weighted_counts = np.zeros_like(counts, dtype=float)
                for j, um in enumerate(unique_midis):
                    mask_um = rounded_midis == um
                    weighted_counts[j] = float(np.sum(valid_weights[mask_um]))
                best_idx = int(np.argmax(weighted_counts))
            else:
                best_idx = int(np.argmax(counts))

            rounded_midi = int(unique_midis[best_idx])

            # Clamp to guitar range
            rounded_midi = int(np.clip(rounded_midi, _GUITAR_MIDI_LO, _GUITAR_MIDI_HI))
            median_freq = float(librosa.midi_to_hz(rounded_midi))

            note_name = librosa.midi_to_note(rounded_midi)
            if note_name is None:
                continue

            notes.append({
                "note": note_name,
                "midi": rounded_midi,
                "frequency": median_freq,
                "start_time": float(onset_t),
                "end_time": float(next_t),
                "duration": float(duration),
                "velocity": float(np.clip(rms / max(global_rms, 1e-9), 0.1, 1.0)),
                "onset": float(onset_t),
            })

        if progress_callback:
            progress_callback(0.92, "Correcting octave errors…")

        notes = self._correct_octave_errors(notes)

        if progress_callback:
            progress_callback(0.95, "Merging notes…")

        merged = self._merge_adjacent_notes(notes, merge_gap=merge_gap)

        if progress_callback:
            progress_callback(1.0, f"Done — {len(merged)} notes")

        return merged

    @staticmethod
    def _correct_octave_errors(notes: List[dict], window: int = 5) -> List[dict]:
        """Fix isolated octave jumps (+/- 12 MIDI) relative to local context.

        An octave error is detected when a note's MIDI value deviates by
        roughly 12 semitones from the local median and shifting it by
        ±12 brings it much closer.
        """
        if len(notes) < 3:
            return notes

        midis = np.array([n["midi"] for n in notes], dtype=float)

        for i in range(len(notes)):
            lo = max(0, i - window)
            hi = min(len(notes), i + window + 1)
            neighbours = np.concatenate([midis[lo:i], midis[i+1:hi]])
            if len(neighbours) == 0:
                continue
            local_med = float(np.median(neighbours))
            cur = midis[i]
            dist_cur = abs(cur - local_med)

            # Only fix if distance is close to an octave (10-14 semitones)
            if dist_cur < 9:
                continue

            for shift in [12, -12]:
                candidate = cur + shift
                if _GUITAR_MIDI_LO <= candidate <= _GUITAR_MIDI_HI:
                    if abs(candidate - local_med) < dist_cur * 0.6:
                        new_midi = int(round(candidate))
                        notes[i] = dict(notes[i])  # copy
                        notes[i]["midi"] = new_midi
                        notes[i]["note"] = librosa.midi_to_note(new_midi)
                        notes[i]["frequency"] = float(librosa.midi_to_hz(new_midi))
                        midis[i] = new_midi
                        break

        return notes

    #legacy segment extraction
    def _extract_notes_from_segment(
        self,
        audio: np.ndarray,
        min_duration: float,
        min_voiced_prob: float,
        time_offset: float,
        use_harmonic: bool,
    ) -> List[dict]:
        processed = librosa.effects.harmonic(audio) if use_harmonic else audio
        f0, times, voiced_probs = self.detect_pitch(processed)
        notes: List[dict] = []
        current_note = None
        note_start_time = None
        current_midi = None
        current_freq = None

        for i, (freq, time) in enumerate(zip(f0, times)):
            voiced_ok = i < len(voiced_probs) and voiced_probs[i] >= min_voiced_prob
            note_name, midi = self.frequency_to_note(freq if voiced_ok else None)

            if note_name is None:
                if current_note is not None:
                    duration = time - note_start_time
                    if duration >= min_duration:
                        notes.append({
                            "note": current_note, "midi": current_midi,
                            "frequency": current_freq,
                            "start_time": note_start_time + time_offset,
                            "end_time": time + time_offset,
                            "duration": duration,
                        })
                    current_note = None
            else:
                if current_note != note_name:
                    if current_note is not None:
                        duration = time - note_start_time
                        if duration >= min_duration:
                            notes.append({
                                "note": current_note, "midi": current_midi,
                                "frequency": current_freq,
                                "start_time": note_start_time + time_offset,
                                "end_time": time + time_offset,
                                "duration": duration,
                            })
                    current_note = note_name
                    current_midi = midi
                    current_freq = freq
                    note_start_time = time

        if current_note is not None and len(times) > 0:
            duration = times[-1] - note_start_time
            if duration >= min_duration:
                notes.append({
                    "note": current_note, "midi": current_midi,
                    "frequency": current_freq,
                    "start_time": note_start_time + time_offset,
                    "end_time": times[-1] + time_offset,
                    "duration": duration,
                })
        return notes

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