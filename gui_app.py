"""Tkinter GUI for Guitar Tab Generator."""
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from audio_processor import AudioProcessor
from pitch_detector import PitchDetector
from tab_generator import GuitarTabGenerator
from self_test import run_sine_test


class GuitarTabApp(tk.Tk):
    """Simple GUI app for generating guitar tabs."""

    def __init__(self):
        super().__init__()
        self.title("Guitar Tab Generator")
        self.geometry("900x600")

        self.audio_proc = AudioProcessor()
        self.pitch_det = PitchDetector()
        self.tab_gen = GuitarTabGenerator()

        self.audio_data = None

        self._build_ui()

    def _build_ui(self):
        control_frame = tk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Duration (sec):").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="5")
        tk.Entry(control_frame, textvariable=self.duration_var, width=6).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Min duration:").pack(side=tk.LEFT)
        self.min_duration_var = tk.StringVar(value="0.1")
        tk.Entry(control_frame, textvariable=self.min_duration_var, width=6).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Min voiced prob:").pack(side=tk.LEFT)
        self.min_voiced_var = tk.StringVar(value="0.75")
        tk.Entry(control_frame, textvariable=self.min_voiced_var, width=6).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Max fret:").pack(side=tk.LEFT)
        self.max_fret_var = tk.StringVar(value="15")
        tk.Entry(control_frame, textvariable=self.max_fret_var, width=4).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Segment (sec):").pack(side=tk.LEFT)
        self.segment_var = tk.StringVar(value="15")
        tk.Entry(control_frame, textvariable=self.segment_var, width=5).pack(side=tk.LEFT, padx=5)

        self.use_harmonic_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Use harmonic", variable=self.use_harmonic_var).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="Record", command=self.on_record).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Load File", command=self.on_load).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Load + Generate", command=self.on_load_and_generate).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Analyze", command=self.on_analyze).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Save Tabs", command=self.on_save).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Run Test", command=self.on_test).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=10)

        self.output = scrolledtext.ScrolledText(self, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    def on_record(self):
        try:
            duration = float(self.duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Duration must be a number.")
            return

        def task():
            self.set_status("Recording...")
            try:
                self.audio_data = self.audio_proc.record_from_microphone(duration)
                self.set_status("Recording complete")
            except Exception as exc:
                self.set_status("Recording failed")
                messagebox.showerror("Error", str(exc))

        threading.Thread(target=task, daemon=True).start()

    def on_load(self):
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        self.set_status("Loading file...")
        try:
            audio, _ = self.audio_proc.load_audio_file(file_path)
            self.audio_data = audio
            self.set_status("File loaded")
        except Exception as exc:
            self.set_status("Load failed")
            messagebox.showerror("Error", str(exc))

    def on_load_and_generate(self):
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        self.set_status("Loading file...")
        try:
            audio, _ = self.audio_proc.load_audio_file(file_path)
            self.audio_data = audio
            self.set_status("File loaded")
            self.on_analyze()
        except Exception as exc:
            self.set_status("Load failed")
            messagebox.showerror("Error", str(exc))

    def on_analyze(self):
        if self.audio_data is None:
            messagebox.showwarning("No audio", "Please record or load audio first.")
            return

        try:
            min_duration = float(self.min_duration_var.get())
            min_voiced = float(self.min_voiced_var.get())
            max_fret = int(self.max_fret_var.get())
            segment_seconds = float(self.segment_var.get()) if self.segment_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Invalid input", "Check analysis parameters.")
            return

        def task():
            self.set_status("Analyzing audio...")
            try:
                self.tab_gen.set_max_fret(max_fret)
                notes = self.pitch_det.extract_notes_from_audio(
                    self.audio_data,
                    min_duration=min_duration,
                    min_voiced_prob=min_voiced,
                    use_harmonic=self.use_harmonic_var.get(),
                    segment_seconds=segment_seconds,
                )
                self.tab_gen.generate_tabs(notes)
                tabs_text = self.tab_gen.format_tabs_as_text()
                self.output.delete("1.0", tk.END)
                self.output.insert(tk.END, tabs_text)
                self.set_status(f"Done. Notes: {len(notes)}")
            except Exception as exc:
                self.set_status("Analysis failed")
                messagebox.showerror("Error", str(exc))

        threading.Thread(target=task, daemon=True).start()

    def on_test(self):
        result = run_sine_test(self.pitch_det)
        message = (
            f"Expected: {result.expected_note}\n"
            f"Detected: {result.detected_note}\n"
            f"Notes: {result.detected_count}\n"
            f"Success: {result.success}"
        )
        messagebox.showinfo("Sine test", message)

    def on_save(self):
        if not self.output.get("1.0", tk.END).strip():
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save tabs",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
        )
        if not file_path:
            return
        try:
            self.tab_gen.save_tabs_to_file(file_path)
            self.set_status("Tabs saved")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    app = GuitarTabApp()
    app.mainloop()
