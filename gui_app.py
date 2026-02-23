"""Tkinter GUI for Guitar Tab Generator."""
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from audio_processor import AudioProcessor
from auto_tune import find_best_extraction
from pitch_detector import PitchDetector
from synth_matcher import optimize_synth_against_original
from tab_checker import check_tabs_against_original
from tab_generator import GuitarTabGenerator
from tab_refiner import refine_tabs_with_original
from self_test import run_sine_test
from tab_synth import synthesize_from_tabs_text


class GuitarTabApp(tk.Tk):
    """Simple GUI app for generating guitar tabs."""

    def __init__(self):
        super().__init__()
        self.title("Guitar Tab Generator")
        self.geometry("1100x700")
        self.minsize(980, 620)

        self._apply_theme()

        self.audio_proc = AudioProcessor()
        self.pitch_det = PitchDetector()
        self.tab_gen = GuitarTabGenerator()

        self.audio_data = None

        self._build_ui()

    def _apply_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        bg = "#0f172a"  # slate-900
        panel = "#111827"  # gray-900
        accent = "#38bdf8"  # sky-400
        text = "#e2e8f0"  # slate-200
        muted = "#94a3b8"  # slate-400

        self.configure(background=bg)
        style.configure("TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Muted.TLabel", background=bg, foreground=muted)
        style.configure(
            "TButton",
            background=panel,
            foreground=text,
            padding=(10, 6),
            focusthickness=3,
            focuscolor=accent,
        )
        style.map(
            "TButton",
            background=[("active", "#1f2937"), ("pressed", "#0b1220")],
            foreground=[("active", "#ffffff")],
        )
        style.configure("Accent.TButton", background=accent, foreground="#0b1120")
        style.map(
            "Accent.TButton",
            background=[("active", "#7dd3fc"), ("pressed", "#0ea5e9")],
            foreground=[("active", "#0b1120")],
        )
        style.configure("TEntry", fieldbackground="#0b1220", foreground=text)
        style.configure("TCheckbutton", background=bg, foreground=text)

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=16, pady=(16, 8))

        title = ttk.Label(header, text="Guitar Tab Generator", font=("Helvetica", 18, "bold"))
        title.pack(side=tk.LEFT)
        subtitle = ttk.Label(
            header,
            text="Audio → Tabs → Synth + Match",
            style="Muted.TLabel",
        )
        subtitle.pack(side=tk.LEFT, padx=(12, 0), pady=(6, 0))

        control_frame = ttk.Frame(self, style="Panel.TFrame")
        control_frame.pack(fill=tk.X, padx=16, pady=8)

        params_frame = ttk.Frame(control_frame, style="Panel.TFrame")
        params_frame.pack(side=tk.LEFT, padx=12, pady=12)

        ttk.Label(params_frame, text="Duration (sec):").grid(row=0, column=0, sticky="w")
        self.duration_var = tk.StringVar(value="5")
        ttk.Entry(params_frame, textvariable=self.duration_var, width=7).grid(row=0, column=1, padx=6)

        ttk.Label(params_frame, text="Min duration:").grid(row=0, column=2, sticky="w")
        self.min_duration_var = tk.StringVar(value="0.1")
        ttk.Entry(params_frame, textvariable=self.min_duration_var, width=7).grid(row=0, column=3, padx=6)

        ttk.Label(params_frame, text="Min voiced prob:").grid(row=0, column=4, sticky="w")
        self.min_voiced_var = tk.StringVar(value="0.75")
        ttk.Entry(params_frame, textvariable=self.min_voiced_var, width=7).grid(row=0, column=5, padx=6)

        ttk.Label(params_frame, text="Max fret:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.max_fret_var = tk.StringVar(value="15")
        ttk.Entry(params_frame, textvariable=self.max_fret_var, width=7).grid(row=1, column=1, padx=6, pady=(8, 0))

        ttk.Label(params_frame, text="Segment (sec):").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.segment_var = tk.StringVar(value="15")
        ttk.Entry(params_frame, textvariable=self.segment_var, width=7).grid(row=1, column=3, padx=6, pady=(8, 0))

        self.use_harmonic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Use harmonic", variable=self.use_harmonic_var).grid(
            row=1, column=4, columnspan=2, sticky="w", padx=6, pady=(8, 0)
        )

        actions_frame = ttk.Frame(control_frame, style="Panel.TFrame")
        actions_frame.pack(side=tk.LEFT, padx=16, pady=12)

        input_frame = ttk.LabelFrame(actions_frame, text="Input", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(input_frame, text="🎙 Record", command=self.on_record).pack(side=tk.LEFT, padx=4)
        ttk.Button(input_frame, text="📂 Load File", command=self.on_load).pack(side=tk.LEFT, padx=4)
        ttk.Button(input_frame, text="⚡ Load + Generate", command=self.on_load_and_generate).pack(
            side=tk.LEFT, padx=4
        )

        process_frame = ttk.LabelFrame(actions_frame, text="Process", padding=10)
        process_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(process_frame, text="🧠 Analyze", command=self.on_analyze, style="Accent.TButton").pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(process_frame, text="✨ Best Quality", command=self.on_best_quality).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(process_frame, text="🛠 Refine w/ Original", command=self.on_refine_with_original).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(process_frame, text="🧪 Check Tabs", command=self.on_check_tabs).pack(
            side=tk.LEFT, padx=4
        )

        output_frame = ttk.LabelFrame(actions_frame, text="Output", padding=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(output_frame, text="💾 Save Tabs", command=self.on_save).pack(side=tk.LEFT, padx=4)
        ttk.Button(output_frame, text="🔊 Render Audio", command=self.on_render_audio).pack(side=tk.LEFT, padx=4)
        ttk.Button(output_frame, text="🎧 Match Original", command=self.on_match_original).pack(
            side=tk.LEFT, padx=4
        )

        utils_frame = ttk.LabelFrame(actions_frame, text="Utilities", padding=10)
        utils_frame.pack(fill=tk.X)
        ttk.Button(utils_frame, text="🧪 Run Test", command=self.on_test).pack(side=tk.LEFT, padx=4)
        ttk.Button(utils_frame, text="❓ Help", command=self.on_help).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Frame(self, style="Panel.TFrame")
        status_bar.pack(fill=tk.X, padx=16, pady=(4, 0))
        ttk.Label(status_bar, textvariable=self.status_var, anchor="w").pack(
            fill=tk.X, padx=10, pady=6
        )

        self.output = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            background="#0b1220",
            foreground="#e2e8f0",
            insertbackground="#e2e8f0",
            font=("Menlo", 12),
        )
        self.output.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 16))

    def on_help(self):
        messagebox.showinfo(
            "Help",
            "\n".join(
                [
                    "Quick start:",
                    "1) Record or Load File",
                    "2) Analyze to generate tabs",
                    "3) Save Tabs or Render Audio",
                    "",
                    "Improve accuracy:",
                    "• Best Quality: auto-tunes extraction parameters",
                    "• Refine w/ Original: correct frets and timing",
                    "• Check Tabs: objective similarity scores",
                    "• Match Original: optimizes synth parameters",
                ]
            ),
        )

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

    def on_best_quality(self):
        if self.audio_data is None:
            messagebox.showwarning("No audio", "Please record or load audio first.")
            return

        try:
            max_fret = int(self.max_fret_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Max fret must be an integer.")
            return

        def task():
            self.set_status("Auto-tuning parameters...")
            try:
                tune = find_best_extraction(self.audio_data, self.pitch_det, use_harmonic=True)
                self.tab_gen.set_max_fret(max_fret)
                self.tab_gen.generate_tabs(tune.notes)
                tabs_text = self.tab_gen.format_tabs_as_text()
                self.output.delete("1.0", tk.END)
                self.output.insert(tk.END, tabs_text)

                self.min_duration_var.set(str(tune.min_duration))
                self.min_voiced_var.set(str(tune.min_voiced_prob))
                self.segment_var.set(str(tune.segment_seconds))
                self.use_harmonic_var.set(tune.use_harmonic)

                self.set_status(
                    f"Best done. Notes: {len(tune.notes)} | "
                    f"md={tune.min_duration}, vp={tune.min_voiced_prob}, seg={tune.segment_seconds}"
                )
            except Exception as exc:
                self.set_status("Auto-tune failed")
                messagebox.showerror("Error", str(exc))

        threading.Thread(target=task, daemon=True).start()

    def on_save(self):
        tabs_text = self.output.get("1.0", tk.END).strip()
        if not tabs_text:
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
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(tabs_text)
            self.set_status("Tabs saved")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def on_refine_with_original(self):
        tabs_text = self.output.get("1.0", tk.END).strip()
        if not tabs_text:
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return

        original_path = filedialog.askopenfilename(
            title="Select original audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if not original_path:
            return

        def task():
            self.set_status("Refining tabs with original audio...")
            try:
                result = refine_tabs_with_original(
                    tabs_text=tabs_text,
                    original_audio_path=original_path,
                )
                self.output.delete("1.0", tk.END)
                self.output.insert(tk.END, result.refined_tabs_text)
                self.set_status(
                    f"Refined: changes={result.changes_count}, step={result.estimated_step_seconds:.3f}s"
                )
            except Exception as exc:
                self.set_status("Refine failed")
                messagebox.showerror("Error", str(exc))

        threading.Thread(target=task, daemon=True).start()

    def on_check_tabs(self):
        tabs_text = self.output.get("1.0", tk.END).strip()
        if not tabs_text:
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return

        original_path = filedialog.askopenfilename(
            title="Select original audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if not original_path:
            return

        def task():
            self.set_status("Checking tabs against original...")
            try:
                result = check_tabs_against_original(
                    tabs_text=tabs_text,
                    original_audio_path=original_path,
                )
                self.set_status(
                    f"Checked: overall={result.overall_score:.4f}, "
                    f"chroma={result.chroma_score:.4f}, onset={result.onset_score:.4f}"
                )
                messagebox.showinfo(
                    "Tabs Check",
                    "\n".join(
                        [
                            f"Overall score: {result.overall_score:.4f}",
                            f"Chroma score: {result.chroma_score:.4f}",
                            f"Onset score: {result.onset_score:.4f}",
                            f"Estimated step: {result.estimated_step_seconds:.3f}s",
                            f"Estimated note: {result.estimated_note_seconds:.3f}s",
                            f"Analyzed seconds: {result.analyzed_seconds:.2f}",
                        ]
                    ),
                )
            except Exception as exc:
                self.set_status("Check failed")
                messagebox.showerror("Error", str(exc))

        threading.Thread(target=task, daemon=True).start()

    def on_render_audio(self):
        tabs_text = self.output.get("1.0", tk.END).strip()
        if not tabs_text:
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save synthesized audio",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not out_path:
            return

        def task():
            self.set_status("Synthesizing guitar audio from tabs...")
            try:
                result = synthesize_from_tabs_text(tabs_text, output_path=out_path, play=True)
                self.set_status(
                    f"Audio rendered: {result.notes_count} notes, {result.duration:.1f}s"
                )
            except Exception as exc:
                self.set_status("Audio render failed")
                messagebox.showerror("Error", str(exc))

        threading.Thread(target=task, daemon=True).start()

    def on_match_original(self):
        tabs_text = self.output.get("1.0", tk.END).strip()
        if not tabs_text:
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return

        original_path = filedialog.askopenfilename(
            title="Select original audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if not original_path:
            return

        out_path = filedialog.asksaveasfilename(
            title="Save matched synthesized audio",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not out_path:
            return

        def task():
            self.set_status("Matching synth to original (this may take a while)...")
            try:
                result = optimize_synth_against_original(
                    tabs_text=tabs_text,
                    original_audio_path=original_path,
                    output_path=out_path,
                )
                self.set_status(
                    f"Matched: score={result.score:.4f}, params={result.params}"
                )
            except Exception as exc:
                self.set_status("Match failed")
                messagebox.showerror("Error", str(exc))

        threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    app = GuitarTabApp()
    app.mainloop()
