"""Premium Tkinter GUI for Guitar Tab Generator."""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font as tkfont

from audio_processor import AudioProcessor
from auto_tune import find_best_extraction
from pitch_detector import PitchDetector
from synth_matcher import optimize_synth_against_original
from tab_checker import check_tabs_against_original
from tab_generator import GuitarTabGenerator
from tab_refiner import refine_tabs_with_original
from self_test import run_sine_test
from tab_synth import synthesize_from_tabs_text


# ── colour palette ──────────────────────────────────────
BG          = "#0f0f0f"
SIDEBAR_BG  = "#1a1a2e"
CARD_BG     = "#16213e"
ACCENT      = "#e94560"
ACCENT_HOVER = "#ff6b81"
TEXT        = "#eaeaea"
MUTED       = "#8892b0"
TAB_BG      = "#0a0a14"
TAB_FG      = "#c8d6e5"
TAB_ACCENT  = "#feca57"
ENTRY_BG    = "#10163a"
BORDER      = "#233554"
SUCCESS     = "#55efc4"
WARN        = "#f6b93b"


class GuitarTabApp(tk.Tk):
    """Redesigned premium GUI for generating guitar tabs."""

    def __init__(self):
        super().__init__()
        self.title("Guitar Tab Generator")
        self.geometry("1280x800")
        self.minsize(1060, 680)
        self.configure(bg=BG)

        self.audio_proc = AudioProcessor()
        self.pitch_det = PitchDetector()
        self.tab_gen = GuitarTabGenerator()
        self.audio_data = None
        self._loaded_path: str | None = None

        self._setup_styles()
        self._build_ui()

    # ── styles ──────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except Exception:
            pass

        s.configure("Sidebar.TFrame", background=SIDEBAR_BG)
        s.configure("Card.TFrame", background=CARD_BG)
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("Sidebar.TLabel", background=SIDEBAR_BG, foreground=TEXT)
        s.configure("Muted.TLabel", background=BG, foreground=MUTED)
        s.configure("Card.TLabel", background=CARD_BG, foreground=TEXT)
        s.configure("TEntry", fieldbackground=ENTRY_BG, foreground=TEXT)
        s.configure("TCheckbutton", background=CARD_BG, foreground=TEXT)

    # ── layout ──────────────────────────────────────────
    def _build_ui(self):
        # ── sidebar ─────────────────────────────────────
        sidebar = tk.Frame(self, bg=SIDEBAR_BG, width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # logo area
        logo_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        logo_frame.pack(fill=tk.X, padx=20, pady=(24, 16))
        tk.Label(logo_frame, text="Guitar Tab", font=("Helvetica", 17, "bold"),
                 fg=TEXT, bg=SIDEBAR_BG).pack(anchor="w")
        tk.Label(logo_frame, text="Generator", font=("Helvetica", 17),
                 fg=MUTED, bg=SIDEBAR_BG).pack(anchor="w")

        self._sep(sidebar)

        # ── Input section ───
        self._section_label(sidebar, "INPUT")
        self._sidebar_btn(sidebar, "Load Audio",          self.on_load,              CARD_BG)
        self._sidebar_btn(sidebar, "Record Mic",          self.on_record,            CARD_BG)

        self._sep(sidebar)

        # ── Process section ───
        self._section_label(sidebar, "PROCESS")
        self._sidebar_btn(sidebar, "Analyze",              self.on_analyze,           ACCENT)
        self._sidebar_btn(sidebar, "Best Quality",         self.on_best_quality,      CARD_BG)
        self._sidebar_btn(sidebar, "Refine w/ Original",   self.on_refine_with_original, CARD_BG)

        self._sep(sidebar)

        # ── Output section ───
        self._section_label(sidebar, "OUTPUT")
        self._sidebar_btn(sidebar, "Render Audio",         self.on_render_audio,      CARD_BG)
        self._sidebar_btn(sidebar, "Match Original",       self.on_match_original,    CARD_BG)
        self._sidebar_btn(sidebar, "Check Tabs",           self.on_check_tabs,        CARD_BG)
        self._sidebar_btn(sidebar, "Save Tabs",            self.on_save,              CARD_BG)

        self._sep(sidebar)
        self._sidebar_btn(sidebar, "Help",                 self.on_help,              CARD_BG)

        # spacer
        tk.Frame(sidebar, bg=SIDEBAR_BG).pack(fill=tk.BOTH, expand=True)

        # status footer
        self.status_var = tk.StringVar(value="Ready")
        status_frame = tk.Frame(sidebar, bg="#0d1025")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(status_frame, textvariable=self.status_var, fg=SUCCESS, bg="#0d1025",
                 font=("Helvetica", 10), anchor="w", wraplength=240).pack(padx=16, pady=12)

        # ── main content ────────────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # top bar with params
        topbar = tk.Frame(main, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        topbar.pack(fill=tk.X, padx=20, pady=(16, 0))

        self._param_entries = {}
        params = [
            ("Duration (s)", "duration", "5"),
            ("Min dur", "min_dur", "0.1"),
            ("Voiced prob", "voiced", "0.75"),
            ("Max fret", "fret", "15"),
            ("Segment (s)", "segment", "15"),
        ]
        for i, (label, key, default) in enumerate(params):
            tk.Label(topbar, text=label, fg=MUTED, bg=CARD_BG,
                     font=("Helvetica", 10)).grid(row=0, column=i*2, padx=(16 if i==0 else 8, 4), pady=12, sticky="w")
            var = tk.StringVar(value=default)
            e = tk.Entry(topbar, textvariable=var, width=7, bg=ENTRY_BG, fg=TEXT,
                         insertbackground=TEXT, relief="flat", font=("Menlo", 11),
                         highlightbackground=BORDER, highlightthickness=1)
            e.grid(row=0, column=i*2+1, padx=(0, 8), pady=12)
            self._param_entries[key] = var

        self.use_harmonic_var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(topbar, text="Harmonic", variable=self.use_harmonic_var,
                            bg=CARD_BG, fg=TEXT, selectcolor=ENTRY_BG, activebackground=CARD_BG,
                            activeforeground=TEXT, font=("Helvetica", 10))
        cb.grid(row=0, column=len(params)*2, padx=12, pady=12)

        # file info bar
        self.file_info_var = tk.StringVar(value="No audio loaded")
        info_bar = tk.Frame(main, bg=BG)
        info_bar.pack(fill=tk.X, padx=20, pady=(10, 0))
        tk.Label(info_bar, textvariable=self.file_info_var, fg=MUTED, bg=BG,
                 font=("Helvetica", 11)).pack(side=tk.LEFT)

        # ── tab display (canvas with scrollbar) ─────────
        tab_frame = tk.Frame(main, bg=TAB_BG, highlightbackground=BORDER, highlightthickness=1)
        tab_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))

        self.tab_canvas = tk.Canvas(tab_frame, bg=TAB_BG, highlightthickness=0)
        sb_y = ttk.Scrollbar(tab_frame, orient=tk.VERTICAL, command=self.tab_canvas.yview)
        sb_x = ttk.Scrollbar(tab_frame, orient=tk.HORIZONTAL, command=self.tab_canvas.xview)
        self.tab_canvas.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tab_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._draw_placeholder()
        self._raw_tabs_text = ""

    # ── widget helpers ──────────────────────────────────
    def _sep(self, parent):
        tk.Frame(parent, height=1, bg=BORDER).pack(fill=tk.X, padx=20, pady=8)

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, fg=MUTED, bg=SIDEBAR_BG,
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=24, pady=(4, 2))

    def _sidebar_btn(self, parent, text: str, command, bg_color: str):
        btn = tk.Button(
            parent, text=text, command=command,
            bg=bg_color, fg=TEXT, activebackground=ACCENT_HOVER, activeforeground="#fff",
            font=("Helvetica", 12), anchor="w", relief="flat", cursor="hand2",
            padx=20, pady=8, bd=0,
        )
        btn.pack(fill=tk.X, padx=12, pady=2)
        btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=ACCENT_HOVER))
        btn.bind("<Leave>", lambda e, b=btn, c=bg_color: b.configure(bg=c))

    def _draw_placeholder(self):
        self.tab_canvas.delete("all")
        self.tab_canvas.create_text(
            400, 180, text="Load audio and click Analyze\nto generate guitar tabs here",
            fill=MUTED, font=("Helvetica", 16), anchor="center", justify="center",
        )

    def _render_tabs_on_canvas(self, tabs_text: str):
        """Draw the tab text onto the canvas with coloured formatting."""
        self.tab_canvas.delete("all")
        self._raw_tabs_text = tabs_text

        pad_x, pad_y = 24, 20
        line_height = 20
        y = pad_y

        tab_font = tkfont.Font(family="Menlo", size=13)
        chord_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        max_x = 800

        for raw_line in tabs_text.split("\n"):
            line = raw_line.rstrip()
            if not line:
                y += 12
                continue

            is_string_line = len(line) >= 2 and line[0] in "eBGDAE" and line[1] == "|"
            is_chord_line = line.startswith("    ") and not is_string_line

            if is_chord_line:
                self.tab_canvas.create_text(pad_x, y, text=line, anchor="nw",
                                            fill=TAB_ACCENT, font=chord_font)
                y += line_height + 2
            elif is_string_line:
                # string name in accent
                self.tab_canvas.create_text(pad_x, y, text=line[:2], anchor="nw",
                                            fill=ACCENT, font=tab_font)
                x = pad_x + tab_font.measure(line[:2])
                i = 2
                while i < len(line):
                    ch = line[i]
                    if ch.isdigit():
                        j = i
                        while j < len(line) and line[j].isdigit():
                            j += 1
                        num_str = line[i:j]
                        self.tab_canvas.create_text(x, y, text=num_str, anchor="nw",
                                                    fill="#ffffff", font=tab_font)
                        x += tab_font.measure(num_str)
                        i = j
                    elif ch == "|":
                        self.tab_canvas.create_text(x, y, text=ch, anchor="nw",
                                                    fill=ACCENT, font=tab_font)
                        x += tab_font.measure(ch)
                        i += 1
                    else:
                        self.tab_canvas.create_text(x, y, text=ch, anchor="nw",
                                                    fill="#2a3a5a", font=tab_font)
                        x += tab_font.measure(ch)
                        i += 1
                max_x = max(max_x, x + 40)
                y += line_height
            else:
                self.tab_canvas.create_text(pad_x, y, text=line, anchor="nw",
                                            fill=MUTED, font=chord_font)
                y += line_height + 4

        self.tab_canvas.configure(scrollregion=(0, 0, max_x, y + 40))

    # ── status / property helpers ───────────────────────
    def set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    @property
    def duration_var(self):
        return self._param_entries["duration"]

    @property
    def min_duration_var(self):
        return self._param_entries["min_dur"]

    @property
    def min_voiced_var(self):
        return self._param_entries["voiced"]

    @property
    def max_fret_var(self):
        return self._param_entries["fret"]

    @property
    def segment_var(self):
        return self._param_entries["segment"]

    def _get_tabs_text(self) -> str:
        return self._raw_tabs_text

    # ── actions ─────────────────────────────────────────
    def on_load(self):
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self.set_status("Loading...")
        try:
            audio, _ = self.audio_proc.load_audio_file(file_path)
            self.audio_data = audio
            self._loaded_path = file_path
            name = os.path.basename(file_path)
            self.file_info_var.set(f"  {name}  ({len(audio)/44100:.1f}s)")
            self.set_status("Audio loaded")
        except Exception as exc:
            self.set_status("Load failed")
            messagebox.showerror("Error", str(exc))

    def on_record(self):
        try:
            duration = float(self.duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Duration must be a number.")
            return
        def task():
            self.set_status("Recording...")
            try:
                self.audio_data = self.audio_proc.record_from_microphone(duration)
                self._loaded_path = None
                self.file_info_var.set(f"  Recorded {duration:.1f}s")
                self.set_status("Recording complete")
            except Exception as exc:
                self.set_status("Recording failed")
                messagebox.showerror("Error", str(exc))
        threading.Thread(target=task, daemon=True).start()

    def on_analyze(self):
        if self.audio_data is None:
            messagebox.showwarning("No audio", "Load or record audio first.")
            return
        try:
            min_dur = float(self.min_duration_var.get())
            min_voiced = float(self.min_voiced_var.get())
            max_fret = int(self.max_fret_var.get())
            seg = float(self.segment_var.get()) if self.segment_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Invalid", "Check parameter values.")
            return

        def task():
            self.set_status("Analyzing audio...")
            try:
                self.tab_gen.set_max_fret(max_fret)
                notes = self.pitch_det.extract_notes_from_audio(
                    self.audio_data, min_duration=min_dur, min_voiced_prob=min_voiced,
                    use_harmonic=self.use_harmonic_var.get(), segment_seconds=seg,
                )
                self.tab_gen.generate_tabs(notes)
                tabs_text = self.tab_gen.format_tabs_as_text()
                self.after(0, lambda: self._render_tabs_on_canvas(tabs_text))
                self.set_status(f"Done - {len(notes)} notes detected")
            except Exception as exc:
                self.set_status("Analysis failed")
                messagebox.showerror("Error", str(exc))
        threading.Thread(target=task, daemon=True).start()

    def on_best_quality(self):
        if self.audio_data is None:
            messagebox.showwarning("No audio", "Load or record audio first.")
            return
        try:
            max_fret = int(self.max_fret_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Max fret must be integer.")
            return

        def task():
            self.set_status("Auto-tuning...")
            try:
                tune = find_best_extraction(self.audio_data, self.pitch_det, use_harmonic=True)
                self.tab_gen.set_max_fret(max_fret)
                self.tab_gen.generate_tabs(tune.notes)
                tabs_text = self.tab_gen.format_tabs_as_text()
                self.after(0, lambda: self._render_tabs_on_canvas(tabs_text))
                self.min_duration_var.set(str(tune.min_duration))
                self.min_voiced_var.set(str(tune.min_voiced_prob))
                self.segment_var.set(str(tune.segment_seconds))
                self.use_harmonic_var.set(tune.use_harmonic)
                self.set_status(f"Best quality - {len(tune.notes)} notes")
            except Exception as exc:
                self.set_status("Auto-tune failed")
                messagebox.showerror("Error", str(exc))
        threading.Thread(target=task, daemon=True).start()

    def on_refine_with_original(self):
        tabs_text = self._get_tabs_text()
        if not tabs_text:
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return
        original_path = filedialog.askopenfilename(
            title="Select original audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All", "*.*")],
        )
        if not original_path:
            return

        def task():
            self.set_status("Refining tabs...")
            try:
                result = refine_tabs_with_original(tabs_text=tabs_text, original_audio_path=original_path)
                self.after(0, lambda: self._render_tabs_on_canvas(result.refined_tabs_text))
                self.set_status(f"Refined: {result.changes_count} changes, step={result.estimated_step_seconds:.3f}s")
            except Exception as exc:
                self.set_status("Refine failed")
                messagebox.showerror("Error", str(exc))
        threading.Thread(target=task, daemon=True).start()

    def on_check_tabs(self):
        tabs_text = self._get_tabs_text()
        if not tabs_text:
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return
        original_path = filedialog.askopenfilename(
            title="Select original audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All", "*.*")],
        )
        if not original_path:
            return

        def task():
            self.set_status("Checking tabs...")
            try:
                r = check_tabs_against_original(tabs_text=tabs_text, original_audio_path=original_path)
                self.set_status(f"overall={r.overall_score:.4f}  chroma={r.chroma_score:.4f}  onset={r.onset_score:.4f}")
                messagebox.showinfo("Tabs Check", "\n".join([
                    f"Overall score:  {r.overall_score:.4f}",
                    f"Chroma score:   {r.chroma_score:.4f}",
                    f"Onset score:    {r.onset_score:.4f}",
                    f"Est. step:      {r.estimated_step_seconds:.3f}s",
                    f"Est. note:      {r.estimated_note_seconds:.3f}s",
                    f"Analyzed:       {r.analyzed_seconds:.1f}s",
                ]))
            except Exception as exc:
                self.set_status("Check failed")
                messagebox.showerror("Error", str(exc))
        threading.Thread(target=task, daemon=True).start()

    def on_render_audio(self):
        tabs_text = self._get_tabs_text()
        if not tabs_text:
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save synthesized audio", defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
        )
        if not out_path:
            return

        def task():
            self.set_status("Synthesizing...")
            try:
                result = synthesize_from_tabs_text(tabs_text, output_path=out_path, play=True)
                self.set_status(f"Rendered: {result.notes_count} notes, {result.duration:.1f}s")
            except Exception as exc:
                self.set_status("Render failed")
                messagebox.showerror("Error", str(exc))
        threading.Thread(target=task, daemon=True).start()

    def on_match_original(self):
        tabs_text = self._get_tabs_text()
        if not tabs_text:
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return
        original_path = filedialog.askopenfilename(
            title="Select original audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All", "*.*")],
        )
        if not original_path:
            return
        out_path = filedialog.asksaveasfilename(
            title="Save matched audio", defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
        )
        if not out_path:
            return

        def task():
            self.set_status("Matching (may take a while)...")
            try:
                result = optimize_synth_against_original(
                    tabs_text=tabs_text, original_audio_path=original_path, output_path=out_path,
                )
                self.set_status(f"Matched: score={result.score:.4f}")
            except Exception as exc:
                self.set_status("Match failed")
                messagebox.showerror("Error", str(exc))
        threading.Thread(target=task, daemon=True).start()

    def on_save(self):
        tabs_text = self._get_tabs_text()
        if not tabs_text:
            messagebox.showwarning("No tabs", "Generate tabs first.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save tabs", defaultextension=".txt",
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

    def on_help(self):
        messagebox.showinfo("Help - Guitar Tab Generator", "\n".join([
            "QUICK START",
            "1.  Load Audio or Record from Mic",
            "2.  Click Analyze to generate tabs",
            "3.  Save Tabs or Render Audio",
            "",
            "IMPROVE ACCURACY",
            "  Best Quality - auto-tune extraction params",
            "  Refine - correct frets using original audio",
            "  Check Tabs - see objective match scores",
            "  Match Original - optimize synth quality",
            "",
            "PARAMETERS",
            "  Duration - mic recording length (seconds)",
            "  Min dur - shortest note to detect",
            "  Voiced prob - confidence filter (0-1)",
            "  Max fret - highest fret to use in tabs",
            "  Segment - process long audio in chunks",
            "  Harmonic - isolate harmonic content first",
        ]))


if __name__ == "__main__":
    app = GuitarTabApp()
    app.mainloop()
