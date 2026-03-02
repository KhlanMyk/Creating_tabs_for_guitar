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
from tab_synth import synthesize_from_tabs_text, synthesize_from_timed_events


# ── colour palette ──────────────────────────────────────
BG          = "#111118"
SIDEBAR_BG  = "#1a1a2e"
CARD_BG     = "#16213e"
ACCENT      = "#e94560"
ACCENT_HOVER = "#ff6b81"
TEXT        = "#eaeaea"
MUTED       = "#8892b0"
DIM         = "#5a6380"
TAB_BG      = "#0a0a14"
TAB_FG      = "#c8d6e5"
TAB_ACCENT  = "#feca57"
ENTRY_BG    = "#10163a"
BORDER      = "#233554"
SUCCESS     = "#55efc4"
WARN        = "#f6b93b"

# section-specific accent colours
SEC_INPUT   = "#00b894"
SEC_PROCESS = "#6c5ce7"
SEC_OUTPUT  = "#fdcb6e"
SEC_INPUT_HOVER   = "#00e6b8"
SEC_PROCESS_HOVER = "#a29bfe"
SEC_OUTPUT_HOVER  = "#ffe8a1"


class GuitarTabApp(tk.Tk):
    """Redesigned premium GUI for generating guitar tabs."""

    def __init__(self):
        super().__init__()
        self.title("Guitar Tab Generator")
        self.geometry("1380x850")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        self.audio_proc = AudioProcessor(sample_rate=22050)
        self.pitch_det = PitchDetector(sample_rate=22050)
        self.tab_gen = GuitarTabGenerator()
        self.audio_data = None
        self._loaded_path: str | None = None
        self._timed_events: list[dict] = []  # real-timing data for synthesis

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

        # progress bar style
        s.configure("Custom.Horizontal.TProgressbar",
                     troughcolor=ENTRY_BG, background=SEC_PROCESS,
                     darkcolor=SEC_PROCESS, lightcolor=SEC_PROCESS_HOVER,
                     bordercolor=BORDER, thickness=14)

    # ── layout ──────────────────────────────────────────
    def _build_ui(self):
        # ── sidebar (scrollable) ─────────────────────────
        sidebar_outer = tk.Frame(self, bg=SIDEBAR_BG, width=310)
        sidebar_outer.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_outer.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_outer, bg=SIDEBAR_BG, highlightthickness=0, width=310)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(sidebar_canvas, bg=SIDEBAR_BG)
        sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw", width=310)

        def _on_sidebar_configure(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
        sidebar.bind("<Configure>", _on_sidebar_configure)
        sidebar_canvas.bind_all("<MouseWheel>",
            lambda e: sidebar_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # logo area
        logo_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        logo_frame.pack(fill=tk.X, padx=20, pady=(20, 6))
        tk.Label(logo_frame, text="Guitar Tab Generator",
                 font=("Helvetica", 15, "bold"), fg=TEXT, bg=SIDEBAR_BG).pack(anchor="w")
        tk.Label(logo_frame, text="Audio  ->  Tablature  ->  Playback",
                 font=("Helvetica", 10), fg=DIM, bg=SIDEBAR_BG).pack(anchor="w", pady=(2, 0))

        # ── STEP 1 ─ Input ──────────────────────────────
        self._section_header(sidebar, "1", "Load Audio", SEC_INPUT,
                             "Start here — load a file or record from microphone")
        self._action_btn(sidebar,
            title="Open File",
            desc="Load MP3, WAV, FLAC, OGG or M4A",
            command=self.on_load,
            accent=SEC_INPUT, hover=SEC_INPUT_HOVER)
        self._action_btn(sidebar,
            title="Record Mic",
            desc="Capture live audio from microphone",
            command=self.on_record,
            accent=SEC_INPUT, hover=SEC_INPUT_HOVER)

        # ── STEP 2 ─ Process ────────────────────────────
        self._section_header(sidebar, "2", "Generate Tabs", SEC_PROCESS,
                             "Convert audio into guitar tablature")
        self._action_btn(sidebar,
            title="Analyze",
            desc="Detect notes and create tabs (main action)",
            command=self.on_analyze,
            accent=ACCENT, hover=ACCENT_HOVER, primary=True)
        self._action_btn(sidebar,
            title="Best Quality",
            desc="Auto-tune all parameters for best result",
            command=self.on_best_quality,
            accent=SEC_PROCESS, hover=SEC_PROCESS_HOVER)
        self._action_btn(sidebar,
            title="Refine with Original",
            desc="Correct frets by comparing to source audio",
            command=self.on_refine_with_original,
            accent=SEC_PROCESS, hover=SEC_PROCESS_HOVER)

        # ── STEP 3 ─ Output ─────────────────────────────
        self._section_header(sidebar, "3", "Export & Verify", SEC_OUTPUT,
                             "Save, render, and check quality")
        self._action_btn(sidebar,
            title="Save Tabs",
            desc="Export tablature to a .txt file",
            command=self.on_save,
            accent=SEC_OUTPUT, hover=SEC_OUTPUT_HOVER)
        self._action_btn(sidebar,
            title="Render Audio",
            desc="Synthesize guitar audio from tabs",
            command=self.on_render_audio,
            accent=SEC_OUTPUT, hover=SEC_OUTPUT_HOVER)
        self._action_btn(sidebar,
            title="Match Original",
            desc="Optimize synth parameters vs original",
            command=self.on_match_original,
            accent=SEC_OUTPUT, hover=SEC_OUTPUT_HOVER)
        self._action_btn(sidebar,
            title="Check Tabs",
            desc="Score tabs accuracy against original",
            command=self.on_check_tabs,
            accent=SEC_OUTPUT, hover=SEC_OUTPUT_HOVER)

        # ── Help ────────────────────────────────────────
        tk.Frame(sidebar, height=1, bg=BORDER).pack(fill=tk.X, padx=16, pady=(12, 6))
        self._action_btn(sidebar,
            title="Help",
            desc="Show quick-start guide",
            command=self.on_help,
            accent=DIM, hover=MUTED)

        # bottom padding
        tk.Frame(sidebar, bg=SIDEBAR_BG, height=16).pack()

        # ── status footer (outside scroll) ──────────────
        self.status_var = tk.StringVar(value="Ready")
        status_frame = tk.Frame(sidebar_outer, bg="#0d1025")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(status_frame, textvariable=self.status_var, fg=SUCCESS, bg="#0d1025",
                 font=("Helvetica", 10), anchor="w", wraplength=280).pack(padx=16, pady=10)

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

        # ── progress bar ────────────────────────────────
        self._progress_frame = tk.Frame(main, bg=BG)
        self._progress_frame.pack(fill=tk.X, padx=20, pady=(6, 0))

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_label = tk.StringVar(value="")

        self._progress_bar = ttk.Progressbar(
            self._progress_frame, variable=self._progress_var,
            maximum=1.0, mode="determinate", length=400,
            style="Custom.Horizontal.TProgressbar")

        self._progress_lbl_widget = tk.Label(
            self._progress_frame, textvariable=self._progress_label,
            fg=MUTED, bg=BG, font=("Helvetica", 10), anchor="w")

        # starts hidden
        self._progress_visible = False

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
    def _section_header(self, parent, step_num: str, title: str, color: str, subtitle: str):
        """Draw a numbered section header with coloured step badge."""
        frame = tk.Frame(parent, bg=SIDEBAR_BG)
        frame.pack(fill=tk.X, padx=16, pady=(14, 4))

        # step badge
        badge = tk.Label(frame, text=f" {step_num} ", font=("Helvetica", 10, "bold"),
                         fg="#fff", bg=color)
        badge.pack(side=tk.LEFT, padx=(0, 8))

        # title + subtitle
        text_frame = tk.Frame(frame, bg=SIDEBAR_BG)
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(text_frame, text=title, font=("Helvetica", 13, "bold"),
                 fg=TEXT, bg=SIDEBAR_BG, anchor="w").pack(anchor="w")
        tk.Label(text_frame, text=subtitle, font=("Helvetica", 9),
                 fg=DIM, bg=SIDEBAR_BG, anchor="w", wraplength=200).pack(anchor="w")

    def _action_btn(self, parent, *, title: str, desc: str, command,
                    accent: str, hover: str, primary: bool = False):
        """Create a button card with title + description."""
        card_bg = ACCENT if primary else CARD_BG
        card = tk.Frame(parent, bg=card_bg, cursor="hand2")
        card.pack(fill=tk.X, padx=16, pady=3)

        inner = tk.Frame(card, bg=card_bg)
        inner.pack(fill=tk.X, padx=14, pady=8)

        # coloured left bar
        bar = tk.Frame(card, bg=accent, width=3)
        bar.place(x=0, y=4, relheight=0.8)

        title_lbl = tk.Label(inner, text=title,
                             font=("Helvetica", 12, "bold") if primary else ("Helvetica", 11),
                             fg="#fff" if primary else TEXT, bg=card_bg, anchor="w")
        title_lbl.pack(anchor="w")

        desc_lbl = tk.Label(inner, text=desc, font=("Helvetica", 9),
                            fg="#ccc" if primary else DIM, bg=card_bg,
                            anchor="w", wraplength=240)
        desc_lbl.pack(anchor="w", pady=(1, 0))

        # click on anything in the card
        for widget in (card, inner, title_lbl, desc_lbl):
            widget.bind("<Button-1>", lambda e, c=command: c())
            widget.configure(cursor="hand2")

        # hover
        def on_enter(_):
            for w in (card, inner, title_lbl, desc_lbl):
                w.configure(bg=hover if not primary else ACCENT_HOVER)

        def on_leave(_):
            for w in (card, inner, title_lbl, desc_lbl):
                w.configure(bg=card_bg)

        for widget in (card, inner, title_lbl, desc_lbl):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

    def _draw_placeholder(self):
        self.tab_canvas.delete("all")
        self.tab_canvas.create_text(
            400, 180, text="Load audio and click Analyze\nto generate guitar tabs here",
            fill=MUTED, font=("Helvetica", 16), anchor="center", justify="center",
        )

    def _render_tabs_on_canvas(self, tabs_text: str):
        """Draw the tab text onto the canvas with coloured formatting.

        Uses bulk text items per line segment (not per-character) for speed.
        """
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
                # Group consecutive same-colour characters into spans
                # string name (first 2 chars)
                self.tab_canvas.create_text(pad_x, y, text=line[:2], anchor="nw",
                                            fill=ACCENT, font=tab_font)
                x = pad_x + tab_font.measure(line[:2])

                # classify rest of line into colour spans
                spans: list[tuple[str, str]] = []  # (text, colour)
                buf = ""
                buf_colour = ""
                for ch in line[2:]:
                    if ch.isdigit():
                        colour = "#ffffff"
                    elif ch == "|":
                        colour = ACCENT
                    else:
                        colour = "#2a3a5a"
                    if colour != buf_colour and buf:
                        spans.append((buf, buf_colour))
                        buf = ""
                    buf += ch
                    buf_colour = colour
                if buf:
                    spans.append((buf, buf_colour))

                for text, colour in spans:
                    self.tab_canvas.create_text(x, y, text=text, anchor="nw",
                                                fill=colour, font=tab_font)
                    x += tab_font.measure(text)

                max_x = max(max_x, x + 40)
                y += line_height
            else:
                self.tab_canvas.create_text(pad_x, y, text=line, anchor="nw",
                                            fill=MUTED, font=chord_font)
                y += line_height + 4

        self.tab_canvas.configure(scrollregion=(0, 0, max_x, y + 40))

    # ── status / property helpers ───────────────────────
    def set_status(self, text: str):
        """Thread-safe status update."""
        self.after(0, lambda: self._do_set_status(text))

    def _do_set_status(self, text: str):
        self.status_var.set(text)

    def _show_progress(self, fraction: float = 0.0, message: str = ""):
        """Thread-safe progress update. Called from background threads."""
        self.after(0, lambda: self._do_show_progress(fraction, message))

    def _do_show_progress(self, fraction: float, message: str):
        if not self._progress_visible:
            self._progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._progress_lbl_widget.pack(side=tk.LEFT, padx=(10, 0))
            self._progress_visible = True
        self._progress_var.set(max(0.0, min(1.0, fraction)))
        self._progress_label.set(message)

    def _hide_progress(self):
        """Thread-safe hide progress bar."""
        self.after(0, self._do_hide_progress)

    def _do_hide_progress(self):
        if self._progress_visible:
            self._progress_bar.pack_forget()
            self._progress_lbl_widget.pack_forget()
            self._progress_visible = False
            self._progress_var.set(0.0)
            self._progress_label.set("")

    def _run_task(self, task_func):
        """Run a function in a background thread with progress support."""
        self._show_progress(0.0, "Starting...")
        def wrapper():
            try:
                task_func()
            finally:
                self._hide_progress()
        threading.Thread(target=wrapper, daemon=True).start()

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
        self._show_progress(0.5, "Loading audio file...")
        try:
            audio, _ = self.audio_proc.load_audio_file(file_path)
            self.audio_data = audio
            self._loaded_path = file_path
            name = os.path.basename(file_path)
            sr = self.audio_proc.sample_rate
            self.file_info_var.set(f"  {name}  ({len(audio)/sr:.1f}s)")
            self.set_status("Audio loaded")
        except Exception as exc:
            self.set_status("Load failed")
            messagebox.showerror("Error", str(exc))
        finally:
            self._hide_progress()

    def on_record(self):
        try:
            duration = float(self.duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Duration must be a number.")
            return
        def task():
            self.set_status("Recording...")
            self._show_progress(0.0, f"Recording {duration:.0f}s...")
            try:
                self.audio_data = self.audio_proc.record_from_microphone(duration)
                self._loaded_path = None
                self.file_info_var.set(f"  Recorded {duration:.1f}s")
                self.set_status("Recording complete")
            except Exception as exc:
                self.set_status("Recording failed")
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        self._run_task(task)

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
                    use_harmonic=self.use_harmonic_var.get(),
                    segment_seconds=seg,
                    use_onset_alignment=True,
                    progress_callback=lambda f, m: self._show_progress(f * 0.9, m),
                )
                self._show_progress(0.95, "Generating tabs...")
                self.tab_gen.generate_tabs(notes)
                self._timed_events = self.tab_gen.get_timed_events()
                tabs_text = self.tab_gen.format_tabs_as_text()
                self.after(0, lambda: self._render_tabs_on_canvas(tabs_text))
                self.set_status(f"Done - {len(notes)} notes detected")
            except Exception as exc:
                self.set_status("Analysis failed")
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        self._run_task(task)

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
                tune = find_best_extraction(
                    self.audio_data, self.pitch_det, use_harmonic=True,
                    progress_callback=lambda f, m: self._show_progress(f * 0.9, m),
                )
                self._show_progress(0.95, "Generating tabs...")
                self.tab_gen.set_max_fret(max_fret)
                self.tab_gen.generate_tabs(tune.notes)
                self._timed_events = self.tab_gen.get_timed_events()
                tabs_text = self.tab_gen.format_tabs_as_text()
                self.after(0, lambda: self._render_tabs_on_canvas(tabs_text))
                self.after(0, lambda: self.min_duration_var.set(str(tune.min_duration)))
                self.after(0, lambda: self.min_voiced_var.set(str(tune.min_voiced_prob)))
                self.after(0, lambda: self.segment_var.set(str(tune.segment_seconds)))
                self.after(0, lambda: self.use_harmonic_var.set(tune.use_harmonic))
                self.set_status(f"Best quality - {len(tune.notes)} notes")
            except Exception as exc:
                self.set_status("Auto-tune failed")
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        self._run_task(task)

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
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        self._run_task(task)

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
                self.after(0, lambda: messagebox.showinfo("Tabs Check", "\n".join([
                    f"Overall score:  {r.overall_score:.4f}",
                    f"Chroma score:   {r.chroma_score:.4f}",
                    f"Onset score:    {r.onset_score:.4f}",
                    f"Est. step:      {r.estimated_step_seconds:.3f}s",
                    f"Est. note:      {r.estimated_note_seconds:.3f}s",
                    f"Analyzed:       {r.analyzed_seconds:.1f}s",
                ])))
            except Exception as exc:
                self.set_status("Check failed")
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        self._run_task(task)

    def on_render_audio(self):
        tabs_text = self._get_tabs_text()
        if not tabs_text and not self._timed_events:
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
                if self._timed_events:
                    # Real-timing synthesis (preserves rhythm)
                    _, result = synthesize_from_timed_events(
                        self._timed_events, output_path=out_path, play=True,
                    )
                else:
                    # Fallback: fixed-step from text
                    result = synthesize_from_tabs_text(tabs_text, output_path=out_path, play=True)
                self.set_status(f"Rendered: {result.notes_count} notes, {result.duration:.1f}s")
            except Exception as exc:
                self.set_status("Render failed")
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        self._run_task(task)

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
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        self._run_task(task)

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
