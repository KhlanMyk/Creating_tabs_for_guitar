"""Premium Tkinter GUI v3 – Guitar Tab Generator.

Improvements over v2:
  • Waveform visualization panel (shows audio waveform after loading)
  • Note timeline panel (coloured blocks for each detected note)
  • Tab toolbar: Copy · Zoom +/− · Reset
  • Keyboard shortcuts: ⌘O open · ⌘S save · ⌘R render · ⌘↩ analyze · ⌘+/− zoom
  • Drag & drop file loading (TkDND when available)
  • Line numbers in tab gutter
  • Elapsed time in progress bar
  • Note statistics bar after analysis
  • Coloured status indicators (info / ok / warn / err)
  • Self-test button
  • Sidebar scroll is scoped (doesn't steal tab-area scroll)
  • Window title reflects loaded file
  • Refined dark colour palette
"""
from __future__ import annotations

import colorsys
import math
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font as tkfont
from typing import List, Dict

from audio_processor import AudioProcessor
from auto_tune import find_best_extraction
from pitch_detector import PitchDetector
from synth_matcher import optimize_synth_against_original
from tab_checker import check_tabs_against_original
from tab_generator import GuitarTabGenerator
from tab_refiner import refine_tabs_with_original
from self_test import run_sine_test
from tab_synth import synthesize_from_tabs_text, synthesize_from_timed_events

BG           = "#0d1117"
SURFACE      = "#161b22"
CARD_BG      = "#1c2333"
SIDEBAR_BG   = "#13161f"
ACCENT       = "#58a6ff"
ACCENT_HOVER = "#79c0ff"
RED          = "#f85149"
RED_HOVER    = "#ff7b72"
GREEN        = "#3fb950"
YELLOW       = "#d29922"
ORANGE       = "#db6d28"
TEXT         = "#e6edf3"
MUTED        = "#8b949e"
DIM          = "#484f58"
TAB_BG       = "#0d1117"
TAB_FG       = "#c9d1d9"
TAB_ACCENT   = "#ffa657"
ENTRY_BG     = "#0d1117"
BORDER       = "#30363d"
SUCCESS      = "#3fb950"
WARN         = "#d29922"

# section colours
SEC_INPUT         = "#39d353"
SEC_INPUT_HOVER   = "#56d364"
SEC_PROCESS       = "#8b5cf6"
SEC_PROCESS_HOVER = "#a78bfa"
SEC_OUTPUT        = "#f0883e"
SEC_OUTPUT_HOVER  = "#f4a261"

# fonts
FONT_TITLE = ("SF Pro Display", "Helvetica Neue", "Helvetica", 15, "bold")
FONT_BODY  = ("SF Pro Text", "Helvetica Neue", "Helvetica", 11)
FONT_SMALL = ("SF Pro Text", "Helvetica Neue", "Helvetica", 9)
FONT_MONO  = ("SF Mono", "Menlo", "Consolas", 13)
FONT_MONO_SM = ("SF Mono", "Menlo", "Consolas", 10)

_NOTES_CHROMA = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pick_font(candidates: tuple, size: int = 11, weight: str = "normal") -> tuple:
    """Return the first available font family from *candidates*."""
    available = set()
    try:
        available = set(tkfont.families())
    except Exception:
        pass
    for name in candidates:
        if name in available:
            return (name, size, weight) if weight != "normal" else (name, size)
    return (candidates[-1], size) if weight == "normal" else (candidates[-1], size, weight)


def _note_colour(note_name: str) -> str:
    """Map note name to a visually distinct colour via HSV."""
    idx = _NOTES_CHROMA.index(note_name) if note_name in _NOTES_CHROMA else 0
    h = idx / 12.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.90)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


class GuitarTabApp(tk.Tk):
    """Redesigned premium GUI v3 for generating guitar tabs."""

    #  init
    def __init__(self):
        super().__init__()
        self.title("🎸 Guitar Tab Generator")
        self.geometry("1440x900")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        self.audio_proc = AudioProcessor(sample_rate=22050)
        self.pitch_det  = PitchDetector(sample_rate=22050)
        self.tab_gen    = GuitarTabGenerator()

        self.audio_data: object | None = None
        self._loaded_path: str | None = None
        self._timed_events: list[dict] = []
        self._detected_notes: list = []
        self._raw_tabs_text: str = ""
        self._tab_font_size: int = 13
        self._task_start_time: float = 0.0

        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()

    #  styles
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

        s.configure("Custom.Horizontal.TProgressbar",
                     troughcolor=ENTRY_BG, background=SEC_PROCESS,
                     darkcolor=SEC_PROCESS, lightcolor=SEC_PROCESS_HOVER,
                     bordercolor=BORDER, thickness=14)

    #  keyboard shortcuts
    def _bind_shortcuts(self):
        """Bind ⌘/Ctrl keyboard shortcuts."""
        mod = "Command" if self.tk.call("tk", "windowingsystem") == "aqua" else "Control"
        self.bind(f"<{mod}-o>", lambda e: self.on_load())
        self.bind(f"<{mod}-s>", lambda e: self.on_save())
        self.bind(f"<{mod}-r>", lambda e: self.on_render_audio())
        self.bind(f"<{mod}-Return>", lambda e: self.on_analyze())
        self.bind(f"<{mod}-equal>", lambda e: self._zoom_in())      # ⌘+
        self.bind(f"<{mod}-minus>", lambda e: self._zoom_out())     # ⌘-
        self.bind(f"<{mod}-0>", lambda e: self._zoom_reset())       # ⌘0

    #  layout builder
    def _build_ui(self):
        sidebar_outer = tk.Frame(self, bg=SIDEBAR_BG, width=310)
        sidebar_outer.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_outer.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_outer, bg=SIDEBAR_BG, highlightthickness=0, width=310)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._sidebar_canvas = sidebar_canvas

        sidebar = tk.Frame(sidebar_canvas, bg=SIDEBAR_BG)
        sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw", width=310)

        def _on_sidebar_configure(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
        sidebar.bind("<Configure>", _on_sidebar_configure)

        # scoped sidebar scroll — only when mouse is over sidebar
        def _enter_sidebar(_):
            sidebar_canvas.bind_all("<MouseWheel>",
                lambda e: sidebar_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        def _leave_sidebar(_):
            sidebar_canvas.unbind_all("<MouseWheel>")
        sidebar_outer.bind("<Enter>", _enter_sidebar)
        sidebar_outer.bind("<Leave>", _leave_sidebar)

        logo_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        logo_frame.pack(fill=tk.X, padx=20, pady=(20, 6))
        tk.Label(logo_frame, text="🎸  Guitar Tab Generator",
                 font=("Helvetica", 15, "bold"), fg=TEXT, bg=SIDEBAR_BG).pack(anchor="w")
        tk.Label(logo_frame, text="Audio → Tablature → Playback",
                 font=("Helvetica", 10), fg=DIM, bg=SIDEBAR_BG).pack(anchor="w", pady=(2, 0))

        # keyboard hints
        mod_sym = "⌘" if self.tk.call("tk", "windowingsystem") == "aqua" else "Ctrl+"
        hints = f"{mod_sym}O open · {mod_sym}S save · {mod_sym}↩ analyze · {mod_sym}R render"
        tk.Label(logo_frame, text=hints, font=("Helvetica", 8), fg=DIM,
                 bg=SIDEBAR_BG, wraplength=250).pack(anchor="w", pady=(4, 0))

        # STEP 1 ─ Input
        self._section_header(sidebar, "1", "Load Audio", SEC_INPUT,
                             "Load a file, drop it on window, or record live")
        self._action_btn(sidebar, title="📂  Open File",
            desc="Load MP3, WAV, FLAC, OGG or M4A",
            command=self.on_load, accent=SEC_INPUT, hover=SEC_INPUT_HOVER)
        self._action_btn(sidebar, title="🎤  Record Mic",
            desc="Capture live audio from microphone",
            command=self.on_record, accent=SEC_INPUT, hover=SEC_INPUT_HOVER)

        # STEP 2 ─ Process
        self._section_header(sidebar, "2", "Generate Tabs", SEC_PROCESS,
                             "Convert audio into guitar tablature")
        self._action_btn(sidebar, title="⚡  Analyze",
            desc="Detect notes and create tabs (main action)",
            command=self.on_analyze, accent=ACCENT, hover=ACCENT_HOVER, primary=True)
        self._action_btn(sidebar, title="✨  Best Quality",
            desc="Auto-tune all parameters for best result",
            command=self.on_best_quality, accent=SEC_PROCESS, hover=SEC_PROCESS_HOVER)
        self._action_btn(sidebar, title="🔧  Refine with Original",
            desc="Correct frets by comparing to source audio",
            command=self.on_refine_with_original, accent=SEC_PROCESS, hover=SEC_PROCESS_HOVER)

        # STEP 3 ─ Output
        self._section_header(sidebar, "3", "Export & Verify", SEC_OUTPUT,
                             "Save, render, playback and check quality")
        self._action_btn(sidebar, title="💾  Save Tabs",
            desc="Export tablature to a .txt file",
            command=self.on_save, accent=SEC_OUTPUT, hover=SEC_OUTPUT_HOVER)
        self._action_btn(sidebar, title="🔊  Render Audio",
            desc="Synthesize guitar audio from tabs",
            command=self.on_render_audio, accent=SEC_OUTPUT, hover=SEC_OUTPUT_HOVER)
        self._action_btn(sidebar, title="🎯  Match Original",
            desc="Optimize synth parameters vs original",
            command=self.on_match_original, accent=SEC_OUTPUT, hover=SEC_OUTPUT_HOVER)
        self._action_btn(sidebar, title="✅  Check Tabs",
            desc="Score tabs accuracy against original audio",
            command=self.on_check_tabs, accent=SEC_OUTPUT, hover=SEC_OUTPUT_HOVER)

        # Extras
        tk.Frame(sidebar, height=1, bg=BORDER).pack(fill=tk.X, padx=16, pady=(12, 6))
        self._action_btn(sidebar, title="🧪  Self Test",
            desc="Run a sine-wave sanity check",
            command=self.on_self_test, accent=DIM, hover=MUTED)
        self._action_btn(sidebar, title="❓  Help",
            desc="Show quick-start guide",
            command=self.on_help, accent=DIM, hover=MUTED)

        tk.Frame(sidebar, bg=SIDEBAR_BG, height=16).pack()

        self.status_var = tk.StringVar(value="Ready")
        self._status_color = SUCCESS
        status_frame = tk.Frame(sidebar_outer, bg="#0b0e17")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_dot = tk.Label(status_frame, text="●", fg=SUCCESS, bg="#0b0e17",
                                    font=("Helvetica", 10))
        self._status_dot.pack(side=tk.LEFT, padx=(16, 4), pady=10)
        self._status_lbl = tk.Label(status_frame, textvariable=self.status_var,
                                    fg=TEXT, bg="#0b0e17", font=("Helvetica", 10),
                                    anchor="w", wraplength=240)
        self._status_lbl.pack(side=tk.LEFT, padx=(0, 16), pady=10)

        main = tk.Frame(self, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # top bar: parameters
        topbar = tk.Frame(main, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        topbar.pack(fill=tk.X, padx=20, pady=(16, 0))

        self._param_entries: dict[str, tk.StringVar] = {}
        params = [
            ("Duration (s)", "duration", "5"),
            ("Min dur", "min_dur", "0.05"),
            ("Voiced prob", "voiced", "0.3"),
            ("Max fret", "fret", "15"),
            ("Segment (s)", "segment", "12"),
        ]
        for i, (label, key, default) in enumerate(params):
            tk.Label(topbar, text=label, fg=MUTED, bg=CARD_BG,
                     font=("Helvetica", 10)).grid(row=0, column=i*2,
                     padx=(16 if i == 0 else 8, 4), pady=12, sticky="w")
            var = tk.StringVar(value=default)
            e = tk.Entry(topbar, textvariable=var, width=7, bg=ENTRY_BG, fg=TEXT,
                         insertbackground=TEXT, relief="flat", font=("Menlo", 11),
                         highlightbackground=BORDER, highlightthickness=1)
            e.grid(row=0, column=i*2+1, padx=(0, 8), pady=12)
            self._param_entries[key] = var

        self.use_harmonic_var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(topbar, text="Harmonic", variable=self.use_harmonic_var,
                            bg=CARD_BG, fg=TEXT, selectcolor=ENTRY_BG,
                            activebackground=CARD_BG, activeforeground=TEXT,
                            font=("Helvetica", 10))
        cb.grid(row=0, column=len(params)*2, padx=12, pady=12)

        # file info bar
        self.file_info_var = tk.StringVar(value="No audio loaded  —  drop a file or click Open")
        info_bar = tk.Frame(main, bg=BG)
        info_bar.pack(fill=tk.X, padx=20, pady=(10, 0))
        tk.Label(info_bar, textvariable=self.file_info_var, fg=MUTED, bg=BG,
                 font=("Helvetica", 11)).pack(side=tk.LEFT)

        viz_frame = tk.Frame(main, bg=BG)
        viz_frame.pack(fill=tk.X, padx=20, pady=(8, 0))

        # waveform canvas
        wf_outer = tk.Frame(viz_frame, bg=BORDER, highlightthickness=0)
        wf_outer.pack(fill=tk.X, pady=(0, 4))
        self._waveform_canvas = tk.Canvas(wf_outer, bg=SURFACE, height=70,
                                          highlightthickness=0)
        self._waveform_canvas.pack(fill=tk.X, padx=1, pady=1)
        self._waveform_canvas.create_text(
            400, 35, text="Waveform — load audio to visualize",
            fill=DIM, font=("Helvetica", 10), anchor="center")

        # note timeline canvas
        nt_outer = tk.Frame(viz_frame, bg=BORDER, highlightthickness=0)
        nt_outer.pack(fill=tk.X, pady=(0, 0))
        self._timeline_canvas = tk.Canvas(nt_outer, bg=SURFACE, height=50,
                                          highlightthickness=0)
        self._timeline_canvas.pack(fill=tk.X, padx=1, pady=1)
        self._timeline_canvas.create_text(
            400, 25, text="Note timeline — analyze audio to see detected notes",
            fill=DIM, font=("Helvetica", 10), anchor="center")

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

        self._progress_elapsed_lbl = tk.Label(
            self._progress_frame, text="", fg=DIM, bg=BG,
            font=("Menlo", 9), anchor="e")

        self._progress_visible = False
        self._elapsed_timer_id: str | None = None

        self._stats_var = tk.StringVar(value="")
        self._stats_frame = tk.Frame(main, bg=BG)
        self._stats_frame.pack(fill=tk.X, padx=20, pady=(2, 0))
        self._stats_lbl = tk.Label(self._stats_frame, textvariable=self._stats_var,
                                   fg=MUTED, bg=BG, font=("Helvetica", 10), anchor="w")
        # hidden initially

        tb = tk.Frame(main, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        tb.pack(fill=tk.X, padx=20, pady=(8, 0))

        for text, cmd in [
            ("📋 Copy", self._copy_tabs),
            ("🔍+ Zoom In", self._zoom_in),
            ("🔍− Zoom Out", self._zoom_out),
            ("↺ Reset", self._zoom_reset),
        ]:
            btn = tk.Label(tb, text=text, fg=MUTED, bg=CARD_BG,
                           font=("Helvetica", 10), cursor="hand2", padx=12, pady=6)
            btn.pack(side=tk.LEFT)
            btn.bind("<Button-1>", lambda e, c=cmd: c())
            btn.bind("<Enter>", lambda e, w=btn: w.configure(fg=TEXT, bg="#252d3d"))
            btn.bind("<Leave>", lambda e, w=btn: w.configure(fg=MUTED, bg=CARD_BG))

        self._font_size_var = tk.StringVar(value=f"{self._tab_font_size}px")
        tk.Label(tb, textvariable=self._font_size_var, fg=DIM, bg=CARD_BG,
                 font=("Menlo", 9), padx=8).pack(side=tk.RIGHT, pady=6)
        tk.Label(tb, text="Font:", fg=DIM, bg=CARD_BG,
                 font=("Helvetica", 9)).pack(side=tk.RIGHT, pady=6)

        tab_outer = tk.Frame(main, bg=BORDER, highlightthickness=0)
        tab_outer.pack(fill=tk.BOTH, expand=True, padx=20, pady=(4, 20))

        # gutter for line numbers
        self._gutter = tk.Canvas(tab_outer, bg="#0a0e17", width=36, highlightthickness=0)
        self._gutter.pack(side=tk.LEFT, fill=tk.Y)

        tab_frame = tk.Frame(tab_outer, bg=TAB_BG)
        tab_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tab_canvas = tk.Canvas(tab_frame, bg=TAB_BG, highlightthickness=0)
        sb_y = ttk.Scrollbar(tab_frame, orient=tk.VERTICAL, command=self._sync_yview)
        sb_x = ttk.Scrollbar(tab_frame, orient=tk.HORIZONTAL, command=self.tab_canvas.xview)
        self.tab_canvas.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        self._gutter.configure(yscrollcommand=sb_y.set)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tab_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # scroll inside tab area
        for cvs in (self.tab_canvas, self._gutter):
            cvs.bind("<MouseWheel>",
                lambda e: (self.tab_canvas.yview_scroll(int(-1*(e.delta/120)), "units"),
                           self._gutter.yview_scroll(int(-1*(e.delta/120)), "units")))

        self._draw_placeholder()

        self._setup_dnd()

    #  sync scrolling for gutter + tab canvas
    def _sync_yview(self, *args):
        self.tab_canvas.yview(*args)
        self._gutter.yview(*args)

    #  drag & drop support
    def _setup_dnd(self):
        """Try TkDND for native drag-and-drop, silently skip if unavailable."""
        try:
            self.tk.eval("package require tkdnd")
            self.tk.eval(f'tkdnd::drop_target register {self._w} *')
            self.tk.eval(f'''
                bind {self._w} <<Drop>> {{
                    set data %D
                    set data [string trim $data {{{{}}}}]
                    event generate {self._w} <<FileDropped>> -data $data
                }}
            ''')
            self.bind("<<FileDropped>>", self._on_file_dropped)
        except Exception:
            pass  # tkdnd not available, that's ok

    def _on_file_dropped(self, event):
        path = str(event.data).strip().strip("{}")
        if os.path.isfile(path):
            self._load_file(path)

    #  widget helpers
    def _section_header(self, parent, step_num: str, title: str, color: str, subtitle: str):
        frame = tk.Frame(parent, bg=SIDEBAR_BG)
        frame.pack(fill=tk.X, padx=16, pady=(14, 4))

        badge = tk.Label(frame, text=f" {step_num} ", font=("Helvetica", 10, "bold"),
                         fg="#fff", bg=color)
        badge.pack(side=tk.LEFT, padx=(0, 8))

        text_frame = tk.Frame(frame, bg=SIDEBAR_BG)
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(text_frame, text=title, font=("Helvetica", 13, "bold"),
                 fg=TEXT, bg=SIDEBAR_BG, anchor="w").pack(anchor="w")
        tk.Label(text_frame, text=subtitle, font=("Helvetica", 9),
                 fg=DIM, bg=SIDEBAR_BG, anchor="w", wraplength=200).pack(anchor="w")

    def _action_btn(self, parent, *, title: str, desc: str, command,
                    accent: str, hover: str, primary: bool = False):
        card_bg = ACCENT if primary else CARD_BG
        card = tk.Frame(parent, bg=card_bg, cursor="hand2")
        card.pack(fill=tk.X, padx=16, pady=3)

        inner = tk.Frame(card, bg=card_bg)
        inner.pack(fill=tk.X, padx=14, pady=8)

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

        for widget in (card, inner, title_lbl, desc_lbl):
            widget.bind("<Button-1>", lambda e, c=command: c())
            widget.configure(cursor="hand2")

        def on_enter(_):
            bg = ACCENT_HOVER if primary else hover
            for w in (card, inner, title_lbl, desc_lbl):
                w.configure(bg=bg)
        def on_leave(_):
            for w in (card, inner, title_lbl, desc_lbl):
                w.configure(bg=card_bg)

        for widget in (card, inner, title_lbl, desc_lbl):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

    #  tab placeholder / rendering
    def _draw_placeholder(self):
        self.tab_canvas.delete("all")
        self._gutter.delete("all")
        self.tab_canvas.create_text(
            400, 160, text="Load audio and click  ⚡ Analyze\nto generate guitar tabs here",
            fill=MUTED, font=("Helvetica", 16), anchor="center", justify="center",
        )
        # mini shortcut hint
        mod = "⌘" if self.tk.call("tk", "windowingsystem") == "aqua" else "Ctrl+"
        self.tab_canvas.create_text(
            400, 210, text=f"Shortcuts: {mod}O open  ·  {mod}↩ analyze  ·  {mod}S save",
            fill=DIM, font=("Helvetica", 11), anchor="center",
        )

    def _render_tabs_on_canvas(self, tabs_text: str):
        """Draw tab text with coloured formatting, line numbers, and zoom support."""
        self.tab_canvas.delete("all")
        self._gutter.delete("all")
        self._raw_tabs_text = tabs_text

        pad_x, pad_y = 24, 20
        line_height = self._tab_font_size + 7
        y = pad_y

        tab_font = tkfont.Font(family="Menlo", size=self._tab_font_size)
        chord_font = tkfont.Font(family="Helvetica", size=self._tab_font_size - 1, weight="bold")
        gutter_font = tkfont.Font(family="Menlo", size=max(8, self._tab_font_size - 3))
        max_x = 800
        line_no = 0

        for raw_line in tabs_text.split("\n"):
            line = raw_line.rstrip()
            line_no += 1

            if not line:
                y += int(line_height * 0.6)
                # gutter: empty line number
                self._gutter.create_text(30, y - int(line_height * 0.3), text=str(line_no),
                                         anchor="e", fill=DIM, font=gutter_font)
                continue

            is_string_line = len(line) >= 2 and line[0] in "eBGDAE" and line[1] == "|"
            is_chord_line = line.startswith("    ") and not is_string_line

            # gutter line number
            self._gutter.create_text(30, y + 2, text=str(line_no), anchor="e",
                                     fill=DIM if not is_string_line else MUTED,
                                     font=gutter_font)

            if is_chord_line:
                self.tab_canvas.create_text(pad_x, y, text=line, anchor="nw",
                                            fill=TAB_ACCENT, font=chord_font)
                y += line_height + 2
            elif is_string_line:
                # string name
                self.tab_canvas.create_text(pad_x, y, text=line[:2], anchor="nw",
                                            fill=ACCENT, font=tab_font)
                x = pad_x + tab_font.measure(line[:2])

                # classify rest into colour spans
                spans: list[tuple[str, str]] = []
                buf = ""
                buf_colour = ""
                for ch in line[2:]:
                    if ch.isdigit():
                        colour = "#ffffff"
                    elif ch == "|":
                        colour = ACCENT
                    else:
                        colour = "#222a3a"
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
        self._gutter.configure(scrollregion=(0, 0, 36, y + 40))

    #  waveform drawing
    def _draw_waveform(self):
        """Draw the audio waveform on the waveform canvas."""
        cvs = self._waveform_canvas
        cvs.delete("all")
        if self.audio_data is None:
            return

        import numpy as np
        data = np.asarray(self.audio_data, dtype=float)
        cvs.update_idletasks()
        w = max(cvs.winfo_width(), 400)
        h = cvs.winfo_height()

        # down-sample into w bins
        n = len(data)
        bin_size = max(1, n // w)
        trimmed = data[:bin_size * w]
        if len(trimmed) == 0:
            return

        reshaped = trimmed[:bin_size * (len(trimmed) // bin_size)].reshape(-1, bin_size)
        maxes = np.max(reshaped, axis=1)
        mins = np.min(reshaped, axis=1)
        peak = max(abs(data.max()), abs(data.min()), 1e-9)

        mid = h / 2
        scale = (h / 2 - 4) / peak

        # filled waveform polygon
        points_top = []
        points_bot = []
        for i, (mx, mn) in enumerate(zip(maxes, mins)):
            x = i
            points_top.append((x, mid - mx * scale))
            points_bot.append((x, mid - mn * scale))

        points_bot.reverse()
        poly = points_top + points_bot
        flat_coords = [c for pt in poly for c in pt]

        if len(flat_coords) >= 6:
            cvs.create_polygon(flat_coords, fill="#1a3a5c", outline=ACCENT, width=0.5,
                               smooth=False)

        # zero line
        cvs.create_line(0, mid, w, mid, fill=DIM, width=1, dash=(4, 4))

        # time labels
        sr = self.audio_proc.sample_rate
        duration = n / sr
        for sec in range(0, int(duration) + 1, max(1, int(duration) // 8)):
            x = int(sec / duration * w) if duration > 0 else 0
            cvs.create_text(x + 2, h - 4, text=f"{sec}s", anchor="sw",
                            fill=DIM, font=("Menlo", 8))

    #  note timeline drawing
    def _draw_note_timeline(self):
        """Draw coloured blocks for each detected note on the timeline canvas."""
        cvs = self._timeline_canvas
        cvs.delete("all")

        if not self._detected_notes:
            cvs.create_text(400, 25,
                text="No notes detected", fill=DIM, font=("Helvetica", 10))
            return

        cvs.update_idletasks()
        w = max(cvs.winfo_width(), 400)
        h = cvs.winfo_height()

        notes = self._detected_notes
        # get time range
        max_time = 0
        for n in notes:
            t = n.get("onset", n.get("start_time", 0)) + n.get("duration", 0.1)
            if t > max_time:
                max_time = t
        if max_time <= 0:
            max_time = 1

        # frequency range for y-mapping
        freqs = [n.get("frequency", 0) for n in notes if n.get("frequency", 0) > 0]
        if not freqs:
            return
        min_freq = min(freqs) * 0.9
        max_freq = max(freqs) * 1.1
        freq_range = max_freq - min_freq if max_freq > min_freq else 1

        for n in notes:
            onset = n.get("onset", n.get("start_time", 0))
            dur = n.get("duration", 0.1)
            freq = n.get("frequency", 0)
            note_name = n.get("note", "C")

            if freq <= 0:
                continue

            # strip octave number for colour
            base = note_name.rstrip("0123456789")
            colour = _note_colour(base)

            x1 = int(onset / max_time * w)
            x2 = max(x1 + 3, int((onset + dur) / max_time * w))
            y_norm = (freq - min_freq) / freq_range
            y1 = int((1 - y_norm) * (h - 12)) + 4
            y2 = min(y1 + 10, h - 2)

            cvs.create_rectangle(x1, y1, x2, y2, fill=colour, outline="", width=0)
            if x2 - x1 > 18:
                cvs.create_text((x1 + x2) // 2, (y1 + y2) // 2,
                                text=note_name, fill="#000", font=("Helvetica", 7, "bold"))

        # also mark onsets on waveform canvas as vertical lines
        wf = self._waveform_canvas
        wf_w = max(wf.winfo_width(), 400)
        wf_h = wf.winfo_height()
        sr = self.audio_proc.sample_rate
        audio_duration = len(self.audio_data) / sr if self.audio_data is not None else max_time
        for n in notes:
            onset = n.get("onset", n.get("start_time", 0))
            x = int(onset / audio_duration * wf_w) if audio_duration > 0 else 0
            wf.create_line(x, 0, x, wf_h, fill="#ffffff20", width=1, dash=(2, 3))

    #  toolbar actions
    def _copy_tabs(self):
        if not self._raw_tabs_text:
            return
        self.clipboard_clear()
        self.clipboard_append(self._raw_tabs_text)
        self.set_status("Tabs copied to clipboard", "ok")

    def _zoom_in(self):
        self._tab_font_size = min(28, self._tab_font_size + 2)
        self._font_size_var.set(f"{self._tab_font_size}px")
        if self._raw_tabs_text:
            self._render_tabs_on_canvas(self._raw_tabs_text)

    def _zoom_out(self):
        self._tab_font_size = max(8, self._tab_font_size - 2)
        self._font_size_var.set(f"{self._tab_font_size}px")
        if self._raw_tabs_text:
            self._render_tabs_on_canvas(self._raw_tabs_text)

    def _zoom_reset(self):
        self._tab_font_size = 13
        self._font_size_var.set(f"{self._tab_font_size}px")
        if self._raw_tabs_text:
            self._render_tabs_on_canvas(self._raw_tabs_text)

    #  status / progress helpers
    def set_status(self, text: str, level: str = "info"):
        """Thread-safe status update with colour level (info/ok/warn/err)."""
        self.after(0, lambda: self._do_set_status(text, level))

    def _do_set_status(self, text: str, level: str):
        colour_map = {"info": ACCENT, "ok": SUCCESS, "warn": WARN, "err": RED}
        colour = colour_map.get(level, ACCENT)
        self.status_var.set(text)
        self._status_dot.configure(fg=colour)

    def _show_progress(self, fraction: float = 0.0, message: str = ""):
        self.after(0, lambda: self._do_show_progress(fraction, message))

    def _do_show_progress(self, fraction: float, message: str):
        if not self._progress_visible:
            self._progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._progress_lbl_widget.pack(side=tk.LEFT, padx=(10, 0))
            self._progress_elapsed_lbl.pack(side=tk.RIGHT, padx=(0, 4))
            self._progress_visible = True
            self._start_elapsed_timer()
        self._progress_var.set(max(0.0, min(1.0, fraction)))
        self._progress_label.set(message)

    def _hide_progress(self):
        self.after(0, self._do_hide_progress)

    def _do_hide_progress(self):
        if self._progress_visible:
            self._progress_bar.pack_forget()
            self._progress_lbl_widget.pack_forget()
            self._progress_elapsed_lbl.pack_forget()
            self._progress_visible = False
            self._progress_var.set(0.0)
            self._progress_label.set("")
            self._stop_elapsed_timer()

    def _start_elapsed_timer(self):
        self._task_start_time = time.time()
        self._tick_elapsed()

    def _tick_elapsed(self):
        if not self._progress_visible:
            return
        elapsed = time.time() - self._task_start_time
        self._progress_elapsed_lbl.configure(text=f"{elapsed:.1f}s")
        self._elapsed_timer_id = self.after(200, self._tick_elapsed)

    def _stop_elapsed_timer(self):
        if self._elapsed_timer_id:
            self.after_cancel(self._elapsed_timer_id)
            self._elapsed_timer_id = None
        self._progress_elapsed_lbl.configure(text="")

    def _run_task(self, task_func):
        self._show_progress(0.0, "Starting...")
        def wrapper():
            try:
                task_func()
            finally:
                self._hide_progress()
        threading.Thread(target=wrapper, daemon=True).start()

    def _show_note_stats(self, notes: list):
        """Show brief note statistics below the progress bar."""
        if not notes:
            self._stats_var.set("")
            self._stats_lbl.pack_forget()
            return

        count = len(notes)
        durations = [n.get("duration", 0) for n in notes]
        onsets = [n.get("onset", n.get("start_time", 0)) for n in notes]
        time_span = max(onsets) - min(onsets) if onsets else 0
        min_d = min(durations) if durations else 0
        max_d = max(durations) if durations else 0

        unique_notes = set(n.get("note", "?") for n in notes)
        self._stats_var.set(
            f"📊  {count} notes  ·  {time_span:.1f}s span  ·  "
            f"dur {min_d:.3f}–{max_d:.3f}s  ·  {len(unique_notes)} unique pitches"
        )
        self._stats_lbl.pack(anchor="w")

    #  property helpers
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

    #  file loading (shared between dialog & dnd)
    def _load_file(self, file_path: str):
        self.set_status("Loading...", "info")
        self._show_progress(0.5, "Loading audio file...")
        try:
            audio, _ = self.audio_proc.load_audio_file(file_path)
            self.audio_data = audio
            self._loaded_path = file_path
            name = os.path.basename(file_path)
            sr = self.audio_proc.sample_rate
            dur = len(audio) / sr
            self.file_info_var.set(f"📂  {name}  ({dur:.1f}s  ·  {sr}Hz)")
            self.title(f"🎸 Guitar Tab Generator — {name}")
            self.set_status(f"Loaded: {name}", "ok")
            self._draw_waveform()
            # reset timeline until new analysis
            self._timeline_canvas.delete("all")
            self._timeline_canvas.create_text(
                400, 25, text="Click ⚡ Analyze to detect notes",
                fill=DIM, font=("Helvetica", 10))
            self._detected_notes = []
            self._stats_var.set("")
            self._stats_lbl.pack_forget()
        except Exception as exc:
            self.set_status("Load failed", "err")
            messagebox.showerror("Error", str(exc))
        finally:
            self._hide_progress()

    #  actions
    def on_load(self):
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self._load_file(file_path)

    def on_record(self):
        try:
            duration = float(self.duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Duration must be a number.")
            return
        def task():
            self.set_status("Recording...", "info")
            self._show_progress(0.0, f"Recording {duration:.0f}s...")
            try:
                self.audio_data = self.audio_proc.record_from_microphone(duration)
                self._loaded_path = None
                self.file_info_var.set(f"🎤  Recorded {duration:.1f}s")
                self.title("🎸 Guitar Tab Generator — Recording")
                self.set_status("Recording complete", "ok")
                self.after(0, self._draw_waveform)
            except Exception as exc:
                self.set_status("Recording failed", "err")
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
            self.set_status("Analyzing audio...", "info")
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
                self._detected_notes = notes
                tabs_text = self.tab_gen.format_tabs_as_text()

                self.after(0, lambda: self._render_tabs_on_canvas(tabs_text))
                self.after(50, self._draw_note_timeline)
                self.after(0, lambda: self._show_note_stats(notes))
                self.set_status(f"Done — {len(notes)} notes detected", "ok")
            except Exception as exc:
                self.set_status("Analysis failed", "err")
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
            self.set_status("Auto-tuning...", "info")
            try:
                tune = find_best_extraction(
                    self.audio_data, self.pitch_det, use_harmonic=True,
                    progress_callback=lambda f, m: self._show_progress(f * 0.9, m),
                )
                self._show_progress(0.95, "Generating tabs...")
                self.tab_gen.set_max_fret(max_fret)
                self.tab_gen.generate_tabs(tune.notes)
                self._timed_events = self.tab_gen.get_timed_events()
                self._detected_notes = tune.notes
                tabs_text = self.tab_gen.format_tabs_as_text()

                self.after(0, lambda: self._render_tabs_on_canvas(tabs_text))
                self.after(50, self._draw_note_timeline)
                self.after(0, lambda: self._show_note_stats(tune.notes))
                self.after(0, lambda: self.min_duration_var.set(str(tune.min_duration)))
                self.after(0, lambda: self.min_voiced_var.set(str(tune.min_voiced_prob)))
                self.after(0, lambda: self.segment_var.set(str(tune.segment_seconds)))
                self.after(0, lambda: self.use_harmonic_var.set(tune.use_harmonic))
                self.set_status(f"Best quality — {len(tune.notes)} notes", "ok")
            except Exception as exc:
                self.set_status("Auto-tune failed", "err")
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
            self.set_status("Refining tabs...", "info")
            try:
                result = refine_tabs_with_original(tabs_text=tabs_text,
                                                    original_audio_path=original_path)
                self.after(0, lambda: self._render_tabs_on_canvas(result.refined_tabs_text))
                self.set_status(
                    f"Refined: {result.changes_count} changes, "
                    f"step={result.estimated_step_seconds:.3f}s", "ok")
            except Exception as exc:
                self.set_status("Refine failed", "err")
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
            self.set_status("Checking tabs...", "info")
            try:
                r = check_tabs_against_original(tabs_text=tabs_text,
                                                original_audio_path=original_path)
                self.set_status(
                    f"overall={r.overall_score:.4f}  chroma={r.chroma_score:.4f}  "
                    f"onset={r.onset_score:.4f}", "ok")
                self.after(0, lambda: messagebox.showinfo("Tabs Check", "\n".join([
                    f"Overall score:  {r.overall_score:.4f}",
                    f"Chroma score:   {r.chroma_score:.4f}",
                    f"Onset score:    {r.onset_score:.4f}",
                    f"Est. step:      {r.estimated_step_seconds:.3f}s",
                    f"Est. note:      {r.estimated_note_seconds:.3f}s",
                    f"Analyzed:       {r.analyzed_seconds:.1f}s",
                ])))
            except Exception as exc:
                self.set_status("Check failed", "err")
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
            self.set_status("Synthesizing...", "info")
            try:
                if self._timed_events:
                    _, result = synthesize_from_timed_events(
                        self._timed_events, output_path=out_path, play=True,
                    )
                else:
                    result = synthesize_from_tabs_text(tabs_text,
                                                       output_path=out_path, play=True)
                self.set_status(
                    f"Rendered: {result.notes_count} notes, {result.duration:.1f}s", "ok")
            except Exception as exc:
                self.set_status("Render failed", "err")
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
            self.set_status("Matching (may take a while)...", "info")
            try:
                result = optimize_synth_against_original(
                    tabs_text=tabs_text, original_audio_path=original_path,
                    output_path=out_path,
                )
                self.set_status(f"Matched: score={result.score:.4f}", "ok")
            except Exception as exc:
                self.set_status("Match failed", "err")
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
            self.set_status("Tabs saved ✓", "ok")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def on_self_test(self):
        def task():
            self.set_status("Running self-test...", "info")
            try:
                result = run_sine_test(self.pitch_det)
                if result.success:
                    self.set_status(
                        f"Self-test passed ✓ — detected {result.detected_note} "
                        f"(expected {result.expected_note})", "ok")
                else:
                    self.set_status(
                        f"Self-test failed ✗ — detected {result.detected_note} "
                        f"(expected {result.expected_note})", "err")
            except Exception as exc:
                self.set_status(f"Self-test error: {exc}", "err")
        self._run_task(task)

    def on_help(self):
        mod = "⌘" if self.tk.call("tk", "windowingsystem") == "aqua" else "Ctrl+"
        messagebox.showinfo("Help — Guitar Tab Generator", "\n".join([
            "QUICK START",
            "1.  Load Audio or drag a file onto the window",
            "2.  Click ⚡ Analyze to generate tabs",
            "3.  Save Tabs or Render Audio",
            "",
            "IMPROVE ACCURACY",
            "  ✨ Best Quality — auto-tune extraction params",
            "  🔧 Refine — correct frets using original audio",
            "  ✅ Check Tabs — see objective match scores",
            "  🎯 Match Original — optimize synth quality",
            "",
            "KEYBOARD SHORTCUTS",
            f"  {mod}O — Open audio file",
            f"  {mod}S — Save tabs to file",
            f"  {mod}R — Render synthesized audio",
            f"  {mod}↩ — Analyze audio",
            f"  {mod}+ / {mod}- — Zoom in/out tabs",
            f"  {mod}0 — Reset zoom",
            "",
            "PARAMETERS",
            "  Duration — mic recording length (seconds)",
            "  Min dur — shortest note to detect",
            "  Voiced prob — confidence filter (0–1)",
            "  Max fret — highest fret to use in tabs",
            "  Segment — process long audio in chunks",
            "  Harmonic — isolate harmonic content first",
        ]))


if __name__ == "__main__":
    app = GuitarTabApp()
    app.mainloop()
