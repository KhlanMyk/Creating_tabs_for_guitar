"""Premium Tkinter GUI for Guitar Tab Generator — v3 Redesign."""
from __future__ import annotations

import math
import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font as tkfont
from tkinter import dnd  # type: ignore[attr-defined]

from audio_processor import AudioProcessor
from auto_tune import find_best_extraction
from pitch_detector import PitchDetector
from synth_matcher import optimize_synth_against_original
from tab_checker import check_tabs_against_original
from tab_generator import GuitarTabGenerator
from tab_refiner import refine_tabs_with_original
from self_test import run_sine_test
from tab_synth import synthesize_from_tabs_text, synthesize_from_timed_events

import numpy as np

# ═══════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════
BG            = "#0d1117"
SURFACE       = "#161b22"
SURFACE2      = "#1c2333"
CARD          = "#1a2234"
CARD_HOVER    = "#222d42"
SIDEBAR_BG    = "#0d1117"
SIDEBAR_EDGE  = "#1b2838"

ACCENT        = "#58a6ff"
ACCENT_GLOW   = "#1f6feb"
ACCENT2       = "#f78166"
ACCENT3       = "#7ee787"
ACCENT4       = "#d2a8ff"
ACCENT5       = "#ffa657"

TEXT          = "#e6edf3"
TEXT2         = "#c9d1d9"
MUTED         = "#8b949e"
DIM           = "#484f58"
FAINT         = "#30363d"

BORDER        = "#21262d"
BORDER_LIGHT  = "#30363d"

# Tab canvas
TAB_BG        = "#0a0e14"
TAB_NUM       = "#79c0ff"
TAB_DASH      = "#1e2736"
TAB_BAR       = "#484f58"
TAB_STRING_LBL = "#f0883e"
TAB_CHORD     = "#d2a8ff"

# Section accents
SEC1          = "#3fb950"   # load
SEC1_HOVER    = "#56d364"
SEC2          = "#58a6ff"   # process
SEC2_HOVER    = "#79c0ff"
SEC3          = "#d2a8ff"   # export
SEC3_HOVER    = "#e2c5ff"
SEC_DANGER    = "#f85149"

# Waveform
WAVE_BG       = "#0d1117"
WAVE_LINE     = "#1f6feb"
WAVE_FILL     = "#1a3a5c"
WAVE_ONSET    = "#f8514960"
WAVE_GRID     = "#161b22"

# Note timeline
TL_BG         = "#0d1117"
TL_NOTE       = "#58a6ff"
TL_NOTE_FILL  = "#1a3a5c"

SUCCESS       = "#3fb950"
WARN          = "#d29922"
ERROR         = "#f85149"

# Keyboard shortcuts
SHORTCUT_OPEN    = "<Control-o>"
SHORTCUT_SAVE    = "<Control-s>"
SHORTCUT_ANALYZE = "<Control-Return>"
SHORTCUT_RENDER  = "<Control-r>"


class GuitarTabApp(tk.Tk):
    """v3 Redesigned GUI with waveform, note timeline, toolbar & shortcuts."""

    def __init__(self):
        super().__init__()
        self.title("  Guitar Tab Generator")
        self.geometry("1480x920")
        self.minsize(1180, 720)
        self.configure(bg=BG)

        # ── state ───────────────────────────────────────
        self.audio_proc = AudioProcessor(sample_rate=22050)
        self.pitch_det  = PitchDetector(sample_rate=22050)
        self.tab_gen    = GuitarTabGenerator()
        self.audio_data: np.ndarray | None = None
        self._loaded_path: str | None = None
        self._timed_events: list[dict] = []
        self._detected_notes: list[dict] = []
        self._raw_tabs_text = ""
        self._tab_font_size = 13
        self._task_start_time: float = 0.0

        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()

    # ═══════════════════════════════════════════════════
    #  STYLES
    # ═══════════════════════════════════════════════════
    def _setup_styles(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except Exception:
            pass

        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT)

        s.configure("Custom.Horizontal.TProgressbar",
                     troughcolor=FAINT, background=ACCENT,
                     darkcolor=ACCENT_GLOW, lightcolor=ACCENT,
                     bordercolor=BORDER, thickness=6)

        s.configure("Toolbar.TButton",
                     background=SURFACE2, foreground=TEXT,
                     borderwidth=0, padding=(10, 4))
        s.map("Toolbar.TButton",
               background=[("active", CARD_HOVER)])

    # ═══════════════════════════════════════════════════
    #  LAYOUT
    # ═══════════════════════════════════════════════════
    def _build_ui(self):
        # ── SIDEBAR ─────────────────────────────────────
        sidebar_outer = tk.Frame(self, bg=SIDEBAR_BG, width=300)
        sidebar_outer.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_outer.pack_propagate(False)

        # thin edge line
        tk.Frame(sidebar_outer, bg=BORDER, width=1).pack(side=tk.RIGHT, fill=tk.Y)

        # scrollable inner
        self._sidebar_canvas = tk.Canvas(sidebar_outer, bg=SIDEBAR_BG,
                                          highlightthickness=0, width=298)
        self._sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(self._sidebar_canvas, bg=SIDEBAR_BG)
        self._sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw", width=298)

        def _on_sb_cfg(e):
            self._sidebar_canvas.configure(scrollregion=self._sidebar_canvas.bbox("all"))
        sidebar.bind("<Configure>", _on_sb_cfg)

        # mousewheel (macOS + linux)
        self._sidebar_canvas.bind("<MouseWheel>",
            lambda e: self._sidebar_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        self._sidebar_canvas.bind("<Button-4>",
            lambda e: self._sidebar_canvas.yview_scroll(-3, "units"))
        self._sidebar_canvas.bind("<Button-5>",
            lambda e: self._sidebar_canvas.yview_scroll(3, "units"))

        # ── Logo ────────────────────────────────────────
        logo_fr = tk.Frame(sidebar, bg=SIDEBAR_BG)
        logo_fr.pack(fill=tk.X, padx=20, pady=(24, 4))
        tk.Label(logo_fr, text="\U0001F3B8", font=("Apple Color Emoji", 22),
                 bg=SIDEBAR_BG, fg=TEXT).pack(side=tk.LEFT, padx=(0, 10))
        title_fr = tk.Frame(logo_fr, bg=SIDEBAR_BG)
        title_fr.pack(side=tk.LEFT)
        tk.Label(title_fr, text="Guitar Tab Generator",
                 font=("SF Pro Display", 16, "bold"), fg=TEXT,
                 bg=SIDEBAR_BG).pack(anchor="w")
        tk.Label(title_fr, text="Audio \u2192 Tabs \u2192 Playback",
                 font=("SF Pro Text", 10), fg=DIM,
                 bg=SIDEBAR_BG).pack(anchor="w")

        self._sep(sidebar)

        # ── STEP 1 — Load ──────────────────────────────
        self._section(sidebar, "\u2460", "Load Audio", SEC1,
                      "Import a song or record from mic")
        self._btn(sidebar, "\U0001F4C2", "Open File", "MP3 WAV FLAC OGG M4A  \u2318O",
                  self.on_load, SEC1, SEC1_HOVER)
        self._btn(sidebar, "\U0001F3A4", "Record Mic", "Live capture from microphone",
                  self.on_record, SEC1, SEC1_HOVER)

        self._sep(sidebar)

        # ── STEP 2 — Process ───────────────────────────
        self._section(sidebar, "\u2461", "Generate Tabs", SEC2,
                      "Detect notes and build tablature")
        self._btn(sidebar, "\u26A1", "Analyze", "Detect notes & create tabs  \u23CE",
                  self.on_analyze, ACCENT2, "#ffb088", primary=True)
        self._btn(sidebar, "\u2728", "Best Quality", "Auto-tune all extraction params",
                  self.on_best_quality, SEC2, SEC2_HOVER)
        self._btn(sidebar, "\U0001F527", "Refine", "Correct frets vs original audio",
                  self.on_refine_with_original, SEC2, SEC2_HOVER)

        self._sep(sidebar)

        # ── STEP 3 — Export ─────────────────────────────
        self._section(sidebar, "\u2462", "Export & Verify", SEC3,
                      "Save tabs, render and compare")
        self._btn(sidebar, "\U0001F4BE", "Save Tabs", "Export tablature to .txt  \u2318S",
                  self.on_save, SEC3, SEC3_HOVER)
        self._btn(sidebar, "\U0001F50A", "Render Audio", "Synthesize guitar WAV  \u2318R",
                  self.on_render_audio, SEC3, SEC3_HOVER)
        self._btn(sidebar, "\U0001F3AF", "Match Original", "Optimize synth vs source",
                  self.on_match_original, SEC3, SEC3_HOVER)
        self._btn(sidebar, "\u2705", "Check Tabs", "Score accuracy against original",
                  self.on_check_tabs, SEC3, SEC3_HOVER)

        self._sep(sidebar)

        self._btn(sidebar, "\u2753", "Help", "Keyboard shortcuts & tips",
                  self.on_help, DIM, MUTED)

        tk.Frame(sidebar, bg=SIDEBAR_BG, height=20).pack()

        # ── Status footer ───────────────────────────────
        self._status_frame = tk.Frame(sidebar_outer, bg=SURFACE)
        self._status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Frame(self._status_frame, bg=BORDER, height=1).pack(fill=tk.X)

        self.status_var = tk.StringVar(value="\u2714  Ready")
        self._status_lbl = tk.Label(self._status_frame, textvariable=self.status_var,
                                     fg=SUCCESS, bg=SURFACE,
                                     font=("SF Mono", 10), anchor="w", wraplength=270)
        self._status_lbl.pack(padx=16, pady=10)

        # ═══════════════════════════════════════════════
        #  MAIN CONTENT
        # ═══════════════════════════════════════════════
        main = tk.Frame(self, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Top parameter bar ───────────────────────────
        topbar = tk.Frame(main, bg=SURFACE, highlightbackground=BORDER,
                          highlightthickness=1)
        topbar.pack(fill=tk.X, padx=16, pady=(12, 0))

        self._param_entries = {}
        params = [
            ("Duration", "duration", "5", "s"),
            ("Min dur", "min_dur", "0.08", "s"),
            ("Voiced", "voiced", "0.55", ""),
            ("Max fret", "fret", "15", ""),
            ("Segment", "segment", "15", "s"),
        ]
        for i, (label, key, default, unit) in enumerate(params):
            f = tk.Frame(topbar, bg=SURFACE)
            f.grid(row=0, column=i, padx=(16 if i == 0 else 6, 6), pady=10, sticky="w")
            tk.Label(f, text=label, fg=MUTED, bg=SURFACE,
                     font=("SF Pro Text", 9)).pack(anchor="w")
            row = tk.Frame(f, bg=SURFACE)
            row.pack(anchor="w")
            var = tk.StringVar(value=default)
            e = tk.Entry(row, textvariable=var, width=5, bg=CARD, fg=TEXT,
                         insertbackground=ACCENT, relief="flat",
                         font=("SF Mono", 11),
                         highlightbackground=BORDER, highlightthickness=1)
            e.pack(side=tk.LEFT)
            if unit:
                tk.Label(row, text=unit, fg=DIM, bg=SURFACE,
                         font=("SF Pro Text", 9)).pack(side=tk.LEFT, padx=(3, 0))
            self._param_entries[key] = var

        self.use_harmonic_var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(topbar, text="\u266A Harmonic", variable=self.use_harmonic_var,
                            bg=SURFACE, fg=TEXT, selectcolor=CARD,
                            activebackground=SURFACE, activeforeground=TEXT,
                            font=("SF Pro Text", 10))
        cb.grid(row=0, column=len(params), padx=16, pady=10)

        # ── File info + notes stats bar ─────────────────
        info_bar = tk.Frame(main, bg=BG)
        info_bar.pack(fill=tk.X, padx=16, pady=(8, 0))

        self.file_info_var = tk.StringVar(value="\U0001F4C1 No audio loaded")
        tk.Label(info_bar, textvariable=self.file_info_var, fg=MUTED, bg=BG,
                 font=("SF Pro Text", 11)).pack(side=tk.LEFT)

        self._notes_info_var = tk.StringVar(value="")
        tk.Label(info_bar, textvariable=self._notes_info_var, fg=ACCENT, bg=BG,
                 font=("SF Mono", 10)).pack(side=tk.RIGHT)

        # ── Progress bar ────────────────────────────────
        self._progress_frame = tk.Frame(main, bg=BG)
        self._progress_frame.pack(fill=tk.X, padx=16, pady=(4, 0))

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_label = tk.StringVar(value="")

        self._progress_bar = ttk.Progressbar(
            self._progress_frame, variable=self._progress_var,
            maximum=1.0, mode="determinate",
            style="Custom.Horizontal.TProgressbar")

        self._progress_lbl_widget = tk.Label(
            self._progress_frame, textvariable=self._progress_label,
            fg=MUTED, bg=BG, font=("SF Mono", 9), anchor="w")

        self._progress_visible = False

        # ── Waveform + Note timeline panel ──────────────
        self._wave_frame = tk.Frame(main, bg=WAVE_BG,
                                     highlightbackground=BORDER, highlightthickness=1)
        self._wave_frame.pack(fill=tk.X, padx=16, pady=(8, 0))

        # waveform canvas (top half)
        self._wave_canvas = tk.Canvas(self._wave_frame, bg=WAVE_BG,
                                       highlightthickness=0, height=70)
        self._wave_canvas.pack(fill=tk.X, expand=True)

        # note timeline canvas (bottom half)
        self._tl_canvas = tk.Canvas(self._wave_frame, bg=TL_BG,
                                     highlightthickness=0, height=30)
        self._tl_canvas.pack(fill=tk.X, expand=True)

        self._draw_wave_placeholder()

        # ── Tab toolbar ─────────────────────────────────
        toolbar = tk.Frame(main, bg=SURFACE2)
        toolbar.pack(fill=tk.X, padx=16, pady=(8, 0))

        self._make_tool_btn(toolbar, "\U0001F4CB Copy", self._copy_tabs)
        self._make_tool_btn(toolbar, "\U0001F50D\u2212", self._zoom_out)
        self._make_tool_btn(toolbar, "\U0001F50D+", self._zoom_in)
        self._make_tool_btn(toolbar, "\u21BA Reset", self._zoom_reset)

        self._tab_info_var = tk.StringVar(value="")
        tk.Label(toolbar, textvariable=self._tab_info_var, fg=DIM, bg=SURFACE2,
                 font=("SF Mono", 9), anchor="e").pack(side=tk.RIGHT, padx=12, pady=4)

        # ── Tab display (canvas) ────────────────────────
        tab_frame = tk.Frame(main, bg=TAB_BG,
                              highlightbackground=BORDER, highlightthickness=1)
        tab_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(4, 16))

        self.tab_canvas = tk.Canvas(tab_frame, bg=TAB_BG, highlightthickness=0)
        sb_y = ttk.Scrollbar(tab_frame, orient=tk.VERTICAL, command=self.tab_canvas.yview)
        sb_x = ttk.Scrollbar(tab_frame, orient=tk.HORIZONTAL, command=self.tab_canvas.xview)
        self.tab_canvas.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tab_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._draw_placeholder()

        # ── Drag & drop zone indicator ──────────────────
        self._setup_dnd()

    # ═══════════════════════════════════════════════════
    #  WIDGET HELPERS
    # ═══════════════════════════════════════════════════
    def _sep(self, parent):
        """Thin horizontal separator."""
        tk.Frame(parent, bg=FAINT, height=1).pack(fill=tk.X, padx=20, pady=(10, 6))

    def _section(self, parent, badge_text: str, title: str, color: str, subtitle: str):
        """Section header with circled number badge."""
        frame = tk.Frame(parent, bg=SIDEBAR_BG)
        frame.pack(fill=tk.X, padx=20, pady=(8, 2))

        tk.Label(frame, text=badge_text, font=("SF Pro Display", 16),
                 fg=color, bg=SIDEBAR_BG).pack(side=tk.LEFT, padx=(0, 8))

        tf = tk.Frame(frame, bg=SIDEBAR_BG)
        tf.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(tf, text=title, font=("SF Pro Display", 13, "bold"),
                 fg=TEXT, bg=SIDEBAR_BG).pack(anchor="w")
        tk.Label(tf, text=subtitle, font=("SF Pro Text", 9),
                 fg=DIM, bg=SIDEBAR_BG, wraplength=200).pack(anchor="w")

    def _btn(self, parent, icon: str, title: str, desc: str, command,
             accent: str, hover: str, primary: bool = False):
        """Sidebar action button with icon, title and description."""
        bg = ACCENT_GLOW if primary else CARD
        hover_bg = hover if not primary else ACCENT

        card = tk.Frame(parent, bg=bg, cursor="hand2",
                        highlightbackground=accent if primary else BORDER,
                        highlightthickness=1)
        card.pack(fill=tk.X, padx=20, pady=2)

        inner = tk.Frame(card, bg=bg)
        inner.pack(fill=tk.X, padx=10, pady=7)

        icon_lbl = tk.Label(inner, text=icon, font=("Apple Color Emoji", 13),
                            bg=bg, fg=TEXT)
        icon_lbl.pack(side=tk.LEFT, padx=(0, 8))

        text_fr = tk.Frame(inner, bg=bg)
        text_fr.pack(side=tk.LEFT, fill=tk.X, expand=True)

        title_lbl = tk.Label(text_fr, text=title,
                             font=("SF Pro Text", 11, "bold") if primary else ("SF Pro Text", 11),
                             fg="#fff" if primary else TEXT, bg=bg, anchor="w")
        title_lbl.pack(anchor="w")

        desc_lbl = tk.Label(text_fr, text=desc, font=("SF Pro Text", 9),
                            fg=TEXT2 if primary else DIM, bg=bg,
                            anchor="w", wraplength=200)
        desc_lbl.pack(anchor="w")

        all_w = (card, inner, icon_lbl, text_fr, title_lbl, desc_lbl)
        for w in all_w:
            w.bind("<Button-1>", lambda e, c=command: c())
            w.configure(cursor="hand2")

        def _enter(_):
            for w in all_w:
                try:
                    w.configure(bg=hover_bg)
                except tk.TclError:
                    pass
        def _leave(_):
            for w in all_w:
                try:
                    w.configure(bg=bg)
                except tk.TclError:
                    pass
        for w in all_w:
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

    def _make_tool_btn(self, parent, text: str, cmd):
        """Tiny toolbar button."""
        b = tk.Label(parent, text=text, fg=MUTED, bg=SURFACE2,
                     font=("SF Pro Text", 10), cursor="hand2", padx=8, pady=4)
        b.pack(side=tk.LEFT, padx=1)
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>", lambda e: b.configure(fg=TEXT, bg=CARD_HOVER))
        b.bind("<Leave>", lambda e: b.configure(fg=MUTED, bg=SURFACE2))

    # ═══════════════════════════════════════════════════
    #  WAVEFORM & TIMELINE
    # ═══════════════════════════════════════════════════
    def _draw_wave_placeholder(self):
        c = self._wave_canvas
        c.delete("all")
        c.create_text(c.winfo_reqwidth() // 2 or 300, 35,
                       text="\U0001F3B5  Load audio to see waveform",
                       fill=DIM, font=("SF Pro Text", 11), anchor="center")
        tc = self._tl_canvas
        tc.delete("all")
        tc.create_text(tc.winfo_reqwidth() // 2 or 300, 15,
                        text="note timeline", fill=FAINT,
                        font=("SF Pro Text", 9), anchor="center")

    def _draw_waveform(self):
        """Render waveform from self.audio_data onto the waveform canvas."""
        c = self._wave_canvas
        c.update_idletasks()
        c.delete("all")

        if self.audio_data is None or len(self.audio_data) == 0:
            self._draw_wave_placeholder()
            return

        w = c.winfo_width() or 800
        h = c.winfo_height() or 70
        mid = h // 2

        # grid lines
        for gy in range(0, h, 15):
            c.create_line(0, gy, w, gy, fill=WAVE_GRID, width=1)

        audio = self.audio_data
        sr = self.audio_proc.sample_rate

        # downsample to one point per pixel
        step = max(1, len(audio) // w)
        n_pts = min(w, len(audio) // step)

        # For each pixel column, get min and max
        coords_top = []
        coords_bot = []
        for i in range(n_pts):
            chunk = audio[i * step: (i + 1) * step]
            if len(chunk) == 0:
                continue
            mn, mx = float(np.min(chunk)), float(np.max(chunk))
            x = i
            y_top = int(mid - mx * mid * 0.9)
            y_bot = int(mid - mn * mid * 0.9)
            coords_top.append((x, y_top))
            coords_bot.append((x, y_bot))

        # fill polygon
        if coords_top:
            poly = []
            for x, y in coords_top:
                poly.extend([x, y])
            for x, y in reversed(coords_bot):
                poly.extend([x, y])
            if len(poly) >= 6:
                c.create_polygon(poly, fill=WAVE_FILL, outline="")

            # top line
            line_pts = []
            for x, y in coords_top:
                line_pts.extend([x, y])
            if len(line_pts) >= 4:
                c.create_line(line_pts, fill=WAVE_LINE, width=1, smooth=True)

        # time labels
        dur = len(audio) / sr
        for sec in range(0, int(dur) + 1, max(1, int(dur // 8))):
            x = int(sec / dur * w) if dur > 0 else 0
            c.create_text(x + 2, h - 3, text=f"{sec}s", fill=DIM,
                          font=("SF Mono", 7), anchor="sw")

    def _draw_note_timeline(self):
        """Draw detected notes as colored blocks on the timeline."""
        tc = self._tl_canvas
        tc.update_idletasks()
        tc.delete("all")

        notes = self._detected_notes
        if not notes or self.audio_data is None:
            tc.create_text((tc.winfo_width() or 400) // 2, 15,
                            text="no notes detected", fill=FAINT,
                            font=("SF Pro Text", 9), anchor="center")
            return

        w = tc.winfo_width() or 800
        h = tc.winfo_height() or 30
        sr = self.audio_proc.sample_rate
        dur = len(self.audio_data) / sr if sr > 0 else 1.0

        for note in notes:
            t0 = note.get("start_time", 0)
            t1 = note.get("end_time", t0 + note.get("duration", 0.1))
            x0 = int(t0 / dur * w) if dur > 0 else 0
            x1 = max(x0 + 2, int(t1 / dur * w))
            # colour by pitch class
            midi = note.get("midi", 60)
            hue = (midi % 12) / 12.0
            r, g, b = _hsv_to_rgb(hue, 0.5, 0.85)
            color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            tc.create_rectangle(x0, 2, x1, h - 2, fill=color, outline="", stipple="")

        # onset markers on waveform
        c = self._wave_canvas
        ww = c.winfo_width() or 800
        for note in notes:
            t0 = note.get("start_time", 0)
            x = int(t0 / dur * ww) if dur > 0 else 0
            c.create_line(x, 0, x, 70, fill="#f8514940", width=1)

    # ═══════════════════════════════════════════════════
    #  TAB CANVAS
    # ═══════════════════════════════════════════════════
    def _draw_placeholder(self):
        self.tab_canvas.delete("all")
        cx = 400
        cy = 120
        self.tab_canvas.create_text(
            cx, cy - 20,
            text="\U0001F3B8",
            fill=DIM, font=("Apple Color Emoji", 36), anchor="center")
        self.tab_canvas.create_text(
            cx, cy + 30,
            text="Load audio and click Analyze\nto generate guitar tabs",
            fill=MUTED, font=("SF Pro Display", 14), anchor="center", justify="center")
        self.tab_canvas.create_text(
            cx, cy + 65,
            text="\u2318O  Open file    \u23CE  Analyze    \u2318S  Save",
            fill=DIM, font=("SF Mono", 10), anchor="center")

    def _render_tabs_on_canvas(self, tabs_text: str):
        """Draw tab text with syntax-highlighted rendering."""
        self.tab_canvas.delete("all")
        self._raw_tabs_text = tabs_text

        pad_x, pad_y = 20, 16
        line_height = self._tab_font_size + 7
        y = pad_y
        sz = self._tab_font_size

        tab_font = tkfont.Font(family="Menlo", size=sz)
        chord_font = tkfont.Font(family="SF Pro Display", size=sz - 1, weight="bold")
        max_x = 800
        line_num = 0

        for raw_line in tabs_text.split("\n"):
            line = raw_line.rstrip()
            if not line:
                y += 8
                continue

            is_string_line = len(line) >= 2 and line[0] in "eBGDAE" and line[1] == "|"
            is_chord_line = line.startswith("    ") and not is_string_line

            if is_chord_line:
                self.tab_canvas.create_text(pad_x, y, text=line, anchor="nw",
                                            fill=TAB_CHORD, font=chord_font)
                y += line_height
            elif is_string_line:
                line_num += 1
                # line number gutter
                self.tab_canvas.create_text(pad_x - 4, y, text=str(line_num),
                                            anchor="ne", fill=FAINT,
                                            font=tkfont.Font(family="Menlo", size=sz - 3))

                # string label (e.g. "e|")
                self.tab_canvas.create_text(pad_x, y, text=line[:2], anchor="nw",
                                            fill=TAB_STRING_LBL, font=tab_font)
                x = pad_x + tab_font.measure(line[:2])

                # colour spans
                spans: list[tuple[str, str]] = []
                buf = ""
                buf_colour = ""
                for ch in line[2:]:
                    if ch.isdigit():
                        colour = TAB_NUM
                    elif ch == "|":
                        colour = TAB_BAR
                    else:
                        colour = TAB_DASH
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
                y += line_height + 2

        self.tab_canvas.configure(scrollregion=(0, 0, max_x, y + 40))

    # ═══════════════════════════════════════════════════
    #  TOOLBAR ACTIONS
    # ═══════════════════════════════════════════════════
    def _copy_tabs(self):
        txt = self._get_tabs_text()
        if txt:
            self.clipboard_clear()
            self.clipboard_append(txt)
            self.set_status("\U0001F4CB Copied to clipboard")

    def _zoom_in(self):
        self._tab_font_size = min(24, self._tab_font_size + 1)
        if self._raw_tabs_text:
            self._render_tabs_on_canvas(self._raw_tabs_text)

    def _zoom_out(self):
        self._tab_font_size = max(8, self._tab_font_size - 1)
        if self._raw_tabs_text:
            self._render_tabs_on_canvas(self._raw_tabs_text)

    def _zoom_reset(self):
        self._tab_font_size = 13
        if self._raw_tabs_text:
            self._render_tabs_on_canvas(self._raw_tabs_text)

    # ═══════════════════════════════════════════════════
    #  DRAG & DROP
    # ═══════════════════════════════════════════════════
    def _setup_dnd(self):
        """Best-effort drag & drop. Works on macOS with TkDND or fallback."""
        try:
            self.drop_target_register("DND_Files")  # type: ignore
            self.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore
        except Exception:
            pass  # TkDND not available — skip silently

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")  # type: ignore
        if os.path.isfile(path):
            self._load_file(path)

    # ═══════════════════════════════════════════════════
    #  KEYBOARD SHORTCUTS
    # ═══════════════════════════════════════════════════
    def _bind_shortcuts(self):
        self.bind("<Command-o>", lambda e: self.on_load())
        self.bind("<Control-o>", lambda e: self.on_load())
        self.bind("<Command-s>", lambda e: self.on_save())
        self.bind("<Control-s>", lambda e: self.on_save())
        self.bind("<Command-r>", lambda e: self.on_render_audio())
        self.bind("<Control-r>", lambda e: self.on_render_audio())
        self.bind("<Command-Return>", lambda e: self.on_analyze())
        self.bind("<Control-Return>", lambda e: self.on_analyze())
        self.bind("<Command-plus>", lambda e: self._zoom_in())
        self.bind("<Command-equal>", lambda e: self._zoom_in())
        self.bind("<Command-minus>", lambda e: self._zoom_out())

    # ═══════════════════════════════════════════════════
    #  STATUS / PROGRESS
    # ═══════════════════════════════════════════════════
    def set_status(self, text: str, kind: str = "info"):
        """Thread-safe status update."""
        colors = {"info": ACCENT, "ok": SUCCESS, "warn": WARN, "err": ERROR}
        c = colors.get(kind, MUTED)
        self.after(0, lambda: self._do_set_status(text, c))

    def _do_set_status(self, text: str, color: str):
        self.status_var.set(text)
        self._status_lbl.configure(fg=color)

    def _show_progress(self, fraction: float = 0.0, message: str = ""):
        self.after(0, lambda: self._do_show_progress(fraction, message))

    def _do_show_progress(self, fraction: float, message: str):
        if not self._progress_visible:
            self._progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._progress_lbl_widget.pack(side=tk.LEFT, padx=(8, 0))
            self._progress_visible = True
        self._progress_var.set(max(0.0, min(1.0, fraction)))
        elapsed = time.time() - self._task_start_time
        ts = f"  [{elapsed:.0f}s]" if elapsed > 1.5 else ""
        self._progress_label.set(f"{message}{ts}")

    def _hide_progress(self):
        self.after(0, self._do_hide_progress)

    def _do_hide_progress(self):
        if self._progress_visible:
            self._progress_bar.pack_forget()
            self._progress_lbl_widget.pack_forget()
            self._progress_visible = False
            self._progress_var.set(0.0)
            self._progress_label.set("")

    def _run_task(self, task_func):
        self._task_start_time = time.time()
        self._show_progress(0.0, "Starting...")
        def wrapper():
            try:
                task_func()
            finally:
                self._hide_progress()
        threading.Thread(target=wrapper, daemon=True).start()

    # ── property helpers ────────────────────────────────
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

    # ═══════════════════════════════════════════════════
    #  ACTIONS
    # ═══════════════════════════════════════════════════
    def _load_file(self, file_path: str):
        """Load audio from a given path (used by both on_load and DnD)."""
        self.set_status("\U0001F4C2 Loading...", "info")
        self._show_progress(0.5, "Loading audio file...")
        try:
            audio, _ = self.audio_proc.load_audio_file(file_path)
            self.audio_data = audio
            self._loaded_path = file_path
            name = os.path.basename(file_path)
            sr = self.audio_proc.sample_rate
            dur_s = len(audio) / sr
            self.file_info_var.set(f"\U0001F3B5 {name}  ({dur_s:.1f}s, {sr}Hz)")
            self.set_status(f"\u2714 Audio loaded — {dur_s:.1f}s", "ok")
            self.after(50, self._draw_waveform)
        except Exception as exc:
            self.set_status(f"\u2716 Load failed", "err")
            messagebox.showerror("Error", str(exc))
        finally:
            self._hide_progress()

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
            self.set_status(f"\U0001F3A4 Recording {duration:.0f}s...", "warn")
            try:
                self.audio_data = self.audio_proc.record_from_microphone(duration)
                self._loaded_path = None
                self.file_info_var.set(f"\U0001F3A4 Recorded {duration:.1f}s")
                self.set_status("\u2714 Recording complete", "ok")
                self.after(50, self._draw_waveform)
            except Exception as exc:
                self.set_status("\u2716 Recording failed", "err")
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
            self.set_status("\u26A1 Analyzing audio...", "info")
            try:
                self.tab_gen.set_max_fret(max_fret)
                notes = self.pitch_det.extract_notes_from_audio(
                    self.audio_data, min_duration=min_dur, min_voiced_prob=min_voiced,
                    use_harmonic=self.use_harmonic_var.get(),
                    segment_seconds=seg,
                    use_onset_alignment=True,
                    progress_callback=lambda f, m: self._show_progress(f * 0.9, m),
                )
                self._detected_notes = notes
                self._show_progress(0.95, "Generating tabs...")
                self.tab_gen.generate_tabs(notes)
                self._timed_events = self.tab_gen.get_timed_events()
                tabs_text = self.tab_gen.format_tabs_as_text()

                # stats
                if notes:
                    dur_range = f"{min(n['duration'] for n in notes):.2f}-{max(n['duration'] for n in notes):.2f}s"
                    time_span = f"{notes[-1]['end_time']:.1f}s"
                else:
                    dur_range = "-"
                    time_span = "-"

                self.after(0, lambda: self._render_tabs_on_canvas(tabs_text))
                self.after(0, lambda: self._notes_info_var.set(
                    f"\u266A {len(notes)} notes  |  span {time_span}  |  dur {dur_range}"))
                self.after(0, lambda: self._tab_info_var.set(
                    f"{len(tabs_text.splitlines())} lines  |  {len(self._timed_events)} events"))
                self.after(50, self._draw_waveform)
                self.after(100, self._draw_note_timeline)
                elapsed = time.time() - self._task_start_time
                self.set_status(f"\u2714 Done — {len(notes)} notes in {elapsed:.1f}s", "ok")
            except Exception as exc:
                self.set_status("\u2716 Analysis failed", "err")
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
            self.set_status("\u2728 Auto-tuning...", "info")
            try:
                tune = find_best_extraction(
                    self.audio_data, self.pitch_det, use_harmonic=True,
                    progress_callback=lambda f, m: self._show_progress(f * 0.9, m),
                )
                self._detected_notes = tune.notes
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
                self.after(0, lambda: self._notes_info_var.set(
                    f"\u266A {len(tune.notes)} notes (auto-tuned)"))
                self.after(50, self._draw_waveform)
                self.after(100, self._draw_note_timeline)
                elapsed = time.time() - self._task_start_time
                self.set_status(
                    f"\u2714 Best quality — {len(tune.notes)} notes in {elapsed:.1f}s", "ok")
            except Exception as exc:
                self.set_status("\u2716 Auto-tune failed", "err")
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
            self.set_status("\U0001F527 Refining tabs...", "info")
            try:
                result = refine_tabs_with_original(tabs_text=tabs_text, original_audio_path=original_path)
                self.after(0, lambda: self._render_tabs_on_canvas(result.refined_tabs_text))
                self.set_status(
                    f"\u2714 Refined: {result.changes_count} changes, step={result.estimated_step_seconds:.3f}s", "ok")
            except Exception as exc:
                self.set_status("\u2716 Refine failed", "err")
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
            self.set_status("\u2705 Checking tabs...", "info")
            try:
                r = check_tabs_against_original(tabs_text=tabs_text, original_audio_path=original_path)
                self.set_status(
                    f"Score: {r.overall_score:.3f}  chroma={r.chroma_score:.3f}  onset={r.onset_score:.3f}", "ok")
                self.after(0, lambda: messagebox.showinfo("Tabs Check", "\n".join([
                    f"Overall score:   {r.overall_score:.4f}",
                    f"Chroma score:    {r.chroma_score:.4f}",
                    f"Onset score:     {r.onset_score:.4f}",
                    f"Est. step:       {r.estimated_step_seconds:.3f}s",
                    f"Est. note:       {r.estimated_note_seconds:.3f}s",
                    f"Analyzed:        {r.analyzed_seconds:.1f}s",
                ])))
            except Exception as exc:
                self.set_status("\u2716 Check failed", "err")
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
            self.set_status("\U0001F50A Synthesizing...", "info")
            try:
                if self._timed_events:
                    _, result = synthesize_from_timed_events(
                        self._timed_events, output_path=out_path, play=True,
                    )
                else:
                    result = synthesize_from_tabs_text(tabs_text, output_path=out_path, play=True)
                self.set_status(
                    f"\u2714 Rendered: {result.notes_count} notes, {result.duration:.1f}s", "ok")
            except Exception as exc:
                self.set_status("\u2716 Render failed", "err")
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
            self.set_status("\U0001F3AF Matching (may take a while)...", "warn")
            try:
                result = optimize_synth_against_original(
                    tabs_text=tabs_text, original_audio_path=original_path, output_path=out_path,
                )
                self.set_status(f"\u2714 Matched: score={result.score:.4f}", "ok")
            except Exception as exc:
                self.set_status("\u2716 Match failed", "err")
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
            self.set_status(f"\u2714 Saved to {os.path.basename(file_path)}", "ok")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def on_help(self):
        messagebox.showinfo("Help — Guitar Tab Generator", "\n".join([
            "QUICK START",
            "  1.  Open File or Record from Mic",
            "  2.  Click Analyze to generate tabs",
            "  3.  Save Tabs or Render Audio",
            "",
            "KEYBOARD SHORTCUTS",
            "  \u2318O / Ctrl+O     Open audio file",
            "  \u2318S / Ctrl+S     Save tabs",
            "  \u2318R / Ctrl+R     Render audio",
            "  \u2318\u23CE / Ctrl+\u23CE   Analyze",
            "  \u2318+/\u2212          Zoom tabs in/out",
            "",
            "IMPROVE ACCURACY",
            "  Best Quality — auto-tune extraction params",
            "  Refine — correct frets using original audio",
            "  Check Tabs — see objective match scores",
            "  Match Original — optimize synth quality",
            "",
            "PARAMETERS",
            "  Duration — mic recording length (seconds)",
            "  Min dur — shortest note to detect",
            "  Voiced — confidence filter (0–1)",
            "  Max fret — highest fret to use in tabs",
            "  Segment — process long audio in chunks",
            "  Harmonic — isolate harmonic content first",
            "",
            "TIPS",
            "  • Drag & drop audio files onto the window",
            "  • Use toolbar to copy tabs or zoom",
            "  • Waveform shows onset markers after analysis",
        ]))


# ── utilities ───────────────────────────────────────────
def _hsv_to_rgb(h: float, s: float, v: float):
    """Convert HSV [0-1] to RGB [0-1]."""
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q


if __name__ == "__main__":
    app = GuitarTabApp()
    app.mainloop()
