#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_gui.py  –  Detección de Objetos  •  Guía por Voz
Features: webcam / imagen / video, pipeline optimizado, snapshot, modelo
  configurable, umbral de confianza, filtro de objetos, espejo, GPU, intervalo
  de narración ajustable, reconexión automática de cámara.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, queue, time, os, tempfile, datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from gtts import gTTS
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from deteccion_objetos import detectar_objetos, guia_usuario

# ─────────────────────────────────────────────────────────
#  PALETA
# ─────────────────────────────────────────────────────────

SB         = "#111827"
SB_SECTION = "#1F2937"
SB_BORDER  = "#374151"
SB_TEXT    = "#F9FAFB"
SB_DIM     = "#9CA3AF"
SB_ACCENT  = "#818CF8"

BG         = "#F8FAFC"
PANEL      = "#FFFFFF"
BORDER_L   = "#E2E8F0"
TXT        = "#0F172A"
TXT_MID    = "#475569"
TXT_DIM    = "#94A3B8"

INDIGO     = "#4F46E5"
INDIGO_H   = "#4338CA"
GREEN      = "#10B981"
GREEN_H    = "#059669"
AMBER      = "#F59E0B"
RED        = "#EF4444"
RED_H      = "#DC2626"

CAM_OFF_BG = "#1E293B"
PREV_BG    = "#0F172A"

F_TITLE  = ("Segoe UI", 14, "bold")
F_BODY   = ("Segoe UI", 10)
F_SMALL  = ("Segoe UI", 9)
F_MONO   = ("Consolas", 10)

YOLO_MODELS = {
    "yolov8n — Rápido":   "yolov8n",
    "yolov8s — Balanceado": "yolov8s",
    "yolov8m — Preciso":  "yolov8m",
}

DET_SIZES = {
    "⚡ Rápida  (320)": 320,
    "⚖️ Balance (416)": 416,
    "🎯 Precisa (640)": 640,
}


# ─────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SafeStepsAI")
        self.root.geometry("1200x770")
        self.root.minsize(960, 660)
        self.root.configure(bg=BG)

        # Dispositivo
        self.device_mode = tk.StringVar(value="Auto")   # Auto | CPU | GPU
        self._gpu = False
        self._device = "cpu"

        # Estado
        self.mode          = tk.StringVar(value="webcam")
        self.selected_file = tk.StringVar(value="")
        self.camera_index  = tk.IntVar(value=0)
        self.narration_on  = tk.BooleanVar(value=True)
        self.det_size_key  = tk.StringVar(value="⚡ Rápida  (320)")
        self.model_key     = tk.StringVar(value="yolov8n — Rápido")
        self.conf_thr      = tk.IntVar(value=35)     # 5–95 (slider), /100 al usar
        self.obj_filter    = tk.StringVar(value="")
        self.mirror_mode   = tk.BooleanVar(value=False)
        self.narr_interval = tk.IntVar(value=4)

        self.model          = None
        self.available_cams = []
        self.is_running     = False
        self.stop_event     = threading.Event()

        # Colas pipeline
        self._display_q = queue.Queue(maxsize=2)
        self._raw_q     = queue.Queue(maxsize=1)
        self.frame_q    = queue.Queue(maxsize=2)
        self._last_dets = []
        self._last_msg  = ""

        # FPS / canvas
        self._fps_cnt  = 0
        self._fps_t    = time.time()
        self._cur_fps  = 0.0
        self._canvas_w = 1
        self._canvas_h = 1

        # Audio
        self._narr_t    = 0.0
        self._narr_msg  = ""
        self._audio_lock = threading.Lock()
        if AUDIO_AVAILABLE:
            try:
                pygame.mixer.init()
            except Exception:
                pass

        # Estado pantalla y frame
        self._last_frame   = None
        self._screen_state = "start"
        self._root_alive   = True
        self._run_gen      = 0

        self._build_ui()
        self._resolve_device()
        self._scan_cameras()
        self._load_model()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._ui_refresh()
        self.root.after(80, self._draw_start_screen)

    # ─────────────────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────────────────

    def _build_ui(self):
        # Contenedor lateral con scroll
        self._sidebar_outer = tk.Frame(self.root, bg=SB, width=240)
        self._sidebar_outer.pack(side="left", fill="y")
        self._sidebar_outer.pack_propagate(False)

        self._sidebar_canvas = tk.Canvas(
            self._sidebar_outer,
            bg=SB,
            highlightthickness=0,
            bd=0
        )
        self._sidebar_scrollbar = ttk.Scrollbar(
            self._sidebar_outer,
            orient="vertical",
            command=self._sidebar_canvas.yview
        )
        self._sidebar_canvas.configure(yscrollcommand=self._sidebar_scrollbar.set)

        self._sidebar_scrollbar.pack(side="right", fill="y")
        self._sidebar_canvas.pack(side="left", fill="both", expand=True)

        self._sidebar = tk.Frame(self._sidebar_canvas, bg=SB, width=240)
        self._sidebar_window = self._sidebar_canvas.create_window(
            (0, 0),
            window=self._sidebar,
            anchor="nw"
        )

        def _on_sidebar_frame_configure(event):
            self._sidebar_canvas.configure(
                scrollregion=self._sidebar_canvas.bbox("all")
            )

        def _on_sidebar_canvas_configure(event):
            self._sidebar_canvas.itemconfigure(
                self._sidebar_window,
                width=event.width
            )

        def _on_mousewheel(event):
            self._sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self._sidebar.bind("<Configure>", _on_sidebar_frame_configure)
        self._sidebar_canvas.bind("<Configure>", _on_sidebar_canvas_configure)

        self._sidebar_canvas.bind(
            "<Enter>",
            lambda e: self._sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        )
        self._sidebar_canvas.bind(
            "<Leave>",
            lambda e: self._sidebar_canvas.unbind_all("<MouseWheel>")
        )

        self._content = tk.Frame(self.root, bg=BG)
        self._content.pack(side="left", fill="both", expand=True)

        self._build_sidebar()
        self._build_header()
        self._build_preview()
        self._build_panel()

    # ─────────── SIDEBAR ─────────────────────────────────

    def _sb_section(self, text):
        tk.Label(self._sidebar, text=text, bg=SB, fg=SB_ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(
                 anchor="w", padx=16, pady=(14, 4))

    def _sb_sep(self):
        tk.Frame(self._sidebar, bg=SB_BORDER, height=1).pack(
            fill="x", padx=16, pady=7)

    def _build_sidebar(self):
        sb = self._sidebar

        # Logo
        logo = tk.Frame(sb, bg=INDIGO)
        logo.pack(fill="x")
        tk.Label(logo, text="SafeStepsAI",
                 bg=INDIGO, fg="#FFFFFF",
                 font=("Segoe UI", 12, "bold"),
                 padx=16, pady=13).pack(anchor="w")

        # ── Modo ─────────────────────────────────────────
        self._sb_section("MODO")
        for txt, val in [("📷  Cámara web",      "webcam"),
                          ("🖼  Imagen estática",  "image"),
                          ("🎬  Archivo de video", "video")]:
            tk.Radiobutton(
                sb, text=txt, value=val, variable=self.mode,
                command=self._on_mode_change,
                bg=SB, fg=SB_TEXT, selectcolor=INDIGO,
                activebackground=SB, activeforeground=SB_ACCENT,
                font=F_BODY, bd=0, highlightthickness=0,
                cursor="hand2", width=20, anchor="w"
            ).pack(anchor="w", padx=16, pady=2)

        # ── Sección cámara ────────────────────────────────
        self.sec_cam = tk.Frame(sb, bg=SB)

        tk.Label(self.sec_cam, text="CÁMARA", bg=SB, fg=SB_ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 4))

        self.combo_cam = ttk.Combobox(self.sec_cam, state="readonly",
                                       font=F_BODY, width=20)
        self.combo_cam.pack(fill="x")
        self.combo_cam.bind("<<ComboboxSelected>>", self._on_cam_select)

        row = tk.Frame(self.sec_cam, bg=SB)
        row.pack(fill="x", pady=(3, 0))
        self.lbl_cam_hint = tk.Label(row, text="Buscando...",
                                      bg=SB, fg=SB_DIM, font=F_SMALL)
        self.lbl_cam_hint.pack(side="left")
        tk.Button(row, text="🔄", bg=SB_SECTION, fg=SB_TEXT,
                  relief="flat", bd=0, padx=6, pady=2, font=F_SMALL,
                  cursor="hand2", activebackground=SB_BORDER,
                  command=self._scan_cameras).pack(side="right")

        tk.Checkbutton(
            self.sec_cam, text="🪞  Espejo de imagen",
            variable=self.mirror_mode,
            bg=SB, fg=SB_TEXT, selectcolor=INDIGO,
            activebackground=SB, activeforeground=SB_ACCENT,
            font=F_BODY, bd=0, highlightthickness=0, cursor="hand2"
        ).pack(anchor="w", pady=(6, 0))

        # ── Sección archivo ───────────────────────────────
        self.sec_file = tk.Frame(sb, bg=SB)

        tk.Label(self.sec_file, text="ARCHIVO", bg=SB, fg=SB_ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 4))

        tk.Button(self.sec_file, text="📂  Seleccionar archivo...",
                  bg=SB_SECTION, fg=SB_TEXT, relief="flat", bd=0,
                  padx=10, pady=6, font=F_BODY, cursor="hand2",
                  activebackground=SB_BORDER,
                  command=self._browse).pack(fill="x")

        self.lbl_file = tk.Label(self.sec_file, text="Sin seleccionar",
                                  bg=SB, fg=SB_DIM, font=F_SMALL,
                                  wraplength=200, justify="left")
        self.lbl_file.pack(anchor="w", pady=(3, 0))

        self._sb_sep()

        # ── Detección ─────────────────────────────────────
        self._sb_section("DETECCIÓN")

        # Confianza
        tk.Label(sb, text="Confianza mínima",
                 bg=SB, fg=SB_DIM, font=F_SMALL).pack(anchor="w", padx=16)

        conf_row = tk.Frame(sb, bg=SB)
        conf_row.pack(fill="x", padx=16, pady=(2, 6))

        self.sld_conf = tk.Scale(
            conf_row, from_=5, to=95, orient="horizontal",
            variable=self.conf_thr,
            bg=SB, fg=SB_TEXT, troughcolor=SB_SECTION,
            highlightthickness=0, bd=0, showvalue=False,
            resolution=5, cursor="hand2",
            command=lambda _: self.lbl_conf_val.config(
                text=f"{self.conf_thr.get()}%")   # mostrar directo, sin *100
        )
        self.sld_conf.set(35)
        self.sld_conf.pack(side="left", fill="x", expand=True)
        self.lbl_conf_val = tk.Label(conf_row, text="35%",
                                      bg=SB, fg=SB_TEXT, font=F_SMALL, width=4)
        self.lbl_conf_val.pack(side="right")

        # Modelo
        tk.Label(sb, text="Modelo YOLO",
                 bg=SB, fg=SB_DIM, font=F_SMALL).pack(anchor="w", padx=16)

        model_row = tk.Frame(sb, bg=SB)
        model_row.pack(fill="x", padx=16, pady=(2, 4))

        self.combo_model = ttk.Combobox(
            model_row, values=list(YOLO_MODELS.keys()),
            state="readonly", font=F_SMALL, width=18)
        self.combo_model.set(self.model_key.get())
        self.combo_model.pack(side="left", fill="x", expand=True, padx=(0, 4))

        tk.Button(model_row, text="🔄", bg=SB_SECTION, fg=SB_TEXT,
                  relief="flat", bd=0, padx=6, pady=2, font=F_SMALL,
                  cursor="hand2", activebackground=SB_BORDER,
                  command=self._reload_model).pack(side="right")

        # Dispositivo
        tk.Label(sb, text="Dispositivo",
                 bg=SB, fg=SB_DIM, font=F_SMALL).pack(anchor="w", padx=16)

        self.combo_device = ttk.Combobox(
            sb,
            values=["Auto", "CPU", "GPU"],
            state="readonly",
            font=F_SMALL,
            width=22
        )
        self.combo_device.set(self.device_mode.get())
        self.combo_device.pack(fill="x", padx=16, pady=(2, 4))
        self.combo_device.bind("<<ComboboxSelected>>", self._on_device_change)

        self.lbl_gpu = tk.Label(sb, text="🔴 Solo CPU",
                                 bg=SB, fg=AMBER, font=F_SMALL)
        self.lbl_gpu.pack(anchor="w", padx=16, pady=(0, 4))

        # Filtro de objetos
        tk.Label(sb, text="Filtrar objetos (vacío = todos)",
                 bg=SB, fg=SB_DIM, font=F_SMALL).pack(anchor="w", padx=16)
        self.ent_filter = tk.Entry(
            sb, textvariable=self.obj_filter,
            bg=SB_SECTION, fg=SB_TEXT, insertbackground=SB_TEXT,
            relief="flat", bd=4, font=F_SMALL)
        self.ent_filter.pack(fill="x", padx=16, pady=(2, 0))
        tk.Label(sb, text="ej: persona, auto, perro",
                 bg=SB, fg=SB_DIM, font=("Segoe UI", 8)).pack(
                 anchor="w", padx=16, pady=(1, 0))

        self._sb_sep()

        # ── Voz ───────────────────────────────────────────
        self._sb_section("VOZ")

        tk.Checkbutton(
            sb, text="🔊  Narración activa",
            variable=self.narration_on,
            bg=SB, fg=SB_TEXT, selectcolor=INDIGO,
            activebackground=SB, activeforeground=SB_ACCENT,
            font=F_BODY, bd=0, highlightthickness=0, cursor="hand2"
        ).pack(anchor="w", padx=16, pady=(0, 6))

        tk.Label(sb, text="Intervalo de narración",
                 bg=SB, fg=SB_DIM, font=F_SMALL).pack(anchor="w", padx=16)

        narr_row = tk.Frame(sb, bg=SB)
        narr_row.pack(fill="x", padx=16, pady=(2, 6))

        self.sld_narr = tk.Scale(
            narr_row, from_=1, to=15, orient="horizontal",
            variable=self.narr_interval,
            bg=SB, fg=SB_TEXT, troughcolor=SB_SECTION,
            highlightthickness=0, bd=0, showvalue=False,
            cursor="hand2",
            command=lambda _: self.lbl_narr_val.config(
                text=f"{self.narr_interval.get()}s")
        )
        self.sld_narr.pack(side="left", fill="x", expand=True)
        self.lbl_narr_val = tk.Label(narr_row, text="4s",
                                      bg=SB, fg=SB_TEXT, font=F_SMALL, width=3)
        self.lbl_narr_val.pack(side="right")

        tk.Label(sb, text="Resolución inferencia",
                 bg=SB, fg=SB_DIM, font=F_SMALL).pack(anchor="w", padx=16)

        self.combo_size = ttk.Combobox(sb, values=list(DET_SIZES.keys()),
                                        state="readonly", font=F_SMALL, width=22)
        self.combo_size.set(self.det_size_key.get())
        self.combo_size.bind("<<ComboboxSelected>>",
                              lambda e: self.det_size_key.set(self.combo_size.get()))
        self.combo_size.pack(fill="x", padx=16, pady=(2, 0))

        self._sb_sep()

        # ── Botones control ───────────────────────────────
        self.btn_start = tk.Button(
            sb, text="▶   INICIAR",
            bg=GREEN, fg="#FFFFFF",
            activebackground=GREEN_H, activeforeground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, pady=9, cursor="hand2",
            command=self._start
        )
        self.btn_start.pack(fill="x", padx=16, pady=(0, 6))

        self.btn_stop = tk.Button(
            sb, text="■   DETENER",
            bg=SB_SECTION, fg=SB_DIM,
            activebackground=RED, activeforeground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, pady=9, cursor="hand2",
            state="disabled",
            command=self._stop
        )
        self.btn_stop.pack(fill="x", padx=16)

        self._sb_sep()

        self.lbl_model = tk.Label(sb, text="⏳  Cargando modelo...",
                                   bg=SB, fg=AMBER,
                                   font=F_SMALL, wraplength=205, justify="left")
        self.lbl_model.pack(anchor="w", padx=16)

        # Botón salir al fondo
        tk.Frame(sb, bg=SB_BORDER, height=1).pack(side="bottom", fill="x", padx=16)
        tk.Button(
            sb, text="⏻   Salir",
            bg=SB, fg=SB_DIM,
            activebackground="#374151", activeforeground="#F87171",
            font=("Segoe UI", 9), relief="flat", bd=0,
            pady=10, cursor="hand2",
            command=self._confirm_exit
        ).pack(side="bottom", fill="x", padx=16, pady=(8, 12))

        self._on_mode_change()

    # ─────────── HEADER ──────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self._content, bg=PANEL,
                        highlightbackground=BORDER_L, highlightthickness=1)
        hdr.pack(fill="x", padx=12, pady=(10, 0))

        tk.Label(hdr, text="Detección de Objetos & Resumen",
                 bg=PANEL, fg=TXT, font=F_TITLE,
                 padx=16, pady=10).pack(side="left")

        right = tk.Frame(hdr, bg=PANEL)
        right.pack(side="right", padx=10)

        tk.Button(right, text="📷  Captura",
                  bg=INDIGO, fg="#FFFFFF",
                  activebackground=INDIGO_H, activeforeground="#FFFFFF",
                  relief="flat", bd=0, padx=10, pady=5,
                  font=F_SMALL, cursor="hand2",
                  command=self._snapshot).pack(side="right", padx=(6, 0))

        self.lbl_fps = tk.Label(right, text="",
                                 bg=PANEL, fg=TXT_DIM, font=F_SMALL)
        self.lbl_fps.pack(side="right", padx=(8, 0))

        self.lbl_status = tk.Label(right, text="● Sin iniciar",
                                    bg=PANEL, fg=TXT_DIM, font=F_SMALL)
        self.lbl_status.pack(side="right")

    # ─────────── PREVIEW ─────────────────────────────────

    def _build_preview(self):
        wrap = tk.Frame(self._content, bg=PANEL,
                         highlightbackground=BORDER_L, highlightthickness=1)
        wrap.pack(fill="both", expand=True, padx=12, pady=8)

        self.canvas = tk.Canvas(wrap, bg=PREV_BG, bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    # ─────────── PANEL INFERIOR ──────────────────────────

    def _build_panel(self):
        panel = tk.Frame(self._content, bg=PANEL,
                          highlightbackground=BORDER_L, highlightthickness=1,
                          height=185)
        panel.pack(fill="x", padx=12, pady=(0, 10))
        panel.pack_propagate(False)

        hdr = tk.Frame(panel, bg=INDIGO)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📋  DETECCIONES",
                 bg=INDIGO, fg="#FFFFFF",
                 font=("Segoe UI", 9, "bold"),
                 padx=12, pady=6).pack(side="left")
        self.lbl_count = tk.Label(hdr, text="",
                                   bg=INDIGO, fg="#C7D2FE",
                                   font=F_SMALL, padx=12)
        self.lbl_count.pack(side="right")

        cols = tk.Frame(panel, bg=PANEL)
        cols.pack(fill="both", expand=True, padx=10, pady=6)

        left = tk.Frame(cols, bg=PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(left, text="Objetos detectados",
                 bg=PANEL, fg=TXT_MID, font=F_SMALL).pack(anchor="w")
        self.txt_dets = tk.Text(
            left, height=7, width=40, bg=BG, fg=TXT, font=F_MONO,
            relief="flat", bd=0, state="disabled",
            cursor="arrow", padx=8, pady=5, wrap="none")
        self.txt_dets.pack(fill="both", expand=True)

        right_col = tk.Frame(cols, bg=PANEL)
        right_col.pack(side="left", fill="both", expand=True)
        tk.Label(right_col, text="Guía de voz",
                 bg=PANEL, fg=TXT_MID, font=F_SMALL).pack(anchor="w")
        self.txt_guide = tk.Text(
            right_col, height=7, width=42,
            bg=BG, fg=INDIGO, font=("Segoe UI", 11),
            relief="flat", bd=0, state="disabled",
            cursor="arrow", padx=8, pady=5, wrap="word")
        self.txt_guide.pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────────
    #  PANTALLAS CANVAS
    # ─────────────────────────────────────────────────────

    def _canvas_center(self):
        cw = max(self._canvas_w, self.canvas.winfo_width(), 100)
        ch = max(self._canvas_h, self.canvas.winfo_height(), 100)
        return cw, ch, cw // 2, ch // 2

    def _draw_start_screen(self):
        self._screen_state = "start"
        self.canvas.delete("all")
        self.canvas.configure(bg=PREV_BG)
        cw, ch, cx, cy = self._canvas_center()
        self.canvas.create_text(cx, cy - 24, text="🎯",
            font=("Segoe UI", 42), fill="#334155", anchor="center")
        self.canvas.create_text(cx, cy + 30, fill="#475569",
            font=("Segoe UI", 13, "bold"), anchor="center",
            text="Selecciona un modo y presiona  ▶ INICIAR")
        self.canvas.create_text(cx, cy + 58, fill="#334155",
            font=("Segoe UI", 10), anchor="center",
            text="Cámara web  •  Imagen  •  Video")

    def _draw_cam_off_screen(self):
        self._screen_state = "cam_off"
        self.canvas.delete("all")
        self.canvas.configure(bg=CAM_OFF_BG)
        cw, ch, cx, cy = self._canvas_center()
        r, oy = 50, cy - 28
        self.canvas.create_oval(cx - r, oy - r, cx + r, oy + r,
                                  fill="#0F172A", outline="#334155", width=2)
        self.canvas.create_text(cx, oy, text="📷",
            font=("Segoe UI", 34), fill="#475569", anchor="center")
        self.canvas.create_text(cx, oy + r + 22,
            text="Cámara apagada",
            font=("Segoe UI", 16, "bold"), fill="#94A3B8", anchor="center")
        self.canvas.create_text(cx, oy + r + 50,
            text="Presiona  ▶ INICIAR  para activarla",
            font=("Segoe UI", 10), fill="#64748B", anchor="center")

    def _draw_loading_screen(self):
        self.canvas.delete("all")
        self.canvas.configure(bg=PREV_BG)
        cw, ch, cx, cy = self._canvas_center()
        self.canvas.create_text(cx, cy, text="Iniciando...",
            font=("Segoe UI", 14), fill="#475569", anchor="center")

    # ─────────────────────────────────────────────────────
    #  EVENTOS
    # ─────────────────────────────────────────────────────

    def _on_canvas_resize(self, event):
        self._canvas_w = event.width
        self._canvas_h = event.height
        if not self.is_running:
            if self._screen_state == "cam_off":
                self._draw_cam_off_screen()
            elif self._screen_state == "start":
                self._draw_start_screen()
            elif self._screen_state == "image_done" and self._last_frame is not None:
                self._draw_frame(self._last_frame)

    def _on_mode_change(self):
        if self.mode.get() == "webcam":
            self.sec_cam.pack(fill="x", padx=16, pady=(0, 4))
            self.sec_file.pack_forget()
        else:
            self.sec_cam.pack_forget()
            self.sec_file.pack(fill="x", padx=16, pady=(0, 4))

    def _on_cam_select(self, _=None):
        i = self.combo_cam.current()
        if 0 <= i < len(self.available_cams):
            self.camera_index.set(self.available_cams[i][0])

    def _browse(self):
        mode = self.mode.get()
        ft = ([("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")]
              if mode == "image" else
              [("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Todos", "*.*")])
        path = filedialog.askopenfilename(filetypes=ft)
        if path:
            self.selected_file.set(path)
            n = os.path.basename(path)
            self.lbl_file.config(
                text=(n if len(n) <= 30 else n[:27] + "..."), fg=SB_TEXT)

    def _set_status(self, txt, color=TXT_DIM):
        self.lbl_status.config(text=txt, fg=color)

    def _resolve_device(self):
        mode = self.device_mode.get()

        if mode == "CPU":
            self._gpu = False
            self._device = "cpu"
        elif mode == "GPU":
            if _TORCH_OK and torch.cuda.is_available():
                self._gpu = True
                self._device = "cuda"
            else:
                self._gpu = False
                self._device = "cpu"
        else:  # Auto
            self._gpu = (_TORCH_OK and torch.cuda.is_available())
            self._device = "cuda" if self._gpu else "cpu"

        gpu_txt = "🟢 GPU activa" if self._gpu else "🔴 Solo CPU"
        gpu_col = GREEN if self._gpu else AMBER
        if hasattr(self, "lbl_gpu"):
            self.lbl_gpu.config(text=gpu_txt, fg=gpu_col)
        if hasattr(self, "lbl_model") and self.model is not None:
            self.lbl_model.config(text=f"Dispositivo actual: {self._device.upper()}", fg=GREEN)

    def _on_device_change(self, _=None):
        self.device_mode.set(self.combo_device.get())
        old_device = self._device
        self._resolve_device()

        if self.is_running:
            messagebox.showinfo(
                "Dispositivo",
                "El cambio se aplicará al volver a cargar el modelo o al iniciar de nuevo."
            )
        elif self.model is not None and old_device != self._device:
            self._reload_model()

    # ─────────────────────────────────────────────────────
    #  SNAPSHOT
    # ─────────────────────────────────────────────────────

    def _snapshot(self):
        if self._last_frame is None:
            messagebox.showinfo("Captura", "No hay imagen disponible para guardar.")
            return
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"captura_{ts}.png"
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")],
            title="Guardar captura"
        )
        if path:
            try:
                bgr = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, bgr)
                messagebox.showinfo("Captura",
                    f"✅ Guardada:\n{os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error al guardar", str(e))

    # ─────────────────────────────────────────────────────
    #  DETECCIÓN DE CÁMARAS
    # ─────────────────────────────────────────────────────

    def _scan_cameras(self):
        self.root.after(0, lambda: self.lbl_cam_hint.config(
            text="Buscando...", fg=SB_DIM))

        def _try(idx):
            for flag in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                try:
                    cap = cv2.VideoCapture(idx, flag)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            return flag
                except Exception:
                    pass
            return None

        def _run():
            found, labels = [], []
            for i in range(10):
                flag = _try(i)
                if flag is not None:
                    found.append((i, flag))
                    labels.append(f"Cámara {i}")
            if not found:
                found  = [(0, cv2.CAP_ANY)]
                labels = ["Cámara 0"]
            self.available_cams = found
            self.root.after(0, lambda: self._set_cams(labels))

        threading.Thread(target=_run, daemon=True).start()

    def _set_cams(self, labels):
        self.combo_cam["values"] = labels
        if labels:
            self.combo_cam.current(0)
            self.camera_index.set(self.available_cams[0][0])
            self.lbl_cam_hint.config(
                text=f"{len(labels)} cámara(s) encontrada(s)", fg=GREEN)
        else:
            self.lbl_cam_hint.config(text="Sin cámaras detectadas", fg=RED)

    # ─────────────────────────────────────────────────────
    #  MODELO
    # ─────────────────────────────────────────────────────

    def _load_model(self, name: str = "yolov8n"):
        self.root.after(0, lambda: self.lbl_model.config(
            text=f"⏳  Cargando {name}.pt...", fg=AMBER))

        def _run():
            try:
                m = YOLO(f"{name}.pt")
                m.to(self._device)
                m(np.zeros((32, 32, 3), dtype=np.uint8), verbose=False)
                self.model = m
                lbl = f"✅  {name}  ({self._device.upper()})"
                self.root.after(0, lambda: self.lbl_model.config(
                    text=lbl, fg=GREEN))
            except Exception as e:
                self.root.after(0, lambda err=e: self.lbl_model.config(
                    text=f"❌  {err}", fg=RED))

        threading.Thread(target=_run, daemon=True).start()

    def _reload_model(self):
        name = YOLO_MODELS.get(self.combo_model.get(), "yolov8n")
        self.model = None
        self._load_model(name)

    # ─────────────────────────────────────────────────────
    #  INICIO / PARADA
    # ─────────────────────────────────────────────────────

    def _start(self):
        if not self.model:
            messagebox.showwarning("Modelo", "El modelo aún se está cargando.")
            return
        mode = self.mode.get()
        if mode in ("image", "video") and not self.selected_file.get():
            messagebox.showwarning("Archivo", "Selecciona un archivo primero.")
            return

        self.is_running = True
        self.stop_event.clear()
        self._run_gen  += 1
        self._last_dets = []
        self._last_msg  = ""
        self._fps_cnt   = 0
        self._fps_t     = time.time()

        self.btn_start.config(state="disabled", bg=BG, fg=TXT_DIM)
        self.btn_stop.config(state="normal", bg=RED, fg="#FFFFFF",
                              activebackground=RED_H)
        self._set_status("● Ejecutando", GREEN)
        self._clear_panel()
        self._draw_loading_screen()

        if mode == "webcam":
            threading.Thread(target=self._capture_thread, daemon=True).start()
            threading.Thread(target=self._detect_thread,  daemon=True).start()
        elif mode == "image":
            threading.Thread(target=self._image_thread,   daemon=True).start()
        else:
            threading.Thread(target=self._video_thread,   daemon=True).start()

    def _stop(self):
        self.stop_event.set()
        self.is_running = False
        self.btn_start.config(state="normal", bg=GREEN, fg="#FFFFFF",
                               activebackground=GREEN_H)
        self.btn_stop.config(state="disabled", bg=SB_SECTION, fg=SB_DIM)
        self._set_status("● Detenido", AMBER)
        self.lbl_fps.config(text="")
        self._clear_panel()
        if self.mode.get() == "webcam":
            self.root.after(150, self._draw_cam_off_screen)
        else:
            self.root.after(150, self._draw_start_screen)

    def _image_done(self):
        self.stop_event.set()
        self.is_running = False
        self._screen_state = "image_done"
        self.btn_start.config(state="normal", bg=GREEN, fg="#FFFFFF",
                               activebackground=GREEN_H)
        self.btn_stop.config(state="disabled", bg=SB_SECTION, fg=SB_DIM)
        self._set_status("● Detección lista", GREEN)
        self.lbl_fps.config(text="")

    # ─────────────────────────────────────────────────────
    #  HELPERS PIPELINE
    # ─────────────────────────────────────────────────────

    def _open_cam(self):
        idx = self.camera_index.get()
        stored = next((f for ci, f in self.available_cams if ci == idx), cv2.CAP_ANY)
        for flag in (stored, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            try:
                cap = cv2.VideoCapture(idx, flag)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        return cap
                    cap.release()
            except Exception:
                pass
        return None

    @staticmethod
    def _put_latest(q: queue.Queue, item):
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass

    def _det_size(self) -> int:
        return DET_SIZES.get(self.det_size_key.get(), 320)

    def _detect_on(self, frame: np.ndarray):
        """YOLO a resolución configurada + filtros de confianza y objeto."""
        try:
            ds = self._det_size()
            oh, ow = frame.shape[:2]
            if oh == 0 or ow == 0:
                return []
            small = cv2.resize(frame, (ds, ds)) if (ds != ow or ds != oh) else frame

            conf = self.conf_thr.get() / 100.0    # slider es 5-95, YOLO necesita 0.05-0.95
            dets, _ = detectar_objetos(self.model, small, conf=conf)

            # Escalar bboxes al frame original
            sx, sy = ow / ds, oh / ds
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                d["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

            # Filtro por nombre de objeto
            flt = self.obj_filter.get().strip()
            if flt:
                allowed = {x.strip().lower() for x in flt.split(',')}
                dets = [d for d in dets
                        if d.get("label_es", "").lower() in allowed
                        or d.get("label_en", "").lower() in allowed]

            return dets
        except Exception:
            return []

    def _apply_mirror(self, frame: np.ndarray) -> np.ndarray:
        if self.mirror_mode.get():
            return cv2.flip(frame, 1)
        return frame

    def _safe_after(self, fn):
        if self._root_alive:
            try:
                self.root.after(0, fn)
            except tk.TclError:
                pass

    # ─────────────────────────────────────────────────────
    #  HILOS
    # ─────────────────────────────────────────────────────

    def _capture_thread(self):
        gen = self._run_gen
        cap = self._open_cam()
        if cap is None:
            self._safe_after(lambda: messagebox.showerror(
                "Error", "No se pudo abrir la cámara.\nVerifica que esté conectada."))
            self._safe_after(self._stop)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        fail_count = 0
        try:
            while not self.stop_event.is_set() and self._run_gen == gen:
                try:
                    ret, bgr = cap.read()
                    if not ret:
                        fail_count += 1
                        if fail_count > 60:          # ~2s de fallos → reconectar
                            cap.release()
                            cap = self._open_cam()
                            if cap is None:
                                break
                            fail_count = 0
                        time.sleep(0.033)
                        continue
                    fail_count = 0
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame = self._apply_mirror(frame)
                    self._put_latest(self._display_q, frame)
                    self._put_latest(self._raw_q, frame)
                    self._fps_cnt += 1
                    if (t := time.time()) - self._fps_t >= 1.0:
                        self._cur_fps = self._fps_cnt / (t - self._fps_t)
                        self._fps_cnt = 0
                        self._fps_t   = t
                except Exception:
                    time.sleep(0.01)
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def _detect_thread(self):
        gen = self._run_gen
        while not self.stop_event.is_set() and self._run_gen == gen:
            try:
                frame = self._raw_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                dets = self._detect_on(frame)
                msg  = guia_usuario(dets)
                self._last_dets = dets
                self._last_msg  = msg
                self._safe_after(lambda d=dets, m=msg: self._update_panel(d, m))
                if (self.narration_on.get() and AUDIO_AVAILABLE
                        and self._should_narrate(msg)):
                    threading.Thread(
                        target=self._audio, args=(msg,), daemon=True).start()
            except Exception:
                pass

    def _image_thread(self):
        try:
            bgr = cv2.imread(self.selected_file.get())
            if bgr is None:
                raise ValueError("No se pudo leer la imagen")
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets  = self._detect_on(frame)
            msg   = guia_usuario(dets)
            boxed = self._draw_boxes(frame, dets)
            self._put_latest(self.frame_q, (boxed, dets, msg))
            if self.narration_on.get() and AUDIO_AVAILABLE:
                threading.Thread(
                    target=self._audio, args=(msg,), daemon=True).start()
        except Exception as e:
            self.root.after(0, lambda err=e: messagebox.showerror("Error", str(err)))
        finally:
            self.root.after(0, self._image_done)

    def _video_thread(self):
        gen = self._run_gen
        cap = cv2.VideoCapture(self.selected_file.get())
        if not cap.isOpened():
            self._safe_after(lambda: messagebox.showerror(
                "Error", "No se pudo abrir el video"))
            self._safe_after(self._stop)
            return
        try:
            while not self.stop_event.is_set() and self._run_gen == gen:
                try:
                    ret, bgr = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    dets  = self._detect_on(frame)
                    msg   = guia_usuario(dets)
                    boxed = self._draw_boxes(frame, dets)
                    self._put_latest(self.frame_q, (boxed, dets, msg))
                    if (self.narration_on.get() and AUDIO_AVAILABLE
                            and self._should_narrate(msg)):
                        threading.Thread(
                            target=self._audio, args=(msg,), daemon=True).start()
                except Exception:
                    time.sleep(0.01)
                time.sleep(0.01)
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self._safe_after(self._stop)

    # ─────────────────────────────────────────────────────
    #  BBOXES
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _draw_boxes(frame: np.ndarray, dets: list) -> np.ndarray:
        if not dets:
            return frame
        img = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            dist   = d.get("distance", "")
            pos    = d.get("position", "")
            danger = dist == "muy cerca" and pos == "al frente"
            color  = (239, 68, 68) if danger else (16, 185, 129)
            conf_v = d.get("confidence", 0)
            label  = f"{d['label_es']}  {conf_v:.0%}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)),
                          (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, max(th + 3, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # ─────────────────────────────────────────────────────
    #  AUDIO
    # ─────────────────────────────────────────────────────

    def _should_narrate(self, msg: str) -> bool:
        now      = time.time()
        interval = max(1, self.narr_interval.get())
        if msg != self._narr_msg and now - self._narr_t >= interval:
            self._narr_t   = now
            self._narr_msg = msg
            return True
        return False

    def _audio(self, text: str):
        if not self._audio_lock.acquire(blocking=False):
            return
        tmp = None
        try:
            tts = gTTS(text=text, lang="es", tld="com.mx")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tmp = f.name
            tts.save(tmp)
            # Re-init only if the mixer is not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.stop()
        except Exception:
            pass
        finally:
            # Do NOT call pygame.mixer.quit() here — it destroys the mixer
            # for all subsequent calls and causes audio to stop working.
            try:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass
            self._audio_lock.release()

    # ─────────────────────────────────────────────────────
    #  BUCLE UI
    # ─────────────────────────────────────────────────────

    def _ui_refresh(self):
        if not self._root_alive:
            return
        try:
            if self.mode.get() == "webcam" and self.is_running:
                try:
                    frame = self._display_q.get_nowait()
                    self._draw_frame(self._draw_boxes(frame, self._last_dets))
                except queue.Empty:
                    pass
                if self._cur_fps > 0:
                    self.lbl_fps.config(text=f"{self._cur_fps:.0f} FPS")
            else:
                try:
                    annotated, dets, msg = self.frame_q.get_nowait()
                    self._draw_frame(annotated)
                    self._update_panel(dets, msg)
                except queue.Empty:
                    pass
        except tk.TclError:
            return
        except Exception:
            pass
        self.root.after(16, self._ui_refresh)

    # ─────────────────────────────────────────────────────
    #  RENDER CANVAS
    # ─────────────────────────────────────────────────────

    def _draw_frame(self, frame_rgb: np.ndarray):
        try:
            cw = self._canvas_w or self.canvas.winfo_width()
            ch = self._canvas_h or self.canvas.winfo_height()
            if cw < 2 or ch < 2:
                return
            h, w = frame_rgb.shape[:2]
            if h == 0 or w == 0:
                return
            scale  = min(cw / w, ch / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            img    = Image.fromarray(
                cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR))
            photo  = ImageTk.PhotoImage(image=img)
            self.canvas.configure(bg=PREV_BG)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
            self.canvas._photo = photo
            self._last_frame   = frame_rgb
        except tk.TclError:
            pass
        except Exception:
            pass

    # ─────────────────────────────────────────────────────
    #  PANEL
    # ─────────────────────────────────────────────────────

    def _update_panel(self, dets: list, msg: str):
        try:
            n = len(dets)
            self.lbl_count.config(
                text=f"{n} objeto{'s' if n != 1 else ''} "
                     f"detectado{'s' if n != 1 else ''}"
            )
            self.txt_dets.config(state="normal")
            self.txt_dets.delete("1.0", "end")
            self.txt_dets.tag_config("danger", foreground=RED)
            self.txt_dets.tag_config("ok",     foreground=GREEN)
            self.txt_dets.tag_config("dim",    foreground=TXT_DIM)

            if dets:
                self.txt_dets.insert("end",
                    f"{'Objeto':<16}{'Posición':<17}{'Distancia':<16}Conf\n", "dim")
                self.txt_dets.insert("end", "─" * 56 + "\n", "dim")
                for d in dets:
                    dist   = d.get("distance", "?")
                    pos    = d.get("position", "")
                    danger = dist == "muy cerca" and pos == "al frente"
                    tag    = "danger" if danger else "ok"
                    icon   = "⚠ " if danger else "  "
                    conf_v = d.get("confidence", 0)
                    self.txt_dets.insert("end",
                        f"{icon}{d['label_es']:<14} "
                        f"{pos:<16} "
                        f"{dist:<15} "
                        f"{conf_v:.0%}\n", tag)
            else:
                self.txt_dets.insert("end", "  Sin detecciones\n", "dim")

            self.txt_dets.config(state="disabled")
            self.txt_guide.config(state="normal")
            self.txt_guide.delete("1.0", "end")
            self.txt_guide.insert("end", f"🔊  {msg}")
            self.txt_guide.config(state="disabled")
        except tk.TclError:
            pass
        except Exception:
            pass

    def _clear_panel(self):
        self.lbl_count.config(text="")
        self.lbl_fps.config(text="")
        for w in (self.txt_dets, self.txt_guide):
            w.config(state="normal")
            w.delete("1.0", "end")
            w.config(state="disabled")

    # ─────────────────────────────────────────────────────
    #  CIERRE
    # ─────────────────────────────────────────────────────

    def _confirm_exit(self):
        if messagebox.askyesno("Salir",
                                "¿Estás seguro que quieres salir?",
                                icon="question"):
            self._root_alive = False
            self.stop_event.set()
            if AUDIO_AVAILABLE:
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass
            try:
                self.root.destroy()
            except Exception:
                pass

    def _on_close(self):
        self._confirm_exit()

    def _safe_after(self, fn):
        if self._root_alive:
            try:
                self.root.after(0, fn)
            except tk.TclError:
                pass


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
