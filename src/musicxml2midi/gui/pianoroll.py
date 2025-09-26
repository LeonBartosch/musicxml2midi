from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import math
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from .models import MidiSong
from .utils import GestureViewBox

LABEL_X_OFF = 0.012

def _ticks_to_seconds_map(tempos, tpb):
    tempos = sorted(tempos or [(0, 120.0)], key=lambda x: x[0])
    segs = []
    last_tick = 0; last_sec = 0.0; last_bpm = tempos[0][1]
    for tick, bpm in tempos:
        if tick > last_tick:
            segs.append((last_tick, tick, last_sec, last_bpm))
            last_sec += ((tick - last_tick) / tpb) * (60.0 / last_bpm)
            last_tick = tick
        last_bpm = bpm
    segs.append((last_tick, 10**12, last_sec, last_bpm))
    def to_sec(t: int) -> float:
        for s0, s1, base, bpm in segs:
            if s0 <= t < s1:
                return base + ((t - s0) / tpb) * (60.0 / bpm)
        s0, _, base, bpm = segs[-1]
        return base + ((t - s0) / tpb) * (60.0 / bpm)
    return to_sec

def _vel_lut_rgba() -> np.ndarray:
    lut = np.zeros((128, 4), dtype=np.ubyte)
    lut[0] = (0, 0, 0, 0)
    for i in range(1, 128):
        t = i / 127.0
        r = int(255 * t); g = int(255 * min(1.0, 2*t)); b = int(255 * (1.0 - t))
        lut[i] = (r, g, b, 255)
    return lut

def _val_from_velocity(vel: int) -> int:
    return max(1, min(127, int(vel)))

def _even_positions(n: int) -> List[float]:
    """Verteile n Punkte gleichmäßig in (0,1): (i+1)/(n+1)."""
    if n <= 0:
        return []
    return [(i + 1) / (n + 1) for i in range(n)]

class PianoRoll(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        pg.setConfigOptions(background=(30,30,30), foreground='w')

        self.plot = self.addPlot(row=0, col=0, viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.plot.hideButtons(); self.plot.setMenuEnabled(False)
        self.plot.getAxis('bottom').setStyle(showValues=False)
        self.plot.getAxis('left').setStyle(showValues=False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.view = self.plot.getViewBox(); self.view.invertY(True)
        self.img = pg.ImageItem(axisOrder='row-major'); self.plot.addItem(self.img)

        self.dyn_plot = self.addPlot(row=1, col=0, viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.dyn_plot.hideButtons(); self.dyn_plot.setMenuEnabled(False)
        self.dyn_plot.getAxis('left').setStyle(showValues=False)
        self.dyn_plot.getAxis('bottom').setStyle(showValues=False)
        self.dyn_plot.setMaximumHeight(60); self.dyn_plot.setMouseEnabled(x=True, y=False)

        self.art_plot = self.addPlot(row=2, col=0, viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.art_plot.hideButtons(); self.art_plot.setMenuEnabled(False)
        self.art_plot.getAxis('left').setStyle(showValues=False)
        self.art_plot.getAxis('bottom').setStyle(showValues=False)
        self.art_plot.setMaximumHeight(60); self.art_plot.setMouseEnabled(x=True, y=False)

        self.dyn_plot.setXLink(self.plot); self.art_plot.setXLink(self.plot)

        self._song: Optional[MidiSong] = None
        self._pixels_per_sec = 200
        self._min_pitch = 21; self._max_pitch = 108
        self._lut = _vel_lut_rgba()

        self._dyn_items: List[pg.GraphicsObject] = []
        self._art_items: List[pg.GraphicsObject] = []
        self._grid_main: List[pg.InfiniteLine] = []
        self._grid_dyn:  List[pg.InfiniteLine] = []
        self._grid_art:  List[pg.InfiniteLine] = []
        self._bar_labels: List[pg.TextItem] = []

        self._ph_main = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((255,0,0), width=2))
        self._ph_dyn  = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((255,0,0), width=2))
        self._ph_art  = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((255,0,0), width=2))
        self.view.addItem(self._ph_main); self.dyn_plot.addItem(self._ph_dyn); self.art_plot.addItem(self._ph_art)

    def set_song(self, song: MidiSong):
        self._song = song
        pitches = [n.pitch for n in song.notes]
        if pitches:
            lo = max(0, min(pitches) - 2); hi = min(127, max(pitches) + 2)
            if hi - lo < 12: hi = min(127, lo + 12)
            self._min_pitch, self._max_pitch = lo, hi
        else:
            self._min_pitch, self._max_pitch = 21, 108

        self._render_roll()
        self._render_dynamics_lane()
        self._render_articulations_lane()
        self._render_grid()

    def set_playhead(self, t: float):
        t = max(0.0, float(t))
        self._ph_main.setPos(t); self._ph_dyn.setPos(t); self._ph_art.setPos(t)

    def _render_roll(self):
        if not self._song or not self._song.notes:
            self.img.clear(); self.plot.setYRange(0,1); self.plot.setXRange(0,3); return
        w = int(max(1, math.ceil(self._song.length * self._pixels_per_sec)))
        h = (self._max_pitch - self._min_pitch + 1)
        arr = np.zeros((h, w), dtype=np.ubyte)
        for n in self._song.notes:
            x0 = int(n.start * self._pixels_per_sec); x1 = int(n.end * self._pixels_per_sec)
            y  = self._max_pitch - n.pitch
            if 0 <= y < h and x1 > x0 >= 0:
                arr[y, x0:x1] = _val_from_velocity(n.velocity)
        self.img.setImage(arr, levels=(0,127), lut=self._lut)
        self.img.setRect(QtCore.QRectF(0, 0, self._song.length, h))
        self.view.setRange(QtCore.QRectF(0, -1.5, max(3.0, self._song.length), h + 1.5))

    def _render_dynamics_lane(self):
        # alte Items entfernen
        for it in getattr(self, "_dyn_items", []):
            try:
                self.dyn_plot.removeItem(it)
            except Exception:
                pass
        self._dyn_items = []

        # Achsen & Range
        self.dyn_plot.setYRange(0, 1)
        self.dyn_plot.setXRange(0, max(3.0, (self._song.length if self._song else 0.0)))
        self.dyn_plot.hideAxis('left')
        self.dyn_plot.hideAxis('bottom')

        if not self._song:
            return

        # Daten
        dyns   = list(self._song.meta.get("dynamics", []))   # [(sec, "mf"), ...]
        wedges = list(self._song.meta.get("wedges",   []))   # [(a_sec, b_sec, "crescendo"/"diminuendo"), ...]

        # >>> Debug:
        print(f"[PR] wedges={len(wedges)} dyns={len(dyns)} first_wedge={wedges[0] if wedges else None}")

        # Farben & Labels
        CRESC_COLOR = ( 80, 200, 255)
        DIM_COLOR   = (255, 180,  80)
        TEXT_COLOR  = (230, 230, 230)
        DOT_COLOR   = (210, 210, 210)

        WLABEL = {"crescendo": "cresc.", "diminuendo": "dim."}
        WCOLOR = {"crescendo": CRESC_COLOR, "diminuendo": DIM_COLOR}

        # --- 1) Voll-hohe Bänder für Hairpins (LinearRegionItem ist robust) ---
        y_top = 1.02
        y_bot = -0.02
        BAND_ALPHA = 140  # etwas kräftiger
        for a, b, kind in wedges:
            try:
                a = float(a); b = float(b)
            except Exception:
                continue
            if not (b > a):
                continue

            col = WCOLOR.get(kind, (180, 180, 180))

            # LinearRegionItem arbeitet in Datenkoordinaten und spannt automatisch die Y-Achse
            region = pg.LinearRegionItem(values=(a, b), orientation=pg.LinearRegionItem.Vertical)
            region.setBrush(pg.mkBrush(col[0], col[1], col[2], BAND_ALPHA))
            region.setZValue(30)
            region.setMovable(False)
            # optional: dünne Outline deaktivieren (sonst grau)
            region.lines[0].setPen(pg.mkPen(0, 0, 0, 0))
            region.lines[1].setPen(pg.mkPen(0, 0, 0, 0))

            self.dyn_plot.addItem(region)
            self._dyn_items.append(region)

        # --- 2) Labels zusammenbauen (links am Startzeitpunkt) ---
        wedge_start: Dict[float, Tuple[str, tuple]] = {}
        for a, b, kind in wedges:
            if b <= a:
                continue
            key = round(float(a), 6)
            wedge_start[key] = (WLABEL.get(kind, kind), WCOLOR.get(kind, TEXT_COLOR))

        combined = []  # [(x, text, has_dot)]
        seen = set()

        for sec, mark in dyns:
            key = round(float(sec), 6)
            if key in wedge_start:
                wlbl, _ = wedge_start[key]
                combined.append((float(sec), f"{mark} {wlbl}", True))
                seen.add(key)
            else:
                combined.append((float(sec), mark, True))

        for a, b, kind in wedges:
            if b <= a:
                continue
            key = round(float(a), 6)
            if key in seen:
                continue
            lbl = WLABEL.get(kind, kind)
            combined.append((float(a), lbl, True))

        combined.sort(key=lambda x: x[0])

        # --- 3) Labels zeichnen (Bullet wie bei Artikulationen) ---
        y_mid = 0.5
        for x, text, use_dot in combined:
            if use_dot:
                dot = pg.ScatterPlotItem(
                    [float(x)], [y_mid],
                    size=7,
                    brush=pg.mkBrush(*DOT_COLOR),
                    pen=pg.mkPen(0, 0, 0, 100)
                )
                self.dyn_plot.addItem(dot); self._dyn_items.append(dot)

            ti = pg.TextItem(text, color=TEXT_COLOR, anchor=(0, 0.5))  # linksbündig
            ti.setPos(float(x) + LABEL_X_OFF, y_mid)
            self.dyn_plot.addItem(ti); self._dyn_items.append(ti)

    def _render_articulations_lane(self):
        for it in self._art_items:
            try: self.art_plot.removeItem(it)
            except Exception: pass
        self._art_items.clear()

        self.art_plot.setYRange(0, 1)
        self.art_plot.setXRange(0, max(3.0, (self._song.length if self._song else 0.0)))
        self.art_plot.hideAxis('left'); self.art_plot.hideAxis('bottom')

        if not self._song:
            return

        ART_COLORS = {
            "slur":          ( 90, 220, 120),
            "staccato":      ( 80, 170, 255),
            "staccatissimo": ( 80, 200, 255),
            "tenuto":        ( 80, 255, 220),
            "accent":        (255, 180,  80),
            "marcato":       (255, 120,  80),
            "pizzicato":     (200, 120, 255),
            "snap-pizzicato":(210, 100, 255),
            "tremolo":       (255, 220,  80),
            "trill":         (120, 220, 255),
            "mordent":       (120, 200, 255),
            "turn":          (120, 180, 255),
            "harmonic":      (180, 220, 255),
            "arco":          (190, 160, 255),
            "mute":          (180, 180, 180),
        }
        DEFAULT = (210, 210, 210)

        # --- Bevorzugt: exakte Tick-Aggregation ---
        arts_by_tick = self._song.meta.get("arts_by_tick")
        if arts_by_tick:
            # Wir brauchen t2s; greifen auf Conductor-Daten in meta zu
            tempos = self._song.meta.get("tempos")  # optional, falls du sie mitgibst
            tpb = self._song.meta.get("tpb")
            # Fallback: Wenn tempos/tpb nicht im Song stecken, nehmen wir eine lineare Sekundenskala an
            if tempos and tpb:
                t2s = _ticks_to_seconds_map(tempos, int(tpb))
                pairs = [(t2s(int(tick)), tags) for (tick, tags) in arts_by_tick]
            else:
                # Song kennt nur Sekundenlänge – dann nehmen wir bereits berechnete arts_by_time
                pairs = self._song.meta.get("arts_by_time") or []

            for sec, tags in pairs:
                # Duplikate pro Zeitpunkt entfernen, Reihenfolge beibehalten
                seen = set()
                tags = [t for t in tags if not (t in seen or seen.add(t))]

                # Gleichmäßig verteilen
                y_positions = _even_positions(len(tags))
                for art, y in zip(tags, y_positions):
                    col = ART_COLORS.get(art, DEFAULT)
                    dot = pg.ScatterPlotItem([float(sec)], [y], size=7,
                                            brush=pg.mkBrush(*col), pen=pg.mkPen(0,0,0,100))
                    self.art_plot.addItem(dot); self._art_items.append(dot)
                    txt = pg.TextItem(art, color=col, anchor=(0, 0.5))
                    txt.setPos(float(sec) + LABEL_X_OFF, y)
                    self.art_plot.addItem(txt); self._art_items.append(txt)
            return

        # --- Fallback: aggregierte Sekundenliste (wenn keine Tick-Daten da sind) ---
        arts_time = self._song.meta.get("arts_by_time")
        if arts_time:
            for sec, tags in arts_time:
                seen = set()
                tags = [t for t in tags if not (t in seen or seen.add(t))]
                y_positions = _even_positions(len(tags))
                for art, y in zip(tags, y_positions):
                    col = ART_COLORS.get(art, DEFAULT)
                    dot = pg.ScatterPlotItem([float(sec)], [y], size=7,
                                            brush=pg.mkBrush(*col), pen=pg.mkPen(0,0,0,100))
                    self.art_plot.addItem(dot); self._art_items.append(dot)
                    txt = pg.TextItem(art, color=col, anchor=(0, 0.5))
                    txt.setPos(float(sec) + LABEL_X_OFF, y)
                    self.art_plot.addItem(txt); self._art_items.append(txt)
            return

        # --- Letzter Fallback: per-Note-Variante ---
        if not self._song.notes:
            return
        for n in self._song.notes:
            if not n.articulations:
                continue
            # Duplikate pro Note entfernen
            seen = set()
            arts = [a for a in n.articulations if not (a in seen or seen.add(a))]

            y_positions = _even_positions(len(arts))
            for art, y in zip(arts, y_positions):
                col = ART_COLORS.get(art, DEFAULT)
                dot = pg.ScatterPlotItem([n.start], [y], size=7,
                                        brush=pg.mkBrush(*col), pen=pg.mkPen(0,0,0,100))
                self.art_plot.addItem(dot); self._art_items.append(dot)
                txt = pg.TextItem(art, color=col, anchor=(0, 0.5))
                txt.setPos(n.start + LABEL_X_OFF, y)
                self.art_plot.addItem(txt); self._art_items.append(txt)

    def _clear_grid(self):
        for coll, plot in (
            (self._grid_main, self.plot),
            (self._grid_dyn,  self.dyn_plot),
            (self._grid_art,  self.art_plot),
        ):
            for it in coll:
                try: plot.removeItem(it)
                except Exception: pass
            coll.clear()
        for t in self._bar_labels:
            try: self.plot.removeItem(t)
            except Exception: pass
        self._bar_labels.clear()

    def _render_grid(self):
        self._clear_grid()
        if not self._song: return
        pen_bar  = pg.mkPen(180, 180, 180, 220, width=2.0)
        pen_beat = pg.mkPen(140, 140, 140, 160, width=1.0)

        def add_lines(plot, beats, bars):
            items=[]
            for x in bars:
                ln = pg.InfiniteLine(pos=float(x), angle=90, pen=pen_bar); plot.addItem(ln); items.append(ln)
            for x in beats:
                ln = pg.InfiniteLine(pos=float(x), angle=90, pen=pen_beat); plot.addItem(ln); items.append(ln)
            return items

        self._grid_main = add_lines(self.plot, self._song.beats, self._song.bars)
        self._grid_dyn  = add_lines(self.dyn_plot, self._song.beats, self._song.bars)
        self._grid_art  = add_lines(self.art_plot, self._song.beats, self._song.bars)

        # Bar-Labels leicht rechts neben die Linie
        for i, x in enumerate(self._song.bars, start=1):
            label = pg.TextItem(str(i), color=(230,230,230))
            label.setAnchor((0.0, 1.0))     # linksbündig
            label.setPos(float(x) + 0.03, 0.0)  # kleiner X-Offset
            self.plot.addItem(label); self._bar_labels.append(label)

        ts_changes: List[Tuple[float,int,int]] = self._song.meta.get("timesig_changes") or []
        for sec, num, den in ts_changes:
            ts_label = pg.TextItem(f"{num}/{den}", color=(255,230,180))
            ts_label.setAnchor((0.0, 1.0))
            ts_label.setPos(float(sec) + 0.03, -0.9)
            self.plot.addItem(ts_label); self._bar_labels.append(ts_label)