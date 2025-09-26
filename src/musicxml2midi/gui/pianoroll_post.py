# gui/pianoroll_post.py
from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import math
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from .models import MidiSong
from .utils import GestureViewBox

def _vel_lut_rgba() -> np.ndarray:
    lut = np.zeros((128, 4), dtype=np.ubyte)
    lut[0] = (0, 0, 0, 0)
    for i in range(1, 128):
        t = i / 127.0
        r = int(255 * t)
        g = 0
        b = int(255 * (1.0 - t))
        lut[i] = (r, g, b, 255)  # Blau -> Rot, volle Deckkraft
    return lut

class PostPanel(QtWidgets.QWidget):
    """
    Unteres Panel: Interpretation -> MIDI
      - große Piano-Roll (Velocity-Färbung)
      - Velocity-Lane (fixe Balkenbreite)
      - eine CC-Lane (Dropdown, Default CC1)
    Wenn embedded_controls=True, wird KEINE eigene Controlleiste angezeigt;
    stattdessen kann cc_selector_widget() extern neben eine Überschrift gesetzt werden.
    """
    def __init__(self, parent=None, embedded_controls: bool = False):
        super().__init__(parent)
        self._song: Optional[MidiSong] = None
        self._pixels_per_sec = 200
        self._min_pitch = 21
        self._max_pitch = 108
        self._lut = _vel_lut_rgba()
        self._embedded = embedded_controls

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # CC-Auswahl
        self._cc_widget = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(self._cc_widget); h.setContentsMargins(0,0,0,0); h.setSpacing(6)
        h.addWidget(QtWidgets.QLabel("CC:"))
        self.ccCombo = QtWidgets.QComboBox()
        h.addWidget(self.ccCombo)

        def _fill_ccs(combo: QtWidgets.QComboBox, default: int):
            labels = [
                (1,  "1  (Modulation)"),
                (2,  "2  (Breath)"),
                (7,  "7  (Volume)"),
                (10, "10 (Pan)"),
                (11, "11 (Expression)"),
                (64, "64 (Sustain)"),
                (67, "67 (Soft Pedal)"),
            ]
            for num, txt in labels:
                combo.addItem(txt, userData=num)
            for n in range(128):
                if all(combo.itemData(i) != n for i in range(combo.count())):
                    combo.addItem(str(n), userData=n)
            for i in range(combo.count()):
                if combo.itemData(i) == default:
                    combo.setCurrentIndex(i); break

        _fill_ccs(self.ccCombo, 1)

        # Nur anzeigen, wenn nicht embedded
        if not self._embedded:
            ctl = QtWidgets.QHBoxLayout(); ctl.setContentsMargins(0,0,0,0)
            ctl.addWidget(self._cc_widget); ctl.addStretch(1)
            root.addLayout(ctl)

        # Graphics
        self.gw = pg.GraphicsLayoutWidget()
        pg.setConfigOptions(antialias=True)
        root.addWidget(self.gw, 1)

        # 0) Große Roll
        self.roll = self.gw.addPlot(row=0, col=0, viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.roll.hideButtons(); self.roll.setMenuEnabled(False)
        self.roll.getAxis('bottom').setStyle(showValues=False)
        self.roll.getAxis('left').setStyle(showValues=False)
        self.roll.setMouseEnabled(x=True, y=False)
        self.view = self.roll.getViewBox(); self.view.invertY(True)
        self.img = pg.ImageItem(axisOrder='row-major'); self.roll.addItem(self.img)

        # 1) Velocity
        self.vel_plot = self.gw.addPlot(row=1, col=0, viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.vel_plot.hideButtons(); self.vel_plot.setMenuEnabled(False)
        self.vel_plot.getAxis('bottom').setStyle(showValues=False)
        self.vel_plot.getAxis('left').setStyle(showValues=False)
        self.vel_plot.setMaximumHeight(90)
        self.vel_plot.setMouseEnabled(x=True, y=False)

        # 2) CC
        self.cc_plot = self.gw.addPlot(row=2, col=0, viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.cc_plot.hideButtons(); self.cc_plot.setMenuEnabled(False)
        self.cc_plot.getAxis('bottom').setStyle(showValues=False)
        self.cc_plot.getAxis('left').setStyle(showValues=False)
        self.cc_plot.setMaximumHeight(90)
        self.cc_plot.setMouseEnabled(x=True, y=False)

        # X-Link
        self.vel_plot.setXLink(self.roll)
        self.cc_plot.setXLink(self.roll)

        # Stores
        self._grid_roll: List[pg.InfiniteLine] = []
        self._grid_vel:  List[pg.InfiniteLine] = []
        self._grid_cc:   List[pg.InfiniteLine] = []
        self._vel_items: List[pg.GraphicsObject] = []
        self._cc_items:  List[pg.GraphicsObject] = []

        # Callbacks
        self.ccCombo.currentIndexChanged.connect(self._render_cc)

    # --- Embedding helper ---
    def cc_selector_widget(self) -> QtWidgets.QWidget:
        """Kleines Widget „CC: [Dropdown]“ zum Einbetten in Header-Leisten."""
        return self._cc_widget

    # ---- Public API ----
    def set_song(self, song: MidiSong):
        self._song = song
        pitches = [n.pitch for n in (song.notes or [])]
        if pitches:
            lo = max(0, min(pitches)-2); hi = min(127, max(pitches)+2)
            if hi-lo < 12: hi = min(127, lo+12)
            self._min_pitch, self._max_pitch = lo, hi
        else:
            self._min_pitch, self._max_pitch = 21, 108

        self._render_roll()
        self._render_velocity_lane()
        self._render_cc()
        self._render_grid()

    # ---- Rendering ----
    def _render_roll(self):
        if not (self._song and self._song.notes):
            self.img.clear(); self.roll.setYRange(0, 1); self.roll.setXRange(0, 3); return

        w = int(max(1, math.ceil(self._song.length * self._pixels_per_sec)))
        h = (self._max_pitch - self._min_pitch + 1)
        arr = np.zeros((h, w), dtype=np.ubyte)

        for n in self._song.notes:
            x0 = int(n.start * self._pixels_per_sec)
            x1 = int(n.end   * self._pixels_per_sec)
            y  = self._max_pitch - n.pitch
            if 0 <= y < h and x1 > x0 >= 0:
                arr[y, x0:x1] = max(1, min(127, int(n.velocity)))

        self.img.setImage(arr, levels=(0,127), lut=self._lut)
        self.img.setRect(QtCore.QRectF(0, 0, self._song.length, h))
        self.view.setRange(QtCore.QRectF(0, -1.5, max(3.0, self._song.length), h + 1.5))

    def _render_velocity_lane(self):
        # alte Items entfernen
        for it in self._vel_items:
            try:
                self.vel_plot.removeItem(it)
            except Exception:
                pass
        self._vel_items.clear()

        self.vel_plot.setYRange(0, 127)
        self.vel_plot.setXRange(0, max(3.0, (self._song.length if self._song else 0.0)))
        if not (self._song and self._song.notes):
            return

        width_px = 6  # konstante Bildschirm-Breite der Balken
        for n in self._song.notes:
            x = float(n.start)
            y1, y2 = 0.0, float(max(0, min(127, int(n.velocity))))
            # Farbe aus LUT (blau->rot)
            rgba = self._lut[max(1, min(127, int(n.velocity)))]
            col = (int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))

            # Vertikaler „Balken“ als Linie mit kosmetischem Pen (Breite in PX)
            curve = pg.PlotCurveItem([x, x], [y1, y2],
                                    pen=pg.mkPen(col, width=width_px, cosmetic=True))
            self.vel_plot.addItem(curve)
            self._vel_items.append(curve)

    def _cc_values(self, cc_num: int) -> List[Tuple[float,int]]:
        if not self._song: return []
        cc_map: Dict[int, List[Tuple[float,int]]] = self._song.meta.get("cc") or {}
        return sorted(cc_map.get(cc_num, []), key=lambda x: x[0])

    def _render_cc(self):
        # alte Items entfernen
        for it in self._cc_items:
            try:
                self.cc_plot.removeItem(it)
            except Exception:
                pass
        self._cc_items.clear()

        self.cc_plot.setYRange(0, 127)
        song = getattr(self, "_song", None)
        self.cc_plot.setXRange(0, max(3.0, song.length if song else 0.0))
        if not song:
            return

        cc_num = self.ccCombo.currentData()
        if cc_num is None:
            cc_num = 1
        cc_num = int(cc_num)

        pts = self._cc_values(cc_num)  # [(time, value)]
        if not pts:
            return

        # sortieren + aufeinanderfolgende identische Punkte entfernen
        pts = sorted(pts, key=lambda x: x[0])
        clean = []
        for t, v in pts:
            if not clean or (abs(t - clean[-1][0]) > 1e-9 or int(v) != int(clean[-1][1])):
                clean.append((float(t), int(v)))
        if not clean:
            return

        times = [t for (t, _v) in clean]
        vals  = [int(v) for (_t, v) in clean]

        # stepMode=True: x braucht eine Kante mehr als y
        end_time = float(song.length) if song.length else (times[-1] + 1.0)
        last_edge = end_time if end_time > times[-1] else (times[-1] + 1e-6)
        xs = times + [last_edge]
        ys = vals  # gleiche Länge wie 'vals'

        pen = pg.mkPen(210, 240, 200, 230, width=2)

        # Kurve (Treppenfunktion)
        curve = pg.PlotCurveItem(xs, ys, stepMode=True, pen=pen)
        self.cc_plot.addItem(curve)
        self._cc_items.append(curve)

        # Marker: hier NICHT xs/ys nehmen, sondern times/vals (gleiche Länge!)
        scat = pg.ScatterPlotItem(times, vals, size=6,
                                brush=pg.mkBrush(210, 240, 200),
                                pen=pg.mkPen(60, 90, 60))
        self.cc_plot.addItem(scat)
        self._cc_items.append(scat)

    def _render_grid(self):
        for coll, plot in ((self._grid_roll, self.roll), (self._grid_vel, self.vel_plot), (self._grid_cc, self.cc_plot)):
            for it in coll:
                try: plot.removeItem(it)
                except Exception: pass
            coll.clear()

        if not self._song: return
        pen_bar  = pg.mkPen(180, 180, 180, 220, width=2.0)
        pen_beat = pg.mkPen(140, 140, 140, 160, width=1.0)

        def add_lines(plot, beats, bars):
            items=[]
            for x in bars:
                ln = pg.InfiniteLine(pos=float(x), angle=90, pen=pen_bar);  plot.addItem(ln); items.append(ln)
            for x in beats:
                ln = pg.InfiniteLine(pos=float(x), angle=90, pen=pen_beat); plot.addItem(ln); items.append(ln)
            return items

        self._grid_roll = add_lines(self.roll,     self._song.beats, self._song.bars)
        self._grid_vel  = add_lines(self.vel_plot, self._song.beats, self._song.bars)
        self._grid_cc   = add_lines(self.cc_plot,  self._song.beats, self._song.bars)