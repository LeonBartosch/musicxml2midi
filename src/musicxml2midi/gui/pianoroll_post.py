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

        # X-Link
        self.vel_plot.setXLink(self.roll)

        # Stores
        self._grid_roll: List[pg.InfiniteLine] = []
        self._grid_vel:  List[pg.InfiniteLine] = []
        self._vel_items: List[pg.GraphicsObject] = []

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

    def _render_grid(self):
        for coll, plot in ((self._grid_roll, self.roll), (self._grid_vel, self.vel_plot)):
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