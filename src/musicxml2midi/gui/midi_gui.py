#!/usr/bin/env python3
"""
Live MIDI Watcher – desktop app (GUI only)
Requests implemented:
1. Velocity lane at the bottom.
2. CC lanes (incl. velocity) are time-synced with Piano Roll.
3. CC lanes show values 0–127.
4. Only horizontal zooming enabled (vertical fixed).
5. File → Load… menu entry (⌘L / Ctrl+L).
6. Bar/Beat grid: bars bold, beats lighter (shown in piano roll + CC lanes).
"""
from __future__ import annotations
import sys, os, time, math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pretty_midi as pm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# --------------------- Data types ---------------------
@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: int
    velocity: int
    channel: int

@dataclass
class CCPoint:
    time: float
    value: int
    channel: int
    cc: int

@dataclass
class MidiSong:
    notes: List[NoteEvent]
    cc1: List[CCPoint]
    cc11: List[CCPoint]
    cc64: List[CCPoint]
    vel: List[CCPoint]
    length: float
    ticks_per_beat: int
    tempo_bpm: float
    file_path: str
    beats: List[float]   # quarter notes
    bars:  List[float]   # bar downbeats

# --------------------- Helpers ---------------------
def make_vel_lut() -> np.ndarray:
    """RGBA-LUT für Velocity: 0=blau (transparent), 127=rot (deckend)."""
    lut = np.zeros((128, 4), dtype=np.ubyte)
    v = (np.arange(128, dtype=np.float32) / 127.0)
    r = (v * 255).astype(np.ubyte)
    g = np.zeros(128, dtype=np.ubyte)
    b = ((1.0 - v) * 255).astype(np.ubyte)
    a = np.where(np.arange(128) == 0, 0, 255).astype(np.ubyte)  # 0 transparent
    lut[:, 0] = r; lut[:, 1] = g; lut[:, 2] = b; lut[:, 3] = a
    return lut

def add_time_grid_to_view(target, beats: List[float], bars: List[float],
                          y_span: Optional[Tuple[float, float]] = None):
    """
    Add vertical lines to a *ViewBox* (beats thin, bars bold).
    `target` can be:
      - a ViewBox
      - a PlotWidget (will resolve to its ViewBox)
      - a PlotItem (will resolve to its ViewBox)
    Returns list of created items so they can be removed later.
    """
    # --- resolve ViewBox robustly ---
    vb = None
    # direct ViewBox?
    if isinstance(target, pg.ViewBox):
        vb = target
    # PlotWidget → PlotItem → ViewBox
    elif hasattr(target, "getPlotItem") and callable(target.getPlotItem):
        try:
            vb = target.getPlotItem().getViewBox()
        except Exception:
            pass
    # PlotItem → ViewBox
    elif hasattr(target, "getViewBox") and callable(target.getViewBox):
        try:
            vb = target.getViewBox()
        except Exception:
            pass

    if vb is None:
        raise TypeError("add_time_grid_to_view: could not resolve a ViewBox from target")

    items = []
    beat_pen = pg.mkPen(150, 150, 150, 100, width=1)
    bar_pen  = pg.mkPen(255, 255, 255, 180, width=2)

    for t in beats:
        ln = pg.InfiniteLine(pos=t, angle=90, movable=False, pen=beat_pen)
        vb.addItem(ln); items.append(ln)
    for t in bars:
        ln = pg.InfiniteLine(pos=t, angle=90, movable=False, pen=bar_pen)
        vb.addItem(ln); items.append(ln)
    return items

# --------------------- MIDI loading ---------------------
def load_midi(path: str) -> MidiSong:
    midi = pm.PrettyMIDI(path)
    tempi_times, tempi_bpms = midi.get_tempo_changes()
    bpm = float(tempi_bpms[0]) if len(tempi_bpms) else 120.0

    notes: List[NoteEvent] = []
    cc1:   List[CCPoint] = []
    cc11:  List[CCPoint] = []
    cc64:  List[CCPoint] = []
    vel:   List[CCPoint] = []

    for inst in midi.instruments:
        ch_guess = 9 if inst.is_drum else 0
        for n in inst.notes:
            notes.append(NoteEvent(n.start, n.end, n.pitch, n.velocity, ch_guess))
            vel.append(CCPoint(n.start, n.velocity, ch_guess, -1))
        for cc in inst.control_changes:
            if cc.number == 1:
                cc1.append(CCPoint(cc.time, cc.value, ch_guess, 1))
            elif cc.number == 11:
                cc11.append(CCPoint(cc.time, cc.value, ch_guess, 11))
            elif cc.number == 64:
                cc64.append(CCPoint(cc.time, cc.value, ch_guess, 64))

    length = float(midi.get_end_time())
    tpq = getattr(midi, 'resolution', 480)

    # Beat (quarters) & bars (downbeats) in seconds
    try:
        beats = list(midi.get_beats())
    except Exception:
        beats = []
    try:
        bars = list(midi.get_downbeats())
    except Exception:
        bars = []
    beats = [t for t in beats if 0 <= t <= length]
    bars  = [t for t in bars  if 0 <= t <= length]

    return MidiSong(
        notes=notes,
        cc1=sorted(cc1, key=lambda x: x.time),
        cc11=sorted(cc11, key=lambda x: x.time),
        cc64=sorted(cc64, key=lambda x: x.time),
        vel=sorted(vel, key=lambda x: x.time),
        length=length, ticks_per_beat=tpq,
        tempo_bpm=bpm, file_path=path,
        beats=beats, bars=bars
    )

# --------------------- Watcher ---------------------
class MidiFileWatcher(FileSystemEventHandler):
    def __init__(self, path: str, on_change):
        super().__init__()
        self.path = os.path.abspath(path)
        self.on_change = on_change
        self._last_sig = 0.0

    def _maybe_signal(self, candidate_path: str):
        # Reagiere, wenn src oder dest genau die beobachtete Datei ist
        if os.path.abspath(candidate_path) == self.path:
            now = time.time()
            if now - self._last_sig > 0.3:  # debounce etwas großzügiger
                self._last_sig = now
                self.on_change()

    def on_modified(self, event):
        self._maybe_signal(event.src_path)

    def on_created(self, event):
        self._maybe_signal(event.src_path)

    def on_moved(self, event):
        # Bei atomarem Save ist meist dest_path == self.path
        dest = getattr(event, "dest_path", None)
        if dest:
            self._maybe_signal(dest)
        else:
            self._maybe_signal(event.src_path)

class GestureViewBox(pg.ViewBox):
    """
    Trackpad:
      • Zwei Finger = Scroll/Pan
      • ⌘ (Command) gehalten = Zoom um Cursor
    zoom_axes: "both" oder "x"
    """
    def __init__(self, zoom_axes: str = "both",
                 zoom_modifier=QtCore.Qt.KeyboardModifier.MetaModifier,
                 allow_y_pan: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_axes = zoom_axes
        self.zoom_modifier = zoom_modifier
        self.allow_y_pan = allow_y_pan
        self.setMouseMode(self.PanMode)

    def wheelEvent(self, ev, axis=None, **kwargs):
        # --- Position robust bestimmen (Qt6/QGraphicsSceneWheelEvent vs QWheelEvent) ---
        try:
            scene_pt = ev.scenePosition()                           # QGraphicsSceneWheelEvent
        except AttributeError:
            try:
                scene_pt = self.mapToScene(ev.position().toPoint()) # QWheelEvent (Qt6)
            except AttributeError:
                scene_pt = self.mapToScene(ev.pos())                # Fallback
        pos_in_view = self.mapSceneToView(scene_pt)

        # Modifier
        try:
            mods = ev.modifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier(0)

        # Delta bestimmen (Pixel bevorzugt; sonst Winkel)
        dx_px = dy_px = 0.0
        try:
            pix = ev.pixelDelta()
            if not pix.isNull():
                dx_px, dy_px = float(pix.x()), float(pix.y())
        except Exception:
            pass
        if dx_px == 0.0 and dy_px == 0.0:
            try:
                ang = ev.angleDelta()
                dx_px, dy_px = ang.x() / 3.0, ang.y() / 3.0
            except Exception:
                d = 0.0
                try: d = float(ev.delta())
                except Exception: d = 0.0
                try: ori = ev.orientation()
                except Exception: ori = QtCore.Qt.Orientation.Vertical
                if ori == QtCore.Qt.Orientation.Horizontal:
                    dx_px = d / 3.0
                else:
                    dy_px = d / 3.0

        if dx_px == 0.0 and dy_px == 0.0:
            ev.ignore(); return

        # --- ⌘ gedrückt -> ZOOM ---
        if mods & (QtCore.Qt.KeyboardModifier.MetaModifier | QtCore.Qt.KeyboardModifier.ControlModifier):
            steps = (dy_px / 40.0) if dy_px != 0.0 else (dx_px / 40.0)
            steps = max(-10.0, min(10.0, steps))
            f = (0.85 ** steps) if steps > 0 else (1.0 / (0.85 ** -steps))
            if self.zoom_axes == "x":
                self.scaleBy(s=(f, 1.0), center=pos_in_view)
            else:
                self.scaleBy(s=(f, f),   center=pos_in_view)
            ev.accept(); return

        # --- Scroll/Pan ---
        xr, yr = self.viewRange()
        vw = max(1.0, float(self.width()))
        vh = max(1.0, float(self.height()))
        tx = -dx_px * (xr[1] - xr[0]) / vw
        ty = -dy_px * (yr[1] - yr[0]) / vh
        if not self.allow_y_pan:             # <— NEU
            ty = 0.0
        self.translateBy(x=tx, y=ty)
        ev.accept()

class FollowViewBox(pg.ViewBox):
    """ViewBox, die Wheel-Ereignisse ignoriert – folgt nur per XLink."""
    def wheelEvent(self, ev, *args, **kwargs):
        ev.ignore()  # nichts tun

# --------------------- Widgets ---------------------
class PianoRoll(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        # Use a PlotItem with our gesture ViewBox (zoom both axes here)
        self.plot = self.addPlot(row=0, col=0, viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.getAxis('bottom').setStyle(showValues=False)
        self.plot.getAxis('left').setStyle(showValues=False)

        self.view = self.plot.getViewBox()
        self.view.invertY(True)         # piano roll flipped vertically

        self.img = pg.ImageItem(axisOrder='row-major')
        self.plot.addItem(self.img)

        self._song: Optional[MidiSong] = None
        self._pixels_per_sec = 200
        self._min_pitch = 21
        self._max_pitch = 108
        self._grid_items: List[pg.InfiniteLine] = []
        self._vel_lut = make_vel_lut()

    def set_song(self, song: MidiSong):
        self._song = song
        # --- Tonhöhenbereich automatisch anpassen ---
        pitches = [n.pitch for n in song.notes]
        if pitches:
            lo = max(0, min(pitches) - 2)       # etwas Puffer unten
            hi = min(127, max(pitches) + 2)     # etwas Puffer oben
            if hi - lo < 12:                    # Mindesthöhe, damit es nicht „platt“ wirkt
                hi = min(127, lo + 12)
            self._min_pitch, self._max_pitch = lo, hi
        else:
            # Fallback
            self._min_pitch, self._max_pitch = 21, 108

        self.render()

        # vorhandenes Grid entfernen & neu setzen
        for it in self._grid_items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self._grid_items = add_time_grid_to_view(self.view, song.beats, song.bars)

    def render(self):
        if not self._song:
            self.img.clear()
            return
        w = int(max(1, math.ceil(self._song.length * self._pixels_per_sec)))
        h = (self._max_pitch - self._min_pitch + 1)
        arr = np.zeros((h, w), dtype=np.ubyte)
        for n in self._song.notes:
            x0 = int(n.start * self._pixels_per_sec)
            x1 = int(n.end * self._pixels_per_sec)
            y = self._max_pitch - n.pitch
            if 0 <= y < h:
                arr[y, x0:x1] = max(0, min(127, n.velocity))
        # farbige Velocity-LUT (blau->rot)
        self.img.setImage(arr, levels=(0,127), lut=self._vel_lut)
        # X auf Sekunden mappen
        self.img.setRect(QtCore.QRectF(0, 0, self._song.length, arr.shape[0]))
        self.view.setRange(QtCore.QRectF(0, 0, self._song.length, arr.shape[0]))

    def set_playhead(self, t: float):
        x = t
        if not hasattr(self, 'ph'):
            self.ph = pg.InfiniteLine(pos=x, angle=90)
            self.view.addItem(self.ph)
        self.ph.setPos(x)

class CCLane(pg.PlotWidget):
    def __init__(self, label: str):
        super().__init__(viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel('left', label)
        self.setYRange(0, 127)
        self.curve = self.plot([], [])
        try:
            self.curve.setPen(pg.mkPen(width=2))
        except Exception:
            pass
        self.ph = pg.InfiniteLine(pos=0, angle=90, movable=False)
        self.addItem(self.ph)
        self._grid_items: List[pg.InfiniteLine] = []

    def set_grid(self, beats: List[float], bars: List[float]):
        for it in self._grid_items:
            try: self.getViewBox().removeItem(it)
            except Exception: pass
        self._grid_items = add_time_grid_to_view(self, beats, bars)

    def set_data(self, pts: List[CCPoint], length: float):
        if not pts:
            self.curve.setData([0, length], [0, 0])
            self.setXRange(0, max(1.0, length))
            return
        # explicit step arrays (robust across pyqtgraph versions)
        t = [p.time for p in pts]
        v = [p.value for p in pts]
        xs: List[float] = []
        ys: List[float] = []
        for i in range(len(t)):
            cur_t = t[i]; cur_v = v[i]
            xs.append(cur_t); ys.append(cur_v)
            next_t = t[i+1] if i+1 < len(t) else cur_t + max(1e-6, 0.001*max(1.0, length))
            xs.append(next_t); ys.append(cur_v)
        self.curve.setData(xs, ys)
        self.setXRange(0, max(1.0, length))

    def set_playhead(self, t: float):
        self.ph.setPos(t)

class VelocityLane(pg.PlotWidget):
    def __init__(self, label: str = "Velocity"):
        super().__init__(viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel('left', label)
        self.setYRange(0, 127)
        self.bar = None
        self.ph = pg.InfiniteLine(pos=0, angle=90, movable=False)
        self.addItem(self.ph)
        self._grid_items: List[pg.InfiniteLine] = []

    def set_grid(self, beats: List[float], bars: List[float]):
        for it in self._grid_items:
            try: self.getViewBox().removeItem(it)
            except Exception: pass
        self._grid_items = add_time_grid_to_view(self, beats, bars)

    def _vel_color(self, v: int) -> QtGui.QColor:
        v = max(0, min(127, int(v)))
        r = int(v / 127 * 255)
        b = int((1 - v / 127) * 255)
        return QtGui.QColor(r, 0, b)

    def set_data_from_notes(self, notes: List[NoteEvent], length: float):
        # Alte Items entfernen
        if isinstance(self.bar, list):
            for it in self.bar:
                try: self.getViewBox().removeItem(it)
                except Exception: pass
        elif self.bar:
            try: self.removeItem(self.bar)
            except Exception: pass
        self.bar = []

        self.setXRange(0, max(1.0, length))

        if not notes:
            return

        vb = self.getViewBox()
        # Pixelbreite der „Balken“ (Gefühlssache: 2–4)
        px = 8

        for n in notes:
            h = max(0, min(127, n.velocity))
            # Eine vertikale Linie von (x,0) bis (x,h)
            line_item = QtWidgets.QGraphicsLineItem(QtCore.QLineF(n.start, 0.0, n.start, float(h)))
            pen = QtGui.QPen()
            pen.setColor(self._vel_color(h))   # blau->rot je Velocity
            pen.setWidth(px)                   # Breite in Pixeln
            pen.setCosmetic(True)              # WICHTIG: pixelkonstant, unabhängig vom Zoom
            pen.setCapStyle(QtCore.Qt.PenCapStyle.SquareCap)
            line_item.setPen(pen)
            vb.addItem(line_item)
            self.bar.append(line_item)

    def set_playhead(self, t: float):
        self.ph.setPos(t)

# --------------------- Main window ---------------------
class Main(QtWidgets.QMainWindow):
    def __init__(self, midi_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Live MIDI Watcher – Piano Roll")
        self.resize(1200, 800)

        # --- Widgets ---
        self.roll = PianoRoll()
        self.cc1  = CCLane("CC1 (Mod)")
        self.cc11 = CCLane("CC11 (Expr)")
        self.cc64 = CCLane("CC64 (Sustain)")
        self.vel  = VelocityLane("Velocity")

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        v.addWidget(self.roll, 6)
        v.addWidget(self.vel,  2)
        v.addWidget(self.cc1,  2)
        v.addWidget(self.cc11, 2)
        v.addWidget(self.cc64, 2)

        # gleiche Achsenbreite links (Y-Labels)
        axis_w = 46
        try:
            self.roll.plot.getAxis('left').setWidth(axis_w)
            for lane in (self.cc1, self.cc11, self.cc64, self.vel):
                lane.getPlotItem().getAxis('left').setWidth(axis_w)
        except Exception:
            pass

        self.status = self.statusBar()

        # Link X-axes (Sekunden) mit Piano Roll
        try:
            self.cc1.getViewBox().setXLink(self.roll.view)
            self.cc11.getViewBox().setXLink(self.roll.view)
            self.cc64.getViewBox().setXLink(self.roll.view)
            self.vel.getViewBox().setXLink(self.roll.view)
        except Exception:
            pass

        # Menü: File → Load
        file_menu = self.menuBar().addMenu("File")
        act_load = QtGui.QAction("Load…", self)
        act_load.setShortcut(QtGui.QKeySequence("Ctrl+L" if sys.platform != "darwin" else "Meta+L"))
        act_load.triggered.connect(self._load_dialog)
        file_menu.addAction(act_load)

        # --- Watchdog & Polling ---
        self.watcher: Optional[MidiFileWatcher] = None
        self.observer: Optional[Observer] = None
        self._watch_path: Optional[str] = None
        self._last_stat: Optional[tuple[float, int]] = None
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(500)  # ms
        self._poll_timer.timeout.connect(self._poll_tick)

        if midi_path:
            self.load_path(midi_path)

    # ------------------ Datei laden ------------------
    def _load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load MIDI", os.getcwd(), "MIDI Files (*.mid *.midi)"
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str):
        try:
            song = load_midi(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return

        self.roll.set_song(song)
        try:
            self.roll.view.setXRange(0, max(1.0, song.length), padding=0)
            for lane in (self.cc1, self.cc11, self.cc64, self.vel):
                lane.setXRange(0, max(1.0, song.length), padding=0)
        except Exception:
            pass

        # Daten setzen
        self.cc1.set_data(song.cc1, song.length)
        self.cc11.set_data(song.cc11, song.length)
        self.cc64.set_data(song.cc64, song.length)
        self.vel.set_data_from_notes(song.notes, song.length)

        # Gridlinien setzen
        self.cc1.set_grid(song.beats, song.bars)
        self.cc11.set_grid(song.beats, song.bars)
        self.cc64.set_grid(song.beats, song.bars)
        self.vel.set_grid(song.beats, song.bars)

        note_n = len(song.notes); cc1_n = len(song.cc1); cc11_n = len(song.cc11); cc64_n = len(song.cc64)
        self.status.showMessage(
            f"Loaded {os.path.basename(path)} | {song.length:.2f}s | "
            f"notes={note_n} cc1={cc1_n} cc11={cc11_n} cc64={cc64_n}",
            6000
        )
        self.setWindowTitle(f"Live MIDI Watcher – {os.path.basename(path)}")

        # Watcher & Polling starten
        self._setup_watcher(path)
        self._start_poll(path)

    # ------------------ File-Watcher ------------------
    def _setup_watcher(self, path: str):
        if self.observer:
            try:
                self.observer.stop(); self.observer.join(1)
            except Exception:
                pass
            self.observer = None

        watch_dir = os.path.dirname(os.path.abspath(path))
        target_path = os.path.abspath(path)

        def on_change():
            QtCore.QTimer.singleShot(300, lambda: self._reload(target_path))

        try:
            from watchdog.observers.polling import PollingObserver
            ObserverCls = PollingObserver
        except Exception:
            from watchdog.observers import Observer
            ObserverCls = Observer

        self.observer = ObserverCls()
        handler = MidiFileWatcher(target_path, on_change)
        self.observer.schedule(handler, watch_dir, recursive=False)
        self.observer.start()

    # ------------------ Polling ------------------
    def _file_signature(self, path: str):
        try:
            st = os.stat(path)
            return (st.st_mtime, st.st_size)
        except Exception:
            return None

    def _start_poll(self, path: str):
        self._watch_path = os.path.abspath(path)
        self._last_stat = self._file_signature(self._watch_path)
        self._poll_timer.start()

    def _stop_poll(self):
        self._poll_timer.stop()
        self._watch_path = None
        self._last_stat = None

    def _poll_tick(self):
        if not self._watch_path:
            return
        sig = self._file_signature(self._watch_path)
        if sig is None:
            return
        if self._last_stat is None:
            self._last_stat = sig
            return
        if sig != self._last_stat:
            self._last_stat = sig
            QtCore.QTimer.singleShot(250, lambda p=self._watch_path: self._reload(p))

    # ------------------ Reload ------------------
    def _reload(self, path: str):
        try:
            song = load_midi(path)
        except Exception as e:
            print("Reload failed:", e)
            return

        self.roll.set_song(song)
        try:
            self.roll.view.setXRange(0, max(1.0, song.length), padding=0)
            for lane in (self.cc1, self.cc11, self.cc64, self.vel):
                lane.setXRange(0, max(1.0, song.length), padding=0)
        except Exception:
            pass

        self.cc1.set_data(song.cc1, song.length)
        self.cc11.set_data(song.cc11, song.length)
        self.cc64.set_data(song.cc64, song.length)
        self.vel.set_data_from_notes(song.notes, song.length)

        self.cc1.set_grid(song.beats, song.bars)
        self.cc11.set_grid(song.beats, song.bars)
        self.cc64.set_grid(song.beats, song.bars)
        self.vel.set_grid(song.beats, song.bars)

        note_n = len(song.notes); cc1_n = len(song.cc1); cc11_n = len(song.cc11); cc64_n = len(song.cc64)
        self.status.showMessage(
            f"MIDI reloaded (file changed) | "
            f"notes={note_n} cc1={cc1_n} cc11={cc11_n} cc64={cc64_n}",
            4000
        )

    # ------------------ Laufende Updates ------------------
    def on_tick(self, t: float):
        self.roll.set_playhead(t)
        self.cc1.set_playhead(t)
        self.cc11.set_playhead(t)
        self.cc64.set_playhead(t)
        self.vel.set_playhead(t)

    # ------------------ Close ------------------
    def closeEvent(self, e: QtGui.QCloseEvent):
        if self.observer:
            self.observer.stop(); self.observer.join(1)
        self._stop_poll()
        return super().closeEvent(e)
    
# --------------------- Entry ---------------------
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    midi_path = sys.argv[1] if len(sys.argv)>1 else None
    w = Main(midi_path)
    w.show()
    sys.exit(app.exec())