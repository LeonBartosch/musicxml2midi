# gui/utils.py
from __future__ import annotations
from typing import List
import pyqtgraph as pg
from PySide6 import QtCore

class GestureViewBox(pg.ViewBox):
    """
    Trackpad:
      • Zwei Finger = Scroll/Pan
      • ⌘ (Command) oder Ctrl gehalten = Zoom um Cursor
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
        # --- Position robust bestimmen ---
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

        # --- ⌘/Ctrl gedrückt -> ZOOM ---
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
        if not self.allow_y_pan:
            ty = 0.0
        self.translateBy(x=tx, y=ty)
        ev.accept()