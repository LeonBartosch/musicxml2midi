# src/musicxml2midi/config.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import copy
import yaml

# Paket-Root: .../src/musicxml2midi
PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = PKG_ROOT / "config.default.yaml"
USER_CFG_PATH = Path.home() / ".config" / "musicxml2midi" / "config.yaml"

def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        # lieber leer zurückgeben als den Core zu crashen
        pass
    return {}

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_config(
    user_path: Optional[Path] = None,
    default_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Lädt die Core-Konfiguration (Default + User-Overrides) und liefert ein gemergtes Dict.
    Enthält u.a. 'ticks_per_beat' (top-level). Humanization/GUI-Teile sind erlaubt,
    stören aber die Core-Nutzung nicht.
    """
    dpath = Path(default_path) if default_path else DEFAULT_CFG_PATH
    upath = Path(user_path) if user_path else USER_CFG_PATH

    defaults = _safe_load(dpath)
    user = _safe_load(upath)
    cfg = _deep_merge(defaults, user)

    # Minimal-Defaults sicherstellen
    cfg.setdefault("ticks_per_beat", 960)

    return cfg

def get_ticks_per_beat(cfg: Dict[str, Any]) -> int:
    """Bequemer Accessor."""
    try:
        return int(cfg.get("ticks_per_beat", 960))
    except Exception:
        return 960