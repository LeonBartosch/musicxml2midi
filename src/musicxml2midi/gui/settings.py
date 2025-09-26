# --- oben: imports/paths ---
from __future__ import annotations
from typing import Dict, Optional
from PySide6 import QtWidgets
import json, yaml
from pathlib import Path

__all__ = ["SettingsDialog"]

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CFG_JSON = APP_DIR / "config.default.json"
DEFAULT_CFG_YAML = APP_DIR / "config.default.yaml"
USER_CFG_PATH    = APP_DIR / "config.user.json"

# ---------------- I/O & Migrations ----------------

# --- Migration: vereinheitlicht Schema ---
def migrate_cfg(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return {"schema_version": 2}
    ver = int(cfg.get("schema_version", 1))
    if ver < 2:
        if "dynamics" in cfg and "velocities" not in cfg:
            cfg["velocities"] = cfg["dynamics"]
        cfg["schema_version"] = 2

    # ccX.seed -> ccX.curve.seed, random_noise -> random (wie gehabt)
    for cc_key in ("cc1", "cc11"):
        cc = cfg.get(cc_key) or {}
        curve = cc.get("curve") or {}
        if "seed" in cc and "seed" not in curve:
            curve["seed"] = cc["seed"]
        if "random_noise" in curve and "random" not in curve:
            curve["random"] = curve.pop("random_noise")
        cc["curve"] = curve
        cfg[cc_key] = cc

    # --- Phrasing defaults ---
    phr = cfg.get("phrasing") or {}
    sl  = phr.get("slur_legato") or {}
    es  = sl.get("early_start") or {}

    sl.setdefault("enabled", True)
    sl.setdefault("overlap_ticks", 30)
    # --- NEU: absolute/ms-Optionen (nur diese drei laut deiner Config) ---
    es.setdefault("ms", 0)                       # early_start.ms
    sl.setdefault("max_advance_ms", 120)         # absoluter Deckel in ms
    sl.setdefault("longest_note_ms", 800)        # Orientierung für Advance
    sl["early_start"] = es

    tim = cfg.get("timing") or {}
    # Migration: alte Beats-Parameter auf neue Frac-Keys spiegeln (nur Default, wenn neu fehlt)
    if "onset_jitter_frac_of_len" not in tim:
        # wenn alter Wert da ist, konservativ auf ≈0.25*onset_beats mappen, sonst Default
        v = float(tim.get("onset_jitter_beats", 0.05))
        tim["onset_jitter_frac_of_len"] = max(0.0, min(0.5, 0.25 if v > 0 else 0.0))
    tim.setdefault("dur_jitter_frac_of_len", 0.15)

    # Guards beibehalten
    tim.setdefault("onset_guard_prev_ticks", 0)
    tim.setdefault("onset_guard_next_ticks", 0)

    cfg["timing"] = tim

    # optional neue Keys mit sanften Defaults
    sl.setdefault("max_advance_frac_of_prev_len", 0.0)

    # bereits vorhandene:
    sl.setdefault("non_slur_min_gap_ticks", 10)
    sl.setdefault("tenuto_min_gap_ticks",   5)
    phr["slur_legato"] = sl
    cfg["phrasing"] = phr

    # --- Slur Advance → Velocity ---
    vel = cfg.get("velocities") or {}
    sav = vel.get("slur_advance_velocity") or {}

    # Defaults
    sav.setdefault("enabled", True)
    sav.setdefault("include_first_note", False)

    # NEU: Skalenmodus + absolute Schwelle in Millisekunden
    # (Default-Werte gern anpassen)
    sav.setdefault("fullscale_ms", 160)               # ~1/3 Beat bei 120 BPM

    sav.setdefault("min_vel", 31)
    sav.setdefault("max_vel", 105)
    sav.setdefault("gamma", 1.0)
    sav.setdefault("mix_old", 0.15)

    # OPTIONALE Migration alter 'fullscale_ticks' -> 'fullscale_ms'
    # Falls vorhanden, ungefähr auf 120 BPM umrechnen (1 Beat = 500 ms @120).
    if "fullscale_ticks" in sav and "fullscale_ms" not in sav:
        tpb = int(cfg.get("ticks_per_beat", 960) or 960)
        full_ticks = max(1, int(sav.get("fullscale_ticks") or 320))
        # 120 BPM: sec = (ticks/tpb) * 0.5 → ms = sec*1000
        approx_ms = int(round((full_ticks / float(tpb)) * 0.5 * 1000.0))
        sav["fullscale_ms"] = approx_ms
        # alten Key sauber entfernen
        try: sav.pop("fullscale_ticks")
        except Exception: pass

    vel["slur_advance_velocity"] = sav
    cfg["velocities"] = vel

    # (Optional) alte Slur-length-Mapping defaults entfernen:
    if "slur_len_velocity" in vel:
        vel.pop("slur_len_velocity")

    return cfg

# --- Defaults laden: erst JSON, sonst YAML ---
def load_defaults() -> dict:
    # 1) JSON bevorzugen
    if DEFAULT_CFG_JSON.exists():
        try:
            cfg = json.loads(DEFAULT_CFG_JSON.read_text(encoding="utf-8")) or {}
            return migrate_cfg(cfg)
        except Exception:
            pass  # fällt auf YAML zurück

    # 2) YAML fallback (für Übergangsphase)
    if DEFAULT_CFG_YAML.exists():
        try:
            cfg = yaml.safe_load(DEFAULT_CFG_YAML.read_text(encoding="utf-8")) or {}
            return migrate_cfg(cfg)
        except Exception:
            pass

    # 3) leere Defaults
    return migrate_cfg({})

def deep_update(base: dict, overrides: dict) -> dict:
    """
    Rekursives Update: überschreibt nur angegebene Werte/Äste.
    """
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def dict_diff(defaults: dict, merged: dict) -> dict:
    """
    Rekursiver Diff: liefert nur Abweichungen von defaults → overrides.
    - Dicts: rekursiv
    - Listen/Tuples: wenn ungleich, komplette Liste speichern
    - Primitives: wenn ungleich, Wert speichern
    """
    out = {}
    for k, v in (merged or {}).items():
        dv = defaults.get(k, None)
        if isinstance(v, dict) and isinstance(dv, dict):
            sub = dict_diff(dv, v)
            if sub:
                out[k] = sub
        elif isinstance(v, (list, tuple)) and isinstance(dv, (list, tuple)):
            if list(v) != list(dv):
                out[k] = v
        elif v != dv:
            out[k] = v
    return out

# --- load_config bleibt gleich, ruft nur load_defaults() ---
def load_config() -> dict:
    defaults = load_defaults()
    if USER_CFG_PATH.exists():
        try:
            overrides = json.loads(USER_CFG_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            overrides = {}
    else:
        overrides = {}
    return deep_update(defaults, overrides)

def save_user_config(cfg: dict):
    """
    Speichere NUR Overrides als JSON (legt Ordner bei Bedarf an).
    """
    USER_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_CFG_PATH.write_text(json.dumps(cfg or {}, indent=2, ensure_ascii=False), encoding="utf-8")

def reset_user_config():
    if USER_CFG_PATH.exists():
        USER_CFG_PATH.unlink()

def merged_config() -> dict:
    """Convenience: geladene (defaults + user) Konfiguration zurückgeben."""
    return load_config()

# ---------------- UI-Panels ----------------

class BeatAccentPanel(QtWidgets.QGroupBox):
    """Metrische Akzentuierung mit Clamp & Regelmatrix."""
    def __init__(self, title: str, cfg: Dict | None, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        cfg = cfg or {}
        self.setChecked(bool(cfg.get("enabled", True)))

        lay = QtWidgets.QVBoxLayout(self)

        # Clamp
        row_bounds = QtWidgets.QHBoxLayout()
        row_bounds.addWidget(QtWidgets.QLabel("Clamp min"))
        self.spn_min = QtWidgets.QSpinBox(); self.spn_min.setRange(1,127); self.spn_min.setValue(int(cfg.get("clamp_min", 1)))
        row_bounds.addWidget(self.spn_min)
        row_bounds.addSpacing(12)
        row_bounds.addWidget(QtWidgets.QLabel("Clamp max"))
        self.spn_max = QtWidgets.QSpinBox(); self.spn_max.setRange(1,127); self.spn_max.setValue(int(cfg.get("clamp_max", 127)))
        row_bounds.addWidget(self.spn_max)
        row_bounds.addStretch(1)
        lay.addLayout(row_bounds)

        # Rules grid
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Rule"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("× mul"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("+ add"), 0, 2)

        self._rule_keys = [
            ("bar_downbeat",   "Beat 1"),
            ("midbar_strong",  "Midbar strong"),
            ("onbeats_other",  "Onbeats other"),
            ("eighth_offbeats","Eighth offbeats (.5)"),
            ("sixteenth_offs", "Sixteenth offs (.25/.75)"),
            ("fallback",       "Fallback"),
        ]
        rules = cfg.get("rules", {}) or {}
        self.mul: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.add: Dict[str, QtWidgets.QSpinBox] = {}
        for r, (key, label) in enumerate(self._rule_keys, start=1):
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            m = QtWidgets.QDoubleSpinBox(); m.setRange(0.5, 2.0); m.setSingleStep(0.01); m.setValue(float(rules.get(key, {}).get("mul", 1.0)))
            a = QtWidgets.QSpinBox(); a.setRange(-32, 32); a.setValue(int(rules.get(key, {}).get("add", 0)))
            self.mul[key] = m; self.add[key] = a
            grid.addWidget(m, r, 1); grid.addWidget(a, r, 2)
        lay.addLayout(grid)

    def result_cfg(self) -> Dict:
        out: Dict = {
            "enabled": self.isChecked(),
            "clamp_min": int(self.spn_min.value()),
            "clamp_max": int(self.spn_max.value()),
            "rules": {},
        }
        for key in self.mul:
            out["rules"][key] = {"mul": float(self.mul[key].value()), "add": int(self.add[key].value())}
        return out
    
class VelocityHumanizePanel(QtWidgets.QGroupBox):
    """
    Humanize for note-on velocities (independent from beat accents):
      - enabled
      - seed
      - mul_jitter (×±, multiplicative around 1.0)
      - add_jitter (±, additive in MIDI velocity units)
    """
    def __init__(self, title: str, cfg: Dict | None, parent=None):
        super().__init__(title, parent)
        cfg = cfg or {}
        self.setCheckable(True)
        self.setChecked(bool(cfg.get("enabled", True)))

        lay = QtWidgets.QFormLayout(self)

        self.spn_seed = QtWidgets.QSpinBox()
        self.spn_seed.setRange(0, 2_000_000_000)
        self.spn_seed.setValue(int(cfg.get("seed", 1337)))
        lay.addRow("Seed:", self.spn_seed)

        self.dbl_mul = QtWidgets.QDoubleSpinBox()
        self.dbl_mul.setRange(0.0, 0.5)
        self.dbl_mul.setSingleStep(0.01)
        self.dbl_mul.setDecimals(3)
        self.dbl_mul.setValue(float(cfg.get("mul_jitter", 0.10)))
        self.dbl_mul.setSuffix(" ×±")
        lay.addRow("Amplitude jitter:", self.dbl_mul)

        self.spn_add = QtWidgets.QSpinBox()
        self.spn_add.setRange(0, 16)
        self.spn_add.setValue(int(cfg.get("add_jitter", 3)))
        self.spn_add.setSuffix(" ±")
        lay.addRow("Add jitter:", self.spn_add)

    def result_cfg(self) -> Dict:
        return {
            "enabled": bool(self.isChecked()),
            "seed": int(self.spn_seed.value()),
            "mul_jitter": float(self.dbl_mul.value()),
            "add_jitter": int(self.spn_add.value()),
        }
    
class SlurAdvanceVelocityPanel(QtWidgets.QGroupBox):
    """
    Slur Advance (Early-Start) → Velocity:
      - wenig Advance => hohe Velocity (max_vel)
      - viel Advance  => niedrige Velocity (min_vel)
      - alte Velocity nur noch als Beimischung (mix_old)
    """
    def __init__(self, title: str, cfg: Dict | None, parent=None):
        super().__init__(title, parent)
        cfg = cfg or {}
        self.setCheckable(True)
        self.setChecked(bool(cfg.get("enabled", True)))

        form = QtWidgets.QFormLayout(self)

        # NEU: absolute Schwelle (ms)
        self.spn_full_ms = QtWidgets.QSpinBox()
        self.spn_full_ms.setRange(1, 4000)   # 1..4000 ms (anpassbar)
        self.spn_full_ms.setSingleStep(5)
        self.spn_full_ms.setValue(int(cfg.get("fullscale_ms", 160)))
        self.spn_full_ms.setSuffix(" ms")
        self.spn_full_ms.setToolTip(
            "So viel Vorziehen (in ms) führt zur maximalen Velocity-Absenkung.\n"
            "Tempo- und Taktunabhängig, gut für DAW/Sample-Workflows."
        )
        form.addRow("Full-scale (absolute):", self.spn_full_ms)

        # Rest wie gehabt
        self.spn_min_vel = QtWidgets.QSpinBox()
        self.spn_min_vel.setRange(1, 127); self.spn_min_vel.setValue(int(cfg.get("min_vel", 31)))
        form.addRow("Min velocity (much advance):", self.spn_min_vel)

        self.spn_max_vel = QtWidgets.QSpinBox()
        self.spn_max_vel.setRange(1, 127); self.spn_max_vel.setValue(int(cfg.get("max_vel", 105)))
        form.addRow("Max velocity (little advance):", self.spn_max_vel)

        self.dbl_gamma = QtWidgets.QDoubleSpinBox()
        self.dbl_gamma.setRange(0.25, 4.0); self.dbl_gamma.setSingleStep(0.05); self.dbl_gamma.setDecimals(2)
        self.dbl_gamma.setValue(float(cfg.get("gamma", 1.0)))
        form.addRow("Curve γ (1=linear):", self.dbl_gamma)

        self.dbl_mix = QtWidgets.QDoubleSpinBox()
        self.dbl_mix.setRange(0.0, 1.0); self.dbl_mix.setSingleStep(0.05); self.dbl_mix.setDecimals(2)
        self.dbl_mix.setValue(float(cfg.get("mix_old", 0.15)))
        form.addRow("Mix old velocity (0..1):", self.dbl_mix)

        self.chk_first = QtWidgets.QCheckBox("Include first slurred note")
        self.chk_first.setChecked(bool(cfg.get("include_first_note", False)))
        form.addRow(self.chk_first)

        self.setLayout(form)

    def result_cfg(self) -> Dict:
        return {
            "enabled": self.isChecked(),
            "fullscale_ms": int(self.spn_full_ms.value()),
            "min_vel": int(self.spn_min_vel.value()),
            "max_vel": int(self.spn_max_vel.value()),
            "gamma": float(self.dbl_gamma.value()),
            "mix_old": float(self.dbl_mix.value()),
            "include_first_note": self.chk_first.isChecked(),
        }
    

class VelocitySection(QtWidgets.QGroupBox):
    def __init__(self, title: str, cfg: Dict | None, parent=None):
        super().__init__(title, parent)
        cfg = cfg or {}
        self.setCheckable(True)
        self.setChecked(bool(cfg.get("enabled", True)))

        root = QtWidgets.QVBoxLayout(self)

        ba_cfg = cfg.get("beat_accent", {}) or {}
        self.ba_panel = BeatAccentPanel("Beat Accents", ba_cfg, self)
        root.addWidget(self.ba_panel)

        vh_cfg = (cfg.get("humanize") or {}) or {}
        self.vh_panel = VelocityHumanizePanel("Humanize (Velocities)", vh_cfg, self)
        root.addWidget(self.vh_panel)

        # NEU: Slur-Längen-Velocity
        sav_cfg = (cfg.get("slur_advance_velocity") or {}) or {}
        self.sav_panel = SlurAdvanceVelocityPanel("Slur advance → Velocity", sav_cfg, self)
        root.addWidget(self.sav_panel)

        root.addStretch(1)

    def result_cfg(self) -> Dict:
        return {
            "enabled": self.isChecked(),
            "beat_accent": self.ba_panel.result_cfg(),
            "humanize": self.vh_panel.result_cfg(),
            "slur_advance_velocity": self.sav_panel.result_cfg(),
        }

class TimingSection(QtWidgets.QGroupBox):
    """
    Timing-Humanize: Onset-Offsets und Längenjitter in Beat-Anteilen.
    """
    def __init__(self, title: str, cfg: Dict | None, parent=None):
        super().__init__(title, parent)
        cfg = cfg or {}
        self.setCheckable(True)
        self.setChecked(bool(cfg.get("enabled", True)))

        lay = QtWidgets.QFormLayout(self)

        self.spn_seed = QtWidgets.QSpinBox()
        self.spn_seed.setRange(0, 2_000_000_000)
        self.spn_seed.setValue(int(cfg.get("seed", 2024)))
        lay.addRow("Seed:", self.spn_seed)

        self.dbl_onset_frac = QtWidgets.QDoubleSpinBox(); self.dbl_onset_frac.setDecimals(3)
        self.dbl_onset_frac.setRange(0.0, 0.5); self.dbl_onset_frac.setSingleStep(0.01)
        self.dbl_onset_frac.setValue(float(cfg.get("onset_jitter_frac_of_len", 0.25)))
        lay.addRow("Onset jitter (± frac of length):", self.dbl_onset_frac)

        self.dbl_dur_frac = QtWidgets.QDoubleSpinBox(); self.dbl_dur_frac.setDecimals(3)
        self.dbl_dur_frac.setRange(0.0, 0.5); self.dbl_dur_frac.setSingleStep(0.01)
        self.dbl_dur_frac.setValue(float(cfg.get("dur_jitter_frac_of_len", 0.15)))
        lay.addRow("Dur jitter (± frac of length):", self.dbl_dur_frac)

    def result_cfg(self) -> Dict:
        return {
            "enabled": self.isChecked(),
            "seed": int(self.spn_seed.value()),
            "onset_jitter_frac_of_len": float(self.dbl_onset_frac.value()),
            "dur_jitter_frac_of_len": float(self.dbl_dur_frac.value()),
        }
    
class PhrasingSection(QtWidgets.QGroupBox):
    """
    Phrasing: Slur-Legato (Overlap, Early Start) + Mindestabstände zwischen NICHT geslurten Noten.
    """
    def __init__(self, title: str, cfg: Dict | None, parent=None):
        super().__init__(title, parent)
        cfg = (cfg or {})
        self.setCheckable(True)
        self.setChecked(bool((cfg.get("slur_legato") or {}).get("enabled", True)))

        lay = QtWidgets.QVBoxLayout(self)

        sl = (cfg.get("slur_legato") or {})

        # --- Slur-Legato (Overlap + Early Start)
        grp_leg = QtWidgets.QGroupBox("Slur Legato", self)
        grp_leg.setCheckable(True)
        grp_leg.setChecked(bool(sl.get("enabled", True)))
        f = QtWidgets.QFormLayout(grp_leg)

        self.spn_overlap = QtWidgets.QSpinBox(); self.spn_overlap.setRange(0, 2000)
        self.spn_overlap.setValue(int(sl.get("overlap_ticks", 30)))
        f.addRow("Overlap (ticks):", self.spn_overlap)

        # --- NEU: Early start (ms) + Deckel/Kappung in ms ---
        self.spn_early_ms = QtWidgets.QSpinBox(); self.spn_early_ms.setRange(0, 5000)
        self.spn_early_ms.setValue(int((sl.get("early_start") or {}).get("ms", 0)))
        self.spn_early_ms.setSuffix(" ms")
        f.addRow("Early start abs (ms):", self.spn_early_ms)

        self.spn_max_adv_ms = QtWidgets.QSpinBox(); self.spn_max_adv_ms.setRange(0, 5000)
        self.spn_max_adv_ms.setValue(int(sl.get("max_advance_ms", 120)))
        self.spn_max_adv_ms.setSuffix(" ms")
        f.addRow("Max advance (ms):", self.spn_max_adv_ms)

        self.spn_longest_ms = QtWidgets.QSpinBox(); self.spn_longest_ms.setRange(0, 20000)
        self.spn_longest_ms.setValue(int(sl.get("longest_note_ms", 800)))
        self.spn_longest_ms.setSuffix(" ms")
        f.addRow("Longest note (ms):", self.spn_longest_ms)

        self.dbl_max_adv = QtWidgets.QDoubleSpinBox(); self.dbl_max_adv.setDecimals(3)
        self.dbl_max_adv.setRange(0.0, 1.0); self.dbl_max_adv.setSingleStep(0.01)
        self.dbl_max_adv.setValue(float(sl.get("max_advance_frac_of_prev_len", 0.0)))
        f.addRow("Max advance (frac of prev len):", self.dbl_max_adv)

        lay.addWidget(grp_leg)

        # --- NEU: Mindestabstände (nicht geslurte) + Tenuto
        grp_gap = QtWidgets.QGroupBox("Mindestabstände (NICHT geslurt)", self)
        g = QtWidgets.QFormLayout(grp_gap)

        self.spn_non_slur_gap = QtWidgets.QSpinBox(); self.spn_non_slur_gap.setRange(0, 2000)
        self.spn_non_slur_gap.setValue(int(sl.get("non_slur_min_gap_ticks", 10)))
        g.addRow("Min gap (ticks):", self.spn_non_slur_gap)

        self.spn_tenuto_gap = QtWidgets.QSpinBox(); self.spn_tenuto_gap.setRange(0, 2000)
        self.spn_tenuto_gap.setValue(int(sl.get("tenuto_min_gap_ticks", 5)))
        g.addRow("Min gap Tenuto (ticks):", self.spn_tenuto_gap)

        lay.addWidget(grp_gap)

        self.grp_leg = grp_leg
        self.grp_gap = grp_gap

    def result_cfg(self) -> Dict:
        return {
            "slur_legato": {
                "enabled": self.isChecked() and self.grp_leg.isChecked(),
                "overlap_ticks": int(self.spn_overlap.value()),
                "early_start": {
                    "ms": int(self.spn_early_ms.value()),
                    "max_advance_ms": int(self.spn_max_adv_ms.value()),
                    "longest_note_ms": int(self.spn_longest_ms.value()),
                },
                "max_advance_frac_of_prev_len": float(self.dbl_max_adv.value()),
                "non_slur_min_gap_ticks": int(self.spn_non_slur_gap.value()),
                "tenuto_min_gap_ticks": int(self.spn_tenuto_gap.value()),
            }
        }

class BaselineSubPanel(QtWidgets.QGroupBox):
    """Baseline-Start/End inkl. Jitter für Notes/Phrases."""
    def __init__(self, title: str, cfg: Dict | None, parent=None):
        super().__init__(title, parent)
        cfg = (cfg or {})
        lay = QtWidgets.QFormLayout(self)

        self.sp_start = QtWidgets.QDoubleSpinBox(); self.sp_start.setRange(0,1); self.sp_start.setSingleStep(0.01)
        self.sp_end   = QtWidgets.QDoubleSpinBox(); self.sp_end.setRange(0,1);   self.sp_end.setSingleStep(0.01)
        self.sp_jst   = QtWidgets.QDoubleSpinBox(); self.sp_jst.setRange(0,1);   self.sp_jst.setSingleStep(0.01)
        self.sp_jen   = QtWidgets.QDoubleSpinBox(); self.sp_jen.setRange(0,1);   self.sp_jen.setSingleStep(0.01)

        self.sp_start.setValue(float(cfg.get("start", 0.0)))
        self.sp_end.setValue(float(cfg.get("end", 0.0)))
        self.sp_jst.setValue(float(cfg.get("start_jitter", 0.0)))
        self.sp_jen.setValue(float(cfg.get("end_jitter", 0.0)))

        lay.addRow("Start (0..1):",        self.sp_start)
        lay.addRow("End (0..1):",          self.sp_end)
        lay.addRow("Start Jitter (±):",    self.sp_jst)
        lay.addRow("End Jitter (±):",      self.sp_jen)

    def result_cfg(self) -> Dict:
        return {
            "start":        float(self.sp_start.value()),
            "end":          float(self.sp_end.value()),
            "start_jitter": float(self.sp_jst.value()),
            "end_jitter":   float(self.sp_jen.value()),
        }

class HumanizeSubPanel(QtWidgets.QGroupBox):
    """
    Reusable Panel für Notes/Phrases:
      - enabled
      - peak_pos_jitter (additiv 0..1)
      - amp_jitter / sharpness_jitter / tail_exp_jitter (multiplikativ in %)
      - optional: Baseline (start/end + jitter)
      - optional: max_boost (nur Phrases)
    """
    def __init__(self, title: str, cfg: Dict | None, parent=None,
                 include_baseline: bool = False,
                 baseline_defaults: Optional[Dict] = None,
                 include_max_boost: bool = False,
                 max_boost_default: float = 0.35):
        super().__init__(title, parent)
        cfg = cfg or {}
        self.setCheckable(True)
        self.setChecked(bool(cfg.get("enabled", True)))

        outer = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.dbl_pos = QtWidgets.QDoubleSpinBox(); self.dbl_pos.setRange(0.0, 0.4); self.dbl_pos.setSingleStep(0.01)
        self.dbl_pos.setValue(float(cfg.get("peak_pos_jitter", 0.05))); self.dbl_pos.setSuffix(" ±")

        self.dbl_amp = QtWidgets.QDoubleSpinBox(); self.dbl_amp.setRange(0.0, 0.5); self.dbl_amp.setSingleStep(0.01)
        self.dbl_amp.setDecimals(3); self.dbl_amp.setValue(float(cfg.get("amp_jitter", 0.10))); self.dbl_amp.setSuffix(" ×±")

        self.dbl_sharp = QtWidgets.QDoubleSpinBox(); self.dbl_sharp.setRange(0.0, 1.0); self.dbl_sharp.setSingleStep(0.05)
        self.dbl_sharp.setDecimals(3); self.dbl_sharp.setValue(float(cfg.get("sharpness_jitter", 0.20))); self.dbl_sharp.setSuffix(" ×±")

        self.dbl_tail = QtWidgets.QDoubleSpinBox(); self.dbl_tail.setRange(0.0, 1.5); self.dbl_tail.setSingleStep(0.05)
        self.dbl_tail.setDecimals(3); self.dbl_tail.setValue(float(cfg.get("tail_exp_jitter", 0.25))); self.dbl_tail.setSuffix(" ×±")

        form.addRow("Peak-Position Jitter", self.dbl_pos)
        form.addRow("Amplitude Jitter",     self.dbl_amp)
        form.addRow("Sharpness Jitter",     self.dbl_sharp)
        form.addRow("Tail-Exp Jitter",      self.dbl_tail)
        outer.addLayout(form)

        self.baseline_panel = None
        if include_baseline:
            bcfg = dict(baseline_defaults or {})
            bcfg.update(cfg.get("baseline", {}) or {})
            self.baseline_panel = BaselineSubPanel("Baseline", bcfg, self)
            outer.addWidget(self.baseline_panel)

        self.spn_max_boost = None
        if include_max_boost:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Max boost (peak cap):"))
            self.spn_max_boost = QtWidgets.QDoubleSpinBox()
            self.spn_max_boost.setRange(0.0, 1.0); self.spn_max_boost.setSingleStep(0.01)
            self.spn_max_boost.setValue(float(cfg.get("max_boost", max_boost_default)))
            row.addWidget(self.spn_max_boost); row.addStretch(1)
            outer.addLayout(row)

    def result_cfg(self) -> Dict:
        out = {
            "enabled":            self.isChecked(),
            "peak_pos_jitter":    float(self.dbl_pos.value()),
            "amp_jitter":         float(self.dbl_amp.value()),
            "sharpness_jitter":   float(self.dbl_sharp.value()),
            "tail_exp_jitter":    float(self.dbl_tail.value()),
        }
        if self.baseline_panel is not None:
            out["baseline"] = self.baseline_panel.result_cfg()
        if self.spn_max_boost is not None:
            out["max_boost"] = float(self.spn_max_boost.value())
        return out

# --- CCcurvePanel: liest/schreibt "curve.seed" und "curve.random" ---
class CCcurvePanel(QtWidgets.QGroupBox):
    """Kurven-Panel (Notes/Phrases/Random) inkl. globalem Seed."""
    def __init__(self, title: str, cfg: Dict | None, parent=None,
                 baseline_defaults=None, include_max_boost=True, max_boost_default=0.35):
        super().__init__(title, parent)
        cfg = cfg or {}
        lay = QtWidgets.QVBoxLayout(self)

        # --- GLOBAL SEED ---
        row_seed = QtWidgets.QHBoxLayout()
        row_seed.addWidget(QtWidgets.QLabel("Seed (global):"))
        self.spn_seed_global = QtWidgets.QSpinBox()
        self.spn_seed_global.setRange(0, 2_000_000_000)
        # Seed liegt im vereinheitlichten Schema in cfg["seed"]
        self.spn_seed_global.setValue(int(cfg.get("seed", 1337)))
        row_seed.addWidget(self.spn_seed_global); row_seed.addStretch(1)
        lay.addLayout(row_seed)

        # --- NOTES ---
        grp_notes = QtWidgets.QGroupBox("Notes", self)
        ln = QtWidgets.QVBoxLayout(grp_notes)
        notes_h_cfg = (cfg.get("notes") or {}).get("humanize", {}) or {}
        self.pnl_notes = HumanizeSubPanel(
            "Humanize (Notes)", notes_h_cfg, self,
            include_baseline=True,
            baseline_defaults=(baseline_defaults or {
                "start": 0.8, "end": 0.8, "start_jitter": 0.0, "end_jitter": 0.0
            }),
            include_max_boost=False
        )
        ln.addWidget(self.pnl_notes)
        lay.addWidget(grp_notes)

        # --- PHRASES ---
        grp_phr = QtWidgets.QGroupBox("Phrases", self)
        lp = QtWidgets.QVBoxLayout(grp_phr)
        phr_h_cfg = (cfg.get("phrases") or {}).get("humanize", {}) or {}
        self.pnl_phr = HumanizeSubPanel(
            "Humanize (Phrases)", phr_h_cfg, self,
            include_baseline=True,
            baseline_defaults=(baseline_defaults or {
                "start": 0.0, "end": 0.0, "start_jitter": 0.0, "end_jitter": 0.0
            }),
            include_max_boost=include_max_boost,
            max_boost_default=float(phr_h_cfg.get("max_boost", max_boost_default))
        )
        lp.addWidget(self.pnl_phr)
        lay.addWidget(grp_phr)

        # --- RANDOM (vereinheitlicht auf "random") ---
        rn_cfg = (cfg.get("random") or {})  # <— statt random_noise
        grp_rn = QtWidgets.QGroupBox("Random Noise (R)", self)
        grp_rn.setCheckable(True)
        grp_rn.setChecked(bool(rn_cfg.get("enabled", True)))
        lr = QtWidgets.QFormLayout(grp_rn)

        self.dbl_rn_x = QtWidgets.QDoubleSpinBox()
        self.dbl_rn_x.setRange(0.0, 5.0); self.dbl_rn_x.setSingleStep(0.05); self.dbl_rn_x.setDecimals(3)
        self.dbl_rn_x.setValue(float(rn_cfg.get("x_scale", 0.4)))

        self.dbl_rn_y = QtWidgets.QDoubleSpinBox()
        self.dbl_rn_y.setRange(0.0, 0.5); self.dbl_rn_y.setSingleStep(0.01); self.dbl_rn_y.setDecimals(3)
        self.dbl_rn_y.setValue(float(rn_cfg.get("y_depth", 0.08)))

        lr.addRow("x-scale (smoothness):", self.dbl_rn_x)
        lr.addRow("y-depth (± amplitude):", self.dbl_rn_y)
        lay.addWidget(grp_rn)
        self.grp_rn = grp_rn

    def result_cfg(self) -> Dict:
        return {
            "seed": int(self.spn_seed_global.value()),
            "notes":   { "humanize": self.pnl_notes.result_cfg() },
            "phrases": { "humanize": self.pnl_phr.result_cfg() },
            "random": {
                "enabled": self.grp_rn.isChecked(),
                "x_scale": float(self.dbl_rn_x.value()),
                "y_depth": float(self.dbl_rn_y.value()),
            },
        }
    
class CCSection(QtWidgets.QGroupBox):
    """CC-Ausgabe-Bereich mit Range-Mapping und Curve-Panel."""
    def __init__(self, title: str, cfg: Dict | None, parent=None,
                 include_max_boost=True, max_boost_default=0.35):
        super().__init__(title, parent)
        cfg = cfg or {}
        self.setCheckable(True)
        self.setChecked(bool(cfg.get("enabled", True)))
        root = QtWidgets.QVBoxLayout(self)

        # Output range
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("Output range:"))
        self.spn_out_min = QtWidgets.QSpinBox(); self.spn_out_min.setRange(0,127)
        self.spn_out_max = QtWidgets.QSpinBox(); self.spn_out_max.setRange(0,127)
        out_min = int(cfg.get("out_min", 0 if "CC1" in title else 32))
        out_max = int(cfg.get("out_max", 127))
        if out_max < out_min: out_min, out_max = out_max, out_min
        self.spn_out_min.setValue(out_min)
        self.spn_out_max.setValue(out_max)
        range_row.addWidget(self.spn_out_min)
        range_row.addWidget(QtWidgets.QLabel("…"))
        range_row.addWidget(self.spn_out_max)
        range_row.addStretch(1)
        root.addLayout(range_row)

        # Curve panel
        self.curve_panel = CCcurvePanel("Curve", cfg.get("curve", {}), self,
                                        include_max_boost=include_max_boost,
                                        max_boost_default=max_boost_default)
        root.addWidget(self.curve_panel)

    def result_cfg(self) -> Dict:
        lo = int(self.spn_out_min.value()); hi = int(self.spn_out_max.value())
        if hi < lo: lo, hi = hi, lo
        return {
            "enabled": self.isChecked(),
            "out_min": lo,
            "out_max": hi,
            "curve": self.curve_panel.result_cfg(),
        }

# ---------------- Dialog ----------------

class SettingsDialog(QtWidgets.QDialog):
    """Hauptdialog: Velocities + CC1 + CC11, Reset/Import/Export, user-overrides speichern."""
    def __init__(self, parent=None, cfg_dict: Dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._cfg = cfg_dict or {}

        root = QtWidgets.QVBoxLayout(self)

        # Scrollbarer Inhalt
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        content = QtWidgets.QWidget(scroll)
        scroll.setWidget(content)
        content_lay = QtWidgets.QVBoxLayout(content)
        content_lay.setSpacing(12)

        # Drei Sektionen: links (Velocities + Timing) | Mitte (CC1) | rechts (CC11)
        row = QtWidgets.QHBoxLayout(); row.setSpacing(12)

        vel_cfg   = self._cfg.get("velocities", self._cfg.get("dynamics", {})) or {}
        cc1_cfg   = self._cfg.get("cc1",  {}) or {}
        cc11_cfg  = self._cfg.get("cc11", {}) or {}
        tim_cfg   = self._cfg.get("timing", {}) or {}

        # linke Spalte: Velocities + Timing + Phrasing (untereinander)
        phr_cfg = self._cfg.get("phrasing", {}) or {}
        left_col = QtWidgets.QWidget(); left_v = QtWidgets.QVBoxLayout(left_col); left_v.setSpacing(12)
        self.sec_vel    = VelocitySection("Velocities", vel_cfg, self)
        self.sec_timing = TimingSection("Humanize (Timing)", tim_cfg, self)
        self.sec_phras  = PhrasingSection("Phrasing", phr_cfg, self)   # NEU
        left_v.addWidget(self.sec_vel)
        left_v.addWidget(self.sec_timing)
        left_v.addWidget(self.sec_phras)  # NEU
        left_v.addStretch(1)

        # mittlere/rechte Spalte: CC1 / CC11
        self.sec_cc1  = CCSection("CC1 (Modwheel)",    cc1_cfg,  self,
                                include_max_boost=True, max_boost_default=0.35)
        self.sec_cc11 = CCSection("CC11 (Expression)", cc11_cfg, self,
                                include_max_boost=True, max_boost_default=0.35)

        row.addWidget(left_col, 1)
        row.addWidget(self.sec_cc1, 1)
        row.addWidget(self.sec_cc11, 1)
        content_lay.addLayout(row)
        root.addWidget(scroll, 1)

        # Buttons (Ok/Cancel + Reset/Import/Export)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btn_reset  = btns.addButton("Reset",   QtWidgets.QDialogButtonBox.ResetRole)
        self.btn_import = btns.addButton("Import…", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_export = btns.addButton("Export…", QtWidgets.QDialogButtonBox.ActionRole)

        btns.accepted.connect(self.on_ok)
        btns.rejected.connect(self.reject)
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        self.btn_import.clicked.connect(self.on_import_clicked)
        self.btn_export.clicked.connect(self.on_export_clicked)

        root.addWidget(btns)

        self.resize(1200, 650)
        self.setSizeGripEnabled(True)

    # ---- UI <-> cfg

    def _repopulate_from_cfg(self, cfg: Dict):
        """Bestehende Panels aus einem cfg-Dict neu setzen (keine Widgets erzeugen)."""
        # Velocities
        vel_cfg = cfg.get("velocities", cfg.get("dynamics", {})) or {}
        self.sec_vel.setChecked(bool(vel_cfg.get("enabled", True)))
        ba = vel_cfg.get("beat_accent", {}) or {}
        self.sec_vel.ba_panel.setChecked(bool(ba.get("enabled", True)))
        self.sec_vel.ba_panel.spn_min.setValue(int(ba.get("clamp_min", 1)))
        self.sec_vel.ba_panel.spn_max.setValue(int(ba.get("clamp_max", 127)))
        rules = ba.get("rules", {}) or {}
        for key, _label in self.sec_vel.ba_panel._rule_keys:
            self.sec_vel.ba_panel.mul[key].setValue(float(rules.get(key, {}).get("mul", 1.0)))
            self.sec_vel.ba_panel.add[key].setValue(int(rules.get(key, {}).get("add", 0)))

        # Velocity Humanize
        vh = (vel_cfg.get("humanize") or {}) or {}
        self.sec_vel.vh_panel.setChecked(bool(vh.get("enabled", True)))
        self.sec_vel.vh_panel.spn_seed.setValue(int(vh.get("seed", 1337)))
        self.sec_vel.vh_panel.dbl_mul.setValue(float(vh.get("mul_jitter", 0.10)))
        self.sec_vel.vh_panel.spn_add.setValue(int(vh.get("add_jitter", 3)))
        
        sav = (vel_cfg.get("slur_advance_velocity") or {}) or {}
        self.sec_vel.sav_panel.setChecked(bool(sav.get("enabled", True)))
        self.sec_vel.sav_panel.spn_full_ms.setValue(int(sav.get("fullscale_ms", 160)))

        self.sec_vel.sav_panel.spn_min_vel.setValue(int(sav.get("min_vel", 31)))
        self.sec_vel.sav_panel.spn_max_vel.setValue(int(sav.get("max_vel", 105)))
        self.sec_vel.sav_panel.dbl_gamma.setValue(float(sav.get("gamma", 1.0)))
        self.sec_vel.sav_panel.dbl_mix.setValue(float(sav.get("mix_old", 0.15)))
        self.sec_vel.sav_panel.chk_first.setChecked(bool(sav.get("include_first_note", False)))

        # Timing (Humanize)
        tim = (cfg.get("timing") or {}) or {}
        self.sec_timing.setChecked(bool(tim.get("enabled", True)))
        self.sec_timing.spn_seed.setValue(int(tim.get("seed", 2024)))
        self.sec_timing.dbl_onset_frac.setValue(float(tim.get("onset_jitter_frac_of_len", 0.25)))
        self.sec_timing.dbl_dur_frac.setValue(float(tim.get("dur_jitter_frac_of_len", 0.15)))

        # Phrasing
        phr = (cfg.get("phrasing") or {})
        sl  = (phr.get("slur_legato") or {})
        es  = (sl.get("early_start") or {})

        self.sec_phras.setChecked(bool(sl.get("enabled", True)))
        self.sec_phras.grp_leg.setChecked(bool(sl.get("enabled", True)))
        self.sec_phras.spn_overlap.setValue(int(sl.get("overlap_ticks", 30)))

        self.sec_phras.spn_early_ms.setValue(int(es.get("ms", 0)))
        self.sec_phras.spn_max_adv_ms.setValue(int(es.get("max_advance_ms", sl.get("max_advance_ms", 120))))
        self.sec_phras.spn_longest_ms.setValue(int(es.get("longest_note_ms", sl.get("longest_note_ms", 800))))

        self.sec_phras.dbl_max_adv.setValue(float(sl.get("max_advance_frac_of_prev_len", 0.0)))
        self.sec_phras.spn_non_slur_gap.setValue(int(sl.get("non_slur_min_gap_ticks", 10)))
        self.sec_phras.spn_tenuto_gap.setValue(int(sl.get("tenuto_min_gap_ticks", 5)))

        # CC1 / CC11
        for sec, name in ((self.sec_cc1, "cc1"), (self.sec_cc11, "cc11")):
            c = cfg.get(name, {}) or {}
            sec.setChecked(bool(c.get("enabled", True)))
            lo = int(c.get("out_min", 0 if name == "cc1" else 32))
            hi = int(c.get("out_max", 127))
            if hi < lo: lo, hi = hi, lo
            sec.spn_out_min.setValue(lo)
            sec.spn_out_max.setValue(hi)

            cv = c.get("curve", {}) or {}
            seed_val = cv.get("seed", c.get("seed", 1337))
            sec.curve_panel.spn_seed_global.setValue(int(seed_val))

            nh = (cv.get("notes") or {}).get("humanize", {}) or {}
            pnl = sec.curve_panel.pnl_notes
            pnl.setChecked(bool(nh.get("enabled", True)))
            pnl.dbl_pos.setValue(float(nh.get("peak_pos_jitter", 0.05)))
            pnl.dbl_amp.setValue(float(nh.get("amp_jitter", 0.10)))
            pnl.dbl_sharp.setValue(float(nh.get("sharpness_jitter", 0.20)))
            pnl.dbl_tail.setValue(float(nh.get("tail_exp_jitter", 0.25)))
            if pnl.baseline_panel:
                b = nh.get("baseline", {}) or {}
                pnl.baseline_panel.sp_start.setValue(float(b.get("start", 0.8)))
                pnl.baseline_panel.sp_end.setValue(float(b.get("end", 0.8)))
                pnl.baseline_panel.sp_jst.setValue(float(b.get("start_jitter", 0.0)))
                pnl.baseline_panel.sp_jen.setValue(float(b.get("end_jitter", 0.0)))

            ph = (cv.get("phrases") or {}).get("humanize", {}) or {}
            ppl = sec.curve_panel.pnl_phr
            ppl.setChecked(bool(ph.get("enabled", True)))
            ppl.dbl_pos.setValue(float(ph.get("peak_pos_jitter", 0.05)))
            ppl.dbl_amp.setValue(float(ph.get("amp_jitter", 0.10)))
            ppl.dbl_sharp.setValue(float(ph.get("sharpness_jitter", 0.20)))
            ppl.dbl_tail.setValue(float(ph.get("tail_exp_jitter", 0.25)))
            if ppl.baseline_panel:
                b = ph.get("baseline", {}) or {}
                ppl.baseline_panel.sp_start.setValue(float(b.get("start", 0.0)))
                ppl.baseline_panel.sp_end.setValue(float(b.get("end", 0.0)))
                ppl.baseline_panel.sp_jst.setValue(float(b.get("start_jitter", 0.0)))
                ppl.baseline_panel.sp_jen.setValue(float(b.get("end_jitter", 0.0)))
            if ppl.spn_max_boost is not None:
                ppl.spn_max_boost.setValue(float(ph.get("max_boost", 0.35)))

            rn = (cv.get("random") or {}) 
            sec.curve_panel.grp_rn.setChecked(bool(rn.get("enabled", True)))
            sec.curve_panel.dbl_rn_x.setValue(float(rn.get("x_scale", 0.4)))
            sec.curve_panel.dbl_rn_y.setValue(float(rn.get("y_depth", 0.08)))

    def _save_overrides(self, merged: dict):
        defaults = load_defaults()
        overrides = dict_diff(defaults, merged)
        save_user_config(overrides)

    # ---- Slots

    def on_ok(self):
        # Die im Dialog sichtbaren (merged) Einstellungen in user.json als Overrides speichern
        try:
            merged = self.result_cfg()
            self._save_overrides(merged)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", f"Could not save user settings:\n{e}")
            return
        self.accept()

    def on_reset_clicked(self):
        ret = QtWidgets.QMessageBox.question(
            self, "Reset settings",
            "Alle Werte auf Werkseinstellungen im Dialog zurücksetzen?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return

        # Werkseinstellungen aus laden → NUR UI
        self._cfg = load_defaults()
        self._repopulate_from_cfg(self._cfg)
        QtWidgets.QMessageBox.information(self, "Reset", "Werkseinstellungen im Dialog aktiv. Mit „OK“ übernehmen.")

    def on_export_clicked(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export settings", str(APP_DIR / "settings.export.json"),
            "JSON (*.json);;YAML (*.yaml *.yml);;All files (*)"
        )
        if not path:
            return

        data = self.result_cfg()  # merged Sicht aus dem UI
        try:
            if path.lower().endswith((".yaml", ".yml")):
                Path(path).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            else:
                Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", f"Could not export settings:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Export", "Einstellungen exportiert.")

    def on_import_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import settings", str(APP_DIR),
            "JSON (*.json);;YAML (*.yaml *.yml);;All files (*)"
        )
        if not path:
            return

        try:
            txt = Path(path).read_text(encoding="utf-8")
            imported = yaml.safe_load(txt) if path.lower().endswith((".yaml", ".yml")) else json.loads(txt)
            imported = migrate_cfg(imported or {})
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import error", f"Could not read settings:\n{e}")
            return

        # NUR UI aktualisieren – KEIN Speichern hier
        self._cfg = imported
        self._repopulate_from_cfg(self._cfg)
        QtWidgets.QMessageBox.information(self, "Import", "Einstellungen geladen. Mit „OK“ übernehmen.")

    # ---- Public API

    def apply_external_cfg(self, merged_cfg: dict):
        self._cfg = dict(merged_cfg or {})
        self._repopulate_from_cfg(self._cfg)
        
    def result_cfg(self) -> Dict:
        """
        Liefert die *merged Sicht* der Einstellungen aus dem Dialog.
        (User-Overrides werden außerhalb via _save_overrides berechnet.)
        """
        out = dict(self._cfg)
        out["velocities"] = self.sec_vel.result_cfg()
        out["dynamics"]   = dict(out["velocities"])  # falls Altpfad noch gebraucht wird
        out["beat_accent"]= dict(out["velocities"].get("beat_accent", {}))
        out["cc1"]    = self.sec_cc1.result_cfg()
        out["cc11"]   = self.sec_cc11.result_cfg()
        out["timing"] = self.sec_timing.result_cfg()
        out["phrasing"] = self.sec_phras.result_cfg()  # enthält deinen slur_legato-Block inkl. ms-Felder
        return out