from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class BeatRule:
    mul: float
    add: float

@dataclass
class BeatAccentConfig:
    enabled: bool
    clamp_min: int
    clamp_max: int
    tol_onbeat: float
    tol_eighth: float
    tol_sixteenth: float
    rules: Dict[str, BeatRule]  # keys siehe unten

@dataclass
class DynamicsConfig:
    mode: str                 # "per_mark" | "linear"
    linear_min: int
    linear_max: int
    per_mark: Dict[str, int]
    fallback: int

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def dynamics_to_base_velocity(mark: Optional[str], cfg: DynamicsConfig) -> int:
    if cfg.mode == "per_mark":
        if mark:
            key = mark.lower()
            if key in cfg.per_mark:
                return int(_clamp(cfg.per_mark[key], 1, 127))
        return int(_clamp(cfg.fallback, 1, 127))
    # linear fallback (z.B. Mapping von -6..+6 auf 0..127 wäre später möglich)
    return int(_clamp(cfg.linear_max, 1, 127))  # simplest default

def categorize_beat(phase_in_beats: float,
                    beat_index_in_bar: int,
                    beats_per_bar: int,
                    cfg: BeatAccentConfig) -> str:
    """
    phase_in_beats: Abstand vom Taktanfang in Beat-Einheiten (0.0 = downbeat, 1.0 = Beat 2, …)
    beat_index_in_bar: 0-basierter ganzzahliger Beatindex (int(phase))
    beats_per_bar: z.B. 4 in 4/4, 3 in 3/4, 2 in 2/2, etc.
    """
    import math
    frac = phase_in_beats - math.floor(phase_in_beats)  # 0.. <1
    # bar downbeat?
    if abs(phase_in_beats - round(phase_in_beats)) <= cfg.tol_onbeat and beat_index_in_bar == 0:
        return "bar_downbeat"
    # midbar strong (nur für gerade Takte sinnvoll)
    if beats_per_bar % 2 == 0 and abs(phase_in_beats - round(phase_in_beats)) <= cfg.tol_onbeat:
        if beat_index_in_bar == (beats_per_bar // 2):
            return "midbar_strong"
    # andere Onbeats (intakte Onbeat)
    if abs(phase_in_beats - round(phase_in_beats)) <= cfg.tol_onbeat:
        return "onbeats_other"
    # Achtel-Offbeat (x.5)
    if abs((frac) - 0.5) <= cfg.tol_eighth:
        return "eighth_offbeats"
    # Sechzehntel-Offbeats (x.25 / x.75)
    if abs(frac - 0.25) <= cfg.tol_sixteenth or abs(frac - 0.75) <= cfg.tol_sixteenth:
        return "sixteenth_offs"
    return "fallback"

def apply_accent(v_in: int, cat: str, cfg: BeatAccentConfig) -> int:
    rule = cfg.rules.get(cat, cfg.rules["fallback"])
    v = v_in * rule.mul + rule.add
    return _clamp(int(round(v)), cfg.clamp_min, cfg.clamp_max)

def shape_velocity(base_velocity: int,
                   phase_in_beats: float,
                   beat_index_in_bar: int,
                   beats_per_bar: int,
                   accent_cfg: BeatAccentConfig) -> int:
    if not accent_cfg.enabled:
        return _clamp(base_velocity, accent_cfg.clamp_min, accent_cfg.clamp_max)
    cat = categorize_beat(phase_in_beats, beat_index_in_bar, beats_per_bar, accent_cfg)
    return apply_accent(base_velocity, cat, accent_cfg)