from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

DEFAULT_TPB = 960
DEFAULT_BPM = 120.0

# --- Pass 1: raw analysis objects ---

@dataclass
class NoteToken:
    start_div: int
    duration_div: int
    midi: Optional[int]    # None for rests
    voice: Optional[str]
    staff: Optional[str]
    tie_start: bool = False
    tie_stop: bool = False
    slur_starts: int = 0
    slur_stops: int = 0
    articulations: List[str] = field(default_factory=list)
    ornaments: List[str] = field(default_factory=list)
    measure_idx: Optional[int] = None

@dataclass
class DirectionEvent:
    time_div: int
    kind: str              # "tempo" | "timesig" | "dynamic" | "wedge"
    payload: dict

@dataclass
class PartAnalysis:
    part_id: str
    part_name: str
    divisions_changes: List[Tuple[int,int]] = field(default_factory=list)  # (time_div, divisions)
    notes: List[NoteToken] = field(default_factory=list)
    directions: List[DirectionEvent] = field(default_factory=list)

@dataclass
class ScoreAnalysis:
    parts: Dict[str, PartAnalysis] = field(default_factory=dict)

# --- Pass 2: processed timeline ---

@dataclass
class NoteEvent:
    start_tick: int
    end_tick: int
    midi: int
    velocity: int
    channel: int
    attrs: Dict[str, bool] = field(default_factory=dict)

@dataclass
class CCEvent:
    tick: int
    cc: int
    value: int

@dataclass
class TrackTimeline:
    name: str
    channel: int
    notes: List[NoteEvent] = field(default_factory=list)
    ccs: List[CCEvent] = field(default_factory=list)

@dataclass
class ConductorTimeline:
    tempos: List[Tuple[int,float]] = field(default_factory=list)           # (tick, bpm)
    timesigs: List[Tuple[int,int,int]] = field(default_factory=list)       # (tick, num, den)

@dataclass
class TimelineBundle:
    conductor: ConductorTimeline
    tracks: Dict[str, TrackTimeline]
    ticks_per_beat: int = DEFAULT_TPB
