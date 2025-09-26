# gui/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class Note:
    pitch: int
    start: float  # seconds
    end: float    # seconds
    velocity: int = 80
    articulations: List[str] = field(default_factory=list)
    dynamic_mark: Optional[str] = None

@dataclass
class MidiSong:
    name: str
    notes: List[Note]
    length: float           # seconds
    beats: List[float]      # beat times in seconds
    bars: List[float]       # bar starts in seconds
    meta: Dict[str, str] = field(default_factory=dict)