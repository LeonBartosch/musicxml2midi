from __future__ import annotations

# Bootstrap
import os, sys
import math
import numpy as np
import hashlib

try:
    import mido
except ImportError:
    mido = None

PROJ_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
SRC_DIR = os.path.join(PROJ_ROOT, "src")

# Erst SRC_DIR nach vorne, PROJ_ROOT maximal ans Ende
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from typing import List, Tuple, Optional, Dict, DefaultDict
from collections import defaultdict
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from .utils import GestureViewBox

from musicxml2midi.analyze import analyze_musicxml
from musicxml2midi.process import build_timelines
from musicxml2midi.timeline import TimelineBundle, TrackTimeline
from musicxml2midi.interpretation import (
    build_L_curve,
    build_phrase_curve,
    build_note_curve,
    build_articulation_curve,
    build_pitch_curve,
    build_random_noise_curve,
    compose_total_curve,
)
from musicxml2midi.gui.models import MidiSong, Note
from musicxml2midi.gui.pianoroll import PianoRoll
from musicxml2midi.gui.pianoroll_post import PostPanel
from musicxml2midi.gui.settings import SettingsDialog
from musicxml2midi.gui.settings import load_config as load_gui_config

# ---------- helpers ----------
# Hilfsfunktionen ans Ende des Files (noch vor MainWindow-Klasse oder als @staticmethods)

def _mk_seconds_to_ticks(bundle: TimelineBundle):
    return seconds_to_ticks_map(bundle.conductor.tempos, bundle.ticks_per_beat)

def _events_from_song(song: MidiSong, bundle: TimelineBundle, channel: int = 0):
    """Baue absolute Tick-Events [(tick, priority, mido.Message/meta)].
    priority: kleinere Zahl = früher; sorgt für sinnvolle Reihenfolge bei gleiche tick.
    """
    s2t = _mk_seconds_to_ticks(bundle)
    evts = []

    # 1) Notes
    for n in (song.notes or []):
        st = int(s2t(float(n.start)))
        en = int(s2t(float(n.end)))
        vel = int(max(1, min(127, int(n.velocity))))
        pitch = int(n.pitch)
        evts.append((st, 20, mido.Message('note_on',  note=pitch, velocity=vel, channel=channel)))
        evts.append((max(en, st+1), 90, mido.Message('note_off', note=pitch, velocity=0,   channel=channel)))

    # 2) CCs aus song.meta["cc"] (Keys sind CC-Nummern, Werte [(sec, val)])
    cc_map = (song.meta or {}).get("cc") or {}
    for cc_num, pts in cc_map.items():
        for t_sec, val in pts:
            tick = int(s2t(float(t_sec)))
            evts.append((tick, 40, mido.Message('control_change', control=int(cc_num), value=int(val), channel=channel)))

    return evts

def _conductor_events(bundle: TimelineBundle):
    """Tempo- und Taktwechsel als absolute Tick-Events."""
    tpb = int(bundle.ticks_per_beat)
    evts = []
    # Tempi: [(tick, bpm)]
    for tick, bpm in sorted(bundle.conductor.tempos, key=lambda x: x[0]):
        tempo = mido.bpm2tempo(float(bpm))  # microsec per beat
        evts.append((int(tick), 0, mido.MetaMessage('set_tempo', tempo=int(tempo))))
    # Time signatures: [(tick, num, den)]
    for tick, num, den in sorted(bundle.conductor.timesigs, key=lambda x: x[0]):
        evts.append((int(tick), 0, mido.MetaMessage('time_signature', numerator=int(num), denominator=int(den))))
    return evts

def _build_midifile_for_song(song: MidiSong, bundle: TimelineBundle, add_conductor_meta: bool = False, channel: int = 0) -> 'mido.MidiFile':
    mf = mido.MidiFile(ticks_per_beat=int(bundle.ticks_per_beat))
    tr = mido.MidiTrack(); mf.tracks.append(tr)

    evts = []
    if add_conductor_meta:
        evts += _conductor_events(bundle)
    evts += _events_from_song(song, bundle, channel=channel)

    # sortieren: (tick, priority, msg)
    evts.sort(key=lambda x: (x[0], x[1]))
    # in Delta konvertieren
    last = 0
    for tick, _prio, msg in evts:
        delta = int(max(0, tick - last))
        msg.time = delta
        tr.append(msg)
        last = tick

    return mf

def _build_midifile_for_conductor_only(bundle: TimelineBundle) -> 'mido.MidiFile':
    mf = mido.MidiFile(ticks_per_beat=int(bundle.ticks_per_beat))
    tr = mido.MidiTrack(); mf.tracks.append(tr)
    evts = _conductor_events(bundle)
    evts.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for tick, _prio, msg in evts:
        delta = int(max(0, tick - last))
        msg.time = delta
        tr.append(msg)
        last = tick
    return mf

def _sanitize_basename(name: str, fallback: str = "export") -> str:
    # Extension weg, Leerraum trimmen
    base = os.path.splitext(name or "")[0].strip()
    # nur sichere Zeichen erlauben
    base = "".join(c if (c.isalnum() or c in " _-.") else "_" for c in base)
    # unschöne Ränder entfernen
    base = base.strip().strip("._-")
    return base or fallback

def _segment_mean(ticks: np.ndarray, L: np.ndarray, t0: int, t1: int) -> float:
    i0 = int(np.searchsorted(ticks, t0, side="left"))
    i1 = int(np.searchsorted(ticks, t1, side="right"))
    if i1 <= i0:
        # Fallback: Sample bei t0
        i0 = int(np.searchsorted(ticks, t0, side="left"))
        i0 = max(0, min(i0, len(L)-1))
        return float(np.clip(L[i0], 0.0, 1.0))
    return float(np.clip(np.mean(L[i0:i1]), 0.0, 1.0))

def _subseed(base_seed: int, tag: str) -> int:
    h = hashlib.blake2b(f"{base_seed}|{tag}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFF  # int32 positiv

def _to01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _rand_unit_symmetric(seed: int) -> float:
    # Deterministic float in [-1, +1] from 32-bit seed
    # Use a small LCG or hash; here we hash and map to [0,1] then to [-1,1]
    h = hashlib.blake2b(seed.to_bytes(8, "big"), digest_size=8).digest()
    u = int.from_bytes(h, "big") / float(2**64 - 1)
    return (u * 2.0) - 1.0

def _map_cc_range_from01(c01: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo_i = int(max(0, min(127, lo)))
    hi_i = int(max(0, min(127, hi)))
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    y = lo_i + (hi_i - lo_i) * _to01(c01)
    return np.clip(np.rint(y), 0, 127).astype(int)

def _array_to_cc_points(xs_sec: List[float], vals_int: np.ndarray) -> List[Tuple[float,int]]:
    """Verdichte Sample-Array zu CC-Punkten (neuer Punkt nur, wenn sich der Wert ändert)."""
    out: List[Tuple[float,int]] = []
    last_v = None
    for t, v in zip(xs_sec, map(int, vals_int.tolist())):
        if last_v is None or v != last_v:
            out.append((float(t), v))
            last_v = v
    return out

def _u01_from(base_seed: int, tag: str) -> float:
    h = hashlib.blake2b(f"{base_seed}|{tag}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(2**64 - 1)  # in [0,1]

def _build_cc_components(analysis_song, ticks, L, tpb, cc_cfg: Dict):
    curve = (cc_cfg.get("curve") or {}) or {}

    # seed: zuerst curve.seed, alternativ top-level cc_cfg.seed, sonst 1337
    base_seed = int(curve.get("seed", cc_cfg.get("seed", 1337)))

    seed_notes   = _subseed(base_seed, "notes")
    seed_phrases = _subseed(base_seed, "phrases")
    seed_noise   = _subseed(base_seed, "random")

    notes_cfg   = (curve.get("notes")   or {})
    phrases_cfg = (curve.get("phrases") or {})
    rn_cfg      = (curve.get("random")  or {})

    N_arr = build_note_curve(
        analysis_song, ticks, L, tpb,
        humanize_cfg={
            "seed": seed_notes,
            "notes": {"enabled": True, **(notes_cfg.get("humanize") or {})},
        },
    )
    P_arr = build_phrase_curve(
        analysis_song, ticks, L, tpb,
        humanize_cfg={
            "seed": seed_phrases,
            "phrases": {"enabled": True, **(phrases_cfg.get("humanize") or {})},
        },
        depth=1.0,
    )
    A_arr = build_articulation_curve(analysis_song, ticks, tpb)
    H_arr = build_pitch_curve(analysis_song, ticks, tpb)

    if bool(rn_cfg.get("enabled", True)):
        r_xscale = float(rn_cfg.get("x_scale", 0.4))
        r_ydepth = float(rn_cfg.get("y_depth", 0.08))
        R_arr = build_random_noise_curve(np.array(ticks), seed_noise, r_xscale, r_ydepth, tpb)
    else:
        R_arr = np.ones_like(ticks, dtype=float)
    return L, P_arr, N_arr, A_arr, H_arr, R_arr

def ticks_to_seconds_map(tempos: List[Tuple[int, float]], tpb: int):
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

def seconds_to_ticks_map(tempos: List[Tuple[int, float]], tpb: int):
    tempos = sorted(tempos or [(0,120.0)], key=lambda x: x[0])
    # erst Tick->Sek-Segmente wie in ticks_to_seconds_map aufbauen
    segs = []
    last_tick = 0; last_sec = 0.0; last_bpm = tempos[0][1]
    for tick, bpm in tempos:
        if tick > last_tick:
            segs.append((last_tick, tick, last_sec, last_bpm))
            last_sec += ((tick - last_tick) / tpb) * (60.0 / last_bpm)
            last_tick = tick
        last_bpm = bpm
    segs.append((last_tick, 10**12, last_sec, last_bpm))

    # daraus eine Sek->Tick-Funktion
    def to_tick(s: float) -> int:
        # passendes Segment per Sekundenbasis finden
        for s0_tick, s1_tick, base_sec, bpm in segs:
            # Segmentdauer in Sek
            seg_sec = ((s1_tick - s0_tick) / tpb) * (60.0 / bpm)
            if s < base_sec + seg_sec:
                # rückrechnen
                dt_sec = s - base_sec
                ticks = s0_tick + int(round((dt_sec / (60.0 / bpm)) * tpb))
                return max(0, ticks)
        # falls dahinter: im letzten Segment extrapolieren
        s0_tick, _, base_sec, bpm = segs[-1]
        dt_sec = s - base_sec
        ticks = s0_tick + int(round((dt_sec / (60.0 / bpm)) * tpb))
        return max(0, ticks)
    return to_tick

def pair_wedges_to_spans(events: List[Tuple[int, str]]) -> List[Tuple[int,int,str]]:
    spans = []
    stack: List[Tuple[int,str]] = []
    for t, typ in sorted(events, key=lambda x: x[0]):
        if typ in ("crescendo", "diminuendo"):
            stack.append((t, typ))
        elif typ == "stop" and stack:
            st, k = stack.pop()
            if t > st:
                spans.append((st, t, k))
    return spans

def _group_by_voice_staff(events) -> DefaultDict[Tuple[str,str], List]:
    groups: DefaultDict[Tuple[str,str], List] = defaultdict(list)
    for ev in events:
        voice = str(ev.attrs.get("voice") or "")
        staff = str(ev.attrs.get("staff") or "")
        groups[(voice, staff)].append(ev)
    for k in groups:
        groups[k].sort(key=lambda e: (e.start_tick, e.end_tick, e.midi))
    return groups

# ---------- build "analysis" song ----------
def build_song_from_bundle(bundle: TimelineBundle, part: TrackTimeline,
                           dyn_events: List[Tuple[int, str]],
                           wedge_spans: List[Tuple[int,int,str]]) -> MidiSong:
    t2s = ticks_to_seconds_map(bundle.conductor.tempos, bundle.ticks_per_beat)

    # --- Noten + Slur onset-basiert, chord-robust ---
    notes: List[Note] = []
    max_tick = 0
    groups = _group_by_voice_staff(part.notes)

    # Aggregation für Artikulations-Lane (optional dedupliziert pro onset in sec)
    arts_by_tick: Dict[int, set] = {}

    for (_voice, _staff), evs in groups.items():
        onsets: Dict[int, List] = {}
        for ev in evs:
            onsets.setdefault(ev.start_tick, []).append(ev)
        onset_ticks = sorted(onsets.keys())

        slur_depth = 0
        have_prev = False

        for t in onset_ticks:
            curr_evs = onsets[t]
            curr_start_sum = sum(int(e.attrs.get("slur_start_n", 0)) for e in curr_evs)
            curr_stop_sum  = sum(int(e.attrs.get("slur_stop_n",  0)) for e in curr_evs)
            place_slur_here = (have_prev and slur_depth > 0)

            onset_tags = set()
            for ev in curr_evs:
                onset_tags.update(ev.attrs.get("tags", []))
            if place_slur_here:
                onset_tags.add("slur")
            if onset_tags:
                arts_by_tick.setdefault(t, set()).update(onset_tags)

            for ev in curr_evs:
                s_tick, e_tick = ev.start_tick, ev.end_tick
                s = t2s(s_tick); e = t2s(e_tick)
                max_tick = max(max_tick, e_tick)
                arts = list(ev.attrs.get("tags", []))
                if place_slur_here: arts.append("slur")
                # dedupe
                seen = set(); arts = [a for a in arts if not (a in seen or seen.add(a))]
                nobj = Note(pitch=ev.midi, start=s, end=e, velocity=ev.velocity, articulations=arts)
                setattr(nobj, "start_tick", s_tick)   # dynamisch anhängen (sofern Note keine __slots__ hat)
                setattr(nobj, "end_tick",   e_tick)
                setattr(nobj, "_vkey", (str(ev.attrs.get("voice") or ""), str(ev.attrs.get("staff") or "")))
                setattr(nobj, "orig_onset_tick", s_tick)
                notes.append(nobj)

            have_prev = True
            slur_depth += curr_start_sum
            slur_depth = max(0, slur_depth - curr_stop_sum)

    length = t2s(max_tick) if notes else 0.0
    if wedge_spans:
        length = max(length, t2s(max(b for _, b, _ in wedge_spans)))
    if dyn_events:
        length = max(length, t2s(max(t for t, _ in dyn_events)))
    length = max(0.0, length) + 0.5

    # Grid
    beats_sec, bars_sec = [], []
    ts = sorted(bundle.conductor.timesigs, key=lambda x: x[0]) or [(0, 4, 4)]
    idx = 0; cur_tick = 0
    while True:
        num, den = ts[idx][1], ts[idx][2]
        blen = int(round(bundle.ticks_per_beat * (4.0 / den)))
        barlen = blen * num
        bars_sec.append(t2s(cur_tick))
        for b in range(num):
            beats_sec.append(t2s(cur_tick + b * blen))
        cur_tick += barlen
        if idx + 1 < len(ts) and cur_tick >= ts[idx + 1][0]:
            idx += 1
        if t2s(cur_tick) > (length + 2.0):
            break

    dyn_times   = [(t2s(t), m) for (t, m) in sorted(dyn_events, key=lambda x: x[0])]
    wedge_times = [(t2s(a), t2s(b), k) for (a,b,k) in wedge_spans]
    ts_changes_sec: List[Tuple[float,int,int]] = [(t2s(t), n, d) for (t, n, d) in bundle.conductor.timesigs]

    arts_time_list = sorted(((t2s(tick), sorted(list(tags))) for tick, tags in arts_by_tick.items()),
                            key=lambda x: x[0])

    return MidiSong(
        name=part.name,
        notes=notes,
        length=length,
        beats=beats_sec,
        bars=bars_sec,
        meta={
            "dynamics": dyn_times,
            "wedges": wedge_times,
            "timesig_changes": ts_changes_sec,
            "arts_by_time": arts_time_list,
            # für PostPanel, falls du später Tick-genau brauchst:
            "tpb": bundle.ticks_per_beat,
            "tempos": bundle.conductor.tempos,
        }
    )

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def _note_phase_in_bar(tick: int, bar_start: int, tpb: int):
    rel = tick - bar_start
    phase = rel / float(tpb)
    return phase, int(math.floor(phase + 1e-9))

def _categorize_beat(phase: float, beat_idx: int, beats_per_bar: int, ba_cfg: dict) -> str:
    tol_on = float(ba_cfg.get("tol_onbeat", 0.08))
    tol_e8 = float(ba_cfg.get("tol_eighth", 0.08))
    tol_s16 = float(ba_cfg.get("tol_sixteenth", 0.06))
    frac = phase - math.floor(phase)
    if abs(phase - round(phase)) <= tol_on and beat_idx == 0:
        return "bar_downbeat"
    if beats_per_bar % 2 == 0 and abs(phase - round(phase)) <= tol_on:
        if beat_idx == (beats_per_bar // 2):
            return "midbar_strong"
    if abs(phase - round(phase)) <= tol_on:
        return "onbeats_other"
    if abs(frac - 0.5) <= tol_e8:
        return "eighth_offbeats"
    if abs(frac - 0.25) <= tol_s16 or abs(frac - 0.75) <= tol_s16:
        return "sixteenth_offs"
    return "fallback"

def _apply_accent(v_in: int, cat: str, ba_cfg: dict) -> int:
    rules = ba_cfg.get("rules", {})
    r = rules.get(cat, rules.get("fallback", {"mul":1.0,"add":0}))
    v = int(round(v_in * float(r.get("mul",1.0)) + int(r.get("add",0))))
    return _clamp(v, int(ba_cfg.get("clamp_min",1)), int(ba_cfg.get("clamp_max",127)))

def _humanize_timing(song: MidiSong, bundle: TimelineBundle, timing_cfg: Dict) -> MidiSong:
    if not (timing_cfg or {}).get("enabled", True):
        return song

    seed = int(timing_cfg.get("seed", 2024))

    # NEU: relative Parameter (Bruchteile der Notenlänge)
    onset_frac  = float(timing_cfg.get("onset_jitter_frac_of_len", 0.25))  # z.B. ±25% der Notenlänge
    length_frac = float(timing_cfg.get("dur_jitter_frac_of_len",   0.15))  # z.B. ±15% der Notenlänge

    # Optional: kleine Guards an den Onset-Fensterkanten
    guard_prev = int(timing_cfg.get("onset_guard_prev_ticks", 0))
    guard_next = int(timing_cfg.get("onset_guard_next_ticks", 0))

    tpb   = bundle.ticks_per_beat
    tempos = bundle.conductor.tempos
    t2s = ticks_to_seconds_map(tempos, tpb)
    s2t = seconds_to_ticks_map(tempos, tpb)

    # Fenster (pro Stimme nach orig_onset sortiert)
    by_vkey: DefaultDict[tuple, List[tuple]] = defaultdict(list)
    for idx, n in enumerate(song.notes):
        st_tick = int(getattr(n, "start_tick", s2t(n.start)))
        en_tick = int(getattr(n, "end_tick",   s2t(n.end)))
        vkey    = getattr(n, "_vkey", ("", ""))
        orig    = int(getattr(n, "orig_onset_tick", st_tick))
        by_vkey[vkey].append((idx, n, st_tick, en_tick, orig))

    windows: Dict[int, Tuple[int, int]] = {}
    for _vkey, arr in by_vkey.items():
        arr.sort(key=lambda x: x[4])  # nach orig_onset_tick
        for k, (idx, _n, st, en, _orig) in enumerate(arr):
            prev_orig = arr[k-1][2] if k > 0 else None  # prev *start_tick* (stabil)
            next_orig = arr[k+1][2] if k+1 < len(arr) else None
            lo = (prev_orig + guard_prev) if prev_orig is not None else -10**12
            hi = (next_orig - guard_next) if next_orig is not None else  10**12
            if hi < lo: hi = lo
            windows[idx] = (lo, hi)

    new_notes: List[Note] = []
    min_len_ticks = 1  # harte Mindestlänge in Ticks

    for idx, n in enumerate(song.notes):
        st_tick = int(getattr(n, "start_tick", s2t(n.start)))
        en_tick = int(getattr(n, "end_tick",   s2t(n.end)))
        note_len_ticks = max(1, en_tick - st_tick)

        tag = f"{n.pitch}|{st_tick}"

        # --- Onset-Jitter (± onset_frac * note_len_ticks)
        r_on = _u01_from(seed, "onset|" + tag) * 2.0 - 1.0
        max_onset_shift = int(round(note_len_ticks * onset_frac))
        d_on_ticks = int(round(r_on * max_onset_shift))

        lo, hi = windows.get(idx, (-10**12, 10**12))
        new_st_tick = _clamp(st_tick + d_on_ticks, lo, hi)

        # --- Längen-Jitter (± length_frac * note_len_ticks)
        r_len = _u01_from(seed, "len|" + tag) * 2.0 - 1.0
        max_len_shift = int(round(note_len_ticks * length_frac))
        d_len_ticks = int(round(r_len * max_len_shift))

        # Endzeit auf Basis *neuen* Starts, mit Mindestlänge
        new_en_tick = max(new_st_tick + min_len_ticks, en_tick + d_len_ticks)

        nn = Note(
            pitch=n.pitch,
            start=t2s(new_st_tick),
            end=t2s(new_en_tick),
            velocity=n.velocity,
            articulations=n.articulations
        )
        setattr(nn, "start_tick", new_st_tick)
        setattr(nn, "end_tick",   new_en_tick)
        setattr(nn, "_vkey", getattr(n, "_vkey", ("", "")))
        setattr(nn, "orig_onset_tick", getattr(n, "orig_onset_tick", st_tick))
        new_notes.append(nn)

    return MidiSong(name=song.name, notes=new_notes, length=song.length, beats=song.beats, bars=song.bars, meta=song.meta)

def _apply_slur_legato(song: MidiSong, bundle: TimelineBundle, phrasing_cfg: Dict) -> MidiSong:
    sl = (phrasing_cfg or {}).get("slur_legato") or {}
    if not sl.get("enabled", True):
        return song

    tpb = bundle.ticks_per_beat
    t2s = ticks_to_seconds_map(bundle.conductor.tempos, tpb)
    s2t = seconds_to_ticks_map(bundle.conductor.tempos, tpb)

    def _bpm_at(tick: int) -> float:
        bpm = 120.0
        for tt, b in bundle.conductor.tempos:
            if tt <= tick: bpm = float(b)
            else: break
        return bpm

    def _sec_per_tick_at(tick: int) -> float:
        return (60.0 / _bpm_at(tick)) / float(tpb)

    # ---- Config (rein relativ) ----
    early_cfg = (sl.get("early_start") or {})
    early_ms  = early_cfg.get("ms", 0)  # zusätzlicher absoluter Offset; darf 0 sein

    max_adv_ms = early_cfg.get("max_advance_ms", sl.get("max_advance_ms", None))
    longest_note_ms = early_cfg.get("longest_note_ms", sl.get("longest_note_ms", None))
    max_adv_frac_prev = float(sl.get("max_advance_frac_of_prev_len", sl.get("max_advance_frac_of_prev_len", 0.50)))

    guard_prev = int(sl.get("guard_prev_onset_ticks", 0))
    guard_next = int(sl.get("guard_next_onset_ticks", 0))

    overlap_ticks = int(sl.get("overlap_ticks", 0))
    overlap_ms    = sl.get("overlap_ms", None)

    non_slur_gap_ticks = int(sl.get("non_slur_min_gap_ticks", 0))
    tenuto_gap_ticks   = int(sl.get("tenuto_min_gap_ticks", 0))
    non_slur_gap_ms    = sl.get("non_slur_min_gap_ms", None)
    tenuto_gap_ms      = sl.get("tenuto_min_gap_ms", None)

    # --- NEU: Obergrenze für die Länge, auf die rel_of_len wirkt (in ms)
    longest_note_ms = sl.get("longest_note_ms", None)

    # Kopie
    notes: List[Note] = []
    for n in song.notes:
        nn = Note(n.pitch, float(n.start), float(n.end), n.velocity, n.articulations)
        setattr(nn, "start_tick", int(getattr(n, "start_tick", s2t(n.start))))
        setattr(nn, "end_tick",   int(getattr(n, "end_tick",   s2t(n.end))))
        setattr(nn, "_vkey", getattr(n, "_vkey", ("", "")))
        setattr(nn, "orig_onset_tick", getattr(n, "orig_onset_tick", int(getattr(n, "start_tick"))))
        notes.append(nn)

    by_vkey: DefaultDict[tuple, List[Note]] = defaultdict(list)
    for nn in notes:
        by_vkey[nn._vkey].append(nn)

    # 2a) Frühstart nur für Slur-Zielnote
    for vkey, arr in by_vkey.items():
        arr.sort(key=lambda x: x.orig_onset_tick)
        for i, cur in enumerate(arr):
            if "slur" not in (cur.articulations or []):
                continue

            st = cur.start_tick
            en = cur.end_tick
            own_len_ticks = max(1, en - st)

            spt = _sec_per_tick_at(st)

            def ms_to_ticks(ms: float) -> int:
                return int(round((ms / 1000.0) / spt))

            own_len_ticks = max(1, en - st)
            spt = _sec_per_tick_at(st)  # sek pro tick an dieser Stelle

            def ms_to_ticks(ms: float) -> int:
                return int(round((ms / 1000.0) / spt))

            # ---- NEU: rein relative Advance-Formel in ms ----
            # L_eff = min(L_ms, longest_note_ms)
            # A_ms  = max_advance_ms * (L_eff / longest_note_ms)
            adv_ms_rel = 0.0
            if (max_adv_ms is not None) and (longest_note_ms is not None) and (longest_note_ms > 0):
                own_len_ms = own_len_ticks * spt * 1000.0
                L_eff = min(float(own_len_ms), float(longest_note_ms))
                adv_ms_rel = float(max_adv_ms) * (L_eff / float(longest_note_ms))

            # zusätzlicher absoluter Offset (kann 0 sein)
            adv_ms_total = max(0.0, float(early_ms) + float(adv_ms_rel))

            base_shift_ticks = ms_to_ticks(adv_ms_total)

            # ---- Caps (wie gehabt) ----
            if i > 0:
                prev = arr[i-1]
                prev_len = max(1, prev.end_tick - prev.start_tick)
                cap_by_prev = int(round(max_adv_frac_prev * prev_len))
            else:
                cap_by_prev = base_shift_ticks

            cap_by_ms = ms_to_ticks(float(max_adv_ms)) if (max_adv_ms is not None) else base_shift_ticks

            applied_shift = min(base_shift_ticks, cap_by_prev, cap_by_ms)

            # Fenster-Guards und Anwenden
            cand = st - applied_shift
            prev_st = (arr[i-1].start_tick + guard_prev) if i > 0 else None
            next_st = (arr[i+1].start_tick - guard_next) if i+1 < len(arr) else None
            if prev_st is not None: cand = max(cand, prev_st)
            if next_st is not None: cand = min(cand, next_st)

            cur.start_tick = max(0, cand)
            cur.start = t2s(cur.start_tick)

    # 2b) Exakter Overlap: ms > ticks
    for vkey, arr in by_vkey.items():
        arr.sort(key=lambda x: x.start_tick)
        for i in range(1, len(arr)):
            prev, cur = arr[i-1], arr[i]
            if "slur" in (cur.articulations or []) and int(prev.pitch) != int(cur.pitch):
                spt_prev = _sec_per_tick_at(prev.start_tick)
                def ms_to_ticks_local(ms: float) -> int:
                    return int(round((ms / 1000.0) / spt_prev))

                # -> wenn overlap_ms gesetzt, hat es Vorrang
                min_ov_ticks_abs = (
                    ms_to_ticks_local(float(overlap_ms))
                    if (overlap_ms is not None)
                    else int(overlap_ticks)
                )

                new_pen = max(prev.start_tick + 1, cur.start_tick + max(0, min_ov_ticks_abs))
                if new_pen != prev.end_tick:
                    prev.end_tick = new_pen
                    prev.end = t2s(prev.end_tick)

    # 2c) Gleicher Pitch: Mindestabstände, ms > ticks
    by_vkey_pitch: DefaultDict[Tuple[tuple, int], List[Note]] = defaultdict(list)
    for nn in notes:
        vkey = getattr(nn, "_vkey", ("", ""))
        by_vkey_pitch[(vkey, int(nn.pitch))].append(nn)

    for arr in by_vkey_pitch.values():
        arr.sort(key=lambda x: x.start_tick)
        for i in range(len(arr) - 1):
            a, b = arr[i], arr[i+1]

            spt_a = _sec_per_tick_at(a.start_tick)
            def ms_to_ticks_a(ms: float) -> int:
                return int(round((ms / 1000.0) / spt_a))

            non_slur_gap = (
                ms_to_ticks_a(float(non_slur_gap_ms))
                if (non_slur_gap_ms is not None)
                else int(non_slur_gap_ticks)
            )
            tenuto_gap = (
                ms_to_ticks_a(float(tenuto_gap_ms))
                if (tenuto_gap_ms is not None)
                else int(tenuto_gap_ticks)
            )

            arts_a = set(a.articulations or [])
            gap_needed = tenuto_gap if "tenuto" in arts_a else non_slur_gap
            if gap_needed <= 0:
                continue

            desired_end = b.start_tick - gap_needed
            if a.end_tick > desired_end:
                a.end_tick = max(a.start_tick + 1, desired_end)
                a.end = t2s(a.end_tick)

    return MidiSong(song.name, notes, song.length, song.beats, song.bars, song.meta)

def _apply_interpretation(bundle: TimelineBundle,
                          track: TrackTimeline,
                          analysis_song: MidiSong,
                          dyn_events: List[Tuple[int, str]],
                          wedge_spans: List[Tuple[int,int,str]],
                          cfg: Dict) -> MidiSong:
    # Sekunden-Events etc. wie bisher ...

    tpb = bundle.ticks_per_beat
    ticks, L = build_L_curve(track, dyn_events, wedge_spans, tpb)
    t2s = ticks_to_seconds_map(bundle.conductor.tempos, tpb)
    xs_sec = [t2s(int(t)) for t in ticks]

    # Hilfs-Lookup für Bar/Beat (closure nutzt tpb)
    def _bar_start_tick(ts_tick: int, num: int, _den: int, tick: int) -> int:
        beats_per_bar = int(num)
        bar_len = beats_per_bar * tpb
        rel = max(0, tick - ts_tick)
        k = rel // bar_len
        return ts_tick + int(k * bar_len)

    vel_cfg  = cfg.get("velocities", cfg.get("dynamics", {})) or {}
    cc1_cfg  = cfg.get("cc1",  {}) or {}
    cc11_cfg = cfg.get("cc11", {}) or {}

    vel_enabled  = bool(vel_cfg.get("enabled",  True))
    cc1_enabled  = bool(cc1_cfg.get("enabled",  True))
    cc11_enabled = bool(cc11_cfg.get("enabled", True))

    # --- Humanize-Timing vorziehen (für CCs & finale Notenzeiten) ---
    timing_cfg = (cfg.get("timing") or {})
    timing_song = _humanize_timing(analysis_song, bundle, timing_cfg)

    # --- NEU: Slur-Legato nach Timing anwenden ---
    phrasing_cfg = (cfg.get("phrasing") or {})
    legato_song = _apply_slur_legato(timing_song, bundle, phrasing_cfg)

    # ---------- 1) Velocity weiterhin auf analysis_song berechnen ----------
    lin_min = int(vel_cfg.get("linear_min", 5))
    lin_max = int(vel_cfg.get("linear_max", 127))
    if lin_max < lin_min: lin_min, lin_max = lin_max, lin_min
    ba_cfg = (cfg.get("beat_accent") or vel_cfg.get("beat_accent") or {}) or {}

    vhum = (vel_cfg.get("humanize") or {}) if vel_enabled else {}
    vh_enabled   = bool(vhum.get("enabled", True))
    vh_seed_base = int(vhum.get("seed", 1337))
    vh_mul_amt   = float(vhum.get("mul_jitter", 0.10))
    vh_add_amt   = int(vhum.get("add_jitter", 3))

    velocities: List[int] = []
    for n in analysis_song.notes:
        if not vel_enabled:
            vel = _clamp(int(getattr(n, "velocity", 64)), 1, 127)
        else:
            s_tick = int(getattr(n, "start_tick", 0))
            e_tick = int(getattr(n, "end_tick", s_tick + 1))
            t_mid  = s_tick + max(1, (e_tick - s_tick)//3)

            dyn = _segment_mean(np.array(ticks), np.array(L), s_tick, t_mid)
            base_v = lin_min + dyn * (lin_max - lin_min)

            offs = 0
            arts = (n.articulations or [])
            if "accent"   in arts: offs += 12
            if "marcato"  in arts: offs += 18
            if "staccato" in arts: offs -= 6
            if "tenuto"   in arts: offs += 3
            vel = int(round(base_v + offs))

            if ba_cfg.get("enabled", True):
                timesigs = bundle.conductor.timesigs
                def _timesig_at(tick: int):
                    last = (0, 4, 4)
                    for tt, num, den in timesigs:
                        if tt <= tick: last = (tt, num, den)
                        else: break
                    return last
                ts_tick, num, den = _timesig_at(s_tick)
                bar_start = _bar_start_tick(ts_tick, num, den, s_tick)
                phase, beat_idx = _note_phase_in_bar(s_tick, bar_start, tpb)
                beats_per_bar = int(num) if int(num) > 0 else 4
                cat = _categorize_beat(phase, beat_idx, beats_per_bar, ba_cfg)
                vel = _apply_accent(vel, cat, ba_cfg)

            if vh_enabled and (vh_mul_amt > 0.0 or vh_add_amt > 0):
                hh_mul = _subseed(vh_seed_base, f"vel_mul|{s_tick}|{n.pitch}")
                hh_add = _subseed(vh_seed_base, f"vel_add|{s_tick}|{n.pitch}")
                r_mul = _rand_unit_symmetric(hh_mul)
                r_add = _rand_unit_symmetric(hh_add)
                vel = vel * (1.0 + r_mul * vh_mul_amt)
                vel = vel + (r_add * vh_add_amt)

            vel = _clamp(int(round(vel)), 1, 127)

        velocities.append(vel)

    # ---------- 1b) Slur-Vorziehen → Velocity (mit wählbarer Skalierung) ----------
    slv = ((cfg.get("velocities") or {}).get("slur_advance_velocity")) or {}
    fullscale_ms = int(slv.get("fullscale_ms", 160))
    sav_enabled   = bool(slv.get("enabled", True))
    include_first = bool(slv.get("include_first_note", False))

    v_min = int(slv.get("min_vel", 31))    # viel Advance → leise
    v_max = int(slv.get("max_vel", 105))   # wenig Advance → laut
    gamma = float(slv.get("gamma", 1.0))
    mix_in_base = float(slv.get("mix_old", 0.2))  # etwas Basis-Dynamik beibehalten (Default erhöht)

    if sav_enabled and len(analysis_song.notes) == len(legato_song.notes) == len(timing_song.notes):
        for i, (n_src, n_tim, n_leg) in enumerate(zip(analysis_song.notes, timing_song.notes, legato_song.notes)):
            arts = set(n_leg.articulations or [])
            is_slur_follower = ("slur" in arts)

            if not include_first and not is_slur_follower:
                continue

            st_leg = int(getattr(n_leg, "start_tick", 0))      # nach Slur/Phrasing
            st_tim = int(getattr(n_tim, "start_tick", st_leg)) # vor Slur
            adv_ticks = max(0, st_tim - st_leg)                # tatsächlicher Vorziehwert in Ticks

            def _bpm_at(tick: int):
                bpm = 120.0
                for tt, b in bundle.conductor.tempos:
                    if tt <= tick: bpm = b
                    else: break
                return float(bpm)

            bpm_here = _bpm_at(st_tim)
            sec_per_tick = (60.0 / bpm_here) / tpb
            denom_ticks = max(1, int(round((fullscale_ms / 1000.0) / sec_per_tick)))
            norm = adv_ticks / float(denom_ticks)

            if gamma != 1.0:
                norm = norm ** gamma
            norm = 0.0 if norm < 0.0 else (1.0 if norm > 1.0 else norm)

            # wenig Advance → v_max, viel Advance → v_min
            v_target = v_max + (v_min - v_max) * norm

            # Basis-Velocity leicht beibehalten, damit Beat-Accent/Humanize nicht „weggebügelt“ wird
            v_new = int(round((1.0 - mix_in_base) * v_target + mix_in_base * int(velocities[i])))

            # optional: noch etwas Jitter drauf (wie bei normalen Velocities)
            jitter_amt = float(slv.get("extra_jitter", 0.0))  # z.B. 2.0
            if jitter_amt > 0:
                hh = _subseed(vh_seed_base, f"sav_jit|{st_tim}|{n_leg.pitch}")
                v_new = v_new + _rand_unit_symmetric(hh) * jitter_amt

            velocities[i] = _clamp(v_new, 1, 127)

    # ---------- 2) CC-Kurven jetzt auf timing_song ----------
    def _build_cc_array(cc_cfg: Dict) -> Optional[np.ndarray]:
        if not (cc_cfg or {}).get("enabled", True): return None
        curve = (cc_cfg.get("curve") or {}) or {}
        base_seed = int(curve.get("seed", cc_cfg.get("seed", 1337)))
        seed_notes   = _subseed(base_seed, "notes")
        seed_phrases = _subseed(base_seed, "phrases")
        seed_noise   = _subseed(base_seed, "random")

        notes_cfg   = (curve.get("notes")   or {})
        phrases_cfg = (curve.get("phrases") or {})
        rn_cfg      = (curve.get("random")  or {})

        L_arr = np.array(L)
        N_arr = build_note_curve(
            legato_song, np.array(ticks), L_arr, tpb,
            humanize_cfg={"seed": seed_notes,
                          "notes": {"enabled": True, **(notes_cfg.get("humanize") or {})}},
        )
        P_arr = build_phrase_curve(
            legato_song, np.array(ticks), L_arr, tpb,
            humanize_cfg={"seed": seed_phrases,
                          "phrases": {"enabled": True, **(phrases_cfg.get("humanize") or {})}},
            depth=1.0,
        )
        A_arr = build_articulation_curve(legato_song, np.array(ticks), tpb)
        H_arr = build_pitch_curve(legato_song, np.array(ticks), tpb)

        if bool(rn_cfg.get("enabled", True)):
            r_xscale = float(rn_cfg.get("x_scale", 0.4))
            r_ydepth = float(rn_cfg.get("y_depth", 0.08))
            R_arr = build_random_noise_curve(np.array(ticks), seed_noise, r_xscale, r_ydepth, tpb)
        else:
            R_arr = np.ones_like(ticks, dtype=float)

        return compose_total_curve(L_arr, P_arr, N_arr, A_arr, H_arr, R_arr)

    cc_map: Dict[int, List[Tuple[float,int]]] = {}
    C01_cc1  = _build_cc_array(cc1_cfg)  if cc1_enabled  else None
    C01_cc11 = _build_cc_array(cc11_cfg) if cc11_enabled else None
    if C01_cc1 is not None:
        vals = _map_cc_range_from01(C01_cc1, int(cc1_cfg.get("out_min", 0)), int(cc1_cfg.get("out_max", 127)))
        cc_map[1] = _array_to_cc_points(xs_sec, vals)
    if C01_cc11 is not None:
        vals = _map_cc_range_from01(C01_cc11, int(cc11_cfg.get("out_min", 32)), int(cc11_cfg.get("out_max", 127)))
        cc_map[11] = _array_to_cc_points(xs_sec, vals)

    # ---------- 3) Finale Noten: Zeiten aus timing_song, Velocity aus oben ----------
    notes_out: List[Note] = []
    for idx, (n_orig, n_t) in enumerate(zip(analysis_song.notes, legato_song.notes)):
        notes_out.append(Note(
            pitch=n_orig.pitch,
            start=float(n_t.start),
            end=float(n_t.end),
            velocity=int(velocities[idx]),
            articulations=n_orig.articulations
        ))

    post = MidiSong(
        name=analysis_song.name + " (MIDI)",
        notes=notes_out,
        length=analysis_song.length,
        beats=analysis_song.beats,
        bars=analysis_song.bars,
        meta={"cc": cc_map}
    )
    # WICHTIG: KEIN _humanize_timing(post, ...) mehr – wir haben timing schon übernommen.
    return post

# ---------- GUI ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("musicxml2midi – Inspector (Analysis & MIDI)")
        self.resize(1280, 900)

        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)

        # --- Top toolbar ---
        tbar = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Open MusicXML…")
        self.btn_settings = QtWidgets.QPushButton("Settings…")
        self.btn_export = QtWidgets.QPushButton("Export MIDI")
        self.btn_export.setEnabled(False)  # kommt später
        tbar.addWidget(self.btn_open)
        tbar.addWidget(self.btn_settings)
        tbar.addWidget(self.btn_export)
        tbar.addStretch(1)
        root.addLayout(tbar)

        # --- Split: left list, right panels (analysis+post) ---
        main = QtWidgets.QHBoxLayout(); root.addLayout(main, 1)

        left = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(left)
        self.part_list = QtWidgets.QListWidget()
        lv.addWidget(QtWidgets.QLabel("Parts"))
        lv.addWidget(self.part_list, 1)

        # Panels
        self.roll = PianoRoll()                         # Analyse (oben)
        # --- Curve Inspector Panel (Controls + Plot) ---
        self.curve_box = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(self.curve_box)
        cv.setContentsMargins(0, 0, 0, 0)

        # Controls-Reihe
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setContentsMargins(4, 2, 4, 2)
        ctrl.addWidget(QtWidgets.QLabel("Interpretation → MIDI"))

        self.chk_show_L     = QtWidgets.QCheckBox("L (Dynamics)")
        self.chk_show_P     = QtWidgets.QCheckBox("P (Phrase)")
        self.chk_show_N     = QtWidgets.QCheckBox("N (Notes)")
        self.chk_show_A     = QtWidgets.QCheckBox("A (Artic)")
        self.chk_show_H     = QtWidgets.QCheckBox("H (Pitch)")
        self.chk_show_R     = QtWidgets.QCheckBox("R (Random Noise)")
        self.chk_show_total = QtWidgets.QCheckBox("Preview (Total)")

        # Default: nur Total aktiv
        for cb in (self.chk_show_L, self.chk_show_P, self.chk_show_N, self.chk_show_A, self.chk_show_H, self.chk_show_R):
            cb.setChecked(False)
        self.chk_show_total.setChecked(True)

        for cb in (self.chk_show_L, self.chk_show_P, self.chk_show_N, self.chk_show_A, self.chk_show_H, self.chk_show_R, self.chk_show_total):
            ctrl.addWidget(cb)

        ctrl.addStretch(1)
        cv.addLayout(ctrl)

        # Plot
        self.cc_plot = PlotWidget(viewBox=GestureViewBox(zoom_axes="x", allow_y_pan=False))
        self.cc_plot.setMenuEnabled(False)
        self.cc_plot.setMouseEnabled(x=True, y=False)
        self.cc_plot.showGrid(x=True, y=True, alpha=0.2)
        cv.addWidget(self.cc_plot, 1)

        # Interpretation (unten)
        self.post = PostPanel(embedded_controls=True)   

        # Header oben
        analysis_box = QtWidgets.QWidget()
        analysis_v = QtWidgets.QVBoxLayout(analysis_box)
        analysis_v.setContentsMargins(0,0,0,0)
        analysis_v.addWidget(QtWidgets.QLabel("Analysis"))
        analysis_v.addWidget(self.roll, 1)

        # Header unten mit CC-Auswahl rechts
        post_box = QtWidgets.QWidget()
        post_v = QtWidgets.QVBoxLayout(post_box)
        post_v.setContentsMargins(0,0,0,0)
        post_hdr = QtWidgets.QHBoxLayout()
        post_hdr.addStretch(1)
        post_v.addLayout(post_hdr)
        post_v.addWidget(self.post, 1)

        # --- Vertikaler Splitter: oben (Analysis), MITTE (L(t)), unten (Post) ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        splitter.addWidget(analysis_box)
        splitter.addWidget(self.curve_box)
        splitter.addWidget(post_box)

        # Stretch-Faktoren (oben : mitte : unten)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)    # L(t) mittlere Höhe
        splitter.setStretchFactor(2, 3)

        # Initialgrößen (Pixel, optional)
        splitter.setSizes([320, 200, 560])

        # Panels X-synchronisieren (oben<->unten)
        self.post.roll.setXLink(self.roll.plot)
        self.cc_plot.setXLink(self.roll.plot)

        main.addWidget(left, 0)
        main.addWidget(splitter, 1)

        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)

        # state
        self.bundle: Optional[TimelineBundle] = None
        self.parts_order: List[str] = []
        self.dyn_by_part: Dict[str, List[Tuple[int, str]]] = {}
        self.wedges_by_part: Dict[str, List[Tuple[int,int,str]]] = {}

        # wiring
        self.btn_open.clicked.connect(self.on_open)
        self.part_list.itemSelectionChanged.connect(self.on_select_part)
        self.btn_settings.clicked.connect(self.on_settings)
        self.btn_export.clicked.connect(self.on_export)

        self.cfg_data = load_gui_config()

        def _replot_curves():
            if getattr(self, "_current_pid", None):
                pid = self._current_pid
                self.show_curves(self.bundle.tracks[pid],
                                self.dyn_by_part.get(pid, []),
                                self.wedges_by_part.get(pid, []))

        for cb in (self.chk_show_L, self.chk_show_P, self.chk_show_N, self.chk_show_A, self.chk_show_H, self.chk_show_R, self.chk_show_total):
            cb.toggled.connect(_replot_curves)

        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel("CC preview:"))
        self.cmb_cc_preview = QtWidgets.QComboBox()
        self.cmb_cc_preview.addItems(["CC1", "CC11"])  # nur eine aktiv
        self.cmb_cc_preview.setCurrentIndex(0)
        ctrl.addWidget(self.cmb_cc_preview)
        self.cmb_cc_preview.currentIndexChanged.connect(_replot_curves)


    # ---- file load & analysis ----
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open MusicXML", "", "MusicXML (*.musicxml *.xml)")
        if not path: return
        self._last_xml_path = path
        self.load_file(path)

    def load_file(self, path: str):
        cfg = self.cfg_data
        tpb = int(cfg.get("ticks_per_beat", 960))
        score = analyze_musicxml(path)
        self.bundle = build_timelines(score, cfg)

        dyns: Dict[str, List[Tuple[int, str]]] = {}
        wedges: Dict[str, List[Tuple[int,int,str]]] = {}

        for pid, pa in score.parts.items():
            def to_tick(div_pos: int) -> int:
                divs = 480
                for td, dv in pa.divisions_changes:
                    if td <= div_pos: divs = dv
                    else: break
                return int(round(div_pos * (tpb / max(1, divs))))

            dls: List[Tuple[int,str]] = []
            wraw: List[Tuple[int,str]] = []
            for d in pa.directions:
                if d.kind == "dynamic" and "mark" in d.payload:
                    dls.append((to_tick(d.time_div), d.payload["mark"].lower()))
                elif d.kind == "wedge" and "type" in d.payload:
                    wraw.append((to_tick(d.time_div), d.payload["type"]))
            dyns[pid] = sorted(dls, key=lambda x: x[0])
            wedges[pid] = pair_wedges_to_spans(wraw)

        self.dyn_by_part = dyns
        self.wedges_by_part = wedges

        self.parts_order = list(self.bundle.tracks.keys())
        self.part_list.clear()
        for pid in self.parts_order:
            self.part_list.addItem(self.bundle.tracks[pid].name)
        if self.parts_order:
            self.part_list.setCurrentRow(0)
            self.show_part(self.parts_order[0])

        from os.path import basename
        self.status.showMessage(f"Loaded {basename(path)} | parts={len(self.parts_order)}", 5000)

        self.btn_export.setEnabled(True)

    def on_select_part(self):
        row = self.part_list.currentRow()
        if 0 <= row < len(self.parts_order):
            self.show_part(self.parts_order[row])

    def show_part(self, pid: str):
        if not self.bundle:
            return
        track = self.bundle.tracks[pid]
        dyns  = self.dyn_by_part.get(pid, [])
        wdg   = self.wedges_by_part.get(pid, [])

        analysis_song = build_song_from_bundle(self.bundle, track, dyns, wdg)
        self.roll.set_song(analysis_song)

        # hier mit allen drei Argumenten
        self.show_curves(track, dyns, wdg)

        self._current_pid = pid
        self.rebuild_post_for_part(pid)

    # ---- actions ----
    def on_settings(self):
        dlg = SettingsDialog(self, cfg_dict=self.cfg_data)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            # Dialog hat gespeichert → neu laden
            self.cfg_data = load_gui_config()
            if getattr(self, "_current_pid", None):
                pid = self._current_pid
                self.rebuild_post_for_part(pid)
                self.show_curves(self.bundle.tracks[pid],
                                self.dyn_by_part.get(pid, []),
                                self.wedges_by_part.get(pid, []))
            self.statusBar().showMessage("Settings applied → MIDI updated.", 2000)

    def rebuild_post_for_part(self, pid: str):
        if not self.bundle: return
        track = self.bundle.tracks[pid]
        dyns  = self.dyn_by_part.get(pid, [])
        wedges = self.wedges_by_part.get(pid, [])

        # Analysis-Song neu bauen (damit start_tick sicher dran ist)
        analysis_song = build_song_from_bundle(self.bundle, track, dyns, wedges)

        post_song = _apply_interpretation(
            bundle=self.bundle,
            track=track,
            analysis_song=analysis_song,
            dyn_events=dyns,
            wedge_spans=wedges,
            cfg=self.cfg_data,   # zentrale Quelle!
        )
        # unten anzeigen
        self.post.set_song(post_song)

    
    def show_curves(self, track: TrackTimeline, dyns, wedges):
        if not self.bundle:
            return
        tpb = self.bundle.ticks_per_beat
        ticks, L_base = build_L_curve(track, dyns, wedges, tpb)

        analysis_song = build_song_from_bundle(self.bundle, track, dyns, wedges)

        # Timing anwenden (wie in _apply_interpretation)
        timing_cfg = (self.cfg_data.get("timing") or {})
        timing_song = _humanize_timing(analysis_song, self.bundle, timing_cfg)

        # Slur-Legato wie in _apply_interpretation
        phrasing_cfg = (self.cfg_data.get("phrasing") or {})
        legato_song = _apply_slur_legato(timing_song, self.bundle, phrasing_cfg)

        t2s = ticks_to_seconds_map(self.bundle.conductor.tempos, tpb)
        xs = [t2s(int(t)) for t in ticks]

        # 4) CC-spezifische Komponenten + gemappte Preview
        sel = self.cmb_cc_preview.currentText() if hasattr(self, "cmb_cc_preview") else "CC1"
        if sel == "CC1":
            cc_cfg = self.cfg_data.get("cc1", {}) or {}
            out_lo = int(cc_cfg.get("out_min", 0))
            out_hi = int(cc_cfg.get("out_max", 127))
        else:  # CC11
            cc_cfg = self.cfg_data.get("cc11", {}) or {}
            out_lo = int(cc_cfg.get("out_min", 32))   # CC11 Default 32!
            out_hi = int(cc_cfg.get("out_max", 127))
        if out_hi < out_lo:
            out_lo, out_hi = out_hi, out_lo
        color  = (255, 255, 255)    # weiß

        # CC-spezifische L/P/N/A/H (inkl. Humanize aus cc_cfg)
        L, P, N, A, H, R = _build_cc_components(legato_song, np.array(ticks), np.array(L_base), tpb, cc_cfg)

        # Total 0..1 (weiche Sättigung wie gehabt)
        C01 = compose_total_curve(L, P, N, A, H, R)
        C01 = np.clip(C01, 0.0, 1.0)

        # --- Preview in gemappter Range (FLOAT, nicht gerundet) ---
        preview = out_lo + (out_hi - out_lo) * C01

        # Plot vorbereiten
        self.cc_plot.clear()
        self.cc_plot.setYRange(0, 127)

        # Ticks/Gitter fest auf 0, 63, 127
        yaxis = self.cc_plot.getPlotItem().getAxis('left')
        yaxis.setTicks([
            [(0, '0'), (63, '63'), (127, '127')],  # Major Ticks (werden gegriddet)
            []                                      # optional: Minor leer
        ])

        # Optional: Komponenten als subtile Hilfslinien in 0..1 (wenn du sie behalten willst):
        # -> ENTWEDER weglassen...
        # -> ODER umrechnen in CC-Range, damit alles die gleiche Skala hat:
        if self.chk_show_L.isChecked():
            self.cc_plot.plot(xs, out_lo + (out_hi - out_lo) * L, pen=pg.mkPen((255, 215, 0),  width=1))
        if self.chk_show_P.isChecked():
            self.cc_plot.plot(xs, out_lo + (out_hi - out_lo) * P, pen=pg.mkPen((0, 200, 120),  width=1))
        if self.chk_show_N.isChecked():
            self.cc_plot.plot(xs, out_lo + (out_hi - out_lo) * N, pen=pg.mkPen((80, 200, 255), width=1))
        if self.chk_show_A.isChecked():
            self.cc_plot.plot(xs, out_lo + (out_hi - out_lo) * A, pen=pg.mkPen((220, 120, 255), width=1))
        if self.chk_show_H.isChecked():
            self.cc_plot.plot(xs, out_lo + (out_hi - out_lo) * H, pen=pg.mkPen((100, 160, 255), width=1))
        if self.chk_show_R.isChecked():
            self.cc_plot.plot(xs, out_lo + (out_hi - out_lo) * R, pen=pg.mkPen((255, 100, 100), width=1))
        if self.chk_show_total.isChecked():
            self.cc_plot.plot(xs, preview, pen=pg.mkPen(color, width=2))

    def on_export(self):
        if mido is None:
            QtWidgets.QMessageBox.warning(
                self, "Export MIDI",
                "Das Paket 'mido' ist nicht installiert.\n\nInstalliere es z.B. mit:\n    pip install mido python-rtmidi"
            )
            return

        if not self.bundle or not self.parts_order:
            QtWidgets.QMessageBox.information(self, "Export MIDI", "Kein Projekt geladen.")
            return

        # --- Basename & Zielordner in EINEM Dialog (Speichern) auswählen ---
        default_base = "export"
        if hasattr(self, "_last_xml_path") and self._last_xml_path:
            default_base = os.path.splitext(os.path.basename(self._last_xml_path))[0]

        # optional: letztes Verzeichnis merken (oder nimm os.getcwd())
        start_dir = os.path.dirname(getattr(self, "_last_xml_path", "")) or os.getcwd()
        default_path = os.path.join(start_dir, f"{default_base}.mid")

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Basename & Ziel wählen", default_path, "MIDI (*.mid);;Alle Dateien (*)"
        )
        if not fname:
            return

        out_dir = os.path.dirname(fname)
        base_in = os.path.splitext(os.path.basename(fname))[0]  # ohne .mid
        base = _sanitize_basename(base_in, fallback=default_base)

        would_overwrite = []
        if os.path.exists(os.path.join(out_dir, f"{base}_conductor.mid")):
            would_overwrite.append(f"{base}_conductor.mid")
        for idx, pid in enumerate(self.parts_order):
            track = self.bundle.tracks[pid]
            safe_name = "".join(c if c.isalnum() or c in " _-." else "_" for c in (track.name or f"part_{idx+1}")).strip().strip("._-") or f"part_{idx+1}"
            fname_part = f"{base}_{idx+1:02d}_{safe_name}.mid"
            if os.path.exists(os.path.join(out_dir, fname_part)):
                would_overwrite.append(fname_part)

        if would_overwrite:
            reply = QtWidgets.QMessageBox.question(
                self, "Dateien überschreiben?",
                "Folgende Dateien existieren bereits und würden überschrieben:\n\n" + "\n".join(would_overwrite) + "\n\nFortfahren?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        # 1) Conductor-Datei
        try:
            mf_cond = _build_midifile_for_conductor_only(self.bundle)
            cond_path = os.path.join(out_dir, f"{base}_conductor.mid")
            mf_cond.save(cond_path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export MIDI", f"Conductor-Export fehlgeschlagen:\n{e}")

        # 2) Alle Parts – jeweils interpretieren und speichern
        errors = []
        for idx, pid in enumerate(self.parts_order):
            try:
                track = self.bundle.tracks[pid]
                dyns  = self.dyn_by_part.get(pid, [])
                wedges = self.wedges_by_part.get(pid, [])

                analysis_song = build_song_from_bundle(self.bundle, track, dyns, wedges)
                post_song = _apply_interpretation(
                    bundle=self.bundle,
                    track=track,
                    analysis_song=analysis_song,
                    dyn_events=dyns,
                    wedge_spans=wedges,
                    cfg=self.cfg_data,
                )

                mf = _build_midifile_for_song(post_song, self.bundle, add_conductor_meta=False, channel=0)

                # Dateiname: <base>_<NN>_<Partname>.mid (sicher gemacht)
                safe_name = "".join(
                    c if c.isalnum() or c in " _-." else "_"
                    for c in (track.name or f"part_{idx+1}")
                ).strip().strip("._-") or f"part_{idx+1}"
                fname = f"{base}_{idx+1:02d}_{safe_name}.mid"
                out_path = os.path.join(out_dir, fname)
                mf.save(out_path)
            except Exception as e:
                errors.append(f"{track.name}: {e}")

        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Export MIDI",
                "Einige Dateien konnten nicht exportiert werden:\n\n" + "\n".join(errors)
            )
        else:
            QtWidgets.QMessageBox.information(self, "Export MIDI", "Export finished.")

def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()

    # --- Auto-Load für schnelle Tests ---
    # Per ENV überschreibbar:  MX2M_AUTOLOAD=/pfad/zur/datei.musicxml
    # Default: examples/mar.musicxml (falls vorhanden)
    autoload = os.environ.get("MX2M_AUTOLOAD", "examples/mar.musicxml")
    try:
        if autoload and os.path.exists(autoload):
            w.load_file(autoload)
    except Exception as e:
        # bewusst still – nur Debug-Print, damit GUI trotzdem startet
        print("[autoload]", e)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()