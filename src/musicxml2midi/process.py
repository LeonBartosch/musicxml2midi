from __future__ import annotations
from typing import Dict, List, Tuple
from .timeline import (
    ScoreAnalysis, TimelineBundle, ConductorTimeline, TrackTimeline,
    NoteEvent, DEFAULT_TPB
)
from .util.time import div_to_ticks
import math

def _timesig_at(timesigs: List[Tuple[int,int,int]], tick: int) -> Tuple[int,int,int]:
    """
    timesigs: List[(tick, num, den)]
    Liefert den zuletzt aktiven Takt an 'tick'.
    """
    last = (0, 4, 4)
    for t, num, den in timesigs:
        if t <= tick:
            last = (t, num, den)
        else:
            break
    return last  # (ts_tick, num, den)

def _bar_start_tick(ts_tick: int, num: int, den: int, tick: int, tpb: int) -> int:
    """
    Bestimmt den Taktanfang, in dem 'tick' liegt, gegeben der letzten Taktangabe.
    Annahme: konstantes Raster bis zur nächsten Taktwechsel-Marke.
    """
    beats_per_bar = num  # einfache Zählzeit-Interpretation (4/4 ⇒ 4, 3/4 ⇒ 3, 2/2 ⇒ 2)
    bar_len = beats_per_bar * tpb
    rel = max(0, tick - ts_tick)
    k = rel // bar_len
    return ts_tick + int(k * bar_len)

def _note_phase_in_bar(tick: int, bar_start: int, tpb: int) -> Tuple[float, int]:
    """
    phase_in_beats: 0.0 am Taktanfang, 1.0 = Beat 2, ...
    beat_index_in_bar: floor(phase) (0-basiert)
    """
    rel = tick - bar_start
    phase = rel / float(tpb)
    return phase, int(math.floor(phase + 1e-9))

def _categorize_beat(phase: float, beat_idx: int, beats_per_bar: int, cfg: dict) -> str:
    tol_on = float(cfg.get("tol_onbeat", 0.08))
    tol_e8 = float(cfg.get("tol_eighth", 0.08))
    tol_s16 = float(cfg.get("tol_sixteenth", 0.06))
    frac = phase - math.floor(phase)
    # Downbeat?
    if abs(phase - round(phase)) <= tol_on and beat_idx == 0:
        return "bar_downbeat"
    # Midbar-Strong bei gerader Taktanzahl (z.B. Beat 3 in 4/4)
    if beats_per_bar % 2 == 0 and abs(phase - round(phase)) <= tol_on:
        if beat_idx == (beats_per_bar // 2):
            return "midbar_strong"
    # andere Onbeats (integer, aber nicht 1 und nicht midbar_strong)
    if abs(phase - round(phase)) <= tol_on:
        return "onbeats_other"
    # Achtel-Offbeats (x.5)
    if abs(frac - 0.5) <= tol_e8:
        return "eighth_offbeats"
    # Sechzehntel-Offs (x.25/x.75)
    if abs(frac - 0.25) <= tol_s16 or abs(frac - 0.75) <= tol_s16:
        return "sixteenth_offs"
    return "fallback"

def _apply_accent(v: int, cat: str, cfg: dict) -> int:
    rules = cfg.get("rules", {})
    rule = rules.get(cat, rules.get("fallback", {"mul":1.0,"add":0}))
    mul = float(rule.get("mul", 1.0))
    add = float(rule.get("add", 0.0))
    vv = int(round(v * mul + add))
    lo, hi = cfg.get("clamp_to", [1, 127])
    return max(int(lo), min(int(hi), vv))

def _divisions_at(div_changes: List[Tuple[int,int]], time_div: int) -> int:
    last = 480
    for td, dv in div_changes:
        if td <= time_div:
            last = dv
        else:
            break
    return last

def _merge_ties(tokens):
    """
    Merge aufeinanderfolgende NoteToken gleicher Pitch/Voice/Staff mit ties.
    Rückgabe-Tupel:
      (start_div, duration_div, midi, voice, staff, tags, slurred_any, slur_start_count, slur_stop_count)

    tags = kombinierte Liste aus articulations + ornaments + technical (bereits im Token enthalten)
    """
    out = []
    tokens_sorted = sorted(tokens, key=lambda t: (t.voice or "", t.staff or "", t.start_div))
    # key -> (start_div, dur_acc, tags_acc, slurred_any, slur_start_cnt, slur_stop_cnt)
    active: Dict[Tuple[str|None, str|None, int], Tuple[int,int,List[str],bool,int,int]] = {}

    for tok in tokens_sorted:
        if tok.midi is None:
            continue
        key = (tok.voice, tok.staff, tok.midi)
        this_slur_start_n = int(tok.slur_starts or 0)
        this_slur_stop_n  = int(tok.slur_stops  or 0)
        this_tags = (tok.articulations or []) + (tok.ornaments or [])

        if tok.tie_stop and key in active:
            sd, dur, tags_acc, slurred_any, s_start, s_stop = active[key]
            dur += tok.duration_div
            tags_acc = (tags_acc or []) + (this_tags or [])
            slurred_any = slurred_any or (this_slur_start_n > 0) or (this_slur_stop_n > 0)
            s_start += this_slur_start_n
            s_stop  += this_slur_stop_n
            active[key] = (sd, dur, tags_acc, slurred_any, s_start, s_stop)
            if not tok.tie_start:
                out.append((sd, dur, tok.midi, tok.voice, tok.staff, tags_acc, slurred_any, s_start, s_stop))
                active.pop(key, None)
        elif tok.tie_start:
            active[key] = (
                tok.start_div,
                tok.duration_div,
                list(this_tags or []),
                ((tok.slur_starts or 0) > 0) or ((tok.slur_stops or 0) > 0),
                this_slur_start_n,
                this_slur_stop_n
            )
        else:
            out.append((tok.start_div, tok.duration_div, tok.midi, tok.voice, tok.staff,
                        list(this_tags or []),
                        ((tok.slur_starts or 0) > 0) or ((tok.slur_stops or 0) > 0),
                        this_slur_start_n, this_slur_stop_n))
    # flush
    for (voice, staff, midi), (sd, dur, tags_acc, slurred_any, s_start, s_stop) in list(active.items()):
        out.append((sd, dur, midi, voice, staff, tags_acc, slurred_any, s_start, s_stop))
    out.sort(key=lambda x: (x[0], x[2]))
    return out

def _length_scale(arts: List[str], cfg: dict) -> float:
    ls = cfg.get("length_scaling", {})
    if "staccatissimo" in (arts or []): return float(ls.get("staccatissimo", 0.40))
    if "staccato"       in (arts or []): return float(ls.get("staccato", 0.55))
    if "tenuto"         in (arts or []): return float(ls.get("tenuto", 0.95))
    return float(ls.get("default", 0.98))

def _apply_legato_overlaps(note_events: List[NoteEvent], overlap_ticks_minmax: Tuple[int,int]=(5,60)):
    if not note_events:
        return
    min_ov, max_ov = overlap_ticks_minmax
    for i in range(1, len(note_events)):
        prev = note_events[i-1]
        curr = note_events[i]
        # Overlap nur wenn beide "slurred" und KEINE Slur-Grenze am Übergang:
        if prev.attrs.get("slurred") and curr.attrs.get("slurred") \
           and not prev.attrs.get("slur_stop") and not curr.attrs.get("slur_start"):
            ov = max(min_ov, min(max_ov, max(1, int(0.02 * (prev.end_tick - prev.start_tick)))))
            prev.end_tick = min(prev.end_tick + ov, curr.start_tick - 1 if curr.start_tick > prev.start_tick else prev.end_tick)

def build_timelines(analysis: ScoreAnalysis, cfg: dict) -> TimelineBundle:
    tpb = int(cfg.get("ticks_per_beat", DEFAULT_TPB))
    conductor = ConductorTimeline()
    tracks: Dict[str, TrackTimeline] = {}

    # Conductor-Timeline
    seen_tempos = set()
    for pa in analysis.parts.values():
        for d in pa.directions:
            if d.kind == "tempo":
                divs = _divisions_at(pa.divisions_changes, d.time_div)
                tick = div_to_ticks(d.time_div, divs, tpb)
                key = (tick, float(d.payload["bpm"]))
                if key not in seen_tempos:
                    conductor.tempos.append((tick, float(d.payload["bpm"])))
                    seen_tempos.add(key)
            elif d.kind == "timesig":
                divs = _divisions_at(pa.divisions_changes, d.time_div)
                tick = div_to_ticks(d.time_div, divs, tpb)
                num = int(d.payload["num"]); den = int(d.payload["den"])
                conductor.timesigs.append((tick, num, den))
    if not conductor.tempos:
        conductor.tempos.append((0, 120.0))
    if not conductor.timesigs:
        conductor.timesigs.append((0, 4, 4))
    conductor.tempos.sort(key=lambda x: x[0])
    conductor.timesigs.sort(key=lambda x: x[0])

    # Parts
    ch = 0
    for part_id, pa in analysis.parts.items():
        tr = TrackTimeline(name=pa.part_name or part_id, channel=(ch % 16))
        ch += 1

        phrases = _merge_ties(pa.notes)
        for (sd, dur, midi, voice, staff, tags, slurred_any, slur_start_n, slur_stop_n) in phrases:
            divs = _divisions_at(pa.divisions_changes, sd)
            start_tick = div_to_ticks(sd, divs, tpb)
            end_tick   = start_tick + max(1, div_to_ticks(dur, divs, tpb))

            # Längen-Skalierung
            length_factor = _length_scale(tags, cfg)  # staccato/tenuto werden im Mapping genutzt
            end_tick = start_tick + max(1, int((end_tick - start_tick) * length_factor))

            # Basis aus Dynamik-Mapping (ppp..fff) – falls später pro-Note 'dyn_mark' vorhanden ist, dort einstecken
            vel = int(cfg.get("dynamics_to_velocity", {}).get(None, 80))

            # --- metrisches Shaping nach Beat-Profil ---
            ba = cfg.get("beat_accent", {})
            if ba.get("enabled", True):
                # Taktangabe & Taktanfang an der Note ermitteln
                ts_tick, num, den = _timesig_at(conductor.timesigs, start_tick)
                bar_start = _bar_start_tick(ts_tick, num, den, start_tick, tpb)
                phase, beat_idx = _note_phase_in_bar(start_tick, bar_start, tpb)
                beats_per_bar = int(num) if int(num) > 0 else 4

                cat = _categorize_beat(phase, beat_idx, beats_per_bar, ba)
                vel = _apply_accent(vel, cat, ba)

            # Normalisierte Artikulationstags (für GUI)
            def norm_tag(t: str) -> str:
                t = (t or "").strip().lower()
                if t.startswith("mute("): return "mute"
                if t in ("trill-mark", "inverted-trill"): return "trill"
                if t in ("inverted-mordent",): return "mordent"
                return t

            tagset = []
            for t in (tags or []):
                nt = norm_tag(t)
                if nt and nt not in tagset:
                    tagset.append(nt)

            tr.notes.append(NoteEvent(
                start_tick=start_tick,
                end_tick=end_tick,
                midi=int(midi),
                velocity=vel,
                channel=tr.channel,
                attrs={
                    # Slur
                    "slurred": bool(slurred_any),
                    "slur_start": bool(slur_start_n > 0),
                    "slur_stop":  bool(slur_stop_n  > 0),
                    "slur_start_n": int(slur_start_n),
                    "slur_stop_n":  int(slur_stop_n),

                    # Klassische Einzel-Flags (nützlich für spätere Humanisierung)
                    "staccato": "staccato" in tagset or "staccatissimo" in tagset,
                    "tenuto":   "tenuto"   in tagset,
                    "accent":   "accent"   in tagset,
                    "marcato":  "marcato"  in tagset,

                    # Weitere Typen (für Anzeige/Logik)
                    "pizzicato":       "pizzicato" in tagset,
                    "snap-pizzicato":  "snap-pizzicato" in tagset,
                    "arco":            "arco" in tagset,
                    "tremolo":         "tremolo" in tagset,
                    "trill":           "trill" in tagset,
                    "mordent":         "mordent" in tagset,
                    "turn":            "turn" in tagset,
                    "harmonic":        "harmonic" in tagset,
                    "mute":            "mute" in tagset,   # Brass-Mute / con sordino

                    # Gruppierung
                    "voice": voice,
                    "staff": staff,

                    # Roh-Tags (für GUI-Anzeige in Klartext)
                    "tags": tagset,
                }
            ))

        tr.notes.sort(key=lambda ev: (ev.start_tick, ev.end_tick, ev.midi))
        _apply_legato_overlaps(tr.notes, (
            int(cfg.get("legato_overlap_ticks", {}).get("min", 5)),
            int(cfg.get("legato_overlap_ticks", {}).get("max", 60)),
        ))
        tracks[part_id] = tr

    return TimelineBundle(conductor=conductor, tracks=tracks, ticks_per_beat=tpb)