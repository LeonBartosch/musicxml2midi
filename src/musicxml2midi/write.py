from __future__ import annotations
import os
import re
import mido
from typing import Dict, Iterable
from .timeline import TimelineBundle, TrackTimeline, NoteEvent, CCEvent

# ---------- interne Helfer ----------

def _bpm_to_micro(bpm: float) -> int:
    return int(round(60_000_000 / max(1e-6, float(bpm))))

def _emit_conductor(track: mido.MidiTrack, tempos, timesigs):
    """Schreibt Tempo- und Takt-Metaevents in einen Track (sortiert & delta-times)."""
    last = 0
    events = []
    for tick, bpm in tempos:
        events.append((tick, ("tempo", bpm)))
    for tick, num, den in timesigs:
        events.append((tick, ("timesig", (num, den))))
    # Reihenfolge: TimeSig vor Tempo bei gleichem Tick
    events.sort(key=lambda x: (x[0], 0 if x[1][0] == "timesig" else 1))
    for tick, payload in events:
        delta = tick - last
        last = tick
        kind = payload[0]
        if kind == "tempo":
            track.append(mido.MetaMessage("set_tempo", tempo=_bpm_to_micro(payload[1]), time=delta))
        elif kind == "timesig":
            num, den = payload[1]
            track.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=delta))

def _emit_track_events(mt: mido.MidiTrack, notes: Iterable[NoteEvent], ccs: Iterable[CCEvent]):
    """Schreibt Note/CC-Events (ohne Tempo/TS) als delta-times in einen Track."""
    evs = []
    for n in notes:
        evs.append((n.start_tick, 1, ("on", n)))
        evs.append((n.end_tick,   0, ("off", n)))  # Off zuerst bei gleichem Tick
    for c in ccs:
        evs.append((c.tick, 2, ("cc", c)))
    evs.sort(key=lambda x: (x[0], x[1]))

    last = 0
    for tick, _, payload in evs:
        delta = tick - last
        last = tick
        kind = payload[0]
        if kind == "on":
            n = payload[1]
            mt.append(mido.Message("note_on", note=n.midi, velocity=n.velocity, channel=n.channel, time=delta))
        elif kind == "off":
            n = payload[1]
            mt.append(mido.Message("note_off", note=n.midi, velocity=0, channel=n.channel, time=delta))
        elif kind == "cc":
            c = payload[1]
            mt.append(mido.Message("control_change", control=c.cc, value=c.value, time=delta))

def _sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\s\-\.\(\)\[\]]+", "_", name.strip())
    name = re.sub(r"\s+", " ", name)
    return name or "Part"

# ---------- öffentliche Writer-APIs ----------

def write_midi_combined(bundle: TimelineBundle, out_path: str):
    """
    (Bestehendes Verhalten) – Eine Datei mit Conductor-Track (Tempo/TS) + allen Parts.
    """
    mid = mido.MidiFile(ticks_per_beat=bundle.ticks_per_beat)

    # Conductor
    t_con = mido.MidiTrack()
    t_con.append(mido.MetaMessage("track_name", name="Conductor", time=0))
    _emit_conductor(t_con, bundle.conductor.tempos, bundle.conductor.timesigs)
    mid.tracks.append(t_con)

    # Parts
    for name, tr in bundle.tracks.items():
        mt = mido.MidiTrack()
        mt.append(mido.MetaMessage("track_name", name=name, time=0))
        _emit_track_events(mt, tr.notes, tr.ccs)
        mid.tracks.append(mt)

    mid.save(out_path)

def write_conductor_only(bundle: TimelineBundle, out_path: str):
    """
    Nur Conductor: Tempo/TS in einer separaten MIDI (kein Notentrack).
    """
    mid = mido.MidiFile(ticks_per_beat=bundle.ticks_per_beat)
    t_con = mido.MidiTrack()
    t_con.append(mido.MetaMessage("track_name", name="Conductor", time=0))
    _emit_conductor(t_con, bundle.conductor.tempos, bundle.conductor.timesigs)
    mid.tracks.append(t_con)
    mid.save(out_path)

def write_parts_separately(
    bundle: TimelineBundle,
    out_dir: str,
    template: str = "{index:02d}-{name}.mid",
    omit_track_meta: bool = False,
):
    """
    Schreibt pro Part eine eigene MIDI **ohne** Tempo/TimeSig.
    - out_dir: Ausgabeverzeichnis (wird angelegt).
    - template: Dateinamen-Template; Platzhalter: {index}, {name}
    - omit_track_meta: wenn True, keine track_name MetaMessage (manche Hosts ignorieren sie)
    """
    os.makedirs(out_dir, exist_ok=True)
    for idx, (name, tr) in enumerate(bundle.tracks.items(), start=1):
        fname = template.format(index=idx, name=_sanitize_filename(name))
        path = os.path.join(out_dir, fname)
        mid = mido.MidiFile(ticks_per_beat=bundle.ticks_per_beat)
        mt = mido.MidiTrack()
        if not omit_track_meta:
            mt.append(mido.MetaMessage("track_name", name=name, time=0))
        _emit_track_events(mt, tr.notes, tr.ccs)
        mid.tracks.append(mt)
        mid.save(path)

# Rückwärtskompatibilität: alter Name beibehalten
def write_midi(bundle: TimelineBundle, out_path: str):
    write_midi_combined(bundle, out_path)