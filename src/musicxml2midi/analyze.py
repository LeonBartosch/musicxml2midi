# src/musicxml2midi/analyze.py
from __future__ import annotations
from xml.etree import ElementTree as ET
from typing import Dict
from .timeline import ScoreAnalysis, PartAnalysis, NoteToken, DirectionEvent
from .util.xml import get_ns, F, FA

STEP_TO_SEMITONE = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
def midi_from_pitch(step: str, alter: int, octave: int) -> int:
    return (octave + 1) * 12 + STEP_TO_SEMITONE[step] + int(alter)

def _local(tag: str) -> str:
    return tag.split('}')[-1] if '}' in tag else tag

def analyze_musicxml(path: str) -> ScoreAnalysis:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = get_ns(root)

    # --- Part-Namen sammeln ---
    part_names: Dict[str,str] = {}
    pl = F(root, "part-list", ns)
    if pl is not None:
        for sp in FA(pl, "score-part", ns):
            pid = sp.attrib.get("id","P1")
            pn = F(sp, "part-name", ns)
            part_names[pid] = (pn.text.strip() if pn is not None and pn.text else pid)

    analysis = ScoreAnalysis()

    # --- Parts ---
    for part in FA(root, "part", ns):
        pid = part.attrib.get("id", "P1")
        pname = part_names.get(pid, pid)
        pa = PartAnalysis(part_id=pid, part_name=pname)

        # Absoluter Cursor in Divisions (über alle Takte)
        abs_div = 0
        current_div = 480

        # --- Measures sequentiell parsen ---
        for mi, meas in enumerate(FA(part, "measure", ns)):
            # Measure-lokaler Cursor (Start = aktueller abs_div)
            t = abs_div

            # Wir gehen die KIND-ELEMENTE des Measures IN REIHENFOLGE durch:
            for child in list(meas):
                tag = _local(child.tag)

                # ---------------- ATTRIBUTES (keine Zeitverschiebung) ----------------
                if tag == "attributes":
                    dv = F(child, "divisions", ns)
                    if dv is not None and dv.text:
                        current_div = int(dv.text)
                        pa.divisions_changes.append((t, current_div))  # an der aktuellen Zeit!
                    time_el = F(child, "time", ns)
                    if time_el is not None:
                        beats = F(time_el, "beats", ns)
                        bt    = F(time_el, "beat-type", ns)
                        if beats is not None and bt is not None:
                            pa.directions.append(DirectionEvent(
                                t, "timesig", {"num": int(beats.text), "den": int(bt.text)}
                            ))
                    continue

                # ---------------- FORWARD/BACKUP (Zeitcursor verschieben) ----------------
                if tag == "forward":
                    dur_el = F(child, "duration", ns)
                    if dur_el is not None and dur_el.text:
                        t += int(dur_el.text)
                    continue

                if tag == "backup":
                    dur_el = F(child, "duration", ns)
                    if dur_el is not None and dur_el.text:
                        t -= int(dur_el.text)
                        if t < abs_div:
                            t = abs_div  # Sicherheitsnetz
                    continue

                # ---------------- DIRECTION (Dynamics / Wedges / Tempo) ----------------
                if tag == "direction":
                    # optionale <offset> relativ zum aktuellen Cursor
                    off_el = F(child, "offset", ns)
                    t_evt = t + (int(off_el.text) if (off_el is not None and off_el.text) else 0)

                    # <sound tempo="...">
                    sound = child.find(".//sound")
                    if sound is not None and "tempo" in sound.attrib:
                        try:
                            pa.directions.append(DirectionEvent(t_evt, "tempo", {"bpm": float(sound.attrib["tempo"])}))
                        except ValueError:
                            pass

                    # direction-type: dynamics / wedge / (evtl. andere)
                    for dtyp in FA(child, "direction-type", ns):
                        dyn = F(dtyp, "dynamics", ns)
                        if dyn is not None:
                            for c in list(dyn):
                                mark = _local(c.tag)
                                pa.directions.append(DirectionEvent(t_evt, "dynamic", {"mark": mark}))
                                break
                        wedge = F(dtyp, "wedge", ns)
                        if wedge is not None:
                            wtype = wedge.attrib.get("type")
                            pa.directions.append(DirectionEvent(t_evt, "wedge", {"type": (wtype or "").strip()}))

                        metr = F(dtyp, "metronome", ns)
                        if metr is not None:
                            per_min = F(metr, "per-minute", ns)
                            if per_min is not None and per_min.text:
                                try:
                                    bpm = float(per_min.text.strip())
                                    pa.directions.append(DirectionEvent(t_evt, "tempo", {"bpm": bpm}))
                                except ValueError:
                                    pass
                    continue
                
                # ---------------- NOTES ----------------
                if tag == "note":
                    is_rest = F(child, "rest", ns) is not None
                    is_chord = F(child, "chord", ns) is not None  # True = gleiche Startzeit wie vorherige Note
                    voice_el = F(child, "voice", ns); voice = voice_el.text.strip() if voice_el is not None and voice_el.text else None
                    staff_el = F(child, "staff", ns); staff = staff_el.text.strip() if staff_el is not None and staff_el.text else None
                    dur_el = F(child, "duration", ns); dur = int(dur_el.text) if dur_el is not None and dur_el.text else 0

                    midi = None
                    if not is_rest:
                        pitch = F(child, "pitch", ns)
                        if pitch is not None:
                            step = F(pitch, "step", ns).text.strip()
                            alter_el = F(pitch, "alter", ns); alter = int(alter_el.text) if (alter_el is not None and alter_el.text) else 0
                            octave = int(F(pitch, "octave", ns).text)
                            midi = midi_from_pitch(step, alter, octave)

                    # ties (sowohl <tie> als auch <notations>/<tied>)
                    tie_start = False; tie_stop = False
                    for t_el in FA(child, "tie", ns):
                        ty = t_el.attrib.get("type")
                        if   ty == "start": tie_start = True
                        elif ty == "stop":  tie_stop  = True
                    for not_el in FA(child, "notations", ns):
                        for t_el in FA(not_el, "tied", ns):
                            ty = t_el.attrib.get("type")
                            if   ty == "start": tie_start = True
                            elif ty == "stop":  tie_stop  = True
                            elif ty == "continue": pass  # ignorieren

                    # slurs / articulations / ornaments / technical / tremolo
                    slur_starts = 0; slur_stops = 0; arts=[]; orns=[]
                    for not_el in FA(child, "notations", ns):
                        # <slur>
                        for sl in FA(not_el, "slur", ns):
                            ty = sl.attrib.get("type")
                            if   ty == "start": slur_starts += 1
                            elif ty == "stop":  slur_stops  += 1
                            elif ty == "continue": pass  # NICHT als start+stop zählen

                        # <articulations>
                        a = F(not_el, "articulations", ns)
                        if a is not None:
                            for c in list(a):
                                arts.append(_local(c.tag))  # staccato, tenuto, accent, marcato, ...

                        # <ornaments> (trill, mordent, turn, etc.)
                        o = F(not_el, "ornaments", ns)
                        if o is not None:
                            for c in list(o):
                                tag2 = _local(c.tag)
                                if tag2 == "tremolo":  # manche Programme schreiben Tremolo hierher
                                    orns.append("tremolo")
                                else:
                                    orns.append(tag2)

                        # <technical> (pizzicato, snap-pizzicato, arco, mute, harmonic, ...)
                        te = F(not_el, "technical", ns)
                        if te is not None:
                            for c in list(te):
                                tag2 = _local(c.tag)
                                if tag2 == "mute":
                                    val = (c.text or "").strip().lower() if c.text else ""
                                    orns.append(f"mute({val})" if val else "mute")
                                else:
                                    orns.append(tag2)

                        # eigenständiges <tremolo> direkt unter <notations>
                        for tr in FA(not_el, "tremolo", ns):
                            orns.append("tremolo")

                    # Note-Token anlegen (Start = aktueller Cursor t)
                    tok = NoteToken(
                        start_div=t,
                        duration_div=dur,
                        midi=midi,
                        voice=voice,
                        staff=staff,
                        tie_start=tie_start,
                        tie_stop=tie_stop,
                        slur_starts=slur_starts,
                        slur_stops=slur_stops,
                        articulations=arts,
                        ornaments=orns,
                        measure_idx=mi,
                    )
                    pa.notes.append(tok)

                    # Zeitcursor nur vorrücken, wenn NICHT chord
                    if dur and not is_chord:
                        t += dur

                    continue

                # Andere Measure-Kinder ignorieren wir vorerst bewusst (harmony, barline, print, etc.)

            # Measure-Ende → absoluten Cursor auf t setzen
            abs_div = t

        analysis.parts[pid] = pa

    return analysis