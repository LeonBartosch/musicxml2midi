# src/musicxml2midi/interpretation.py
from typing import List, Tuple
import numpy as np
from musicxml2midi.gui.models import MidiSong, Note
import hashlib

ORDER = ["ppp","pp","p","mp","mf","f","ff","fff"]
LEVEL = {m: i/(len(ORDER)-1) for i,m in enumerate(ORDER)}  # linear 0..1

def _u01_from_int(i: int) -> float:
    # deterministische Uniform[0,1)
    h = hashlib.blake2b(str(i).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(2**64)

def _rand_for_tag(seed: int, tag: tuple) -> float:
    """
    Deterministische Uniform[0,1) pro (seed, tag).
    Achtung: Python's hash() ist per-Process gesalzen → nicht verwenden.
    """
    h = hashlib.blake2b(repr((int(seed), tuple(tag))).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(2**64)

def _jitter_add(base: float, span: float, r: float, lo: float=None, hi: float=None) -> float:
    # additiver jitter: base +/- span (r in [0,1))
    out = base + (2*r - 1.0) * span
    if lo is not None: out = max(lo, out)
    if hi is not None: out = min(hi, out)
    return out

def _jitter_mul(base: float, rel: float, r: float, lo: float=None, hi: float=None) -> float:
    # multiplikativ: base * (1 +/- rel)
    fac = 1.0 + (2*r - 1.0) * rel
    out = base * fac
    if lo is not None: out = max(lo, out)
    if hi is not None: out = min(hi, out)
    return out

def _mark_to_level(mark: str) -> float:
    m = (mark or "").lower().strip()
    if m.startswith("subito "): m = m.split(" ", 1)[1]
    return LEVEL.get(m, LEVEL["mf"])

def _nearest_mark_from_level(x: float) -> str:
    vals = [LEVEL[m] for m in ORDER]
    mids = [(vals[i]+vals[i+1])*0.5 for i in range(len(vals)-1)]
    for i, mid in enumerate(mids):
        if x <= mid: return ORDER[i]
    return ORDER[-1]

def _step_mark(cur: str, direction: str) -> str:
    try: i = ORDER.index(cur)
    except ValueError: i = ORDER.index("mf")
    if (direction or "").startswith("cres"): i = min(i+1, len(ORDER)-1)
    else:                                     i = max(i-1, 0)
    return ORDER[i]

def _first_mark_at_or_after(dyns: List[Tuple[int,str]], tick: int) -> int:
    lo, hi = 0, len(dyns)
    while lo < hi:
        mid = (lo+hi)//2
        if dyns[mid][0] < tick: lo = mid+1
        else: hi = mid
    return lo

def _ease_s(start: float, end: float, s: np.ndarray, alpha: float=2.0) -> np.ndarray:
    s2 = np.power(s, alpha)
    sE = s2 / (s2 + np.power(1-s, alpha) + 1e-12)
    return start + (end-start)*sE

def _sample_L_at_tick(ticks: np.ndarray, L: np.ndarray, t_tick: int) -> float:
    """Interpoliert L(t) an einem Ticks-Wert t_tick (0..1)."""
    return float(np.interp(float(t_tick), ticks.astype(float), L.astype(float)))

def _sample_L_mean(ticks: np.ndarray, L: np.ndarray, t0: int, t1: int) -> float:
    """Mittelt L über [t0, t1) — fallback auf Einzel-Sample, wenn zu kurz."""
    i0 = int(np.searchsorted(ticks, t0, side="left"))
    i1 = int(np.searchsorted(ticks, t1, side="right"))
    if i1 <= i0:
        return _sample_L_at_tick(ticks, L, t0)
    return float(np.clip(np.mean(L[i0:i1]), 0.0, 1.0))

def _phrase_params_from_dyn(d: float) -> tuple[float, float, float]:
    """
    Mappe Dynamiklevel d (0..1) auf (peak_pos, sharpness, tail_exp)
    für Phrasen: laut = Peak weiter links, schärfer, bauchiger.
    """
    peak_pos  = 0.50 - 0.49 * d     # ppp≈0.50 … fff≈0.15
    sharpness = 0.80 + 1.20 * d     # ppp≈0.8  … fff≈2.0
    tail_exp  = 1.80 + 2.70 * d     # ppp≈1.8  … fff≈3.5
    return float(np.clip(peak_pos, 0.01, 0.99)), float(max(0.2, sharpness)), float(max(0.5, tail_exp))

def _note_params_from_dyn(d: float) -> tuple[float, float, float]:
    """
    Mappe Dynamiklevel d (0..1) auf (peak_pos, sharpness, tail_exp)
    für Einzelnoten: etwas milder als Phrasen.
    """
    peak_pos  = 0.50 - 0.45 * d     # ppp≈0.48 … fff≈0.23
    sharpness = 1.00 + 0.80 * d     # ppp≈1.0  … fff≈1.8
    tail_exp  = 1.30 + 0.90 * d     # ppp≈1.3  … fff≈2.2
    return float(np.clip(peak_pos, 0.01, 0.99)), float(max(0.2, sharpness)), float(max(0.5, tail_exp))

def _segment_baseline(seg_len: int, start: float, end: float) -> np.ndarray:
    if seg_len <= 0:
        return np.array([], dtype=float)
    r = np.linspace(0.0, 1.0, seg_len)
    return (1.0 - r) * float(start) + r * float(end)

def _jitter01(val: float, span: float, r: float) -> float:
    # val +/- span, dann sauber in [0,1] einklemmen
    return float(np.clip(val + (2*r - 1.0) * span, 0.0, 1.0))

def shaped_bell(n: int,
                peak_pos: float = 0.5,
                sharpness: float = 1.0,
                tail_exp: float = 1.0) -> np.ndarray:
    """
    Asymmetrische Glockenkurve 0→1→0 mit separater Kontrolle:
      - peak_pos:   0..1, Position des Maximums (0.5 = symmetrisch, →0 = Peak nach links)
      - sharpness:  >1 = schärfer (steilere Flanken), <1 = flacher
      - tail_exp:   >1 = 'dickbauchig' (rechter Abfall verzögert, dann schneller)
                     1 = normal, 2..4 = zunehmend bauchig
    """
    if n <= 1:
        return np.ones(max(1, n))

    peak_pos = float(np.clip(peak_pos, 1e-6, 1-1e-6))
    sharpness = max(1e-6, float(sharpness))
    tail_exp  = max(1e-6, float(tail_exp))

    x = np.linspace(0.0, 1.0, n)
    env = np.zeros_like(x)

    # Linke Seite (0 .. peak_pos): 0→1 mit schärfe-kontrolliertem Anstieg
    left_mask = x <= peak_pos
    if np.any(left_mask):
        xl = x[left_mask] / peak_pos                     # 0..1
        prog_l = xl ** (1.0 / sharpness)                 # sharpness>1 -> schneller hoch
        env[left_mask] = np.sin(0.5 * np.pi * np.clip(prog_l, 0, 1))

    # Rechte Seite (peak_pos .. 1): 1→0 mit 'Bauch'
    right_mask = ~left_mask
    if np.any(right_mask):
        # u: Fortschritt von Peak (0) bis Ende (1)
        u = (x[right_mask] - peak_pos) / (1.0 - peak_pos)
        prog_r = np.clip(u, 0, 1) ** (tail_exp / sharpness)  # tail_exp>1 -> länger oben, dann schneller runter
        env[right_mask] = np.cos(0.5 * np.pi * prog_r)

    m = env.max()
    if m > 0:
        env /= m
    return env

def build_L_curve(track,
                  dyn_events: List[Tuple[int, str]],
                  wedge_spans: List[Tuple[int,int,str]],
                  tpb: int,
                  step_ticks: int = 10,
                  target_tol_beats: float = 0.75,
                  alpha: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Linear Dry Dynamics: Marken = Plateaus, Hairpins = absolute Rampen, danach halten."""
    # Ende ermitteln
    max_tick = 0
    for n in getattr(track, "notes", []):
        end_tick = getattr(n, "end_tick", getattr(n, "end", 0))
        if isinstance(end_tick, (int,float)): max_tick = max(max_tick, int(end_tick))
    if wedge_spans: max_tick = max(max_tick, max(b for _,b,_ in wedge_spans))
    if dyn_events:  max_tick = max(max_tick, max(t for t,_ in dyn_events))
    if max_tick <= 0: max_tick = tpb

    tgrid = np.arange(0, max_tick+1, max(1, int(step_ticks)), dtype=int)

    # 1) Baseline = Plateaus aus Marken (nie zurückspringen)
    dyns = sorted(dyn_events or [(0, "mf")], key=lambda x: x[0])
    L = np.empty_like(tgrid, dtype=float)
    cur = _mark_to_level(dyns[0][1]); L[:] = cur
    for tick, mark in dyns[1:]:
        pos = int(np.searchsorted(tgrid, tick, side="left"))
        cur = _mark_to_level(mark)
        L[pos:] = cur

    # 2) Hairpins: absolute Ramp + halten bis nächste Marke
    tol_ticks = int(round(target_tol_beats * tpb))
    for a, b, kind in sorted(wedge_spans or [], key=lambda x: x[0]):
        if b <= a: continue
        i0 = int(np.searchsorted(tgrid, a, side="left"))
        i1 = int(np.searchsorted(tgrid, b, side="right")) - 1
        if i1 <= i0: continue

        L_start = float(L[i0])
        # Zielmarke nahe Ende?
        j = _first_mark_at_or_after(dyns, b)
        target = None
        if j < len(dyns) and dyns[j][0] <= b + tol_ticks:
            target = _mark_to_level(dyns[j][1])
        # sonst: exakt 1 Stufe rauf/runter
        if target is None:
            cur_mark = _nearest_mark_from_level(L_start)
            target   = LEVEL[_step_mark(cur_mark, (kind or "").lower())]

        seg = tgrid[i0:i1+1]
        s = (seg - seg[0]) / max(1, (seg[-1]-seg[0]))
        L[i0:i1+1] = _ease_s(L_start, target, s, alpha=alpha)

        # halten bis zur nächsten Marke
        stop_tick = dyns[j][0] if j < len(dyns) else (tgrid[-1]+1)
        stop_idx  = int(np.searchsorted(tgrid, stop_tick, side="left"))
        if stop_idx > i1+1:
            L[i1+1:stop_idx] = target

    return tgrid, np.clip(L, 0.0, 1.0)

def note_envelope(duration: float,
                  dyn_level: float,
                  t_res: int = 128,
                  attack_min: float = 0.05,
                  attack_max: float = 0.4) -> np.ndarray:
    """
    Erzeugt eine Note-Hüllkurve (0..1), abhängig von Dauer und Dynamik.
    
    duration: Sekunden (oder normierte Einheiten)
    dyn_level: 0..1 (ppp=0, fff=1)
    t_res: Anzahl Samples
    attack_min/max: relative Attack-Länge bei ff/pp
    """
    if duration <= 0:
        return np.ones(1)

    # Attack-Länge abhängig von Dynamik (ff = schnell, pp = langsam)
    att_frac = attack_max - dyn_level * (attack_max - attack_min)
    att_len = int(max(1, round(t_res * att_frac)))
    dec_len = max(1, t_res - att_len)

    # Attack = Sigmoid
    s = np.linspace(-3, 3, att_len)
    attack = 1 / (1 + np.exp(-s))   # 0→1

    # Decay = linear oder exp
    decay = np.linspace(1, 0.7, dec_len)  # etwas abschwellen

    env = np.concatenate([attack, decay])
    env = env / env.max()  # normieren auf 1
    return env

def build_note_curve(
    song: MidiSong,
    ticks: np.ndarray,
    L: np.ndarray,
    tpb: int,
    humanize_cfg: dict | None = None,
) -> np.ndarray:
    """
    'Phrase-artig', aber pro Note:
      - Werte nur im Notensegment setzen, Rest NaN lassen
      - Überlappungen: hartes Maximum
      - Danach Forward-Fill → letzten Wert halten (kein Reset)
    """
    hcfg  = (humanize_cfg or {}).get("notes", {}) or {}
    hseed = int((humanize_cfg or {}).get("seed", 0))

    h_enabled = bool(hcfg.get("enabled", False))
    j_pos   = float(hcfg.get("peak_pos_jitter", 0.0))
    j_amp   = float(hcfg.get("amp_jitter", 0.0))
    j_sharp = float(hcfg.get("sharpness_jitter", 0.0))
    j_tail  = float(hcfg.get("tail_exp_jitter", 0.0))

    bl       = (hcfg.get("baseline", {}) or {})
    bl_start = float(bl.get("start", 0.8))
    bl_end   = float(bl.get("end",   0.8))
    bl_js    = float(bl.get("start_jitter", 0.0))
    bl_je    = float(bl.get("end_jitter",   0.0))

    # Wichtig: dieser Wert dient NUR als Startwert fürs Forward-Fill,
    # nicht als Füllwert VOR dem Forward-Fill.
    outside_base = float(hcfg.get("outside_baseline", 1.0))

    # 1) NaN-Leinwand wie bei build_phrase_curve
    N = np.full_like(ticks, np.nan, dtype=float)

    for n in getattr(song, "notes", []):
        s_tick = getattr(n, "start_tick", None)
        e_tick = getattr(n, "end_tick",   None)
        if s_tick is None or e_tick is None or e_tick <= s_tick:
            continue

        i0 = int(np.searchsorted(ticks, s_tick, side="left"))
        i1 = int(np.searchsorted(ticks, e_tick, side="right"))
        if i1 <= i0:
            continue

        seg_len = i1 - i0

        # Formparameter aus L (frühes Drittel), note-spezifische Mappings
        t_mid = s_tick + max(1, (e_tick - s_tick)//3)
        dyn = _sample_L_mean(ticks, L, s_tick, t_mid)
        peak_pos, sharpness, tail_exp = _note_params_from_dyn(dyn)

        # Artikulationen
        arts = (n.articulations or [])
        if "accent" in arts or "marcato" in arts:
            peak_pos  = max(0.01, peak_pos - 0.08)
            sharpness *= 1.25
        if "staccato" in arts:
            tail_exp  = max(1.0, tail_exp * 0.75)
        if "tenuto" in arts:
            tail_exp  *= 1.15

        # Humanize
        amp = 1.0
        s0, e0 = bl_start, bl_end
        if h_enabled:
            tag  = ("note", int(getattr(n, "pitch", 0)), int(s_tick))
            r_pos = _rand_for_tag(hseed, tag + ("pos",))
            r_amp = _rand_for_tag(hseed, tag + ("amp",))
            r_sh  = _rand_for_tag(hseed, tag + ("sharp",))
            r_te  = _rand_for_tag(hseed, tag + ("tail",))
            r_bs  = _rand_for_tag(hseed, tag + ("bstart",))
            r_be  = _rand_for_tag(hseed, tag + ("bend",))

            peak_pos  = _jitter_add(peak_pos, j_pos, r_pos, lo=0.01, hi=0.99)
            amp       = _jitter_mul(1.0, j_amp, r_amp, lo=0.7, hi=1.5)
            sharpness = _jitter_mul(sharpness, j_sharp, r_sh, lo=0.2, hi=5.0)
            tail_exp  = _jitter_mul(tail_exp, j_tail,  r_te,  lo=0.5, hi=6.0)
            s0        = _jitter01(s0, bl_js, r_bs)
            e0        = _jitter01(e0, bl_je, r_be)

        env  = shaped_bell(seg_len, peak_pos=peak_pos, sharpness=sharpness, tail_exp=tail_exp)
        base = _segment_baseline(seg_len, s0, e0)

        # Spitze normiert um 1.0 (± Humanize), innerhalb der Note wieder runter auf e0
        peak = 1.0 * amp
        seg_curve = base + (peak - base) * env  # beginnt ~s0, steigt zu ~1, fällt zu ~e0

        # Schreiben wie bei Phrase: einfach „oben drauf“ (hartes Maximum)
        if np.all(np.isnan(N[i0:i1])):
            N[i0:i1] = seg_curve
        else:
            N[i0:i1] = np.fmax(N[i0:i1], seg_curve)

    # Falls keine Note: Outside-Baseline
    if np.all(np.isnan(N)):
        return np.ones_like(ticks, dtype=float) * outside_base

    # 2) Forward-Fill genau wie bei build_phrase_curve → letzten Wert halten
    out = np.copy(N)
    last = outside_base
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]

    return np.clip(out, 0.0, None)

# NEU: Phrasen aus MidiSong (Onset-basiert, Slur mit vorheriger Note)
def extract_phrases_from_song(song: MidiSong) -> List[Tuple[int, int]]:
    """
    Liefert Phrasen als (start_tick, end_tick) aus dem Analysis-Song.
    Regel:
      - Ein Onset OHNE 'slur' startet eine neue Phrase.
      - Alle folgenden Onsets MIT 'slur' gehören zu dieser Phrase.
      - Trifft wieder ein Onset OHNE 'slur' ein, wird die vorige Phrase geschlossen und eine neue gestartet.
    Hinweis: Onset = alle Noten mit gleichem start_tick (Akkord).
    """
    notes = sorted(getattr(song, "notes", []), key=lambda n: getattr(n, "start_tick", 0))
    if not notes:
        return []

    # Onsets bauen
    onsets: List[Tuple[int, List[Note]]] = []
    cur_tick = None
    cur_group: List[Note] = []
    for n in notes:
        st = getattr(n, "start_tick", None)
        if st is None:
            continue
        if cur_tick is None or st != cur_tick:
            if cur_group:
                onsets.append((cur_tick, cur_group))
            cur_tick = st
            cur_group = [n]
        else:
            cur_group.append(n)
    if cur_group:
        onsets.append((cur_tick, cur_group))

    phrases: List[Tuple[int, int]] = []
    phrase_start: int | None = None
    prev_end_of_group: int = 0

    for onset_tick, group in onsets:
        # Onset „hat slur“, wenn irgendeine Note im Akkord 'slur' trägt
        onset_has_slur = any("slur" in (getattr(n, "articulations", []) or []) for n in group)
        group_end = max(getattr(n, "end_tick", onset_tick) or onset_tick for n in group)

        if not onset_has_slur:
            # neuer Start
            if phrase_start is not None:
                # schließe alte Phrase am Ende des VORigen Onsets
                phrases.append((phrase_start, prev_end_of_group))
            phrase_start = onset_tick
        # sonst: wir sind innerhalb derselben Phrase und laufen einfach weiter

        prev_end_of_group = group_end

    # letzte offene Phrase schließen
    if phrase_start is not None and prev_end_of_group > phrase_start:
        phrases.append((phrase_start, prev_end_of_group))

    return phrases

def build_phrase_curve(song: MidiSong, ticks: np.ndarray, L: np.ndarray, tpb: int,
                       humanize_cfg: dict | None = None,
                       depth: float = 1.0) -> np.ndarray:
    # statt Nullen: NaN → danach vorwärts auffüllen
    P = np.full_like(ticks, np.nan, dtype=float)
    phrases = extract_phrases_from_song(song)
    if not phrases:
        # keine Phrasen → alles 0 (oder falls du lieber 1 willst: np.ones_like(ticks))
        return np.zeros_like(ticks, dtype=float)

    hseed = int((humanize_cfg or {}).get("seed", 0))
    hcfg  = (humanize_cfg or {}).get("phrases", {}) or {}
    h_enabled = bool(hcfg.get("enabled", False))
    j_pos   = float(hcfg.get("peak_pos_jitter", 0.0))
    j_amp   = float(hcfg.get("amp_jitter", 0.0))
    j_sharp = float(hcfg.get("sharpness_jitter", 0.0))
    j_tail  = float(hcfg.get("tail_exp_jitter", 0.0))
    max_boost = float(hcfg.get("max_boost", 0.35))
    bl      = (hcfg.get("baseline", {}) or {})
    bl_start = float(bl.get("start", 0.0))
    bl_end   = float(bl.get("end",   0.0))
    bl_js    = float(bl.get("start_jitter", 0.0))
    bl_je    = float(bl.get("end_jitter",   0.0))

    for s_tick, e_tick in phrases:
        i0 = int(np.searchsorted(ticks, s_tick, side="left"))
        i1 = int(np.searchsorted(ticks, e_tick, side="right"))
        if i1 <= i0:
            continue

        seg_len = i1 - i0
        dyn = _sample_L_at_tick(ticks, L, s_tick)
        peak_pos, sharpness, tail_exp = _phrase_params_from_dyn(dyn)

        # Humanize: Parameter + Baseline
        amp = 1.0
        s0, e0 = bl_start, bl_end
        if h_enabled:
            tag = ("phrase", int(s_tick), int(e_tick))
            r_pos = _rand_for_tag(hseed, tag + ("pos",))
            r_amp = _rand_for_tag(hseed, tag + ("amp",))
            r_sh  = _rand_for_tag(hseed, tag + ("sharp",))
            r_te  = _rand_for_tag(hseed, tag + ("tail",))
            r_bs  = _rand_for_tag(hseed, tag + ("bstart",))
            r_be  = _rand_for_tag(hseed, tag + ("bend",))

            peak_pos = _jitter_add(peak_pos, j_pos, r_pos, lo=0.01, hi=0.99)
            amp      = _jitter_mul(1.0, j_amp, r_amp, lo=0.7, hi=1.8)
            sharpness= _jitter_mul(sharpness, j_sharp, r_sh, lo=0.2, hi=5.0)
            tail_exp = _jitter_mul(tail_exp, j_tail, r_te, lo=0.5, hi=6.0)
            s0 = _jitter01(s0, bl_js, r_bs)
            e0 = _jitter01(e0, bl_je, r_be)

        env  = shaped_bell(seg_len, peak_pos=peak_pos, sharpness=sharpness, tail_exp=tail_exp)
        base = _segment_baseline(seg_len, s0, e0)

        peak = base + depth * (1.0 - base)
        peak = base + amp * (peak - base)
        if max_boost > 0.0:
            peak = np.minimum(peak, 1.0 + max_boost)

        seg_curve = base + (peak - base) * env

        # Nur in den Phrasenbereich schreiben; Rest bleibt NaN
        P[i0:i1] = seg_curve

    # --- Forward-Fill: zwischen Phrasen den letzten Wert halten ---
    out = np.copy(P)
    last = 0.0  # vor der ersten Phrase bleibt 0.0; wenn du 1.0 möchtest, setze hier 1.0
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]

    return out

def articulation_envelope(
    arts: List[str],
    seg_len: int,
    base_level: float = 1.0,
    depth_accent: float = 0.25,
    depth_marcato: float = 0.35,
    depth_stacc: float = 0.30,
    depth_tenuto: float = 0.15,
) -> np.ndarray:
    """
    Zeitlich geformte Artikulationshüllkurve (multiplikativ, startet/endet bei 1).
    - accent/marcato: frühe Glocke (Impuls)
    - tenuto: rechte-schiefe Glocke (dickbauchig), KEIN Plateau
    - staccato: weicher Right-Taper (Kosinus)
    Articulationen multiplizieren sich sinnvoll.
    """
    if seg_len <= 1:
        return np.ones(max(1, seg_len), dtype=float) * base_level

    env = np.ones(seg_len, dtype=float) * base_level

    # Helper: add a bell-shaped boost (0..1..0) as a multiplicative factor
    def boost_bell(peak_pos: float, sharpness: float, tail_exp: float, depth: float) -> np.ndarray:
        bell = shaped_bell(seg_len, peak_pos=peak_pos, sharpness=sharpness, tail_exp=tail_exp)
        # -> multipliziert 1..(1+depth)..1
        return 1.0 + depth * bell

    # accent: kurzer, knackiger Impuls früh
    if "accent" in arts:
        env *= boost_bell(peak_pos=0.12, sharpness=2.2, tail_exp=1.1, depth=depth_accent)

    # marcato: etwas länger & kräftiger als accent
    if "marcato" in arts:
        env *= boost_bell(peak_pos=0.18, sharpness=2.5, tail_exp=1.2, depth=depth_marcato)

    # tenuto: rechte-schiefe, dickbauchige Glocke (kein Plateau)
    # Peak später, sanfter Anstieg, verzögerter Abfall → wirkt dem natürlichen Decay entgegen
    if "tenuto" in arts:
        env *= boost_bell(peak_pos=0.75, sharpness=0.85, tail_exp=1.6, depth=depth_tenuto)

    # staccato: weiches, spates Absenken < 1 (gegen Ende)
    if "staccato" in arts:
        x = np.linspace(0.0, 1.0, seg_len)
        start = 0.55  # ab hier beginnt das „Kurzhalten“-Gefühl
        u = np.clip((x - start) / (1.0 - start), 0.0, 1.0)
        # Kosinus-Taper: 1 → (1 - depth_stacc) ganz am Ende, ohne Ecken
        taper = 1.0 - depth_stacc * (0.5 - 0.5 * np.cos(np.pi * u))
        env *= taper

    return env

def build_articulation_curve(song: MidiSong, ticks: np.ndarray, tpb: int) -> np.ndarray:
    """
    A-Kurve: Artikulationshüllkurven pro Note, mit zeitlichen Formen (Impulse, Sustain).
    """
    A = np.ones_like(ticks, dtype=float)

    for n in song.notes:
        s_tick = getattr(n, "start_tick", None)
        e_tick = getattr(n, "end_tick", None)
        if s_tick is None or e_tick is None or e_tick <= s_tick:
            continue

        i0 = int(np.searchsorted(ticks, s_tick, side="left"))
        i1 = int(np.searchsorted(ticks, e_tick, side="right"))
        if i1 <= i0:
            continue

        seg_len = i1 - i0
        env = articulation_envelope(n.articulations or [], seg_len)

        A[i0:i1] *= env

    return A

def build_pitch_curve(
    song: MidiSong,
    ticks: np.ndarray,
    tpb: int,
    depth: float = 0.15,
    smooth_window: int = 20  # leichte Glättung (>=1)
) -> np.ndarray:
    """
    H-Kurve (Pitch-Height): center_pitch = Mitte der vorkommenden Pitch-Range.
    - Zwischen Noten wird linear interpoliert (kein Reset auf 1).
    - Optional leichte Glättung gegen Kanten.
    """
    H = np.ones_like(ticks, dtype=float)

    notes = [
        n for n in getattr(song, "notes", [])
        if getattr(n, "start_tick", None) is not None
        and getattr(n, "end_tick",   None) is not None
        and n.end_tick > n.start_tick
    ]
    if not notes:
        return H

    # ---- 1) center_pitch als Mitte der Range bestimmen ----
    min_pitch = float(min(n.pitch for n in notes))
    max_pitch = float(max(n.pitch for n in notes))
    if max_pitch == min_pitch:
        center_pitch = min_pitch  # alle Noten gleich
    else:
        center_pitch = 0.5 * (min_pitch + max_pitch)

    # ---- 2) pro-Tick Werte mit NaNs füllen (zwischen Noten) ----
    vals = np.full_like(ticks, np.nan, dtype=float)

    def _merge(old, new):
        if np.isnan(old): 
            return new
        # Geometrisches Mittel bei Überlappungen
        return float(np.sqrt(max(1e-12, old) * max(1e-12, new)))

    lo, hi = 1.0 - depth, 1.0 + depth

    for n in notes:
        i0 = int(np.searchsorted(ticks, n.start_tick, side="left"))
        i1 = int(np.searchsorted(ticks, n.end_tick,   side="right"))
        if i1 <= i0:
            continue

        diff = float(n.pitch) - center_pitch
        factor = 1.0 + depth * (diff / 12.0)
        factor = max(lo, min(hi, factor))  # clamp

        seg = vals[i0:i1]
        if np.all(np.isnan(seg)):
            vals[i0:i1] = factor
        else:
            mask = np.isnan(seg)
            seg[mask]  = factor
            seg[~mask] = np.array([_merge(o, factor) for o in seg[~mask]])
            vals[i0:i1] = seg

    # Falls alles NaN (sollte nicht passieren) → 1
    if np.all(np.isnan(vals)):
        return H

    # ---- 3) Lücken linear auffüllen ----
    idx = np.where(~np.isnan(vals))[0]
    first, last = idx[0], idx[-1]
    vals[:first] = vals[first]
    vals[last+1:] = vals[last]
    x_known = ticks[idx].astype(float)
    y_known = vals[idx].astype(float)
    x_all   = ticks[first:last+1].astype(float)
    vals[first:last+1] = np.interp(x_all, x_known, y_known)

    # ---- 4) Optionale Glättung ----
    if smooth_window and smooth_window > 1:
        k = int(smooth_window)
        if k % 2 == 0: 
            k += 1  # ungerade
        pad = k // 2
        kernel = np.ones(k, dtype=float) / k
        vals = np.convolve(np.pad(vals, (pad, pad), mode="edge"), kernel, mode="valid")

    return np.clip(vals, lo, hi)

def build_random_noise_curve(
    ticks: np.ndarray,
    seed: int,
    x_scale: float = 0.4,   # Knotendistanz (Grain) in Beats
    y_depth: float = 0.08,  # ±Amplitude um 1.0
    tpb: int | None = None, # für Beats→Samples
    post_smooth_beats: float = 0.0,  # kleines optionales Nachglätten (0..0.25)
    interp: str = "quintic" # "linear" | "cosine" | "quintic"
) -> np.ndarray:
    """
    Lokales, aber glattes Jittern:
    - Alle 'x_scale' Beats wird ein Zufallswert gezogen (N(0,1)) → Knoten.
    - Dazwischen weiche Interpolation (quintic/cosine/linear).
    - Optional minimales Nachglätten über ein kurzes Beat-Fenster.

    Rückgabe: Kurve ≥ 0, um 1.0 herum (1 ± y_depth).
    """
    n = int(len(ticks))
    if n <= 2 or y_depth <= 0.0:
        return np.ones(n, dtype=float)

    rng = np.random.default_rng(int(seed))

    # --- Grain von Beats → Samples ---
    dt_ticks = float(np.median(np.diff(ticks))) if n > 2 else 1.0
    if dt_ticks <= 0.0:
        dt_ticks = 1.0

    if tpb and tpb > 0:
        samples_per_beat = max(1, int(round(tpb / dt_ticks)))
        knot_every = max(1, int(round(samples_per_beat * max(1e-3, float(x_scale)))))
    else:
        # Fallback ohne tpb: relativ zum Array
        base = max(8, n // 64)
        knot_every = max(1, int(round(base * (x_scale / 0.4))))

    # --- Knoten aufspannen ---
    knot_idx = np.arange(0, n, knot_every, dtype=int)
    if knot_idx[-1] != n - 1:
        knot_idx = np.append(knot_idx, n - 1)
    knot_vals = rng.standard_normal(len(knot_idx))

    # --- Interpolationsvorbereitung ---
    # Segmentindex pro Sample und lokale Phase t in [0,1)
    seg_id = np.minimum((np.arange(n) // knot_every), len(knot_idx) - 2).astype(int)
    seg_off = np.arange(n) - seg_id * knot_every
    seg_len = np.minimum(knot_every, (knot_idx[seg_id + 1] - knot_idx[seg_id]))
    t = np.clip(seg_off / np.maximum(1, seg_len), 0.0, 1.0)

    if interp == "cosine":
        # Cosine fade: 0..1 weich
        s = 0.5 * (1 - np.cos(np.pi * t))
    elif interp == "linear":
        s = t
    else:
        # Quintic smoothstep: 6t^5 - 15t^4 + 10t^3 (sehr smooth, C2)
        s = t * t * t * (t * (6*t - 15) + 10)

    v0 = knot_vals[seg_id]
    v1 = knot_vals[seg_id + 1]
    r_full = (1.0 - s) * v0 + s * v1

    # --- (Optional) sehr leichtes Nachglätten über kurzes Beat-Fenster ---
    if post_smooth_beats and tpb and tpb > 0:
        win = int(round(max(1e-3, float(post_smooth_beats)) * samples_per_beat))
        if win > 1:
            # symmetrisches gleitendes Mittel, kantenbewusst
            k = np.ones(win, dtype=float) / float(win)
            pad = win // 2
            r_full = np.convolve(np.pad(r_full, (pad, pad), mode="edge"), k, mode="valid")

    # --- Normalisieren & um 1.0 herum legen ---
    mu  = float(np.mean(r_full))
    std = float(np.std(r_full)) or 1.0
    r_full = (r_full - mu) / std

    return np.clip(1.0 + y_depth * r_full, 0.0, None)

def _soft_clip(x, knee=0.15):
    y = np.copy(x)
    above = x > 1.0
    y[above] = 1.0 + (1.0 - np.exp(-(x[above] - 1.0)/max(1e-6, knee))) * knee
    # unterhalb 0 hart clippen (vereinfacht)
    y[x < 0.0] = 0.0
    return y

def compose_total_curve(L, P, N, A, H, R) -> np.ndarray:
    return _soft_clip(L * P * N * A * H * R, knee=0.2)