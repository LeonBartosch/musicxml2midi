from __future__ import annotations
import argparse, pathlib, sys, traceback
from . import analyze, process, write
from .config import load_config

def main(argv=None):
    p = argparse.ArgumentParser(description="MusicXML -> MIDI (Blueprint CLI)")
    p.add_argument("--in", dest="infile", required=True, help="Input MusicXML (.musicxml/.xml)")
    p.add_argument("--out", dest="outfile", required=False, help="Output MIDI file (.mid) – combined (default)")
    p.add_argument("--config", dest="config", default=None, help="YAML config (defaults applied if omitted)")

    # Neue Optionen für Split-Output
    p.add_argument("--no-combined", action="store_true", help="Do not write the combined MIDI file")
    p.add_argument("--conductor-out", dest="conductor_out", default=None, help="Write a conductor-only MIDI (tempo/time signatures)")
    p.add_argument("--parts-out-dir", dest="parts_out_dir", default=None, help="Write one MIDI per part into this directory (without tempo/timesig)")

    args = p.parse_args(argv)

    in_path = pathlib.Path(args.infile).expanduser().resolve()
    if not in_path.exists():
        print(f"[cli] ERROR: Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    print(f"[cli] infile = {in_path}")

    try:
        score = analyze.analyze_musicxml(str(in_path))
        bundle = process.build_timelines(score, cfg)
    except Exception:
        traceback.print_exc()
        sys.exit(2)

    wrote_anything = False

    # 1) Combined (Standard), falls nicht abgeschaltet
    if not args.no_combined:
        out_path = None
        if args.outfile:
            out_path = pathlib.Path(args.outfile).expanduser().resolve()
        else:
            # Default: gleiche Basis wie Input + .mid
            out_path = in_path.with_suffix(".mid")
        write.write_midi_combined(bundle, str(out_path))
        print(f"[cli] combined  -> {out_path}")
        wrote_anything = True

    # 2) Conductor separat
    if args.conductor_out:
        cond_path = pathlib.Path(args.conductor_out).expanduser().resolve()
        write.write_conductor_only(bundle, str(cond_path))
        print(f"[cli] conductor -> {cond_path}")
        wrote_anything = True

    # 3) Parts separat
    if args.parts_out_dir:
        out_dir = pathlib.Path(args.parts_out_dir).expanduser().resolve()
        write.write_parts_separately(bundle, str(out_dir))
        print(f"[cli] parts     -> {out_dir}")
        wrote_anything = True

    if not wrote_anything:
        print("[cli] WARNING: no output produced (use --out, --conductor-out, --parts-out-dir or omit --no-combined).")

    total_notes = sum(len(pa.notes) for pa in score.parts.values())
    print(f"[cli] Done. parts={len(score.parts)} notes={total_notes} tpb={bundle.ticks_per_beat}")