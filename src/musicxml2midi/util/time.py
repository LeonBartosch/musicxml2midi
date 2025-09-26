from __future__ import annotations

def div_to_ticks(div_val: int, divisions: int, tpb: int) -> int:
    if divisions <= 0:
        divisions = 480
    return int(round(div_val * (tpb / divisions)))
