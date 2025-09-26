from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import copy, yaml

def _deep_merge(a: Dict, b: Dict) -> Dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

class ConfigManager:
    def __init__(self, default_path: Path, user_path: Path):
        self.default_path = Path(default_path)
        self.user_path = Path(user_path)
        self.defaults: Dict[str, Any] = {}
        self.user: Dict[str, Any] = {}
        self.data: Dict[str, Any] = {}

    def load(self):
        self.defaults = yaml.safe_load(self.default_path.read_text(encoding="utf-8")) if self.default_path.exists() else {}
        self.user = yaml.safe_load(self.user_path.read_text(encoding="utf-8")) if self.user_path.exists() else {}
        self.data = _deep_merge(self.defaults, self.user)

    def replace(self, new_data: Dict[str, Any]):
        self.data = _deep_merge(self.defaults, new_data)

    def save_user(self):
        # speichert nur Abweichungen
        def diff(defs, cur):
            if not isinstance(defs, dict) or not isinstance(cur, dict):
                return copy.deepcopy(cur)
            out = {}
            for k, v in cur.items():
                if k not in defs:
                    out[k] = copy.deepcopy(v)
                else:
                    d = diff(defs[k], v)
                    if d != {} and d != defs[k]:
                        out[k] = d
            return out
        self.user = diff(self.defaults, self.data)
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_path.write_text(yaml.safe_dump(self.user, sort_keys=False), encoding="utf-8")