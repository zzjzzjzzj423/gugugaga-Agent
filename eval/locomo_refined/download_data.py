from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
BASE_URL = "https://raw.githubusercontent.com/jianglong-nie/MiniMem/main/benchmarks/locomo_refined/data"
FILES = ("conversations.jsonl", "questions.jsonl")


def main() -> int:
    target_dir = ROOT / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = target_dir / name
        if target.exists():
            print(f"Keeping existing {target}")
            continue
        with urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
            data = response.read()
        target.write_bytes(data)
        print(f"Downloaded {target} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

