"""Trích xuất RADARSAT-1 CEOS L0 leader thành JSON."""

import argparse
import json
from pathlib import Path

from radarsat1_processing import read_leader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leader", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("output/radarsat-1") / f"{args.leader.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(read_leader(args.leader), indent=2, ensure_ascii=False) + "\n")
    print(f"Đã ghi {output}")


if __name__ == "__main__":
    main()
