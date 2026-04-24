#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parent
    cmd = [sys.executable, str(base / "tracking_extra.py"), "--algorithm", "norfair", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
