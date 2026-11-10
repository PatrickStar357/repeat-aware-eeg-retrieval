"""Render the paper figures from recorded data with no network requests."""
from pathlib import Path
import argparse
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=["en", "zh"], default=["en"])
    parser.add_argument("--figures", nargs="+", type=int, choices=range(1, 7),
                        default=list(range(1, 7)))
    args = parser.parse_args()
    subprocess.run([
        sys.executable, str(root / "figures" / "draw_figures.py"),
        "--workspace", str(root / "outputs"),
        "--inputs", str(root / "figures" / "source_data"),
        "--output", str(root / "outputs" / "figures"),
        "--languages", *args.languages,
        "--figures", *map(str, args.figures),
    ], check=True)
