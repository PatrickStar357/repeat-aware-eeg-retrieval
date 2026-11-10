"""Replay the main matched comparison from the delivered frozen outputs."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "artifacts" /
                       "reproduce_frozen.py"), run_name="__main__")
