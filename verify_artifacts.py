"""Check delivered files against the repository's SHA256 manifest."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for row in manifest["files"]:
        path = root / row["file"]
        if not path.is_file():
            errors.append({"file": row["file"], "error": "missing"})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            errors.append({"file": row["file"], "error": "sha256 mismatch"})
    predictions = list((root / "artifacts" / "predictions").rglob("*.npz"))
    if len(predictions) != 180:
        errors.append({"error": "expected 180 frozen prediction files"})
    print(json.dumps({"files_checked": len(manifest["files"]),
                      "prediction_files": len(predictions), "errors": errors,
                      "status": "passed" if not errors else "failed"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
