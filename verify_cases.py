"""Verify six complete candidate galleries against four frozen scorers."""
from pathlib import Path
import json
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    record = json.loads((root / "artifacts/results/case_gallery_records.json").read_text(encoding="utf-8"))
    paths = {
        "NICE_global": root / "artifacts/case_comparisons/within10_NICE_global.npz",
        "ATM_global": root / "artifacts/case_comparisons/within10_ATM_global.npz",
        "anchor_nll": root / "artifacts/predictions/sub-01/seed2028/within10_anchor_nll.npz",
        "hybrid_block_mlp": root / "artifacts/predictions/sub-01/seed2028/within10_ours.npz",
    }
    for model, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            for case in record["cases"]:
                index = case["query_index"]
                assert str(data["query_ids"][index]) == case["query_id"]
                assert data["candidate_ids"][index].tolist() == [v["id"] for v in case["candidates"]]
                assert int(data["targets"][index]) == case["target_index"]
                order = np.argsort(-data["logits"][index], kind="stable")
                rank = int(np.flatnonzero(order == data["targets"][index])[0] + 1)
                chosen = case["models"][model]
                assert int(order[0]) == chosen["selected_index"]
                assert str(data["candidate_ids"][index, order[0]]) == chosen["selected_image"]
                assert rank == chosen["target_rank"]
    assert sum(len(c["candidates"]) for c in record["cases"]) == 60
    unselected = sum(not v["selected_by"] for c in record["cases"] for v in c["candidates"])
    assert unselected == 44
    print(json.dumps({"cases": 6, "candidate_records": 60, "model_choices_checked": 24,
                      "unselected_candidates": unselected, "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
