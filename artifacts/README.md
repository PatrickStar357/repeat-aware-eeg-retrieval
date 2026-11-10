# Compare and Weigh: numerical experiment artifacts

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python verify_artifacts.py
python reproduce.py
python verify_cases.py
```

`predictions/` contains 180 main-comparison frozen outputs: ten participants,
three seeds, three galleries and two scorers. `case_comparisons/` adds two
NICE/ATM outputs. The replay checks 66 aggregate metrics with maximum absolute
difference 0.0; the case verifier checks 24 choices and target ranks across
six queries and 60 candidate records.

`results/` retains every configuration: 135 task/configuration means,
4,050 individual run rows and 1,350 participant means. The 45 configurations
are evaluated on three supplied-gallery tasks. Separate JSON files retain
cost, temporal/category diagnostics, paired comparisons and case records.

Original implementation and preprocessing are in `../data/code/`.
Pinned external-input acquisition inventories are in `../data/metadata/`.
See `../DATA_AVAILABILITY.md` for exact data routes and `../REPRODUCIBILITY.md`
for protocol and training requirements. Raw EEG, original photographs,
checkpoints and large feature tensors are not redistributed. The included
replay computes metrics from frozen predictions rather than training models.

In `provenance.json`, distributed `path`/`file` fields are repository-relative;
`source` fields retain original project paths. In `prediction_provenance.json`,
`path` fields are relative to this `artifacts/` directory. The repository-root
manifest identifies all distributed file hashes. Result fractions become
percentages by multiplying by 100 when reported in the manuscript.
