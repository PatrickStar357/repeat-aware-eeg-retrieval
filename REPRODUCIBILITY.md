# Numerical reproduction and source implementation

## Main matched comparison

Validated replay environment: Python 3.12.14 and NumPy 2.3.5.

```bash
python -m pip install -r requirements.txt
python verify_artifacts.py
python reproduce.py
python verify_cases.py
```

The main replay reads 180 frozen prediction files: two scorers, three gallery
tasks, ten participants and three seeds. It computes Top-1, Top-5, mAP, Brier,
NLL, AURC, ECE15, wrong-answer confidence and three fixed-coverage risks.
Seeds are averaged within participant, then participants within condition.
All 66 checked aggregate metrics match the recorded summary with a maximum
absolute difference of 0.0. New reports are written to `outputs/`.

`artifacts/results/case_gallery_records.json` preserves six query records,
their 60 candidates, four scorers' choices and target ranks. `verify_cases.py`
checks 24 model choices directly against the frozen arrays, including the 44
candidate records selected by none of the scorers. Two NICE/ATM case-comparison
files supplement the 180 main-pair outputs.

This route reads saved predictions and does not run model training or download
data. It verifies output traceability rather than independent retraining.

## Original implementation and inputs

`data/code/` preserves the original source layout. There are 35 original
implementation/preparation files: 31 sources in the recorded experiment
inventory and four additional recording/index/preprocessing sources. Selected NICE, ATM and
SATTC definitions are unchanged and retain their original MIT notices.

`data/metadata/` contains the recorded acquisition inventories for the
non-rest EEG inputs, canonical stimulus identity metadata and the two
DirectT intra-subject RN50 baseline feature files. It records input sizes,
checksums, original URLs and the feature-repository revision. Inspect the
plan, without network access or data downloads, with:

```bash
python data/code/download_datasets.py --groups raw images features --plan-only
```

Omitting `--plan-only` starts the large external-input downloads. Keep the
explicit `--groups raw images features`; the unchanged original downloader
also supports a different dataset outside this submission. Here `images`
covers canonical identity/licensing metadata, not stimulus photograph bytes.

Full training additionally requires indexed source EEG, preprocessing outputs,
RN50 feature inputs, trained/model assets and a compatible GPU environment.
`requirements-training.txt` lists libraries referenced by these implementations;
it is not a lockfile of the historical GPU training environment. Recorded
configuration and source hashes remain in the result summaries and provenance.
The selected upstream files supply definitions extracted by the experiment
adapters; they are source snapshots, not standalone upstream training packages.

## Experimental units and interpretation

- Ten participant-specific models, with three seeds averaged within participant.
- Four EEG recordings per query and supplied K10/K200 candidate galleries.
- Concept membership supplied for the primary within10 task.
- Disjoint EEG-supervised fitting, calibration and evaluation concepts.
- Evaluation-pool inspection during development; reported comparisons retain
  their exploratory interpretation.
- SATTC uses the documented adapted label-free postprocessor and additional
  unlabeled batch context, distinct from the inductive scorers.
- All configurations and single-metric alternatives remain in the full results.

The source-to-evidence map is in `docs/EXPERIMENT_MAP.md`. Original data access
and external inputs are described in `DATA_AVAILABILITY.md`.
