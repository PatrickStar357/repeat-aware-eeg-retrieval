# Repeat-Aware Scoring for Calibrated EEG-to-Image Retrieval

Experiment artifacts for **Compare and Weigh**: model and preprocessing source,
complete result tables, frozen predictions, and numerical verification tools.

## Reproduce and verify

Use Python 3.12:

```bash
python -m pip install -r requirements.txt
python verify_artifacts.py
python reproduce.py
python verify_cases.py
```

The main replay checks 66 aggregate metrics from 180 frozen prediction files
covering ten participants, three seeds, three gallery tasks and two scorers.
The verified maximum absolute difference from the recorded summary is 0.0.
The case verifier checks 24 choices and target ranks against four scorers for
six queries and their 60 candidate records. Two additional NICE/ATM output
files support these comparisons. Generated replay reports go to `outputs/`.

## Included materials

```text
data/code/                    Original implementation and preprocessing source
  experiments/                Training, scoring, fusion, evaluation and controls
  upstream/                   Selected source snapshots with original licenses
data/metadata/                Pinned EEG and RN50 input acquisition inventories
artifacts/
  predictions/                180 main-pair frozen outputs
  case_comparisons/           Two additional NICE/ATM outputs for case verification
  results/                    Full result matrices, diagnostics and case records
  reproduce_frozen.py         Numerical replay implementation
  prediction_provenance.json  Output provenance and source hashes
reproduce.py                  Main replay entry point
verify_cases.py               Four-scorer case verification
verify_artifacts.py           File integrity verification
requirements.txt              Validated replay dependencies
requirements-training.txt     Dependencies referenced by original training source
artifact_manifest.json        Distributed file inventory
checksums.sha256              SHA256 integrity record
```

The source includes 35 original implementation/preparation files and selected
NICE, ATM and SATTC definitions with their MIT notices. The complete summaries
retain all 45 configurations for the three reported gallery tasks.
`docs/EXPERIMENT_MAP.md` connects the evidence to its implementation and data.

## Data access and protocol

Three seeds are averaged within each participant; the ten participants are the
statistical units. Tasks use participant-specific R4 queries and supplied
K10/K200 galleries, with concept identity provided for within10. The exploratory
development history and baseline adaptations remain specified in
`REPRODUCIBILITY.md` and the manuscript.

The replay reconstructs metrics from original frozen outputs. Retraining also
requires the original indexed EEG, visual features, model assets and a compatible
GPU environment. Access routes and excluded external inputs are described in
`DATA_AVAILABILITY.md`.

The recorded input versions, file sizes and checksums are included. To inspect
the acquisition plan without downloading any data, run:

```bash
python data/code/download_datasets.py --groups raw images features --plan-only
```

Use the explicit groups shown above. The preserved download script also has
options for other original-project datasets, whose inventories are outside
this submission. The `images` group here retrieves canonical identity and
licensing metadata; original stimulus photographs are not bundled.

Publication graphics, layout checks and image illustrations are maintained
separately from this numerical experiment package.

## Citation and rights

See `CITATION.cff` for author metadata and `LICENSES.md` for rights and third-party
notices. Cite the manuscript and original datasets/methods for reused material.
