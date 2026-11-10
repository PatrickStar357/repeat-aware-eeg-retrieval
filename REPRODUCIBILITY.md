# Reproduction routes and protocol

## Route 1: original frozen-output replay

Validated with Python 3.12.14 and NumPy 2.3.5:

```bash
python verify_artifacts.py
python reproduce.py
```

`artifacts/predictions/` contains two scorers (`ours` and `anchor_nll`) for
within10, cross10 and official200, for participants 01--10 and seeds
2028/2029/2030. Each file contains logits and target indices. The replay computes
Top-1, Top-5, mAP, Brier, NLL, AURC, ECE15, wrong-answer confidence and the
three fixed coverage risks, then averages seeds within participant and
participants within condition. The aggregate total is 66 checked metrics.

The recorded replay discrepancy is zero. The six fixed query ranks are also
checked for the main pair. No training, feature extraction or model downloads
occur in this route. New reports are written to `outputs/`.

## Route 2: figure reconstruction

```bash
python -m pip install -r requirements-figures.txt
python make_figures.py
```

The recorded rendering environment uses Matplotlib 3.11.2 and Pillow 13.0.0.
English uses Arial with DejaVu Sans fallback. Chinese requires the installed
Windows SimHei font. Font substitution can change geometry, so inspect a fresh
export before reusing it. Delivered font, alignment and collision reports refer
to the delivered SVG/PDF geometry; installing no optional audit helper is not
reported as a fresh alignment pass.

Figure 6 uses six documented CC0 concept photographs and 60 candidate-record
cards. Its 24 model selections and ranks are checked with `python verify_cases.py`,
including the 44 records selected by none of the models. Two additional
NICE/ATM case-comparison output files supplement the 180 main-pair predictions.
No original stimulus photograph is used or required for figure reconstruction.

Figures 3 and 4 use every one of the ten participant means for each displayed
method. Figure 5 uses all five displayed covariance alternatives and all ten
participants; its final column is their mean. Fixed source values, native units,
and exclusions are documented in `figures/plotted_data/` and the figure captions.

## Route 3: original training and analysis source

`data/code/` preserves the original source layout: paths defined relative to
the original data root remain under the repository's `data/`. There are 38
original implementation/preparation files, including the original 31 experiment
files plus seven recording/index/source-preparation files. Selected NICE, ATM
and SATTC source definitions are included unchanged with their MIT notices.

Full training additionally requires the original indexed EEG data, preprocessing
outputs, RN50 feature inputs, model assets and a compatible GPU environment.
`requirements-training.txt` is a dependency list from the source imports, not
a lockfile of the historical training environment. The raw-data route is not
executed by `reproduce.py`. No new training or independent evaluation run is
claimed by this artifact release.

Main source entry points and their evidence roles are listed in
`docs/EXPERIMENT_MAP.md`. Recorded configuration and source hashes remain in
`artifacts/results/summary.json` and the provenance records.

## Experimental units and interpretation

- Ten participant-specific models; three seeds averaged within each participant.
- Four EEG recordings per query (R4), with supplied K10/K200 galleries.
- Concept identity is provided for the primary within10 task.
- EEG-supervised training/calibration/evaluation concepts are disjoint.
- The evaluation pool was inspected during development. Reported effects are
  exploratory; the recorded paired comparisons do not remove all selection
  effects from the 45-configuration search.
- SATTC is an adapted label-free structural postprocessor with extra unlabeled
  batch context, distinguished from the inductive scorers in the manuscript.
- Single-metric alternatives and every configuration remain in the full results.

The repository supports inspection of these stated conditions and outputs. It
does not assert EAB-level self-contained retraining, an artifact badge, or an
independent replication certificate.
