# Manuscript-to-experiment map

| Evidence | Included material | Verification or implementation |
|---|---|---|
| Full 45-configuration, three-gallery matrix | `artifacts/results/summary.json`, `all_task_means.csv`, `individual_runs.csv`, `participant_seed_means.csv` | All conditions retained; main pair checked by `reproduce.py` |
| Main-pair ranking, proper scores and fixed-coverage risks | `artifacts/predictions/`, `artifacts/reproduce_frozen.py` | `python reproduce.py` |
| Participant-level controls and covariance effects | `artifacts/results/summary.json`, `participant_seed_means.csv`, `final_comparisons.json` | Original participant/seed data and paired comparisons |
| Six-query, four-scorer choices | `artifacts/results/case_gallery_records.json`, `artifacts/case_comparisons/` | `python verify_cases.py` |
| Query costs | `artifacts/results/cost_benchmark_v89.json` | Original hardware/timing records |
| Temporal and category diagnostics | `artifacts/results/temporal_occlusion_v88.json`, `category_results_v149.json` | Original diagnostic records |
| Recording preparation/indexing | `data/code/preprocess_eeg.py`, `preprocess_train_stream_v40.py`, `build_data_index.py` | Requires original data and metadata index |
| Pinned input acquisition | `data/metadata/`, `data/code/download_datasets.py` | `python data/code/download_datasets.py --groups raw images features --plan-only` |
| Matched NICE/ATM backbone training | `data/code/experiments/train_all_backbones_v42.py`, `train_seed_replication_v61.py` | Requires processed EEG, RN50 features and GPU inputs |
| Candidate prediction and geometry | `data/code/experiments/cross_trial_likelihood_v6.py`, `compare_noise_controls.py`, `discriminant_evidence_v40.py` | Original numerical implementations |
| Fusion and compact evaluation | `data/code/experiments/score_stacking_v24.py`, `final_choice_screen_v75.py`, `compact_evidence_audit_v86.py` | Original fitting/evaluation source |
| Adapted SATTC baseline | `data/code/experiments/sattc_adapter_v63.py`, `closeout_experiments_v64.py` | Pinned source definitions and documented batch context |

In `artifacts/provenance.json`, `path`/`file` fields are relative to the
repository root; `source` fields retain the original project locations.
In `artifacts/prediction_provenance.json`, `path` is relative to `artifacts/`
and `source` retains the original output location. Every distributed file
and its integrity hash is listed in `artifact_manifest.json`.
