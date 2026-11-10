# Manuscript-to-artifact map

| Evidence | Included location | Reproduction route |
|---|---|---|
| Full 45-configuration / three-gallery results | `artifacts/results/summary.json`, `all_task_means.csv`, `individual_runs.csv`, `participant_seed_means.csv` | Inspect all conditions; main matched pair replayed by `reproduce.py` |
| Main pair's ranks, probability scores and fixed coverage risks | `artifacts/predictions/`, `artifacts/reproduce_frozen.py` | `python reproduce.py` |
| Figures 3--4 | `figures/plotted_data/fig03_fig04_participant_values.csv` | `python make_figures.py --figures 3 4` |
| Figure 5 covariance controls | `figures/plotted_data/fig05_covariance_participant_values.csv` | `python make_figures.py --figures 5` |
| Six fixed cases | `artifacts/results/qualitative_cases_v151.json` | Main-pair ranks checked in replay; all four recorded scorers retained in case log |
| Costs | `artifacts/results/cost_benchmark_v89.json` | Recorded hardware/timing summaries; not a new benchmark on the reader's hardware |
| Time and category diagnostics | `artifacts/results/temporal_occlusion_v88.json`, `category_results_v149.json` | Recorded diagnostics; source retained |
| Recording preparation and indexing | `data/code/preprocess_eeg.py`, `preprocess_train_stream_v40.py`, `build_data_index.py` | Requires original source data and metadata index |
| NICE / ATM matched backbone training | `data/code/experiments/train_all_backbones_v42.py`, `train_seed_replication_v61.py` | Requires preprocessed recordings, RN50 features, upstream assets and GPU environment |
| Candidate prediction and geometry | `data/code/experiments/cross_trial_likelihood_v6.py`, `compare_noise_controls.py`, `discriminant_evidence_v40.py` | Original implementation record |
| Fusion / compact selection and evaluation | `data/code/experiments/score_stacking_v24.py`, `final_choice_screen_v75.py`, `compact_evidence_audit_v86.py` | Original implementation and configuration record |
| SATTC adapted structural baseline | `data/code/experiments/sattc_adapter_v63.py`, `closeout_experiments_v64.py` | Pinned upstream functions and explicitly reported extra batch context |

Historical paths inside provenance records describe the source project. The
actual distributed locations and hashes are listed in `artifact_manifest.json`.
No command above claims to reconstruct excluded raw data or checkpoints.

Complete Figure 6 galleries: `figures/source_data/figure6_gallery_cases.json`; run `python verify_cases.py`. Additional NICE/ATM outputs are in `artifacts/case_comparisons/`. Figure 6 uses CC0 concepts and record cards; original stimulus images remain external.
