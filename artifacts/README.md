# Compare and Weigh: experiment and frozen-prediction artifacts

The archive contains original experiment code, complete 45-configuration
summary tables for three galleries, participant/seed records, category and
input diagnostics, measured costs, and the six fixed case records. No original
THINGS stimulus photographs or raw EEG are redistributed.

## Reproduce the main comparison without training

Install NumPy, then run:

    python reproduce_frozen.py

The script reads 180 NPZ files: 10 participants x 3 seeds x 3 galleries x two
matched scorers. It recomputes 66 aggregate metrics and checks the six fixed
case ranks for these two scorers. The delivered check has maximum absolute
deviation 0.0 from the recorded summary. Three seeds are averaged within each
participant; participants, not queries or seeds, are the ten statistical units.

All logits are frozen original experiment outputs, not newly trained results.
This verifies metric/result traceability, not an independent training replication.
The evaluation-pool development history reported in the manuscript still applies.

## Retraining inputs

THINGS-EEG2: NEMAR nm000232 v1.1.1; original dataset/publication citations and
selection/preprocessing are in the paper. Original THINGS photos remain under
their owners' terms and are obtained from the original dataset, not this archive.
Trained weights, processed EEG, and large feature banks are not included.
The training scripts retain their original project-relative dependencies and
are not presented as a self-contained Docker environment.

Source result tables use fractions for ranking/confidence; paper percentages
multiply by 100. Method identifiers remain those of the frozen experiment log.
code/ contains the original implementation/analysis record; reproduce_frozen.py
is the standalone entry point for checking the main reported comparison.
