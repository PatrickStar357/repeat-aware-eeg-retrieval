# Data and code availability

## Materials included here

The repository contains original implementation and preprocessing source,
the 45-configuration summary matrix for the three reported gallery tasks,
participant/seed records, main-comparison frozen logits and target indices,
cost and diagnostic summaries, and figure source data. `artifact_manifest.json`
maps every distributed file to a SHA256 digest. The original summary units are
fractions; manuscript percentages multiply by 100.

The main replay uses 180 frozen prediction files and requires NumPy only.
All numerical data used by Figures 3--5 are also exported as participant-level
CSV files under `figures/plotted_data/`.

## Reused public data

THINGS-EEG2 is obtained from its original repositories and used under their
original access conditions. The recorded project data route is NEMAR nm000232
v1.1.1; do not silently substitute a different snapshot when retraining.

- Dataset record: <https://github.com/nemarDatasets/nm000232>
- Original project: <https://osf.io/3jk45/>
- Original code: <https://github.com/gifale95/eeg_encoding>
- Dataset paper: <https://doi.org/10.1016/j.neuroimage.2022.119754>

Original EEG recordings, processed recording arrays, stimulus photographs,
trained models and feature banks are not included in this repository. Use the
original data controllers' access routes and terms. Public availability or a
dataset citation is not treated as a blanket right to redistribute photographs.

## Concept illustrations

The six photographs in `figures/source_data/cc0/` come from the documented
THINGSplus-CC0 collection. Their source/member identifiers, CRCs and file
hashes are in the accompanying `provenance.json`.

- Official CC0 archive: <https://osf.io/download/wb36u/>
- THINGSplus paper: <https://doi.org/10.3758/s13428-023-02110-8>

They illustrate object concepts. They are not substituted experimental stimuli
or displayed model outputs; the same six recorded query ranks remain in the
case log.

## Artifact inventory

Distributed files and their SHA256 digests are indexed in
`artifact_manifest.json` and `checksums.sha256`. Raw-data access routes and
omitted external assets are listed above.

Figure 6 pairs documented CC0 concept illustrations with the 60 original
candidate IDs and hashes. The cards and choices identify experimental records;
the concept photos are not replacements for the corresponding stimuli.
