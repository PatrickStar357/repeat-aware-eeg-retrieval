# Data and implementation availability

## Included numerical materials

The package contains original implementation/preprocessing source, all
45-configuration summaries for three gallery tasks, participant/seed records,
frozen logits and target indices, measured costs, diagnostic summaries and
six-query candidate records. `artifact_manifest.json` and `checksums.sha256`
identify every distributed file.

There are 180 main-pair frozen prediction files and two additional NICE/ATM
case-comparison files. The NumPy replay checks the main comparison; the case
verifier checks all four scorers' recorded choices. Result fractions are
multiplied by 100 when expressed as percentages in the manuscript.

## Original data and external inputs

THINGS-EEG2 is accessed under its original conditions. The recorded project
route is NEMAR nm000232 v1.1.1; a different snapshot must not be silently
substituted for the recorded experiment.

- Dataset record: <https://github.com/nemarDatasets/nm000232>
- Original project: <https://osf.io/3jk45/>
- Original code: <https://github.com/gifale95/eeg_encoding>
- Dataset paper: <https://doi.org/10.1016/j.neuroimage.2022.119754>

The recorded NEMAR acquisition entries are in
`data/metadata/nemar_manifest.json` (non-rest EEG/BIDS files and four stimulus
identity/licensing metadata files). Each entry retains its original download
URL, byte size and checksum. The input inventory contains no raw EEG arrays.

The released RN50 feature route is
<https://huggingface.co/datasets/Haitao999/things-eeg> at revision
`944f0d50e37e125eb6b7b98d14a41571c9ed0ef7`.
`data/metadata/things_all.json` records the exact train/test DirectT
intra-subject RN50 baseline feature files, with their sizes and content hashes;
`things_repo.json` records the release revision. These are acquisition
metadata, not redistributed feature tensors. `load_visual_features.py`
checks the input hashes and canonical class/filename identity.

To inspect the pinned plan, run from the repository root:

```bash
python data/code/download_datasets.py --groups raw images features --plan-only
```

Removing `--plan-only` starts the external downloads. Use the explicit groups
shown: the preserved original script has additional options outside this
submission. The included `images` inventory supplies canonical identity and
licensing metadata. Original photographs, if needed for recomputing visual
features, must be obtained separately from their original dataset providers.

Raw/processed EEG arrays, original stimulus photographs, trained checkpoints
and large visual feature banks are obtained through the original providers
and are not redistributed in this package. Reproduction from the included
frozen outputs needs none of these external assets.

## Record identity

The six-query case record contains the original query and candidate identifiers,
hashes, target indices and model choices. It contains no replacement stimuli or
synthetic prediction values. Publication illustrations and layout materials are
kept outside this numerical artifact repository.
