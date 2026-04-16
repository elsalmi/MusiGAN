# MusiGAN Reproducibility Notes

## Current state

This repo is notebook-first and legacy-framework-first. It contains enough code
to understand the experiment, but it is not yet a clean one-command training
package.

Committed implementation artifacts:

- `Models/CudNN_final.ipynb`
- `Models/GAN_final.ipynb`
- `Models/gan_final.py`
- `Data/notes`
- Existing generated MIDI and HDF5 artifacts under `Output midi/`

## Legacy environment

The original README and notebooks point to a stack around:

- Python 3.x.
- TensorFlow 1.x / Keras-era APIs.
- `music21`.
- GPU-oriented CuDNN recurrent layers.

Expected friction:

- CuDNN layer APIs changed across TensorFlow/Keras versions.
- `Models/gan_final.py` contains a local absolute path for MIDI discovery.
- Notebook outputs are useful evidence, but they are not a scripted CI pipeline.
- Existing outputs were not regenerated during this documentation pass.

## Safe rerun order

1. Create an isolated legacy Python environment.
2. Install TensorFlow/Keras versions compatible with the notebooks.
3. Install `music21`.
4. Replace local absolute MIDI paths with repo-relative paths or a configured
   `MIDI_DATA_DIR`.
5. Parse MIDI into `Data/notes`.
6. Run the LSTM/GRU notebook for baseline generated outputs.
7. Run the GAN notebook or script for adversarial outputs.
8. Save generated outputs outside the repo unless they are clearly licensed for
   redistribution.
9. Fill in `reports/SAMPLE_REPORT.md` with listening notes and symbolic metrics.

## Reproducibility target

The next engineering pass should add:

- `requirements-legacy.txt` or `environment.yml`.
- A small `scripts/prepare_notes.py` parser.
- A small `scripts/generate_sample.py` entrypoint.
- A fixture dataset made from synthetic or permissively licensed MIDI.
- Exported metrics under `reports/`.

## What was not done in this pass

- No model retraining.
- No new MIDI samples.
- No new checkpoints.
- No cleanup or deletion of existing legacy artifacts.
