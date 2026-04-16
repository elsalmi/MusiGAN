# MusiGAN

Creative music generation experiments using LSTM, GRU, and GAN models over MIDI
note sequences.

## TL;DR

MusiGAN explores symbolic music generation rather than audio synthesis. The repo
contains legacy notebooks, a standalone GAN script, serialized note data,
trained/output artifacts, and generated MIDI examples from LSTM, GRU, and GAN
experiments.

The portfolio-ready framing is intentionally careful:

- This is a creative ML research artifact, not a production music model.
- Evaluation is listening-first, supported by proposed symbolic metrics.
- MIDI provenance matters. Some committed source material is game/anime-derived
  and should not be treated as a clean redistributable benchmark.
- No new datasets, checkpoints, or generated media are added in this upgrade
  pass.

## What is in this repo

| Artifact | Purpose |
| --- | --- |
| `Models/CudNN_final.ipynb` | Legacy LSTM/GRU notebook using CuDNN recurrent layers |
| `Models/GAN_final.ipynb` | Legacy GAN notebook for symbolic sequence generation |
| `Models/gan_final.py` | Scripted GAN training/generation path |
| `Data/notes` | Serialized note/chord token stream used by generation code |
| `Data/Pokemon MIDIs/` | 307 MIDI files observed locally |
| `Data/Piano MIDIs/` | 92 MIDI files observed locally |
| `Output midi/GAN midi/` | Existing GAN-generated MIDI and HDF5 artifacts |
| `Output midi/GRU midi/` | Existing GRU-generated MIDI artifacts |
| `Output midi/LSTM midi/` | Existing LSTM-generated MIDI artifact |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Legacy environment and rerun notes |
| [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) | Data source, risk, and licensing notes |
| [docs/README_EVAL.md](docs/README_EVAL.md) | Listening rubric and proposed symbolic metrics |
| [reports/SAMPLE_REPORT.md](reports/SAMPLE_REPORT.md) | Generated-sample tracking template |

## Problem

Generate novel symbolic music sequences from a corpus of MIDI files, then compare
whether LSTM, GRU, or GAN-style approaches produce more usable musical sketches.

This is not an accuracy-maximization project. A useful output is one that has
musical continuity, rhythmic coherence, and enough variation to be worth human
arrangement.

## Data

The project works with MIDI files, which encode note events rather than raw
waveforms. The committed corpus includes:

- 307 Pokemon MIDI files.
- 92 piano MIDI files from game and media themes.
- A serialized note stream at `Data/notes`.

Important caveat: these files may include copyrighted compositions or derivative
transcriptions. Treat them as legacy research inputs for local experimentation,
not as a clean public training dataset. See
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).

## Method

The core pipeline is:

1. Parse MIDI files with `music21`.
2. Convert notes/chords/rests into symbolic tokens.
3. Encode token sequences as integer windows.
4. Train sequence models:
   - CuDNNLSTM.
   - CuDNNGRU.
   - GAN generator/discriminator experiments.
5. Convert predicted symbolic sequences back into MIDI with `music21`.
6. Evaluate by listening and by inspecting symbolic structure.

## Results and artifacts

The repo currently contains generated outputs and trained artifacts, but the
evaluation is not yet packaged as a formal benchmark.

| Track | Current evidence |
| --- | --- |
| LSTM | Generated MIDI artifact in `Output midi/LSTM midi/` |
| GRU | Generated MIDI artifacts in `Output midi/GRU midi/` |
| GAN | Generated MIDI and HDF5 artifacts in `Output midi/GAN midi/` |
| Evaluation | Listening rubric and proposed symbolic metrics now documented |

The strongest honest claim is that GAN outputs were the preferred creative
direction during exploration, while LSTM/GRU outputs tended to become repetitive.
That preference is qualitative until the sample report is filled in with scored
listening notes and symbolic metrics.

## Reproduce

This is a legacy notebook-first project. The original stack used Python 3,
TensorFlow 1.x/Keras-era APIs, `music21`, and GPU-oriented CuDNN layers. The
current code is useful for audit and refactoring, but should not be treated as a
one-command modern training pipeline.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before rerunning.

## Demo plan

A safer public demo should avoid redistributing training MIDI or questionable
generated files. The recommended demo shape is:

1. Use a small synthetic or clearly licensed MIDI seed.
2. Generate a short symbolic continuation.
3. Display piano-roll or token-level structure.
4. Include listening-rubric scores and symbolic metrics.
5. Make provenance and licensing warnings visible.

## What I would improve next

1. Move legacy data/checkpoints into external artifact storage with clear
   licensing gates.
2. Add a pinned legacy environment for exact reruns.
3. Refactor `Models/gan_final.py` to remove local absolute paths.
4. Export a small generated-sample table with listening scores.
5. Add symbolic metrics such as pitch-class distribution, repetition rate,
   interval distribution, note-density, and unique n-gram rate.
