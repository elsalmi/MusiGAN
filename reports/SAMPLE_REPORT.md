# MusiGAN Sample Report

## Purpose

This report tracks generated MIDI samples without adding new copyrighted source
data, generated media, or checkpoints.

No new samples were generated during this documentation pass.

## Current artifact inventory

| Location | Current role | Public-demo status |
| --- | --- | --- |
| `Output midi/LSTM midi/` | Existing LSTM generated MIDI artifact | Needs provenance review before public reuse |
| `Output midi/GRU midi/` | Existing GRU generated MIDI artifacts | Needs provenance review before public reuse |
| `Output midi/GAN midi/` | Existing GAN generated MIDI and HDF5 artifacts | Needs provenance review before public reuse |

## Evidence summary

The original project notes say GAN outputs were preferred because they trained
faster and sounded less repetitive than LSTM/GRU outputs. This remains a
qualitative observation until samples are scored with the listening rubric in
`docs/README_EVAL.md`.

## Sample tracking template

| Sample ID | Model | Source artifact | Listening score | Symbolic metrics | Redistribution status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Pending | Pending | Pending | Pending | Pending | Pending | Fill after provenance-safe sample review |

## Next sample review pass

1. Select one LSTM, one GRU, and one GAN generated MIDI from existing outputs.
2. Confirm whether each sample can be shared publicly.
3. Score each sample with the listening rubric.
4. Compute symbolic metrics from the MIDI event stream.
5. Add short notes on repetition, coherence, and arrangement potential.
6. Publish only cleared samples or synthetic examples.
