# MusiGAN Evaluation Notes

## Goal

Evaluate generated symbolic music as musical sketches, not as classification
outputs. A good sample should be coherent enough for a human to arrange, while
still showing variation beyond memorized source patterns.

## Listening rubric

Score each generated sample from 1 to 5 on each dimension.

| Dimension | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Continuity | Abrupt or broken | Some phrase continuity | Clear phrase flow |
| Rhythm | Erratic timing | Mostly steady | Intentional-feeling groove |
| Pitch coherence | Random jumps | Some tonal center | Strong melodic contour |
| Variation | Repetitive loop | Some changes | Balanced repetition and novelty |
| Playability | Awkward or impossible | Partly playable | Easy to interpret on piano/MIDI |
| Arrangement potential | Not useful | Could be edited | Strong seed for arrangement |

Recommended qualitative notes:

- Best phrase or moment.
- Worst failure mode.
- Whether the sample sounds copied, repetitive, or incoherent.
- Whether the sample would be useful as a seed for human composition.

## Proposed symbolic metrics

These metrics can be computed without distributing audio:

- Pitch-class histogram divergence from the training corpus.
- Interval distribution divergence.
- Note-density per bar or per fixed timestep.
- Rest ratio.
- Unique token ratio.
- Repeated n-gram rate.
- Longest repeated subsequence.
- Duration distribution.
- Chord-vs-note ratio.
- Out-of-vocabulary or malformed-token rate.

## Suggested report format

Each generated sample should be tracked with:

- Model family: LSTM, GRU, or GAN.
- Source checkpoint or notebook.
- Seed strategy.
- Number of generated tokens.
- Listening rubric scores.
- Symbolic metric snapshot.
- Redistribution status.
- Notes on whether the sample should be public, private, or discarded.

## Current evaluation status

The repo contains generated MIDI files, but it does not yet contain a completed
score table. The current portfolio claim should stay qualitative until
`reports/SAMPLE_REPORT.md` is filled in with evaluated samples.
