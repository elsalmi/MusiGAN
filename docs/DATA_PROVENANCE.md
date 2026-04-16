# MusiGAN Data Provenance

## Summary

MusiGAN uses MIDI data, not raw audio. MIDI stores symbolic performance events
such as pitch, duration, offset, instrument, and velocity.

Observed committed data:

| Location | Observed contents |
| --- | --- |
| `Data/Pokemon MIDIs/` | 307 `.mid` files |
| `Data/Piano MIDIs/` | 92 `.mid` files |
| `Data/notes` | Serialized note/chord/rest token stream |

## Licensing and redistribution warning

The committed MIDI files appear to include game and media themes, including
Pokemon and other soundtrack-derived material. These files should be treated as
legacy local research inputs, not as a clean open benchmark or a portfolio-safe
redistribution dataset.

Practical rule:

- Do not add more copyrighted MIDI files to this repo.
- Do not package questionable MIDI files into a public demo.
- Do not claim the dataset is open, clean, or cleared for commercial use unless
  each source is verified.
- Prefer synthetic, public-domain, Creative Commons, or explicitly licensed MIDI
  for future demonstrations.

## Current portfolio-safe wording

Use wording like:

- "Legacy MIDI corpus used for symbolic music generation experiments."
- "Generated-sample evaluation is documented, but training-data provenance needs
  cleanup before any public demo."
- "No new data or generated media were added in this pass."

Avoid wording like:

- "Open benchmark dataset."
- "Copyright-free MIDI dataset."
- "Production-ready music generation model."
- "Generated samples are safe to redistribute."

## Cleanup plan

1. Inventory each MIDI file source.
2. Move questionable source MIDI out of the repo into private/local storage.
3. Keep only clearly licensed examples or synthetic fixtures in git.
4. Add a small manifest with file name, source URL, license, and allowed use.
5. Regenerate public sample reports from cleared material.
