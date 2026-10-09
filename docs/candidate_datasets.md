# Candidate public datasets for registration validation

Public multichannel microscopy data for validating optomerge's aligners on data
*other* than P.'s own OptoSplit movies — the "generalise for the wider community"
goal. The registration problem optomerge solves (two or more channels imaged onto
one sensor / two cameras, needing sub-pixel alignment) is shared by single-molecule
FRET (donor/acceptor split on one camera), multicolor SMLM/super-resolution (bead
channel mapping), and dual-camera cytoskeleton imaging. All entries below are that
problem.

Compiled 2026-07-10. Links were live at compile time; re-check availability and
licence before use.

## Top pick — Deep-LASI multi-color smFRET

- **Link:** https://zenodo.org/records/7561162 (DOI 10.5281/… via Zenodo record;
  paper: https://www.nature.com/articles/s41467-023-42272-9)
- **Licence:** CC-BY 4.0 (re-use with attribution).
- **What:** Raw 1-, 2-, and 3-color single-molecule TIRF movies of a tunable
  L-shaped DNA origami (Cy3B donor / Atto647N acceptor, third color added), plus
  titration and dynamics series. Donor/acceptor channels are split onto one camera
  — the exact "two halves of one frame" geometry optomerge handles, with bead-based
  channel mapping in the workflow.
- **Why it's the best match:** same optical setup as OptoSplit (spectral splitter,
  one sensor), real (not simulated) data, well-documented, permissive licence.
- **Size / access:** 48.7 GB total, but the `Figure_*.zip` files download
  individually (~0.5–7 GB each); `Data_Overview.xlsx` at the record root indexes
  them. Good small-ish starting points: `Figure_2.zip` (488 MB), `Figure_3.zip`
  (988 MB).
- **Suggested aligner:** donor vs acceptor are similar point-source patterns →
  try `--ecc` (euclidean) first, then `--keypoint` if the shift is large; `--mi`
  if intensities differ strongly between channels.

## Secondary options

### Multi-camera simultaneous TIRF + IRM of microtubules
- **Link:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11451753/ (biorxiv:
  https://www.biorxiv.org/content/10.1101/2024.08.28.610099)
- **What:** Dual-camera setup imaging dynamic microtubules and associated proteins
  (TIRF fluorescence on one camera, interference reflection on another) — two
  detectors that must be registered to each other.
- **Why:** Closest to P.'s heads-on-filaments case (line-like filaments + point-like
  motors), and a *two-camera* registration rather than split-frame, which exercises
  a different part of the problem.
- **Caveat:** confirm the data-availability statement / repository link in the paper
  before relying on it; not verified to be a downloadable raw archive.
- **Suggested aligner:** `--feature` (filament + motor) for the biology case;
  `--ecc`/`--keypoint` for the cross-camera bead/fiducial registration.

### Traces (smFRET analysis tool with example data)
- **Link:** https://github.com/stephlj/Traces
- **What:** MATLAB smFRET analysis package that ships a *small* example two-color
  dataset with the standard bead-based channel-mapping workflow (acceptor→donor
  pixel map).
- **Why:** smallest, fastest way to get a real two-color movie + a known bead
  mapping for an end-to-end smoke test of the new aligners.

### Hellenkamp multi-laboratory smFRET benchmark
- **Link:** https://www.nature.com/articles/s41592-018-0085-0 (preprint:
  https://arxiv.org/abs/1710.03807)
- **What:** Blind 20-lab study on dye-labeled dsDNA duplexes; widely used reference
  two-color smFRET data. Useful as a cross-lab robustness check.
- **Caveat:** raw-data location not confirmed from the article page; check the
  data-availability statement / corresponding authors.

### TetraSpeck fluorescent bead calibration stacks
- **What (not one dataset):** TetraSpeck beads fluoresce in all channels and are the
  standard fiducial for measuring/validating channel registration — imaging sparse
  beads, localizing each in every channel, and fitting the channel-to-channel map.
- **Where to find:** many multicolor SMLM/super-resolution deposits include a bead
  calibration stack alongside the biology data (search the repositories below).
- **Why:** gives *ground-truth* registration (known bead correspondences), so it's
  the cleanest way to quantify aligner accuracy rather than just eyeball overlays.

## Repositories to browse for more

- **BioImage Archive** — https://www.ebi.ac.uk/bioimage-archive/ (filter for
  multichannel TIRF / SMLM; often includes bead calibration).
- **Image Data Resource (IDR)** — https://idr.openmicroscopy.org/
- **EMPIAR** — https://www.ebi.ac.uk/empiar/ (more EM, some correlative).
- **Zenodo** — https://zenodo.org/ — search "smFRET raw", "two-color TIRF",
  "TetraSpeck registration", "dual-view".

## Notes for when validating

- optomerge expects the two channels packed into one frame (split-frame). For
  *two-camera* datasets (e.g. the TIRF/IRM one) the two images come as separate
  files/stacks — they'd need concatenating side-by-side into one frame, or a small
  reader tweak, before the current pipeline ingests them.
- Bead datasets give quantitative accuracy (residual bead distance after
  registration); biology datasets give qualitative overlay quality. Use a bead
  stack to pick per-method defaults, then confirm on real structure.
- Start with the smallest download (Traces example or a single Deep-LASI
  `Figure_*.zip`) to shake out I/O + channel-order handling before committing to
  the 48 GB set.
