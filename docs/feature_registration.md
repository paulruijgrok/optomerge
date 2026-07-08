# Feature-distance registration

A registration algorithm tailored to **point-vs-line** channels: molecular
"heads" that sit on gliding filaments and co-move with them. It is an opt-in
alternative to the default phase-correlation aligner, selected with
`--feature` (or `[alignment].method = "feature"`).

## Why it exists

Phase correlation registers two images by their shared texture. When the two
OptoSplit channels image *different-looking* structures — a point-like head in
one and a line-like filament in the other — they share almost no texture, so the
correlation peak is weak and easily captured by noise (the "seeing-double"
failure).

But the head and filament are one physical object that co-moves: in every frame
the head sits on the filament (at a tip or along its length). That gives a
direct, geometry-based objective that does not depend on the channels looking
alike — **find the transform that makes the detected heads land on the
filaments.**

## How it works

For a set of frames sampled evenly across the movie (`feature_frames`):

1. Detect head centroids in the moving (red) channel: threshold at
   `mean + head_sigma·std`, connected components, keep blobs with area in
   `[head_min_area, head_max_area]` (rejects noise specks and non-point-like
   blobs).
2. Build the filament mask in the reference (green) channel: threshold at
   `mean + filament_sigma·std`, then **despeckle** (drop components smaller than
   `filament_min_area`) so background noise is not treated as filament. Take its
   distance transform (distance to the nearest filament pixel).
3. Score a candidate translation by the (weighted, capped) mean distance from
   every head to the nearest filament, and minimise it over a bounded grid
   (`±max_shift`) with parabolic sub-pixel refinement.

Because the head is on the filament in *every* frame, that distance goes to zero
at the correct registration regardless of where along the filament the head sits
— so it is robust to tip- vs mid-filament labels.

### Robustness

- **Orphan heads** — a head with no filament nearby (a red-only object, or a
  filament too faint to detect) would otherwise inflate the cost and bias the
  fit. Its distance is clipped to `distance_cap`, so it contributes a constant
  and cannot move the optimum.
- **Stuck objects** (`deweight_stuck`) — a stuck object sits at the same pixel in
  many sampled frames, so it is counted once per frame and can dominate the
  field. With de-weighting each head is weighted by `1/persistence` (persistence
  = distinct frames with a head within `stuck_radius`), so each physical object
  gets one vote and moving objects restore field coverage.

## Configuration (`[alignment]`)

| key | default | meaning |
|-----|---------|---------|
| `method` | `phase` | set to `feature` to select this aligner |
| `feature_frames` | 60 | frames sampled to fit the alignment; raise for sparse/dim movies |
| `max_shift` | 0 → 30 | translation search half-width (px); feature mode needs a bound, so 0 means 30 |
| `head_sigma` | 3.0 | head threshold, std-devs above the moving-channel mean |
| `filament_sigma` | 2.0 | filament threshold, std-devs above the reference-channel mean |
| `head_min_area` / `head_max_area` | 4 / 60 | keep only point-like head blobs (px) |
| `filament_min_area` | 20 | despeckle: drop filament-mask blobs smaller than this (px) |
| `distance_cap` | 6.0 | orphan-head robustness threshold (px); also the inlier threshold for the score |
| `deweight_stuck` | false | weight heads by 1/persistence so stuck objects count once |
| `stuck_radius` | 2.0 | px within which cross-frame detections are the same stuck object |

## Diagnostics

A `test_pipeline.py --feature` run writes **`diag_06`**: for several sampled
frames it overlays the filament outline, each detected head (red ✕), and where
the fit lands it (yellow ○ = on filament / used, blue ○ = orphan / ignored),
with an on-filament count per frame. This is the view for tuning the detection
thresholds, `distance_cap`, and `deweight_stuck`.

## Convention note

`transform_image` is an inverse (destination→source) warp: applying it with
translation `(t1, t2)` moves a feature by `(-t2 row, -t1 col)`. The objective
evaluates the head position *after* that warp — `(row - t2, col - t1)` — so the
fitted `t` is exactly the value the pipeline applies. The end-to-end test
(`test_end_to_end_recovers_shift`) guards this by applying the real transform and
asserting the head lands on the filament.

## Roadmap / future work

- **Constrained rotation + scale.** Translation-only leaves a *field-dependent*
  residual on some movies (worse toward one edge), the signature of a small
  optical rotation/scale between the two OptoSplit halves that a single
  translation cannot remove. The next development step is to extend the search to
  a small, bounded rotation/scale on top of the same robust distance metric —
  seeded from the translation fit and tightly bounded (e.g. ±1–2°, ±2%). Because
  the objective is head-to-filament distance rather than image cross-correlation,
  the extra degrees of freedom are far less prone to the over-fitting that made
  us pin rotation/scale in the phase-correlation aligner.
- **Better detectors.** Detection is deliberately simple (threshold + connected
  components). The downstream FASTrack package has stronger object/filament
  detection that could be borrowed for hard movies (faint filaments, dense
  fields), kept behind the same `frame_features` seam.

## Generalising for the wider OptoSplit community

This algorithm is tailored to head/filament data — an important but *niche* case.
To serve general users of optomerge, the following are flagged for future work:

- **General-purpose alternative aligners** behind the same `Aligner` seam, for
  channels that are not point-vs-line: **ECC** (enhanced correlation coefficient,
  handles affine + illumination differences), **mutual-information** registration
  (the standard for genuinely different-modality channels), and **feature-based**
  (ORB/SIFT keypoints). Any of these would be a more robust default than plain
  phase correlation for cross-structure channels.
- **Method auto-selection / guidance.** Help users pick an aligner from their
  data (e.g. fall back from phase correlation to a feature/MI method when the
  correlation peak is weak), and document when to reach for each.
- **A structure-agnostic feature objective.** The head-to-filament idea
  generalises to "minimise the distance between the salient features of channel A
  and channel B"; exposing the detector and the distance target as pluggable
  pieces would let non-filament users reuse the geometry-based approach.
- **Broader validation.** Test across more OptoSplit datasets and channel types,
  and tune the default parameters so they work out-of-the-box for the largest
  fraction of users (tracked alongside the general channel-order / segmentation
  robustness work).
