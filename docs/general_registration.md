# General-purpose registration

`optomerge` ships two channel-pair-specific aligners — `phase` (FFT phase
correlation) and `feature` (head-to-filament distance, see
[feature_registration.md](feature_registration.md)). This page documents the
three *general* aligners added for users whose data is not the point-vs-line
OptoSplit case: any two-image registration where the channels share structure, or
are multimodal, or are offset by more than a small shift.

All three live in `optomerge.general_registration`, plug into the same `Aligner`
seam (`align(reference, moving) -> Transform`), operate on the normalised channel
projections, and require OpenCV (`pip install opencv-python-headless`). They are
opt-in; the default `phase` behaviour is unchanged.

## Which method for which data

| Method | CLI | Best when | Needs |
|---|---|---|---|
| **ECC** | `--ecc` | Channels image *similar* structure; intensities related by gain+bias (same beads/cells in two colours). Sub-pixel, dense. | Small–moderate shift, shared texture |
| **Mutual information** | `--mi` | *Multimodal* channels whose brightnesses are statistically dependent but not linearly related (a stain vs a label). | Shared structure, bounded shift |
| **Keypoint (ORB/SIFT)** | `--keypoint` | Larger shifts/rotations, partial overlap; richly textured images. | Enough corner-like, matchable texture |

Rule of thumb: try `ecc` first for two similar-looking channels; reach for `mi`
when the channels look genuinely different; use `keypoint` when the offset is too
large for a local intensity optimiser to lock on, or when there is a rotation.

## The methods

### ECC — `EccAligner`

OpenCV's Enhanced Correlation Coefficient (`cv2.findTransformECC`). A dense,
gradient-based optimiser that maximises a correlation criterion invariant to an
affine change of intensity (gain + bias), so unequal channel brightness does not
bias the fit. It refines from the identity, so it wants the true shift to be
modest (tens of pixels) and the channels to share visible structure.

```bash
python run_optomerge.py --ecc --ecc-motion euclidean --max-shift 20
```

`--ecc-motion` sets the degrees of freedom: `translation`, `euclidean` (rotation +
translation, default), or `affine` (adds scale/shear — over-fits easily, use only
with evidence). `--ecc-gauss-blur` (px, odd) pre-smooths each iteration; larger
widens the capture range at the cost of precision. The reported score is the ECC
coefficient (0–1). `max_shift`, if set, rejects a fit whose translation exceeds it
(returns identity), guarding against ECC wandering when the channels share little.

### Mutual information — `MutualInformationAligner`

Maximises the mutual information between the two channels: how much knowing one
channel's intensity tells you about the other's. MI is maximal at registration for
*any* consistent intensity relationship, not just a linear one, which makes it the
robust choice for multimodal channels where correlation fails. The fit is a
bounded coarse-to-fine search — a grid over integer translations seeds a
derivative-free refinement over the enabled degrees of freedom. MI is evaluated on
the joint intensity histogram over the region where the warped moving channel is
in-bounds, so out-of-frame padding cannot inflate the score.

```bash
python run_optomerge.py --mi --mi-bins 32 --max-shift 20            # translation only
python run_optomerge.py --mi --fit-rotation --fit-scale --max-shift 20   # + bounded rot/scale
```

`--mi-bins` sets the joint-histogram resolution. Rotation/scale are opt-in via
`--fit-rotation` / `--fit-scale` and bounded by `feature_rotation_max_deg` /
`feature_scale_max_pct` (shared with the feature aligner). `max_shift` bounds the
search (defaults to 20 px if left at the `0` "unconstrained" sentinel). The score
is the final mutual information (nats).

### Keypoint matching — `FeatureMatchAligner`

Detects keypoints (ORB or SIFT), matches descriptors with a Lowe ratio test, and
robustly fits a transform to the surviving correspondences with RANSAC. Unlike the
intensity methods this is not a local optimiser, so it copes with larger shifts and
rotations and with partial overlap — but it needs enough shared, corner-like
texture in *both* channels to find matches, so it suits beads/speckle/cell-edge
images rather than sparse point sources.

```bash
python run_optomerge.py --keypoint --keypoint-detector sift --keypoint-motion similarity
```

`--keypoint-detector` is `orb` (fast, license-free) or `sift` (more robust to
scale/contrast; OpenCV ≥ 4.4). `--keypoint-motion` is `similarity` (rotation +
uniform scale + translation, default) or `affine` (adds shear/anisotropy). Further
knobs in `[alignment]`: `keypoint_max_features`, `keypoint_ratio` (Lowe threshold,
lower = stricter), `keypoint_min_matches`, `keypoint_ransac_thresh` (px). The score
is the RANSAC inlier fraction. Too few matches/inliers returns identity rather than
a bad fit.

## Convention note

Every method estimates the **destination → source** (reference-grid → moving-grid)
affine in top-left-origin pixel coordinates, then converts it with
`Transform.from_affine_matrix`, the exact inverse of the pipeline's own
`_build_cv2_matrix`. So the fitted `Transform` is precisely what the pipeline
applies to the moving channel — the same sign/axis convention as the `phase` and
`feature` aligners. The round-trip is verified to ~1e-14 in
`tests/test_general_registration.py`.

Library calls that fail to converge (no texture, too few matches, mismatched
shapes) return an identity `Transform` with `score = 0` rather than raising, so one
hard channel pair cannot sink a batch run.

## Failure modes

- **ECC** can converge to a wrong local optimum if the channels share little
  structure or the shift is large; use `--max-shift` as a guard and/or a larger
  `--ecc-gauss-blur`, or switch to `keypoint`.
- **MI** is only as good as its coarse grid seed for large shifts; raise
  `max_shift` (cost grows quadratically) or pre-crop.
- **Keypoint** returns identity when detectors find too little texture (e.g. a
  sparse point channel) — that is the case the `feature` aligner exists for.

## First comparison on real data (2026-07-10/11)

All 31 raw movies in `test_data/` (MyLOV1Char, MyLOVChar, MyLOVChar4), channel
order forced to `top_green_fils_bottom_red_heads`, `--reuse-alignment none`,
default acceptance limits (|t| <= 25 px, |rot| <= 2 deg, |scale-1| <= 2 %):

| method | rejected by acceptance | crashed | wall time (31 movies) |
|---|---|---|---|
| `phase` | 9 (scale 2.7-5 %, one translation of ~120 px) | 0 | 4m 11s |
| `ecc` | 2 (rotation 2.9-3.7 deg, MyLOVChar ch12) | 0 | 4m 31s |
| `mi` | 0 | 0 | 3m 47s |

Acceptance is a sanity gate, not an accuracy measure: these numbers show which
methods produce *plausible* transforms, not which is most accurate. Overlays
still need a visual check, and the feature aligner (the tailored method for
heads-on-filaments data) was not part of this sweep. Logs: `output_{phase,ecc,mi}/run_log.txt`
(not committed).

## Roadmap / future work

- **Method auto-selection / guidance** — pick an aligner from channel statistics
  (texture overlap, intensity relationship) instead of a manual `--align-method`.
  Deliberately deferred until the methods are validated on real datasets.
- **Structure-agnostic feature objective** and borrowing FASTrack detectors for
  hard movies (tracked in the feature-registration roadmap).
- **Masked / multi-frame variants** — currently these aligners fit on the channel
  projection; a per-frame or masked-overlap variant could help low-SNR movies.
- **Broader validation** across public multichannel datasets (see
  [`candidate_datasets.md`](candidate_datasets.md)) to tune the defaults for the
  wider community.
