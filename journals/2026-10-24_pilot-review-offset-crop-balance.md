# 2026-10-24 — Pilot review: red offset bug, geometric crop, outlier-capped vmax

Notion mirror: *2026/10/24 Pilot review: red offset bug, geometric crop,
outlier-capped vmax* under the Optomerge log hub. Session on Sherlock
(`sh04-ln04`, with jobs on `possu`), branch `feature/sherlock`.

## TL;DR

- **Write-up:** the 2026-10-10/11 work (Sherlock setup, exclusion, pilot) was
  journaled and mirrored to Notion (`40a15fb`).
- **P.'s visual review of the pilot:**
  - Some movies are visibly offset (e.g. 20170803 ch08 ROI02 01, red low/right),
    possibly more so in sparse movies.
  - A few have a poor red/green balance.
- **Diagnosis:** a new script, `hpc/runs/MyLOV1Char/diagnose_alignment.py`, ran
  on all 27 movies. Four findings:
  1. **The red offset was a bug, not a bad fit.** The feature aligner fits with
     the red/green crops top-left aligned, but the merge centres them in a padded
     canvas. Unequal crops gave a red offset of about half the size difference.
     - 20170803 was 4–6 px low.
     - After subtracting the predicted offset, the residual was median 0.40 px.
     - The fit itself is stable within 0.25 px across frame subsets.
     - Filament density was not a factor.
  2. **Balance:** one bright green aggregate (3.8–10× p99.9) set `vmax` in three
     movies, so green was crushed (R/G up to 3.85).
  3. **Over-cropping:** the crop was signal-based (time-mean > 0.01 after
     background subtraction), so sparse movies were cut to where objects go, down
     to 19% of the field. The existing `min_size_*_frac` only checks the
     *detected channel*, not the output crop.
  4. **ECC/MI** silently return identity on unequal crops.
- **Fixes**, as agreed with P. (both crop and cap are new defaults):
  - Canvas-frame conversion of bare-crop fits (`41b4b37`).
  - `[processing] crop = "footprint"` and `norm_cap = 2.0` (`e997fe4`).
- **Result on the pilot** (re-run into `aligned_feature_v2`, 27/27 ok):

  | | Pilot | v2 |
  |---|---|---|
  | Residual red offset | median 1.42, max 6.00 px | median 0.24, max 0.49 px |
  | R/G balance | 0.21–3.85 | 0.18–1.31 |
  | Output area (fraction of channel) | 0.19–0.98 | 0.94–1.00 |

- Tests: 161 passed, 1 skipped (7 new).

## What changed and why

### Diagnostic (`hpc/runs/MyLOV1Char/diagnose_alignment.py`)
For each movie it re-runs the calibration exactly as the pilot did (it reproduced
the logged transforms exactly), then records:
- **Crop geometry:** red/green crop sizes and the offset predicted by the
  padding.
- **Fit diagnostics:** detections and filament density; the cost basin of the
  objective; refits on 3 disjoint frame subsets; and a refit with a
  filament-centreline (skeleton) objective.
- **Residual in the output:** the red offset measured in the *written output*, by
  fitting R heads onto G centrelines.
- **Comparison with the old composites:** red and green each template-matched
  against the lab's old `* RGB.tif` composites.
- **Intensity:** `vmax` vs p99.9, and the output R/G balance.

It writes `summary.csv`, `summary.png` and a PNG per movie. `--aligned` selects
which output folder to check.

Pilot evidence (before the fix):

| Movie | Padding prediction | Measured in output | vs old composite |
|---|---|---|---|
| 20170803 ch08 ROI02 01 | (+6, 0) | (+5.8, +0.5) | (+5.9, +1.4) |
| 20181002 ch12 ROI01 03 | (+2, −6) | (+1.4, −5.8) | |

All 27 movies fell on the 1:1 line in `summary.png`.

### Offset fix (`41b4b37`)
- `Aligner.fits_padded_frame`: defaults to False; True for phase correlation,
  which centre-pads the pair itself.
- `calibration.canvas_transform` / `_to_padded_frame` convert a fit with
  t′ = t + (o_mov − o_ref) + (M − I)e. This is exact, including the 0/0.5 px
  rotation-centre difference.
  - It's applied in `_build_calibration`, so the single, robust and conserved
    calibrators all get it.
  - `test_pipeline.py` and `profile_pipeline.py` use it too; the
    feature-detection overlay keeps the raw fit.
- Keypoint gets the same conversion. ECC/MI still need equal crops (open item).
- Re-run `aligned_feature_padfix`: residual median 0.37 px, max 0.74 px.
  20170803 now agrees with the old composites to about 1 px.

### Crop + intensity (`e997fe4`)
P. chose both as new defaults.

- **`crop = "footprint"`:** the bounding box where the reference and every warped
  moving channel have data. It uses the valid-pixel masks (outside the scrub
  mask), padded and warped exactly like the data, with fully-covered pixels only.
  - Legacy `crop = "signal"` is kept.
  - A degenerate (empty) footprint falls back to the signal rule.
- **`norm_cap = 2.0`:** `vmax` ≤ vmin + 2·(p99.9 − vmin) of the despeckled
  projection.
  - Simulated beforehand: only 4/27 movies change, and the R/G range drops from
    0.19–3.85 to 0.19–1.13.
  - A pure p99.9 limit was rejected: it changes every movie and can clip sparse
    heads (see the `_despeckle` docstring).
  - Limit: it catches only outliers covering < 0.1% of the channel.
- **`Calibration.notes`:** QC messages. The runner logs them as `[QC] ...`.
- **Docs:** README known-limitations, `optomerge.example.toml`, and
  `docs/feature_registration.md` ("Fit frame vs merge canvas").

## Commands

```bash
# diagnostics (one movie to test, then all), possu
srun -p possu -c 16 --mem=64G -t 00:30:00 bash -c 'source hpc/env.sh; \
  python hpc/runs/MyLOV1Char/diagnose_alignment.py $SCRATCH/optomerge_runs/MyLOV1Char/diagnostics_2026-10-24 --workers 16'

# re-run the pilot with the fixes into a new folder, then diagnose it
RUN=hpc/runs/MyLOV1Char; OUT=$SCRATCH/optomerge_runs/MyLOV1Char/aligned_feature_v2
for d in 20170803 20180620 20181002; do
  python run_optomerge.py --config $RUN/base.toml $RUN/$d.toml --output $OUT/$d --workers 6
done
python $RUN/diagnose_alignment.py $SCRATCH/optomerge_runs/MyLOV1Char/diagnostics_2026-10-24_v2 \
  --workers 12 --aligned aligned_feature_v2

# tests (in a job)
srun -p possu -c 4 --mem=16G -t 00:20:00 bash -c 'source hpc/env.sh; python -m pytest -q'
```

Possu had only ~13 free cores today; 32-CPU requests were refused, so the jobs
used 12 CPUs.

## Files / commits

| Commit | What |
|---|---|
| `40a15fb` | `journals/2026-10-10_…`, `hpc/runs/MyLOV1Char/diagnose_alignment.py` |
| `41b4b37` | `registration.py` (`fits_padded_frame`), `calibration.py` (`canvas_transform`), `test_pipeline.py`, `profile_pipeline.py`, `docs/feature_registration.md`, tests; `--aligned` in the diagnostic |
| `e997fe4` | `movie.py` (`_footprint_crop`), `pipeline.py`, `calibration.py` (`norm_cap`, `notes`), `config.py`, `run_optomerge.py`, `optomerge.example.toml`, README, `tests/test_crop_and_limits.py` |
| `64b1d86` | this journal (part 1) |
| `b456e6a` | ECC/MI equal-shape trimming (`fit_in_canvas`) |
| `26bc5b8` | `compare_aligners.sbatch` |
| `bd722ff` | MI coarse-search sign fix |
| `76b5a90` / `7c3e88f` | objects normalisation / merge |
| `59fb6c0` | `RGBMovie` docstring |

Outputs are in `$SCRATCH/optomerge_runs/MyLOV1Char/`:
- `aligned_feature/`: the original pilot
- `aligned_feature_padfix/`: offset fix only
- `aligned_feature_v2/`: all fixes
- `diagnostics_2026-10-24{,_padfix,_v2}/`
- `scratch_scripts/footprint.py`: a one-off comparison of footprint vs old crops

## Part 2 (same day): ECC/MI on unequal crops, MI sign bug, "objects" normalisation

P. asked to (1) fix the unequal-crop handling for ECC and MI and compare them on
the pilot, and (2) set brightness from the typical intensity of the
heads/filaments, letting very bright blobs saturate.

### ECC/MI (`b456e6a`, `bd722ff`)
- **Unequal crops.** `Aligner.needs_equal_shapes` is now True for ECC and MI.
  `calibration.fit_in_canvas` (which replaces `canvas_transform`) trims both
  crops to their common top-left region, fits, and converts to the canvas.
  - New test: ECC on unequal crops recovers the exact canvas transform.
- **MI sign bug** (pre-existing). The coarse translation grid compared
  `moving[r - dy]` instead of `moving[r + dy]`, so its seed had the opposite
  sign.
  - The refinement rescued it on smooth synthetic textures, which is why the old
    test passed, but not on real data (about 24 px off on 20170803).
  - Fixed, with a test of the coarse search alone (`refine=False`).
  - The July MI comparison in `docs/general_registration.md` predates both fixes,
    and with unequal crops MI returned identity anyway.
- **Comparison on the pilot**
  (`hpc/runs/MyLOV1Char/compare_aligners.sbatch`, jobs 49386587 and 49393166):

  | | Median red offset | Max | Movies > 1 px | vs old composites (median) |
  |---|---|---|---|---|
  | feature | 0.24 px | 0.49 px | 0/27 | 0.57 px |
  | ECC | 1.10 px | 4.38 px | 16/26 (1 rejected: rot 2.19°) | 1.70 px |
  | MI (fixed) | 1.59 px | 11.3 px | 18/27 | 1.70 px |

  Feature is the right method for head/filament data. The residual measure may
  favour it, but the old-composite check is independent and gives the same
  ranking. ECC and MI may still suit OptoSplit data with similar structure in
  both channels.

### "objects" normalisation (`76b5a90`, merged as `7c3e88f`)
Developed in a local clone (`../optomerge-objnorm`, branch
`feature/object-norm`) so the running comparison jobs weren't affected; this
system's git 1.8 has no `worktree`.

- **New default:** `[processing] norm_mode = "objects"` with `norm_target = 0.5`.
  `"range"` is the previous rule.
- **How it works:**
  - Objects are components (≥ 4 px) of the background-subtracted,
    Gaussian(1)-smoothed sampled frames above median + 8 × the raw noise σ.
  - Each object votes with its peak (brightest 3×3 mean); the typical object is
    the median vote.
  - `vmin` is p1 of the frames; `vmax − vmin` = typical peak / 0.5.
- **Three issues found and fixed on the way:**
  1. *Clipping before background subtraction.* The merge clipped at 1 before
     subtracting the background. `Channel.normalized(clip_high=False)` now lets
     the writer clip instead.
  2. *Detector found noise.* The first detector (3×3 median, 4σ, pixel or
     component median) mostly found noise blobs: 70–320 per frame at about 4×
     the noise. Overlays confirmed that the 8σ detector marks exactly the heads
     and filaments.
  3. *`vmin` clipped the background.* With the range-mode `vmin` (min of the max
     projection), 87–94% of raw pixels sat below it and were clipped to 0, so
     objects were measured from the wrong level.
- **Result on the pilot** (`aligned_feature_v3d`, 27/27 ok):

  | | range mode (v2) | objects mode (v3d) | Target |
  |---|---|---|---|
  | Green typical object (0–255) | 68 | 120 [108–127] | 128 |
  | Red typical object (0–255) | 83 | 119 [99–175] | 128 |
  | R/G | 1.25 | 1.00 [0.80–1.63] | 1 |

  - End-to-end trace on one movie: calibration 209/190 raw counts vs output
    199/208.
  - Object pixels saturated: 2–7%, and 13% for 20170803 ch08 red (large bright
    stationary aggregates, as intended).
- **Tests:** 169 passed, 1 skipped.

Outputs in `$SCRATCH/optomerge_runs/MyLOV1Char/`:
- `aligned_ecc/`, `aligned_mi/`, `aligned_feature_v3{,b,c,d}/`
- `diagnostics_{ecc,mi,objnorm_*}/`
- `scratch_scripts/{object_levels,object_detectors*,trace_green}.py`
  (one-off analysis)

## Open items / next steps

- **P. to look at `aligned_feature_v3d`**, the current defaults (all fixes plus
  objects normalisation), rather than v2. Then merge `feature/sherlock` →
  `main`.
- `norm_target = 0.5` is a first choice; P. may prefer more or less headroom.
- The ECC/MI results doc (`docs/general_registration.md`, July section) should
  get a note that those numbers predate the two fixes, plus the new pilot
  comparison.
- The clone `../optomerge-objnorm` can be deleted (merged as `7c3e88f`).
- **`reuse_alignment = "first"`** reuses one movie's canvas-frame transform for
  movies whose crop sizes may differ. The transform should be stored in a
  crop-independent frame (raw-frame coordinates).
- Range mode only: the `vmax` cap misses outliers larger than 0.1% of the channel.
- Possibly fold the diagnostic into a general `diagnose_run.py` for any
  feature-aligned run.
- Carried over: `--channel-order auto`, sidecar alignment save/load, GPU,
  Deep-LASI validation, stale `SPEEDUP_PROPOSAL.md`.
