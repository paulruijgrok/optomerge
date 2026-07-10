# Benchmarks

Performance notes for optomerge. Numbers are indicative (one machine, one
dataset); re-measure with `profile_pipeline.py` and `run_optomerge.py --workers`
on your own hardware.

## Single-movie: OpenCV backend

`profile_pipeline.py` on a 512×512, 800-frame, two-channel movie (`MyLOVChar4/ch3`,
12 logical CPUs):

| backend | total | merge (step 5) | note |
|---------|------:|---------------:|------|
| scipy   | 114.8s | 46.6s | background subtraction ~70% of the merge |
| cv2     | 23.8s | 13.3s | **4.8× overall** |

OpenCV is a default dependency for this reason. After cv2 the merge is dominated
by the affine transform (~4s, on the pow2-padded canvas) and RGB assembly
(norm/crop/build/concat, ~6s); background subtraction is ~3s.

## Batch: file-level parallelism (`--workers`)

`run_optomerge.py` over 6 movies (`MyLOV1Char/20170803`, ~800 frames each), same
12-CPU machine, cv2 backend, each worker capped to ~cores/workers threads. Two
runs: float64 merge (before Phase 3) and float32 merge (`precision = "single"`,
now the default). Speedup is vs the float64 single-worker baseline (49.6s):

| workers | float64 | float32 | float32 speedup |
|--------:|--------:|--------:|----------------:|
| 1 | 49.6s | – | 1.00× (baseline) |
| 2 | 39.3s | 37.2s | 1.33× |
| 3 | 35.6s | 33.1s | 1.50× |
| 4 | 39.2s | 34.5s | 1.44× |
| 6 | 43.0s | **32.1s** | **1.55×** |
| 8 | – | 36.1s | 1.37× |

**Reading it.** A single movie already saturates all cores (the kernels thread
over frames), so parallelism only overlaps the not-fully-parallel per-movie work
(load, alignment, RGB assembly). Throughput is bounded by memory: each movie
holds large intermediates (the pow2-padded moving channel and the accumulated RGB
buffer), so several in flight exhaust RAM / memory bandwidth before they exhaust
cores. Under float64 throughput peaked at ~3 workers then *declined* (6 workers
was slower than 3). Halving the per-movie memory with float32 lowered every point
and moved the sweet spot to 6 workers (best 35.6s → 32.1s) — with more headroom,
more movies fit before the memory wall. This is the memory-bound signature, and
exactly why cluster nodes with ample RAM/core will scale further.

**Scaling.** File-level parallelism is embarrassingly parallel and the right
primitive for scale-out. On a machine with ample RAM per core (e.g. a cluster
node) the curve should keep climbing toward the core count rather than peaking
early. For a cluster, a job array can point every node at the same `--output`
(resume-by-skip means each node processes its share and skips others' finished
files), with `--workers` set to that node's core count. `--reuse-alignment first`
is kept sequential (per-set dependency) and is not parallelised.

**Guidance.** Pick `--workers` to fit memory, not just cores: on a RAM-limited
laptop a few workers is the sweet spot; on a high-memory node scale toward the
core count.

## Merge precision (float32)

The merge runs in **float32** by default (`[processing].precision = "single"`),
halving its multi-GB intermediates and the accumulated RGB buffer vs float64.
This is effectively free: cv2 already computes in float32 internally and the
output is 8-/16-bit, so the rendered result is unchanged (the 8-bit render is
byte-identical to the float64 path in tests). Calibration/alignment stay float64.
Set `precision = "double"` for bit-for-bit float64 fidelity. Lower per-movie
memory both speeds the single-movie path and lets more `--workers` fit in RAM
before the memory wall — i.e. it raises the laptop sweet spot and improves
cluster scaling.

## Future work (not yet done)

- **Tighter transform padding.** `zero_pad_images` pads each channel to the next
  power of two (and its `+0.1` nudge doubles dims that are already powers of two,
  so a 256×512 channel becomes 512×1024 — a 4× area blow-up). The moving-channel
  transform (~4s, the top single-movie kernel after cv2) and its bg-sub both run
  on that oversized canvas. Padding only to a snug margin (content + fitted
  translation + rotation/scale reach) is an estimated ~2.5–3× cut on those steps
  (~3s/movie, ~13% of single-movie latency) and a matching drop in that
  intermediate's memory. **Caveat:** the transform is centred on the image, so
  the new padding must keep the content exactly centred (even padding per side)
  to preserve the rotation centre, guarded by a byte-identical output regression
  test. Flagged; needs a fidelity sign-off before implementing.
- **GPU (CuPy).** `grey_opening` / `map_coordinates` / FFT have GPU drop-ins;
  not yet ported.
