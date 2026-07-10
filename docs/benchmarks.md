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
12-CPU machine, cv2 backend, each worker capped to ~cores/workers threads:

| workers | total | speedup |
|--------:|------:|--------:|
| 1 | 49.6s | 1.00× |
| 2 | 39.3s | 1.26× |
| 3 | 35.6s | **1.39×** |
| 4 | 39.2s | 1.27× |
| 6 | 43.0s | 1.15× |

**Reading it.** A single movie already saturates all cores (the kernels thread
over frames), so parallelism only overlaps the not-fully-parallel per-movie work
(load, alignment, RGB assembly). On this laptop throughput peaks at ~3 workers
and then *declines*: each movie holds multi-GB float64 intermediates (the
pow2-padded moving channel is ~512×1024×800 ≈ 3.3 GB, the RGB buffer ~2 GB), so
several in flight exhaust RAM / memory bandwidth before they exhaust cores.

**Scaling.** File-level parallelism is embarrassingly parallel and the right
primitive for scale-out. On a machine with ample RAM per core (e.g. a cluster
node) the curve should keep climbing toward the core count rather than peaking
early. For a cluster, a job array can point every node at the same `--output`
(resume-by-skip means each node processes its share and skips others' finished
files), with `--workers` set to that node's core count. `--reuse-alignment first`
is kept sequential (per-set dependency) and is not parallelised.

**Guidance.** Pick `--workers` to fit memory, not just cores: on a RAM-limited
laptop a few workers is the sweet spot; on a high-memory node scale toward the
core count. Reducing the per-movie memory footprint (float32 intermediates,
tighter transform padding) would raise the laptop sweet spot *and* speed the
single-movie path — the planned next step.
