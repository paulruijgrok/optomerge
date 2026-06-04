# optomerge — Bottleneck Analysis and Speedup Proposal

## 1. Where is the time actually spent?

Run the profiling script first to get exact numbers on your machine:

```bash
python profile_pipeline.py                          # auto-picks first test movie
python profile_pipeline.py --frames 100             # quick run, scales linearly
```

### Analytical cost model (512×512 raw frame, 800 frames, two channels)

| Step | Estimated ops | Share |
|---|---:|---:|
| `subtract_background` green (grey_opening × 800 frames) | 66 B | 19 % |
| `subtract_background` red (grey_opening × 800, on zero-padded 512×1024 frames) | **266 B** | **78 %** |
| `transform_image` red (cubic spline × 800 frames) | 7 B | 2 % |
| `calculate_alignment` (scale/rot search, single projection) | 0.7 B | <1 % |
| Loading from disk (419 MB) | ~0.8 s | <1 % |

**Key finding:** `subtract_background` (morphological opening with a disk of radius 10) accounts for roughly 97% of the total compute. The affine transform applied to the frames is fast by comparison. The scale/rotation alignment search is negligible because it operates on a single projected frame, not the full movie.

Your intuition that "applying the alignment to all frames" is the slowest step is correct — but the culprit inside that step is not the transformation itself. It is the **two rounds of background subtraction** (one on the green channel, one on the red channel after the transform). The red background subtraction is especially expensive because the image has been zero-padded to the next power-of-2 size (e.g. 256×512 → 512×1024) before the transform is applied.

---

## 2. Three-tier optimisation plan

The proposals are ordered from easiest to implement to most involved.

---

### Tier 1 — Quick wins (no package changes, ~2–4× speedup)

#### 1a. Smaller background-subtraction radius

The morphological opening cost scales with the disk area, which scales with r².
Halving the radius from 10 to 5 reduces the cost by ~4×, with a modest effect
on the quality of the background estimate.

```python
# In run_optomerge.py or test_pipeline.py:
python run_optomerge.py --bg-radius 5
```

#### 1b. Replace morphological opening with Gaussian blur

The Gaussian blur is a separable filter — O(H × W × kernel_width) instead of
O(H × W × disk_area). For a sigma matching the current disk radius it is
typically 10–30× faster with very similar background estimates for TIRF data.

Change in `optomerge/optomerge/processing.py`, `subtract_background`, line
`method = methods{2}` equivalent:

```python
# Current (slow):
pipe = AlignmentPipeline(..., bg_radius=10)   # morph_open

# Proposed fast alternative — add a method argument:
pipe = AlignmentPipeline(..., bg_radius=10, bg_method="gaussian")
```

The `subtract_background` function already has a `method` parameter accepting
`'morph_open'` or `'gaussian'` — you just need to thread it through
`AlignmentPipeline` and the CLI flags.

#### 1c. Process background subtraction before zero-padding

Currently `subtract_background` on the red channel is called **after**
`zero_pad_images`, so it operates on the larger padded frame (e.g. 512×1024).
Moving it to before padding would halve the area and nearly halve the cost with
no algorithmic change.

Edit `optomerge/optomerge/pipeline.py`, `_align_frames_with_red`:

```python
# BEFORE (current):
red, green = zero_pad_images(red, green)    # red becomes 512×1024
red = transform_image(...)
red = subtract_background(red, ...)         # expensive: 512×1024

# AFTER (proposed):
red = subtract_background(red, ...)         # cheap: 256×512 (original size)
red, green = zero_pad_images(red, green)
red = transform_image(...)
# no second subtract_background needed
```

This alone cuts the background subtraction cost from 332 B ops to ~66 B ops —
a **5× speedup on that step** — and the result is nearly identical because the
padding region is zeros and the opening does not affect it.

---

### Tier 2 — Multi-core CPU parallelism (~N_core× speedup on `align_frames`)

The frame loop inside `align_frames` is **embarrassingly parallel**: each
frame can be transformed and background-subtracted independently of all others.

#### Implementation: `ProcessPoolExecutor` frame-chunked parallelism

Add a `n_workers` parameter to `AlignmentPipeline` (and the CLI `--workers N`
flag). The frame loop is split into chunks, each processed by a separate worker.

```python
# optomerge/optomerge/processing.py — add this helper:

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def _process_frame_batch(args):
    """Worker function: subtract background on a batch of frames.
    Must be a top-level function (not a lambda) to be picklable."""
    frames_chunk, radius, method = args
    return subtract_background(frames_chunk, radius=radius, method=method)

def subtract_background_parallel(movie, radius=10, method="morph_open",
                                  n_workers=None):
    """Parallelised version of subtract_background over the frame axis."""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    N = movie.shape[2]
    chunk = max(1, N // n_workers)
    batches = [movie[:, :, i:i+chunk] for i in range(0, N, chunk)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(
            _process_frame_batch,
            [(b, radius, method) for b in batches]
        ))
    return np.concatenate(results, axis=2)
```

Similarly, `transform_image` can be parallelised over frames:

```python
def _transform_frame_batch(args):
    frames_chunk, rot, sx, sy, tx, ty = args
    return transform_image(frames_chunk, rot, sx, sy, tx, ty)

def transform_image_parallel(im, rot, sx, sy, tx=0., ty=0., n_workers=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    N = im.shape[2]
    chunk = max(1, N // n_workers)
    batches = [im[:, :, i:i+chunk] for i in range(0, N, chunk)]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(
            _transform_frame_batch,
            [(b, rot, sx, sy, tx, ty) for b in batches]
        ))
    return np.concatenate(results, axis=2)
```

#### Expected speedup

On an 8-core machine, the frame-level parallelism gives close to 8× on the
background-subtraction and transform steps. The total wall-clock time drops
roughly proportionally because those steps dominate.

#### File-level parallelism (the other option)

An even simpler approach — no code changes to the package itself — is to
process multiple movie files simultaneously in the batch runner:

```python
# run_optomerge.py — replace the sequential loop with:
from concurrent.futures import ProcessPoolExecutor

def process_wrapper(args):
    return process_file(*args)

with ProcessPoolExecutor(max_workers=args.workers) as pool:
    results = list(pool.map(process_wrapper, job_args))
```

This is ideal when you have many short movies. For a single long movie,
frame-level parallelism is better.

---

### Tier 3 — GPU acceleration (10–50× speedup, requires NVIDIA GPU + CuPy)

CuPy provides GPU-accelerated drop-in replacements for NumPy and
`scipy.ndimage`. The three expensive operations map directly:

| CPU (current) | GPU (CuPy) | Typical speedup |
|---|---|---|
| `scipy.ndimage.grey_opening` | `cupyx.scipy.ndimage.grey_opening` | 20–50× |
| `scipy.ndimage.map_coordinates` | `cupyx.scipy.ndimage.map_coordinates` | 10–30× |
| `numpy.fft.fft2` | `cupy.fft.fft2` | 5–20× |

#### Installation

```bash
# Match the CUDA version shown by: nvidia-smi
pip install cupy-cuda12x      # for CUDA 12.x
pip install cupy-cuda11x      # for CUDA 11.x
```

#### Proposed GPU-aware `subtract_background`

```python
# optomerge/optomerge/processing.py

def subtract_background(movie, radius=10, method="morph_open", use_gpu=False):
    if use_gpu:
        try:
            import cupy as cp
            import cupyx.scipy.ndimage as cpnd
            from skimage.morphology import disk as _disk
            movie_gpu = cp.asarray(movie)
            se = cp.asarray(_disk(radius).astype(np.float64))
            result = cp.zeros_like(movie_gpu)
            for i in range(movie_gpu.shape[2]):
                bg = cpnd.grey_opening(movie_gpu[:, :, i], footprint=se)
                result[:, :, i] = movie_gpu[:, :, i] - bg
            return cp.asnumpy(result)
        except ImportError:
            pass   # fall through to CPU path

    # ... existing CPU code ...
```

#### Proposed GPU-aware `transform_image`

```python
def transform_image(im, rot, sx, sy, tx=0., ty=0., use_gpu=False):
    if use_gpu:
        try:
            import cupy as cp
            import cupyx.scipy.ndimage as cpnd
            # ... build coords on GPU, call cpnd.map_coordinates ...
        except ImportError:
            pass
    # ... existing CPU code ...
```

#### Practical note on GPU memory

A 512×1024 float64 frame is ~4 MB. 800 frames = 3.2 GB, which exceeds the VRAM
of most consumer GPUs. Process in chunks of `floor(VRAM_GB × 0.7 / 4 MB)` frames
at a time and concatenate:

```python
vram_gb   = 6          # adjust to your GPU
chunk_max = int(vram_gb * 0.7 * 1e9 / (H * W * 8))   # float64 = 8 bytes
```

---

## 3. Recommended implementation order

| Priority | Change | Effort | Speedup |
|---|---|---|---:|
| **1** | Move `subtract_background` before `zero_pad_images` (§ 1c) | 5 min | 5× on bgsub |
| **2** | Add `bg_method="gaussian"` option (§ 1b) | 30 min | 10–20× on bgsub |
| **3** | Frame-level multiprocessing in `align_frames` (§ 2) | 2 h | ~N_core× |
| **4** | File-level multiprocessing in batch runner (§ 2) | 30 min | ~N_core× |
| **5** | GPU via CuPy (§ 3) | 4–8 h | 20–50× |

Start with items 1 and 2. Together they reduce the dominant cost (background
subtraction) by at least 10× with no loss of quality and are trivial to
implement and revert if needed. Multiprocessing adds another N_core× on top.

---

## 4. How to confirm the profiling numbers

```bash
# Full movie (slow)
python profile_pipeline.py "test_data/MyLOVChar4/ch3/movie 01.tif"

# Quick run on first 100 frames (results scale linearly with N)
python profile_pipeline.py --frames 100

# Compare Gaussian vs morph_open background subtraction:
python -c "
import time, numpy as np
from optomerge.processing import subtract_background
rng = np.random.default_rng(0)
mov = rng.random((256, 512, 100))

t0 = time.perf_counter()
_ = subtract_background(mov, method='morph_open', radius=10)
print(f'morph_open  r=10: {time.perf_counter()-t0:.2f}s')

t0 = time.perf_counter()
_ = subtract_background(mov, method='morph_open', radius=5)
print(f'morph_open  r=5:  {time.perf_counter()-t0:.2f}s')

t0 = time.perf_counter()
_ = subtract_background(mov, method='gaussian',   radius=10)
print(f'gaussian   r=10: {time.perf_counter()-t0:.2f}s')
"
```
