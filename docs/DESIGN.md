# The object-oriented design of optomerge

This document describes the architecture of the `optomerge` package. The
object-oriented public API lives in `src/optomerge/` as the domain classes
described below; the validated numerical routines they delegate to live in the
private `src/optomerge/_kernels/` subpackage.

## The objects

The sketch you provided maps cleanly onto a small set of classes. I split them
into two groups: the **domain objects** that hold data, and the **seams** —
abstractions whose only job is to make the three future goals (faster, more
robust, more general) cheap to reach.

### Domain objects

- **`Movie`** (abstract) — an ordered stack of frames plus pixel metadata.
  Owns the behaviour every movie shares: `shape`, lazy `frame(i)`, iteration in
  `bunches()`, projections, and `save()`. Crucially it holds a *reader* rather
  than a raw array, so a movie can be larger than RAM.
- **`RawMovie(Movie)`** — a movie straight off disk, channels still spatially
  packed in each frame. Adds `find_layout()` (channel detection) and
  `channels()` (splitting).
- **`RGBMovie(Movie)`** — the merged, aligned, false-coloured output. Adds the
  output-only concerns: colour-plane mapping, the shared crop range, and
  float→uint16 scaling. Its `from_channels()` classmethod is the OO home of the
  current `_align_frames_with_red` / `_align_frames_no_red`.
- **`Frame`** — one 2-D image plus its index/time and per-frame metadata (e.g.
  the blue-LED flag that currently rides along as a loose `blue_led_frames`
  array).
- **`FrameBunch`** — a contiguous block of frames processed as one unit. This is
  the granularity chosen from object motion *and* compute budget, exactly as you
  described. It is also the unit of work the executor sees, which is what makes
  the speed work tractable later.
- **`Channel`** — one crop carrying a single molecular species. It bundles the
  pixels with the bounding box, mask, and the normalisation limits that the
  current code passes around as four loose floats.
- **`ScrubImage`** — the enlarged helper derived from a `Channel` to drive
  sub-pixel alignment. It remembers its upscale factor so a transform fitted in
  scrub space can be mapped back to native pixels.

### Seams (composition, not inheritance)

- **`ChannelLayout` / `ChannelSpec`** — *data* describing how channels are
  packed and which output colour each maps to. The two hard-coded orders today
  become two factory calls; a 3-up or left/right layout becomes a new instance.
- **`Aligner`** (abstract) + **`PhaseCorrelationAligner`** — the registration
  strategy. Every aligner returns a `Transform` from the same `align()` call.
- **`Transform`** — a fitted geometric mapping that knows how to `apply()`
  itself. Replaces the loose `AlignmentParams` + scattered `transform_image`.
- **`MovieReader` / `MovieWriter`** (abstract) + TIFF concretes — pluggable I/O
  backends selected by file extension.
- **`MergePipeline`** — a thin orchestrator that wires the above together and
  walks the movie bunch by bunch. It owns no algorithms.

## How the pieces talk

```
                 MergePipeline.run()
                        |
        opens   RawMovie  ──reader──►  MovieReader (TIFF/ND2/HDF5…)
                  │
   find_layout()  │  uses           ChannelLayout ── ChannelSpec[]
                  ▼
   for each FrameBunch:                     Aligner.align(ref, moving)
        channels() ─► [Channel] ──► ScrubImage ───────────►  Transform
                          │                                     │
                          └──────────► RGBMovie.from_channels(…, transforms)
                                                  │
                                          save() ─┴─► MovieWriter (TIFF/Zarr/MP4)
```

Read it as: the pipeline is the recipe; the domain objects hold state and do the
work; the seams are the four places you swap implementations.

## Is OO the right call here?

Mostly yes — but the win comes from *bundling state with its invariants* and
from *strategy/backend seams*, not from a deep inheritance tree.

**Where OO clearly helps.**

1. *Killing parameter soup.* The current `align_frames()` function takes
   eighteen positional arguments: `bounds1, scrub1, bounds2, scrub2, t1, t2,
   rot, s1, s2, green_min, green_max, red_min, red_max, …`. Several of these are
   the same fact about two different channels. That is exactly the situation
   where an object pays off: a `Channel` owns its own bounds, mask, and
   intensity limits, so it is structurally impossible to normalise the red
   channel with the green channel's limits. This is the single biggest
   *robustness* improvement, and it is pure OO.
2. *Pluggable seams for generality.* "More input files / more output formats /
   more alignment methods" is the textbook case for the Strategy pattern.
   `MovieReader`, `MovieWriter`, and `Aligner` turn each of those axes into "add
   a subclass," with no edits to the pipeline. The current code expresses these
   choices as `if bounds2 is None` branches and hard-coded format calls, which
   grow combinatorially as you add cases.
3. *A real `Movie` hierarchy.* `RawMovie` and `RGBMovie` genuinely *are* movies
   for the purposes of shape/iteration/save, but interpret a frame differently.
   That is a legitimate use of inheritance (LSP holds for the shared surface),
   and it gives `RGBMovie` a natural place for output-only concerns.

**Where to be careful.**

1. *Don't over-deepen the hierarchy.* `Channel` should not sprout
   `GreenChannel` / `RedChannel` subclasses — the colour is data, not type.
   Likewise `FrameBunch` is one class, not a taxonomy. Keep inheritance for
   `Movie` and for the abstract seams; use composition everywhere else.
2. *Performance is a backend concern, not a class concern.* The "faster" goal
   (chunking, GPU, lazy reads) should be satisfied by the `MovieReader` /
   `FrameBunch` / `Transform.apply` boundaries, which can be reimplemented
   (Dask, CuPy, memory-mapped) without changing the public objects. OO helps
   *only* by giving those swaps a stable interface; it does not make code faster
   by itself, and naive object-per-pixel/object-per-frame designs can hurt. The
   design keeps pixels in bulk numpy arrays inside the objects for this reason.
3. *Keep a functional core under the OO shell.* The numerically heavy routines
   (phase correlation, warping, segmentation) are best left as pure functions
   that the methods call. That keeps them testable in isolation and easy to
   accelerate, with the classes providing the ergonomic, safe surface on top.

**Verdict.** OO is beneficial for optomerge, primarily as (a) data+invariant
bundling on `Channel`/`Movie` and (b) Strategy/backend seams for the three
extension axes. The recommendation is a *shallow* hierarchy (`Movie` → `RawMovie`
/ `RGBMovie`, plus abstract `Aligner`/`Reader`/`Writer`) wrapping a functional
numerical core — not a deep class tree.

## Suggested migration path

The redesign reuses the existing algorithms rather than discarding them:

1. Keep `processing.py`, `registration.py`, `transform.py`, `segmentation.py` as
   the functional core.
2. Implement `TiffReader`/`TiffWriter` by calling the existing `io.py`.
3. Fill `PhaseCorrelationAligner.align` by calling `calculate_alignment`.
4. Fill `RGBMovie.from_channels` by porting the two `_align_frames_*` methods
   into one channel-count-agnostic path.
5. Port the two channel orders into `ChannelLayout` factories that wrap
   `find_channel_bounds`.

Each step is independently testable against the current pipeline's output, so
the migration can be validated frame-for-frame.
```
