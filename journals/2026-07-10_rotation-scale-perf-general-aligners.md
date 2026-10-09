# 2026-07-10/11 — Rotation/scale, performance pass, general aligners (backfill)

*Written retroactively on 2026-10-09 from the git history, run logs and the
2026-07-10 Notion entry. The general-aligner work and the 31-movie comparison
were never journaled at the time.*

Notion mirrors: *2026-07-10 — Rotation/scale registration + performance pass*
(written at the time) and *2026/07/11 General-purpose aligners + first real-data
comparison (backfill)*, both under the Optomerge log hub.

## TL;DR

- **Constrained rotation + scale** on the feature aligner (opt-in
  `--fit-rotation` / `--fit-scale`, bounded ±2° / ±2 %), validated on
  `MyLOVChar4/ch3` movie 03. Merged as **PR #4**.
- **Performance pass** (branch `perf/framebunch-parallelism`), merged as **PR #5**
  on 2026-07-10 11:59:
  - OpenCV became a core dependency: 114.8 s → 23.8 s per movie.
  - `--workers N` file-level parallelism with per-worker thread caps.
  - The merge runs in float32 by default.
  - Benchmarks are in `docs/benchmarks.md`.
- **General-purpose aligners** (ECC, mutual information, ORB/SIFT keypoints) on
  `feature/general-aligners`. Committed locally 07-10 22:19, **not pushed**.
  They were later rebased and merged on 2026-10-09.
- **Candidate public validation datasets** compiled in
  `docs/candidate_datasets.md` (left uncommitted until 2026-10-09).
- **First real-data comparison** of phase / ECC / MI on all 31 test movies
  (07-11), summarised below.
- 27 raw MyLOV1Char movies copied into `test_data/` for broader validation.

## What changed and why

### Rotation + scale (PR #4, `82a19de`)
The translation-only feature aligner left a field-dependent residual, the
signature of a small optical rotation/scale between the two OptoSplit halves. A
bounded Nelder–Mead re-refine of (t, rot, isotropic scale), seeded at the
translation optimum, fixes it. It is off by default.

### Performance (PR #5)
The work was profiling-first; see `docs/benchmarks.md`.
- **Windowed bg-sub:** `bec137b` was tried and reverted (`9460924`). It was slower
  on real data because the content bbox is roughly the whole canvas.
- **OpenCV as a core dependency** (`0257aa8`): bg-sub was about 70 % of the merge,
  and cv2 is roughly 5–20× faster on it.
- **`--workers`** (`a1d1d1a`) plus the oversubscription fix (`d993e9f`, which caps
  cv2/BLAS threads per worker).
- **float32 merge** (`413594c`): this halves memory and moved the laptop sweet spot
  from 3 to 6 workers (best 32.1 s for 6 movies, 1.55×).

### General aligners (`bca13f4`, rebased to `1cb50e6`)
This is for OptoSplit-style data that isn't heads-on-filaments.
- New `general_registration.py`: `EccAligner`, `MutualInformationAligner` and
  `FeatureMatchAligner`.
- `Transform.from_affine_matrix` converts library affines into the native
  convention (round-trip ~1e-14).
- Wired into the config (`method = ecc|mi|keypoint`), the CLI
  (`--ecc/--mi/--keypoint`) and `optomerge.example.toml`.
- 15 new tests. Docs are in `docs/general_registration.md`.

### First comparison on real data (07-11)
All 31 movies (MyLOV1Char, MyLOVChar, MyLOVChar4) were run with
`--channel-order top_green_fils_bottom_red_heads` and `--reuse-alignment none`
under default acceptance limits:

| method | rejected | crashed | time | rejection reasons |
|---|---|---|---|---|
| phase | 9 | 0 | 4m 11s | scale 2.7–5 %; movie 03 translation (−57, 104) px |
| ecc | 2 | 0 | 4m 31s | rotation 2.9–3.7° (MyLOVChar ch12 01, 03) |
| mi | 0 | 0 | 3m 47s | — |

Logs are in `output_{phase,ecc,mi}/run_log.txt` (git-ignored). Acceptance is a
plausibility gate, not an accuracy score. The feature aligner was not part of
this sweep.

## Commands (reconstructed)

```bash
python run_optomerge.py --channel-order top_green_fils_bottom_red_heads --align-method phase --output output_phase
python run_optomerge.py --channel-order top_green_fils_bottom_red_heads --ecc --output output_ecc
python run_optomerge.py --channel-order top_green_fils_bottom_red_heads --mi  --output output_mi
```

## Files / commits

| Commit | What |
|---|---|
| `82a19de` / PR #4 | Constrained rotation + scale on the feature aligner |
| `bec137b`, `9460924` | Windowed bg-sub, then reverted |
| `0257aa8` | OpenCV as a core dependency |
| `a1d1d1a`, `d993e9f` | `--workers`, plus the per-worker thread cap |
| `ddfd44d`, `43b46f8`, `1f46283` | `docs/benchmarks.md` |
| `413594c` | float32 merge |
| `406d3c2` / PR #5 | Perf branch merged to `main` |
| `bca13f4` → `1cb50e6` | General aligners (rebased 2026-10-09) |
| `docs/candidate_datasets.md` | Public datasets (committed 2026-10-09 as `70d4f83`) |

## Open items / next steps (as of 07-11)

- Push and merge `feature/general-aligners` (done 2026-10-09).
- Do a visual overlay check of MI vs feature.
- Validate on public data (Deep-LASI first).
- Tighter transform padding: about 3 s per movie, but it needs a fidelity sign-off
  because of the rotation centre.
- Port GPU (CuPy).
- `--channel-order auto` is still unreliable.
- `SPEEDUP_PROPOSAL.md` is stale and should be marked as such.
- Sidecar save/load of a fitted alignment.
