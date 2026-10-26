# 2026-10-26 — v3d sign-off, merge to main

Notion mirror: *2026/10/26 v3d sign-off, merge to main* under the Optomerge log
hub. Session on Sherlock (`sh03-ln08`, tests in a `possu` job), branch
`feature/sherlock`.

## TL;DR

- **P. signed off `aligned_feature_v3d`:** registration, intensity balance and
  crop size all look good on the pilot. These are the current defaults (all
  2026-10-24 fixes plus `norm_mode = "objects"`).
- **`norm_target = 0.5` is kept** as the brightness default.
- **Docs:** `docs/general_registration.md` now marks the July ECC/MI sweep as
  superseded and adds the 2026-10-24 pilot accuracy comparison (`36310d5`).
- **Merged `feature/sherlock` → `main`** (fast-forward, 17 commits) and pushed
  both branches. Tests: 169 passed, 1 skipped.
- `aligned_feature_v3d` is **not** copied off `$SCRATCH`. P. will regenerate:
  more of the dataset is being uploaded, and everything will be reprocessed in
  one go once the algorithm is final.

## What changed and why

### Docs (`36310d5`)
- The July comparison in `docs/general_registration.md` predates two fixes:
  - ECC/MI returned identity on unequal crops (now `fit_in_canvas`).
  - MI's coarse search had its sign reversed (`bd722ff`).

  Those rows are now flagged as historical.
- A new section has the pilot accuracy table (feature 0.24 px median, 0/27 > 1 px;
  ECC 1.10 px; MI 1.59 px). It comes from the 2026-10-24 journal.
- The convention note said "mismatched shapes return identity". It now explains
  how unequal crops are handled.

### Merge
`main` had no commits of its own, so `git merge --ff-only feature/sherlock`.

## Commands

```bash
# tests (in a job)
srun -p possu -c 4 --mem=16G -t 00:20:00 bash -c 'source hpc/env.sh; python -m pytest -q'
# -> 169 passed, 1 skipped (job 49664370)

git checkout main && git pull --ff-only origin main
git merge --ff-only feature/sherlock
git push origin main feature/sherlock
```

## Files / commits

| Commit | What |
|---|---|
| `36310d5` | `docs/general_registration.md`: July ECC/MI sweep superseded; pilot accuracy comparison |
| `92b4ffe..36310d5` | `main` fast-forwarded to `feature/sherlock` |
| (this commit) | `journals/2026-10-26_v3d-signoff-merge-main.md` |

## Open items / next steps

- **P. is uploading more of the dataset.** Once the algorithm is final, reprocess
  everything in one batch run (SLURM, `--num-shards`/`--workers`). Size the job
  with a small representative test first.
- Clone `~/code/optomerge-objnorm` can be deleted. It's fully merged with no local
  changes; the agent's `rm` was denied, so P. will delete it by hand.
- **`reuse_alignment = "first"`:** store the transform in a crop-independent
  (raw-frame) form.
- Range mode only: the `vmax` cap misses outliers larger than 0.1% of the channel.
- Possibly fold the diagnostic into a general `diagnose_run.py`.
- Carried over: `--channel-order auto`, sidecar alignment save/load, GPU,
  Deep-LASI validation, stale `SPEEDUP_PROPOSAL.md`.
