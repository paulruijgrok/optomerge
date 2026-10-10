# optomerge — project guide for Claude

Python package to excise and register multichannel sub-images from raw
OptoSplit-style movie stacks (several channels packed into one camera frame).
`src/` layout. Pluggable aligners behind the `Aligner` seam: `phase`, `feature`,
`ecc`, `mi`, `keypoint`. Keep it general for any OptoSplit-style data;
data-specific heuristics should be opt-in.

## Entry points

- `run_optomerge.py` is the batch runner. It is fail-isolated, resumable,
  `--workers N`, and `--num-shards/--shard-index`.
- `hpc/` holds the SLURM scripts (see `hpc/README.md`).
- `test_pipeline.py` and `profile_pipeline.py` are diagnostics and profiling.

## One project, two machines

The MacBook and Sherlock (Stanford HPC) are one project, kept in sync through
GitHub (`origin`). Sherlock just runs the work on larger resources. Pull before
starting a session and push when it ends, on either machine. Don't keep
machine-local copies of anything that belongs to the project: code, skills,
journals and `CLAUDE.md` all live in git. Sherlock specifics (env in
`$GROUP_HOME/envs/optomerge`, data on `$SCRATCH`, `possu` partition) are in
`hpc/README.md`.

## Conventions

- Follow the **pipeline-engineering-standards** skill
  (`.claude/skills/pipeline-engineering-standards/`, tracked in the repo):
  - Run the full `pytest` suite before every commit.
  - Write real commit and PR descriptions (what/why, verification, future work).
  - Sweep for dead code before committing.
  - Optimise in order: correctness → single-core → parallel.
- Work on a feature branch and merge to `main` only when complete and green.
- Set `--channel-order` explicitly, because `auto` is unreliable.
- Docs live in `docs/`. The README is a map that links out to them.

## Session journaling (standard operating procedure)

At the end of any session with non-trivial work, record it by default; don't wait
to be asked. Non-trivial means changes to code, config or dependencies,
debugging, or analysis runs.

1. **Local journal**: `journals/YYYY-MM-DD_<short-slug>.md` in this repo. This is
   the source of truth. Include:
   - what changed and why
   - exact commands
   - files and commits touched
   - an explicit *open items / next steps* section
2. **Notion mirror**: a dated sub-page titled `YYYY/MM/DD <short title>` under the
   **Optomerge log** hub (<https://app.notion.com/p/399daf1f293b80b8be8ffa12b41596ac>).
   Mirror the local journal's structure: **TL;DR**, **Detailed narrative**, a
   **Files / commits** table, and **Open items / next steps**. Point back to the
   local journal file.
