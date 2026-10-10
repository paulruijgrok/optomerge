---
name: pipeline-engineering-standards
description: How P. likes to build and maintain automated scientific analysis pipelines (Python), plus behavioral guardrails for any coding task — surface assumptions instead of guessing, keep changes minimal and surgical, verify work against explicit success criteria before calling it done. Covers README standards, pre-commit testing, commit/PR description format, cautious dead-code sweeps, correctness-first-then-performance optimization order, and a standard batch-processing framework for unattended multi-dataset runs. Use whenever creating/modifying a pipeline repo, starting any non-trivial coding task, writing a README, preparing a commit or PR, adding an analysis step, optimizing performance, cleaning up code, or building a batch runner — in FASTplus, FASTrack, optomerge, or any analysis-pipeline repo. Trigger even without the word "skill" — e.g. "about to commit this", "clean up this file", "speed this up", "add batch mode", "fix the bug", or any request to write/edit code.
---

# Pipeline Engineering Standards

Working conventions for building automated scientific analysis pipelines, and for how to approach any coding task on them. These apply across repos (FASTplus, FASTrack, optomerge, and others) — they're how P. works, not project-specific rules. When in doubt, prefer the convention here over inventing a new pattern.

The throughline: pipelines get handed datasets unattended and must produce trustworthy, reproducible results, or fail loudly and leave the rest of the run intact. Every practice below serves that goal — correctness and traceability first, convenience second.

**Tradeoff:** these guidelines bias toward caution and thoroughness over speed. For genuinely trivial changes (a typo, a one-line config tweak), use judgment rather than ceremony — but default to following them, especially for anything touching analysis logic or shared infrastructure.

## Part 1 — How to approach any task

These apply to any coding work in these repos, before you get to the pipeline-specific conventions in Part 2.

### 1. Think before coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly before implementing. If something is genuinely uncertain and matters to the outcome, ask rather than guess.
- If multiple reasonable interpretations of a request exist, present them — don't silently pick one and hope it's right.
- If a simpler approach exists than the one implied by the request, say so. Push back when warranted; a scientific pipeline is exactly the place where a well-reasoned "are you sure?" is worth more than quiet compliance.
- If something is unclear, stop and name what's confusing rather than working around the ambiguity.

### 2. Simplicity first

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested — this cuts against the temptation to over-generalize a config schema or CLI before there's a second real use case for it.
- No error handling for impossible scenarios.
- If a first pass comes out much longer than the problem seems to warrant, that's a signal to look for a simpler version before moving on.

Ask: "would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting while making an unrelated change.
- Don't refactor things that aren't broken as a side effect of a different task.
- Match existing style, even where you'd personally do it differently.
- If you notice unrelated dead code or a stale comment while working nearby, mention it rather than deleting it on the spot — see §8 for where that cleanup happens deliberately instead.
- Do remove imports, variables, or functions that *your own* change made unused — that's a direct consequence of your edit, not unrelated cleanup, and leaving it behind is itself a loose end.

The test: every changed line should trace directly to the task at hand. (§8 below covers the separate, deliberate hygiene pass — this section is about not smuggling cleanup into unrelated diffs.)

### 4. Goal-driven execution

Define success criteria. Loop until verified, not until it looks plausible.

Translate the task into something checkable:

- "Add validation" → write tests for invalid inputs, then make them pass.
- "Fix the bug" → write a test that reproduces it, then make it pass.
- "Refactor X" → confirm tests pass before and after, with no behavior change.
- "Speed this up" → establish a measured baseline first, then verify the speedup against it (see §9 — profile before optimizing, don't assume).

For multi-step tasks, state a brief plan before diving in:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you work through a task independently without checking in every step. Weak criteria ("make it work") tend to produce plausible-looking code that hasn't actually been confirmed correct — which is a particular risk in analysis pipelines, where a wrong-but-plausible number is worse than an obvious crash.

## Part 2 — Pipeline & repo conventions

### 5. README quality

The README is the entry point for both P. and anyone else (including future-you) returning to a repo after months away. A repo without a good README is effectively undocumented, no matter how good the code is.

Every repo's `README.md` should contain, in this order:

1. **One-paragraph summary** — what the repo does and what kind of data/problem it addresses.
2. **Quick start** — the smallest possible sequence of commands to go from a fresh clone to a working result on sample data. This should be copy-pasteable and actually tested, not aspirational.
3. **Detailed installation** — environment setup (conda/venv), dependency versions, any non-Python dependencies (e.g. compiled extensions, external tools), and known platform gotchas.
4. **Pipelines overview** — a short (2-4 sentence) description of *each* implemented pipeline/analysis mode, what input it expects and what output it produces. Each entry links out to a dedicated page (`docs/<pipeline>.md`) for full details — algorithm description, parameters, config schema, examples. The README stays a map; the docs pages hold the territory.
5. **Batch processing** — pointer to the batch framework (see §10) and where its docs live.
6. **Project status / repo map** — brief note on what's stable vs. experimental, and where to find tests, configs, and scripts.

Keep the README itself short. If a section is growing past a few paragraphs, that's a sign it wants to become its own `docs/` page with a link from the README instead.

Use `references/readme_template.md` in this skill as the starting skeleton for new repos or major README rewrites.

### 6. Run the full test suite before every commit

Tests run before the commit happens, not after, and not just for the files that changed. A green local test suite is the minimum bar for a commit to exist — it's what keeps `main` (and every commit on it) bisectable and trustworthy, and it's the mechanism that makes the "verify" half of §4 real rather than aspirational.

- Run the complete suite (`pytest` or equivalent), not a subset, even when the change feels small or unrelated to most of the codebase. Small, "obviously safe" changes are exactly the ones that cause surprising breakage elsewhere.
- If tests are slow, that's a problem to solve (markers for slow/integration tests, parallel test execution) rather than a reason to skip them before commit.
- If a test fails and the fix isn't obvious, do not commit around it (e.g. by narrowing the diff or skipping the test) without flagging it explicitly — silently weakening test coverage to get a commit through defeats the purpose.
- If the repo has no tests yet for the code being touched, treat adding at least minimal coverage as part of the change, not a follow-up.

### 7. Commit and PR descriptions

Every commit (and PR, where applicable) gets a real description, not just a one-line summary of the diff. The goal is that someone reading the log later understands *why* the change happened and what's known to still be missing, without having to read the diff itself.

Use `references/pr_commit_template.md` for the exact structure. In short, every description includes:

- **What changed and why** — the motivation, not just a restatement of the diff.
- **How it was verified** — which tests were run, and any manual verification for things tests don't cover (e.g. visual inspection of an output plot). This is where the success criteria from §4 get written down.
- **Future work** — a short, explicit section noting follow-ups, known limitations, or things intentionally deferred. This is what keeps planned work from getting lost between conversations — capture it here even if it also lives in an issue tracker.

### 8. Sweep for loose ends before every commit — flag, don't silently delete

Before committing — and periodically even outside the commit cycle — scan the touched files (and ideally the broader codebase) for leftovers that accumulate silently in fast-moving analysis code:

- Commented-out code blocks left over from earlier approaches.
- Functions, classes, or imports that are no longer called from anywhere.
- Stale `TODO`/`FIXME` comments that no longer apply, or that have quietly become permanent.
- Config options, CLI flags, or parameters that are accepted but no longer do anything.
- Print/debug statements left in from troubleshooting.

This doesn't need to be a separate ceremony — fold it into the pre-commit check alongside the test run. A practical approach:

```bash
# unused functions/imports/variables
vulture . --min-confidence 80

# unused imports and basic dead-code lints
ruff check . --select F401,F841

# leftover commented-out code and stale markers (manual scan of the diff)
git diff --staged | grep -nE '^\+.*#.*\b(TODO|FIXME|XXX)\b'
```

**How to handle what you find**, consistent with §3 (surgical changes):

- Dead code that *your own change* just orphaned (an import, helper, or branch only your edit made unused): remove it. That's a direct consequence of the change you're making, not unrelated cleanup.
- Pre-existing dead code unrelated to the current change: don't delete it silently, even if you're confident it's unused. Flag it — in the commit description's Future work section, or directly to P. — and let it be a deliberate decision rather than a side effect. In scientific code especially, something that looks like dead code can turn out to be a reference implementation, a disabled-but-documented alternate method, or something still called via a path you didn't grep for (dynamic dispatch, notebooks, external scripts).
- If you're going to do a dedicated cleanup pass (as opposed to cleanup incidental to another change), say so up front and treat it as its own task with its own commit, so the diff is easy to review on its own terms.

### 9. Optimization order: correctness, then single-core, then parallel

When implementing or improving an analysis routine, work through these phases in order, and don't start a later phase until the earlier one is solid:

1. **Correctness first.** Get the algorithm right on a single core with the simplest implementation that's clearly correct. Validate against known/expected outputs, edge cases, and (where possible) an independent reference implementation or hand-computed example. Correctness bugs hidden behind a fast, parallel implementation are much harder to find later.
2. **Single-processor performance.** Once correct, profile (`cProfile`, `line_profiler`, or similar) and optimize the single-threaded path — better algorithms/data structures, vectorization (NumPy/SciPy), avoiding redundant work. Most real-world speedups come from this step, and it keeps the code easy to reason about and debug. Profile before optimizing — don't guess at the bottleneck (this is §4's "measured baseline" applied to performance work specifically).
3. **Parallelize last.** Only after the single-core path is correct and reasonably fast, parallelize across the available hardware (multiprocessing, joblib, GPU where applicable). Parallel code is harder to debug, so it should wrap a trusted, already-fast serial implementation rather than be the first version written.

A useful gut check: if you can't currently explain why the serial version's output is correct, it's too early to parallelize it.

### 10. Standard batch processing framework

Every pipeline that's used for real analysis (as opposed to a one-off script) should ship with a batch mode for running it unattended across many datasets, each with its own config. The framework should be designed so a multi-hour overnight run survives individual dataset failures and leaves a clear record of what happened.

Core requirements:

- **Per-dataset config.** Each dataset gets its own config file (or a base config plus per-dataset overrides). Don't hard-code dataset-specific parameters into the batch script itself.
- **Fail-isolated execution.** One dataset's failure (bad input, crash, timeout) must not stop the rest of the batch. Wrap each dataset's run in its own error boundary, log the failure with enough detail to debug later, and move on.
- **Structured logging per run.** Each dataset's run gets its own log output (and ideally its own output directory), plus a batch-level summary at the end: how many succeeded, how many failed, and why.
- **Resumability.** Re-running a batch should skip datasets that already completed successfully (unless explicitly forced), so a partially-failed overnight run can be resumed without redoing finished work.
- **No interactive prompts.** Anything that would require human input mid-run (confirmations, missing-parameter prompts) should instead fail that dataset and log it — unattended means unattended.

`scripts/batch_runner_template.py` in this skill is a ready-to-adapt implementation of this pattern (config discovery, per-dataset try/except with logging, resume-by-default, summary report at the end). Start new batch frameworks from it rather than rebuilding the pattern from scratch each time.

## Applying this skill

When starting any non-trivial task, run through Part 1 first — state assumptions, sketch a plan with verify steps, and confirm the simplest viable approach before writing code.

When asked to start a new pipeline repo, scaffold the README from the template and set up the batch runner pattern early — retrofitting batch mode after a pipeline has grown organically is more work than building on it from the start.

When asked to review code, commit, or open a PR, walk through §6-8 explicitly (tests, description, loose-ends sweep) rather than assuming they're implicitly covered.

When asked to "speed this up," check which phase of §9 the code is actually in before reaching for parallelism — it's a common shortcut to jump straight to multiprocessing on code that hasn't been profiled or correctness-checked yet.

**Signals these guidelines are working:** diffs that contain only what the task required, fewer rewrites caused by overcomplicated first attempts, clarifying questions arriving before implementation rather than corrections arriving after, and dead code that gets flagged and deliberately decided on rather than silently appearing or disappearing.
