# Commit / PR description template

Applies to both individual commit messages (for substantive commits) and PR descriptions. For small/trivial commits (typo fixes, formatting), a one-line summary is fine — this template is for anything that changes behavior, adds a feature, or fixes a non-trivial bug.

```markdown
## Summary

What changed and why. Focus on motivation and intent, not a restatement of the diff —
someone reading this in a year should understand the *reason* the change happened.

## Changes

- Bullet list of the substantive changes, grouped logically if there are several.

## Verification

- Full test suite: pass/fail, and note any new tests added.
- Any manual verification performed (e.g. visual check of output plots, comparison
  against a reference dataset) that automated tests don't cover.

## Future work

- Explicit follow-ups, known limitations, or things intentionally deferred out of
  scope for this change. Empty is fine if there's genuinely nothing — but check
  before leaving it out.
```

**Example:**

```markdown
## Summary

Switches the gliding velocity calculation from a fixed-window average to a
Savitzky-Golay filter, fixing systematic underestimation of velocity for
short tracks (<20 frames) flagged in the optomerge cross-validation.

## Changes

- Replace `compute_velocity()` fixed-window logic with `scipy.signal.savgol_filter`
- Add `window_length`/`polyorder` to the track-analysis config schema
- Update default config and docs/track_analysis.md with new parameters

## Verification

- Full test suite passes (`pytest`, 142 passed)
- Added regression tests comparing output against a hand-computed 10-frame track
- Manually spot-checked velocity traces for 5 sample tracks against the old method

## Future work

- The new filter parameters are not yet auto-tuned per frame rate — currently
  the user must set them manually for non-default capture rates.
- Consider exposing filter choice (SG vs. moving average) as a config option
  rather than a hard switch, for backward comparison.
```
