#!/usr/bin/env python
"""Diagnose the MyLOV1Char pilot: alignment residuals and red/green intensity balance.

For every raw movie of the three recording dates this re-runs the calibration
exactly as the pilot did (same base.toml + per-date TOML), then measures:

* crop geometry: red vs green channel-crop sizes and the offset that the merge's
  centred zero-padding introduces relative to the frame the feature aligner fits in;
* fit stability: the production fit repeated on 3 disjoint frame subsets;
* the fit with a filament-*centreline* (skeleton) objective instead of the mask;
* the residual red-vs-green offset measured directly in the written output
  (heads in R fitted onto the green-filament centreline);
* the red-vs-green offset relative to the lab's earlier "* RGB.tif" composite;
* intensity: vmax vs robust percentiles of the despeckled max projection, and the
  red/green balance of our output and of the old composite.

Writes summary.csv and one PNG per movie plus summary.png into OUTDIR.

Usage (inside a job, from the repo root):
    python hpc/runs/MyLOV1Char/diagnose_alignment.py OUTDIR [--workers N] [--only SUBSTR]
        [--aligned aligned_feature]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RUN = Path(__file__).resolve().parent
DATES = ("20170803", "20180620", "20181002")
DATE_COLORS = {"20170803": "#2a78d6", "20180620": "#eb6834", "20181002": "#1baf7a"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _skeleton_aligner(base):
    """Copy of ``base`` whose objective is distance to the filament centreline."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from optomerge.feature_registration import FeatureDistanceAligner

    class SkeletonAligner(FeatureDistanceAligner):
        def _detect(self, reference, moving):
            green = np.asarray(reference.data, dtype=np.float64)
            red = np.asarray(moving.data, dtype=np.float64)
            if green.ndim == 2:
                green, red = green[:, :, None], red[:, :, None]
            per_frame = []
            for k in range(red.shape[2]):
                rows, cols, fil = self.frame_features(green[:, :, k], red[:, :, k])
                if rows.size == 0 or not fil.any():
                    continue
                skel = skeletonize(fil)
                per_frame.append((rows, cols, ndimage.distance_transform_edt(~skel)))
            return per_frame

    al = SkeletonAligner.__new__(SkeletonAligner)
    al.__dict__.update(base.__dict__)
    return al


def _with(al, **kw):
    out = al.__class__.__new__(al.__class__)
    out.__dict__.update(al.__dict__)
    out.__dict__.update(kw)
    return out


def _pad_offset(shape_g, shape_r):
    """(row, col) displacement of red content vs the aligner's top-left frame.

    The aligner treats red pixel (r, c) as sitting over green pixel (r, c) at
    identity. The merge centres each crop in a common power-of-2 canvas, so red
    pixel (r, c) actually sits over green pixel (r + dr, c + dc).
    """
    from optomerge._kernels.transform import _next_pow2
    Hg, Wg = shape_g[:2]
    Hr, Wr = shape_r[:2]
    P, Q = _next_pow2(max(Hg, Hr)), _next_pow2(max(Wg, Wr))
    return ((P - Hr) // 2 - (P - Hg) // 2, (Q - Wr) // 2 - (Q - Wg) // 2)


def _landscape(al, per_frame, w, t1, t2, rot, half=3.0, step=0.25):
    """Capped-distance cost on a translation grid around (t1, t2), rotation fixed."""
    g = np.arange(-half, half + 1e-9, step)
    J = np.empty((g.size, g.size))
    for i, d2 in enumerate(g):
        for j, d1 in enumerate(g):
            d = al._affine_distances(per_frame, t1 + d1, t2 + d2, rot, 1.0, 1.0)
            J[i, j] = float((np.minimum(d, al.distance_cap) * w).sum() / w.sum())
    return g, J


def _basin_stats(g, J, tol):
    """Extent of the near-minimum basin (cost < min + tol): principal std devs + angle."""
    ii, jj = np.nonzero(J <= J.min() + tol)
    pts = np.stack([g[jj], g[ii]], 1)                     # (col, row) offsets
    if len(pts) < 3:
        return 0.0, 0.0, float("nan")
    ev, evec = np.linalg.eigh(np.cov(pts.T))
    ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))  # long axis, image coords
    return float(np.sqrt(max(ev[1], 0))), float(np.sqrt(max(ev[0], 0))), ang


def _match(template_img, search_img, margin=12):
    """Sub-pixel (row, col) of template_img's inset centre region inside search_img."""
    import cv2
    t = template_img[margin:-margin, margin:-margin].astype(np.float32)
    s = search_img.astype(np.float32)
    if t.shape[0] > s.shape[0] or t.shape[1] > s.shape[1]:
        return None
    res = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (x, y) = cv2.minMaxLoc(res)

    def _sub(line, k):
        if 0 < k < len(line) - 1:
            a, b, c = line[k - 1], line[k], line[k + 1]
            den = a - 2 * b + c
            return k + (0.5 * (a - c) / den if den else 0.0)
        return float(k)
    return _sub(res[:, x], y) - margin, _sub(res[y, :], x) - margin, float(peak)


def _balance(rgb_mean, rgb_frames):
    """Red/green ratios of bright-signal levels: p99.5 of mean projection, p99.9 of frames."""
    out = {}
    for name, img, q in (("mean", rgb_mean, 99.5), ("frame", rgb_frames, 99.9)):
        r = np.percentile(img[..., 0], q)
        g = np.percentile(img[..., 1], q)
        out[name] = float(r / g) if g > 0 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# per-movie analysis
# --------------------------------------------------------------------------- #

def analyse(raw_path: str, date: str, outdir: str, aligned: str = "aligned_feature") -> dict:
    import cv2
    cv2.setNumThreads(1)
    import tifffile
    from optomerge.config import Settings
    from optomerge.movie import RawMovie
    from optomerge.calibration import _compute_limits, _despeckle, _sample_frame_stack

    raw = Path(raw_path)
    settings = Settings.from_toml(str(RUN / "base.toml"), str(RUN / f"{date}.toml"))
    cal = settings.build_calibrator()
    al = cal.aligner
    in_root = Path(settings.io.input)
    out_root = Path(settings.io.output).parent.parent / aligned / date
    rel = raw.relative_to(in_root)
    ours_path = out_root / rel.parent / (raw.stem + "_aligned.tif")
    old_path = raw.with_name(raw.stem + " RGB.tif")

    rec = {"date": date, "movie": raw.stem, "subdir": str(rel.parent)}

    # -- calibration, exactly as the pilot ----------------------------------
    movie = RawMovie.open(raw)
    maxp = movie.max_projection(cal.projection_frames)
    meanp = movie.mean_projection(cal.projection_frames)
    layout = cal.layout.resolve(maxp, meanp)
    nf = cal.feature_frames
    stack = _sample_frame_stack(movie, nf, limit=cal.projection_frames)
    ch = {c.name: c for c in layout.split(stack)}
    g_ch, r_ch = ch["green"], ch["red"]
    t = al.align(g_ch, r_ch)
    rec.update(t1=t.t1, t2=t.t2, rot_deg=float(np.degrees(t.rot)), score=t.score)

    gb, rb = (np.asarray(s.bounds) for s in layout.specs)
    rec.update(green_bounds=gb.tolist(), red_bounds=rb.tolist(),
               green_shape=list(g_ch.data.shape[:2]), red_shape=list(r_ch.data.shape[:2]))
    pr, pc = _pad_offset(g_ch.data.shape, r_ch.data.shape)
    rec.update(pad_off_row=pr, pad_off_col=pc)

    # -- detections / density at the fitted transform -----------------------
    per_frame = al._detect(g_ch, r_ch)
    w = al._stuck_weights(per_frame)
    fil_frac, heads = [], []
    G, R = np.asarray(g_ch.data, float), np.asarray(r_ch.data, float)
    for k in range(G.shape[2]):
        rows, _, fil = al.frame_features(G[:, :, k], R[:, :, k])
        fil_frac.append(fil.mean())
        heads.append(rows.size)
    d = al._affine_distances(per_frame, t.t1, t.t2, t.rot, t.s1, t.s2)
    inl = d < al.distance_cap
    rec.update(frames_used=len(per_frame), heads_per_frame=float(np.mean(heads)),
               filament_frac=float(np.mean(fil_frac)),
               inlier_frac=float(inl.mean()) if d.size else float("nan"),
               inlier_median_d=float(np.median(d[inl])) if inl.any() else float("nan"))

    # -- cost basin of the production objective -----------------------------
    g_ax, J = _landscape(al, per_frame, w, t.t1, t.t2, t.rot)
    long_sd, short_sd, ang = _basin_stats(g_ax, J, tol=0.02)
    rec.update(basin_long_sd=long_sd, basin_short_sd=short_sd, basin_angle=ang)

    # -- stability: 3 disjoint frame subsets of the same size ----------------
    big = _sample_frame_stack(movie, 3 * nf, limit=cal.projection_frames)
    bch = {c.name: c for c in layout.split(big)}
    subs = []
    for s in range(3):
        sg = SimpleNamespace(data=bch["green"].data[:, :, s::3])
        sr = SimpleNamespace(data=bch["red"].data[:, :, s::3])
        ts = al.align(sg, sr)
        subs.append((ts.t1, ts.t2, np.degrees(ts.rot)))
    subs = np.asarray(subs)
    rec.update(sub_t1_sd=float(subs[:, 0].std(ddof=1)), sub_t2_sd=float(subs[:, 1].std(ddof=1)),
               sub_rot_sd=float(subs[:, 2].std(ddof=1)),
               sub_t1=subs[:, 0].round(2).tolist(), sub_t2=subs[:, 1].round(2).tolist())

    # -- centreline objective -------------------------------------------------
    sk = _skeleton_aligner(al)
    ts = sk.align(g_ch, r_ch)
    rec.update(skel_t1=ts.t1, skel_t2=ts.t2, skel_rot_deg=float(np.degrees(ts.rot)))
    sk_pf = sk._detect(g_ch, r_ch)
    sg_ax, SJ = _landscape(sk, sk_pf, sk._stuck_weights(sk_pf), ts.t1, ts.t2, ts.rot)

    # -- intensity limits ------------------------------------------------------
    limits = _compute_limits(layout, maxp, settings.processing.norm_exclude)
    desp = {c.name: c.data for c in layout.split(_despeckle(maxp))}
    for name in ("green", "red"):
        vmin, vmax = limits[name]
        img = desp[name].ravel()
        p999, p9999 = np.percentile(img, [99.9, 99.99])
        rec[f"{name}_vmin"], rec[f"{name}_vmax"] = vmin, vmax
        rec[f"{name}_vmax_over_p999"] = float((vmax - vmin) / max(p999 - vmin, 1e-9))
        rec[f"{name}_px_above_half_vmax"] = int((img >= vmin + 0.5 * (vmax - vmin)).sum())
    del maxp, meanp, stack, big

    # -- our output -------------------------------------------------------------
    ours = tifffile.imread(ours_path)                    # (T, H, W, 3) uint8
    rec["out_shape"] = list(ours.shape[1:3])
    rec["out_area_frac"] = float(ours.shape[1] * ours.shape[2] / (G.shape[0] * G.shape[1]))
    idx = np.linspace(0, ours.shape[0] - 1, 3 * nf).astype(int)
    o_mean = ours.mean(axis=0)
    o_frames = ours[idx].astype(np.float64)
    bal = _balance(o_mean, o_frames)
    rec.update(out_RG_mean=bal["mean"], out_RG_frame=bal["frame"],
               out_sat_R=float((ours[idx, :, :, 0] == 255).mean()),
               out_sat_G=float((ours[idx, :, :, 1] == 255).mean()))

    # residual offset in the written output (translation only, both objectives)
    og = SimpleNamespace(data=np.moveaxis(o_frames[..., 1], 0, 2))
    orr = SimpleNamespace(data=np.moveaxis(o_frames[..., 0], 0, 2))
    for tag, a in (("mask", al), ("skel", sk)):
        a8 = _with(a, max_shift=8.0, step=0.5, fit_rotation=False, fit_scale=False)
        tr = a8.align(og, orr)
        # heads at (row, col) land on filament at (row - t2, col - t1): red sits +t off.
        rec[f"res_{tag}_row"], rec[f"res_{tag}_col"] = tr.t2, tr.t1
    del ours

    # -- old composite -------------------------------------------------------------
    if old_path.exists():
        old = tifffile.imread(old_path)                  # (T, H, W, 3) uint8
        old_mean = old.mean(axis=0)
        ob = _balance(old_mean, old[idx].astype(np.float64))
        rec.update(old_shape=list(old.shape[1:3]), old_RG_mean=ob["mean"], old_RG_frame=ob["frame"])
        mg = _match(o_mean[..., 1], old_mean[..., 1])
        mr = _match(o_mean[..., 0], old_mean[..., 0])
        if mg and mr:
            # our red content is displaced by (p_g - p_r) relative to the old composite
            rec.update(vs_old_row=mg[0] - mr[0], vs_old_col=mg[1] - mr[1],
                       vs_old_peak_g=mg[2], vs_old_peak_r=mr[2])
        del old
    else:
        old_mean = None

    _figure(rec, o_mean, old_mean, g_ax, J, sg_ax, SJ, t, ts, Path(outdir))
    return rec


def _figure(rec, o_mean, old_mean, g_ax, J, sg_ax, SJ, t, ts, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rg(img):
        x = np.zeros(img.shape[:2] + (3,))
        for c in (0, 1):
            v = img[..., c]
            hi = np.percentile(v, 99.7)
            x[..., c] = np.clip(v / hi, 0, 1) if hi > 0 else 0
        return x

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), gridspec_kw={"width_ratios": [1.4, 1.4, 1, 1]})
    ax[0].imshow(rg(o_mean)); ax[0].set_title(f"ours (mean), R/G={rec['out_RG_mean']:.2f}")
    if old_mean is not None:
        ax[1].imshow(rg(old_mean)); ax[1].set_title(f"old RGB.tif (mean), R/G={rec.get('old_RG_mean', np.nan):.2f}")
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])
    for a, gx, JJ, name, tt in ((ax[2], g_ax, J, "mask objective", t), (ax[3], sg_ax, SJ, "centreline objective", ts)):
        ext = [gx[0], gx[-1], gx[-1], gx[0]]
        im = a.imshow(JJ, extent=ext, cmap="Greys_r")
        a.plot(0, 0, "+", color="#2a78d6", ms=12, mew=2)
        a.set_title(f"{name}\nfit t=({tt.t1:.2f}, {tt.t2:.2f})", fontsize=9)
        a.set_xlabel("Δt1 (col, px)"); a.set_ylabel("Δt2 (row, px)")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"{rec['movie']}\npad offset (row,col)=({rec['pad_off_row']},{rec['pad_off_col']})  "
                 f"output residual (row,col) mask=({rec['res_mask_row']:.2f},{rec['res_mask_col']:.2f}) "
                 f"centreline=({rec['res_skel_row']:.2f},{rec['res_skel_col']:.2f})  "
                 f"vs old=({rec.get('vs_old_row', np.nan):.2f},{rec.get('vs_old_col', np.nan):.2f})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"{rec['date']}_{rec['movie'][-12:].replace(' ', '_')}_{rec['subdir'].split('/')[0]}.png", dpi=90)
    plt.close(fig)


def _summary_figure(rows, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for date in DATES:
        rr = [r for r in rows if r["date"] == date]
        c = DATE_COLORS[date]
        ax[0].scatter([r["res_skel_col"] for r in rr], [r["res_skel_row"] for r in rr],
                      s=40, color=c, label=date, edgecolor="white", linewidth=1)
        ax[1].scatter([r["pad_off_row"] for r in rr], [r["res_skel_row"] for r in rr],
                      s=40, color=c, label=f"{date} row", marker="o", edgecolor="white", linewidth=1)
        ax[1].scatter([r["pad_off_col"] for r in rr], [r["res_skel_col"] for r in rr],
                      s=40, color=c, label=f"{date} col", marker="s", edgecolor="white", linewidth=1)
        ax[2].scatter([r["filament_frac"] * 100 for r in rr],
                      [np.hypot(r["res_skel_row"], r["res_skel_col"]) for r in rr],
                      s=40, color=c, label=date, edgecolor="white", linewidth=1)
    ax[0].axhline(0, color="#bbb", lw=0.8); ax[0].axvline(0, color="#bbb", lw=0.8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("red offset, col (px, + = right)"); ax[0].set_ylabel("red offset, row (px, + = down)")
    ax[0].set_title("Residual red offset in output (centreline fit)")
    lim = [-6, 6]
    ax[1].plot(lim, lim, color="#bbb", lw=0.8)
    ax[1].set_xlabel("predicted by padding mismatch (px)"); ax[1].set_ylabel("measured residual (px)")
    ax[1].set_title("Padding prediction vs measured (○ row, □ col)")
    ax[2].set_xlabel("filament mask area (% of field)"); ax[2].set_ylabel("|residual| (px)")
    ax[2].set_title("Residual vs filament density")
    for a in ax:
        a.grid(color="#eee", lw=0.6); a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "summary.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--only", default=None, help="only movies whose path contains this")
    ap.add_argument("--aligned", default="aligned_feature",
                    help="output folder (next to aligned_feature) holding the movies to check")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(RUN.parents[2]))
    from run_optomerge import _collect_tif_files
    from optomerge.config import Settings

    jobs = []
    for date in DATES:
        s = Settings.from_toml(str(RUN / "base.toml"), str(RUN / f"{date}.toml"))
        for f in _collect_tif_files(Path(s.io.input), exclude=s.io.exclude):
            if args.only is None or args.only in str(f):
                jobs.append((str(f), date))
    print(f"{len(jobs)} movies, {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(analyse, f, d, str(outdir), args.aligned): f for f, d in jobs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                rows.append(r)
                print(f"[ok] {r['date']} {r['movie'][-14:]}  t=({r['t1']:.2f},{r['t2']:.2f}) "
                      f"pad=({r['pad_off_row']},{r['pad_off_col']}) "
                      f"res=({r['res_skel_row']:.2f},{r['res_skel_col']:.2f}) "
                      f"R/G={r['out_RG_mean']:.2f}", flush=True)
            except Exception:
                print(f"[FAIL] {futs[fut]}\n{traceback.format_exc()}", flush=True)

    rows.sort(key=lambda r: (r["date"], r["subdir"], r["movie"]))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(outdir / "summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    if rows:
        _summary_figure(rows, outdir)
    print(f"wrote {outdir}/summary.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
