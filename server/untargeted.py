"""Untargeted LC-MS (metabolomics / lipidomics): raw files → feature table → statistics.

The chain is the one OpenMS/UmetaFlow uses for small molecules — mass traces → elution peaks → isotope-
grouped features (FeatureFindingMetabo), pose-clustering RT alignment, KD-tree linking into a consensus
map — followed by what every metabolomics paper then does by hand: re-integrating missing values from the
raw data, grouping adducts of one compound, putative annotation, blank / QC / detection-rate filtering,
QC-based drift correction (QC-RLSC), PQN normalisation, PCA / PLS-DA and univariate tests.

Retention time is seconds in the code and minutes on every axis and table.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import chem, msio, peaks
from .common import (BLANK_COLOR, DOWN, NS, PALETTE, QC_COLOR, UP, UserFacingError, commas, fmt_n, log_exc,
                     versions)

LIB_PATH = Path(__file__).resolve().parent / "data" / "metabolites.csv"
STD_COLOR = "#8c6d31"
PRIMARY = {1: "[M+H]+", -1: "[M-H]-"}


# ====================================================================== small helpers

def _f(params, key, default, auto=None):
    """Numeric parameter: (value, source). Empty / 'auto' strings fall back to the auto or default value."""
    v = params.get(key)
    if isinstance(v, str) and v.strip().lower() in ("", "auto", "none", "nan"):
        v = None
    if v is not None:
        try:
            return float(v), "user"
        except (TypeError, ValueError):
            pass
    if auto is not None:
        return float(auto), "auto"
    return float(default), "default"


def _b(params, key, default):
    v = params.get(key)
    if v is None or v == "":
        return bool(default), "default"
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on"), "user"
    return bool(v), "user"


def _s(params, key, default, options=None):
    v = params.get(key)
    if v is None or str(v).strip() == "":
        return default, "default"
    v = str(v).strip()
    if options and v.lower() not in [o.lower() for o in options]:
        return default, "default"
    if options:
        v = next(o for o in options if o.lower() == v.lower())
    return v, "user"


def _bh(p):
    """Benjamini–Hochberg adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if not n:
        return out
    pv = p[ok]
    o = np.argsort(pv)
    ranked = pv[o] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[o] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def _cv(x, axis=1):
    """Coefficient of variation (%) along an axis, ignoring missing values; NaN when fewer than 2 values."""
    import warnings
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(x, axis=axis)
        s = np.nanstd(x, axis=axis, ddof=1)
        cv = 100 * s / m
    cv[n < 2] = np.nan
    return cv


def _fid(mz, rt_s):
    return f"M{mz:.4f}T{rt_s / 60:.2f}"


def _short(s, n=28):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _jsonable(v, nd=4):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, nd) if np.isfinite(f) else None


def _pg(job, title, note=""):
    """Open a parameter group, or re-open it if an earlier step already created one with this title."""
    for g in job.result["params_panel"]:
        if g["title"] == title:
            job._pgroup = g
            return
    job.param_group(title, note)


# ====================================================================== sample sheet

def _sheet(files, params) -> pd.DataFrame:
    """One row per file (same order as files): sample, group, type, batch, order."""
    rows = params.get("sheet") or []
    auto = msio.match_sheet(files, None)
    lut = {}
    for r in rows:
        f = str(r.get("file") or "").strip()
        if f:
            lut.setdefault(f, r)
            lut.setdefault(Path(f).name, r)
            lut.setdefault(msio._stem(Path(f).name), r)
        s = str(r.get("sample") or "").strip()
        if s:
            lut.setdefault("sample:" + s.lower(), r)
    out = []
    for k, f in enumerate(files):
        r = (lut.get(str(f)) or lut.get(Path(f).name) or lut.get(msio._stem(Path(f).name))
             or lut.get("sample:" + msio._stem(Path(f).name)) or auto[k])

        def g(key, default=None):
            v = r.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)) or str(v).strip().lower() in ("", "nan", "none"):
                return default
            return v
        typ = str(g("type", auto[k]["type"])).strip().lower()
        typ = {"pooled": "qc", "pool": "qc", "std": "standard", "blk": "blank"}.get(typ, typ)
        if typ not in ("sample", "qc", "blank", "standard"):
            typ = msio.infer_type(Path(f).name)
        try:
            order = float(g("order", k + 1))
        except (TypeError, ValueError):
            order = float(k + 1)
        out.append({"file": str(f), "sample": str(g("sample", auto[k]["sample"])).strip(),
                    "group": str(g("group", "")).strip() if typ == "sample" else "",
                    "type": typ, "batch": str(g("batch", "1")).strip() or "1", "order": order})
    df = pd.DataFrame(out)
    # unique sample names (two files called "QC" would otherwise collide in every table)
    seen = {}
    names = []
    for s in df["sample"]:
        seen[s] = seen.get(s, 0) + 1
        names.append(s if seen[s] == 1 else f"{s}_{seen[s]}")
    df["sample"] = names
    df["run"] = np.arange(len(df))
    return df


def _group_order(params, sheet: pd.DataFrame) -> list[str]:
    """Groups in the order they first appear in the sample sheet (the order the user wrote them)."""
    rows = params.get("sheet") or []
    order = []
    for r in rows:
        g = str(r.get("group") or "").strip()
        t = str(r.get("type") or "sample").strip().lower()
        if g and g.lower() != "nan" and t in ("sample", "") and g not in order:
            order.append(g)
    for g in sheet.loc[sheet["type"] == "sample", "group"]:
        if g and g not in order:
            order.append(g)
    return order


def _colours(sheet, groups):
    gc = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    col = []
    for _, r in sheet.iterrows():
        col.append(QC_COLOR if r.type == "qc" else BLANK_COLOR if r.type == "blank" else STD_COLOR if r.type == "standard"
                   else gc.get(r.group, "#6b8fb3"))
    return col, gc


def _legend(ax, sheet, gc, **kw):
    hs, ls = [], []
    for g, c in gc.items():
        hs.append(plt.Line2D([], [], color=c, marker="o", lw=0, ms=6))
        ls.append(g)
    ungrouped = ((sheet.type == "sample") & (sheet.group == "")).any()
    if ungrouped:
        hs.append(plt.Line2D([], [], color="#6b8fb3", marker="o", lw=0, ms=6)); ls.append("sample")
    for t, c, lab in (("qc", QC_COLOR, "QC"), ("blank", BLANK_COLOR, "blank"), ("standard", STD_COLOR, "standard")):
        if (sheet.type == t).any():
            hs.append(plt.Line2D([], [], color=c, marker="o", lw=0, ms=6)); ls.append(lab)
    ax.legend(hs, ls, fontsize=8, **kw)


# ====================================================================== fast XIC for gap filling

class _MS1Index:
    """All MS1 peaks of one run sorted by m/z, so an XIC is two binary searches and a bincount instead of a
    Python loop over every spectrum — gap filling asks for thousands of them."""

    def __init__(self, run, pol):
        sel = _ms1_sel(run, pol)
        self.rt = run.rt[sel]
        starts = run.offsets[sel]
        lens = run.offsets[sel + 1] - starts
        total = int(lens.sum())
        scan = np.repeat(np.arange(len(sel), dtype=np.int32), lens)
        first = np.repeat(np.cumsum(lens) - lens, lens)
        pos = np.arange(total) - first + np.repeat(starts, lens)
        mz = run.mz[pos]
        o = np.argsort(mz, kind="stable")
        self.mz = mz[o]
        self.it = run.intensity[pos][o]
        self.scan = scan[o]

    def xic(self, mz, ppm, t0, t1):
        s0 = int(np.searchsorted(self.rt, t0))
        s1 = int(np.searchsorted(self.rt, t1, side="right"))
        if s1 - s0 < 3:
            return self.rt[s0:s1], np.zeros(max(s1 - s0, 0))
        tol = mz * ppm * 1e-6
        a = int(np.searchsorted(self.mz, mz - tol))
        b = int(np.searchsorted(self.mz, mz + tol, side="right"))
        sc = self.scan[a:b]
        m = (sc >= s0) & (sc < s1)
        y = np.bincount(sc[m] - s0, weights=self.it[a:b][m].astype(float), minlength=s1 - s0)
        return self.rt[s0:s1], y


def _ms1_sel(run, pol):
    m = run.ms_level == 1
    if pol and (run.polarity[m] == pol).any():
        m &= (run.polarity == pol) | (run.polarity == 0)
    return np.flatnonzero(m)


# ====================================================================== OpenMS feature finding

def _oms_param(obj, values: dict):
    p = obj.getDefaults() if hasattr(obj, "getDefaults") else obj.getParameters()
    for k, v in values.items():
        p.setValue(k, v)
    obj.setParameters(p)


def _mass_traces(exp, ppm, noise, min_len):
    import pyopenms as oms
    mtd = oms.MassTraceDetection()
    _oms_param(mtd, {"mass_error_ppm": float(ppm), "noise_threshold_int": float(noise), "chrom_peak_snr": 3.0,
                     "min_trace_length": float(min_len)})
    traces = []
    mtd.run(exp, traces, 0)
    return traces


def _estimate_fwhm(run, pol, ppm, noise):
    """Typical chromatographic peak width (FWHM, s) from the 200 most intense mass traces of one run."""
    exp = msio.to_openms(run, levels=(1,), polarity=pol if (run.polarity == pol).any() else None)
    traces = _mass_traces(exp, ppm, noise, 2.0)
    if not traces:
        return None, 0
    ints = np.array([t.getIntensity(False) for t in traces])
    top = np.argsort(ints)[::-1][:200]
    fw = []
    for k in top:
        t = traces[int(k)]
        if t.getSize() < 5:
            continue
        try:
            fw.append(float(t.estimateFWHM(False)))
        except Exception:  # noqa: BLE001
            continue
    fw = np.array([x for x in fw if np.isfinite(x) and x > 0])
    if len(fw) < 10:
        return None, len(fw)
    return float(np.median(fw)), len(fw)


def _find_features(run, pol, cfg):
    """MassTraceDetection → ElutionPeakDetection → FeatureFindingMetabo for one run → (FeatureMap, DataFrame)."""
    import pyopenms as oms
    exp = msio.to_openms(run, levels=(1,), polarity=pol if (run.polarity == pol).any() else None)
    exp.sortSpectra(True)
    traces = _mass_traces(exp, cfg["ppm"], cfg["noise"], cfg["min_trace"])
    epd = oms.ElutionPeakDetection()
    _oms_param(epd, {"chrom_fwhm": float(cfg["fwhm"]), "chrom_peak_snr": 3.0, "width_filtering": "fixed",
                     "min_fwhm": float(cfg["min_fwhm"]), "max_fwhm": float(cfg["max_fwhm"]),
                     "masstrace_snr_filtering": "false"})
    split = []
    epd.detectPeaks(traces, split)
    ffm = oms.FeatureFindingMetabo()
    _oms_param(ffm, {"chrom_fwhm": float(cfg["fwhm"]), "local_rt_range": float(max(10.0, cfg["fwhm"])),
                     "local_mz_range": 6.5, "charge_lower_bound": 1, "charge_upper_bound": 3,
                     "isotope_filtering_model": "metabolites (5% RMS)" if cfg["iso"] else "none",
                     "mz_scoring_13C": "true" if cfg["iso"] else "false",
                     "remove_single_traces": "true" if cfg["single"] else "false",
                     "report_convex_hulls": "true", "enable_RT_filtering": "true"})
    fm = oms.FeatureMap()
    chroms = []
    ffm.run(split, fm, chroms)
    fm.setUniqueIds()
    fm.ensureUniqueId()
    rows = []
    for f in fm:
        hulls = f.getConvexHulls()
        if len(hulls):
            bb = hulls[0].getBoundingBox()
            lo, hi = bb.minPosition()[0], bb.maxPosition()[0]
        else:
            lo = hi = f.getRT()
        rows.append({"uid": f.getUniqueId(), "mz": f.getMZ(), "rt": f.getRT(), "intensity": f.getIntensity(),
                     "charge": f.getCharge(), "rt_start": lo, "rt_end": hi,
                     "fwhm": float(f.getMetaValue("FWHM")) if f.metaValueExists("FWHM") else np.nan,
                     "n_traces": int(f.getMetaValue("num_of_masstraces")) if f.metaValueExists("num_of_masstraces") else 1})
    return fm, pd.DataFrame(rows, columns=["uid", "mz", "rt", "intensity", "charge", "rt_start", "rt_end", "fwhm", "n_traces"]), len(traces)


# ====================================================================== the pipeline

def run(job, files, params):
    t_start = time.time()
    params = dict(params or {})
    files = [Path(f) for f in files]
    if not files:
        raise UserFacingError("No data files were given. Add the raw files (or mzML) of your study.")
    import pyopenms as oms  # noqa: F401 - fail early with a clear import error
    job.result["versions"] = versions("pyopenms", "numpy", "scipy", "pandas", "statsmodels", "scikit-learn", "matplotlib")

    sheet = _sheet(files, params)
    runs = msio.load_runs(job, files, 2, 25)

    # ------------------------------------------------------------------ polarity
    job.step(26, "Checking polarity")
    npos = sum(int(((r.ms_level == 1) & (r.polarity > 0)).sum()) for r in runs)
    nneg = sum(int(((r.ms_level == 1) & (r.polarity < 0)).sum()) for r in runs)
    pol_choice, pol_src = _s(params, "polarity", "auto", ["auto", "positive", "negative"])
    if pol_choice == "positive":
        pol = 1
    elif pol_choice == "negative":
        pol = -1
    else:
        pol = 1 if npos >= nneg else -1
    pname = "positive" if pol > 0 else "negative"
    _pg(job, "Samples & polarity", "Which files are analysed and in which ion mode.")
    if npos == 0 and nneg == 0:
        pol_why = "The files do not record polarity. Positive mode was assumed; if the data are negative mode set this to negative, or adducts and annotations will be wrong."
        job.flag("warn", "The files do not say which ion mode they were acquired in; <b>positive mode was assumed</b>. If they are negative mode, set polarity = negative and re-analyse.")
    elif npos and nneg and pol_choice == "auto":
        other = "negative" if pol > 0 else "positive"
        pol_why = f"The files contain both modes ({commas(npos)} positive and {commas(nneg)} negative MS1 scans). The dominant mode ({pname}) was analysed; set this to {other} and re-analyse for the other half."
        job.flag("warn", f"Your runs mix ion modes ({commas(npos)} positive, {commas(nneg)} negative MS1 scans). Positive and negative data contain different ions of the same compounds and cannot be merged feature-by-feature, so only <b>{pname} mode</b> was analysed. Re-analyse with polarity = {other} for the rest.")
    else:
        pol_why = f"All MS1 scans are {pname} mode." if pol_choice == "auto" else f"Set to {pname} by you."
    job.param("polarity", "Ion mode analysed", pol_choice if pol_choice != "auto" else "auto", pol_why, pol_src, "select",
              options=["auto", "positive", "negative"], resolved=pname)

    keep, dropped = [], []
    for k, r in enumerate(runs):
        sel = _ms1_sel(r, pol)
        if len(sel) < 10 or ((r.polarity[r.ms_level == 1] == -pol).all() and (r.ms_level == 1).any()):
            dropped.append(sheet["sample"][k])
            continue
        keep.append(k)
    if dropped:
        job.flag("warn", f"{len(dropped)} file(s) have no {pname}-mode MS1 scans and were left out: <b>{', '.join(dropped[:12])}</b>"
                         + (f" and {len(dropped) - 12} more" if len(dropped) > 12 else "") + ".")
    if not keep:
        raise UserFacingError(f"None of the files has {pname}-mode MS1 spectra. Check the polarity setting.")
    sheet = sheet.iloc[keep].reset_index(drop=True)
    runs = [runs[k] for k in keep]
    file_index = {k_new: k_old for k_new, k_old in enumerate(keep)}   # position in job.result["runs"]
    groups_all = _group_order(params, sheet)
    col, gc = _colours(sheet, groups_all)
    n_s, n_q, n_b = int((sheet.type == "sample").sum()), int((sheet.type == "qc").sum()), int((sheet.type == "blank").sum())
    job.param("_design", "Files used", f"{len(sheet)} files: {n_s} samples, {n_q} QCs, {n_b} blanks"
              + (f", {int((sheet.type == 'standard').sum())} standards" if (sheet.type == "standard").any() else ""),
              "From the sample sheet (type column); types missing from the sheet were inferred from file names (QC/pool, blank, std).",
              "data", "fixed")

    ctx = dict(job=job, params=params, sheet=sheet, runs=runs, pol=pol, col=col, gc=gc, groups_all=groups_all,
               file_index=file_index, n_files_total=len(files))

    _raw_qc(ctx)
    _detect(ctx)
    _align(ctx)
    _link(ctx)
    _gap_fill(ctx)
    _ms2_link(ctx)
    _annotate(ctx)
    _filters(ctx)
    _drift(ctx)
    _qc_cv_filter(ctx)
    _normalise(ctx)
    _stats_compute(ctx)
    _multivariate(ctx)
    _univariate(ctx)
    _outputs(ctx)
    job.method("Runtime", f"{(time.time() - t_start) / 60:.1f} min for {len(runs)} files.")


# ====================================================================== step 1 — raw-data QC

def _raw_qc(ctx):
    job, sheet, runs, pol, col, gc, params = ctx["job"], ctx["sheet"], ctx["runs"], ctx["pol"], ctx["col"], ctx["gc"], ctx["params"]
    job.step(27, "Raw-data quality control")
    job.section("rawqc", "Step 1", "Raw-data quality control",
                "Before any feature is found: did every injection spray, elute and record a comparable amount of signal?")
    tic, bpc, tt = [], [], []
    for r in runs:
        sel = _ms1_sel(r, pol)
        tt.append(r.rt[sel])
        tic.append(r.tic[sel])
        bpc.append(r.bpi[sel])
    total = np.array([float(np.trapezoid(y, t)) if len(t) > 1 else float(y.sum()) for t, y in zip(tt, tic)])
    med_int = []
    for r in runs:
        sel = _ms1_sel(r, pol)
        take = sel[:: max(1, len(sel) // 60)]
        v = np.concatenate([r.intensity[r.offsets[i]:r.offsets[i + 1]] for i in take]) if len(take) else np.array([])
        med_int.append(float(np.median(v)) if len(v) else 0.0)
    med_int = np.array(med_int)
    ctx["tic_total"] = total

    # TIC overlay
    fig, ax = plt.subplots(figsize=(10, 3.8))
    zorder = {"blank": 1, "sample": 2, "standard": 2, "qc": 3}
    for k in range(len(runs)):
        ax.plot(tt[k] / 60, tic[k], color=col[k], lw=0.9 if sheet.type[k] == "qc" else 0.7,
                alpha=0.9 if sheet.type[k] != "sample" else 0.75, zorder=zorder.get(sheet.type[k], 2))
    ax.set_xlabel("retention time (min)")
    ax.set_ylabel("total ion current")
    _legend(ax, sheet, gc, loc="upper right", ncol=2)
    nonblank = sheet.type.isin(["sample", "qc"]).values
    ref_total = np.median(total[nonblank]) if nonblank.any() else np.median(total)
    dev = np.log2(np.where(total > 0, total, np.nan) / ref_total)
    fold, fsrc = _f(params, "tic_outlier_fold", 2.0)
    _pg(job, "Raw-data QC")
    job.param("tic_outlier_fold", "Flag samples whose total signal differs by more than (fold)", fold,
              "A sample with half or double the median total ion current usually means a failed or partial injection, "
              "a spray problem or a very different sample amount — worth checking before trusting it.", fsrc)
    bad = [sheet["sample"][k] for k in range(len(runs)) if nonblank[k] and np.isfinite(dev[k]) and abs(dev[k]) > np.log2(fold)]
    # agreement of the QC chromatograms with each other
    qidx = np.flatnonzero(sheet.type.values == "qc")
    qc_r = None
    if len(qidx) >= 2:
        lo = max(tt[k].min() for k in qidx)
        hi = min(tt[k].max() for k in qidx)
        grid = np.linspace(lo, hi, 600)
        Y = np.array([np.interp(grid, tt[k], tic[k]) for k in qidx])
        cc = np.corrcoef(Y)
        qc_r = float(cc[np.triu_indices(len(qidx), 1)].min())
    job.figure("tic_overlay", "Total ion chromatograms", fig, wide=True,
               how="Each line is one injection: the summed signal of every ion over the run. Replicates of the same kind of "
                   "sample should trace nearly the same shape. QC injections (black) are aliquots of one pooled sample, so "
                   "they should lie on top of each other; blanks (grey) should be far lower except for solvent/contaminant peaks.",
               yours=(f"Total signal varies {fmt_n(total[nonblank].min() if nonblank.any() else total.min())}–{fmt_n(total[nonblank].max() if nonblank.any() else total.max())} "
                      f"across samples and QCs ({2 ** np.nanmax(np.abs(dev[nonblank])) if nonblank.any() else 1:.1f}-fold at most from the median). ")
               + (f"The {len(qidx)} QC chromatograms correlate with each other at r ≥ <b>{qc_r:.3f}</b>"
                  + (" — the instrument was stable." if qc_r > 0.95 else " — noticeably different; look for retention shifts or loss of signal.") if qc_r is not None else "")
               + (f" Outside ±{fold:g}-fold: <b>{', '.join(bad)}</b>." if bad else ""))

    # per-sample intensity bars in injection order
    o = np.argsort(sheet.order.values, kind="stable")
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    x = np.arange(len(o))
    axs[0].bar(x, total[o], color=[col[k] for k in o])
    axs[0].set_ylabel("total MS1 signal\n(TIC area)")
    axs[0].axhline(ref_total, color="#888", lw=0.8, ls="--")
    axs[1].bar(x, med_int[o], color=[col[k] for k in o])
    axs[1].set_ylabel("median peak\nintensity")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([sheet["sample"][k] for k in o], rotation=90, fontsize=7)
    axs[1].set_xlabel("injection order →")
    _legend(axs[0], sheet, gc, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=6)
    # a trend with injection order in the QCs is the drift that step 8 corrects
    trend = ""
    if len(qidx) >= 3:
        rq = np.corrcoef(sheet.order.values[qidx], np.log(np.maximum(total[qidx], 1)))[0, 1]
        slope = np.polyfit(sheet.order.values[qidx], np.log2(np.maximum(total[qidx], 1)), 1)[0]
        span = sheet.order.values.max() - sheet.order.values.min()
        trend = (f" QC total signal changes by about <b>{100 * (2 ** (slope * span) - 1):+.0f}%</b> from the first to the last injection (r = {rq:.2f})"
                 + (" — a drift that the QC-based correction (step 8) is designed to remove." if abs(slope * span) > 0.15 else " — the run was stable."))
    job.figure("intensity_order", "Signal per injection, in run order", fig, wide=True,
               how="Bars follow the order the samples were injected. A steady slope across the run (often downward as the "
                   "ion source gets dirty) is instrument drift; a single short bar is a failed injection. The dashed line is the median.",
               yours=(f"Median total signal {fmt_n(ref_total)}." + trend + (f" <b>{', '.join(bad)}</b> stand(s) out." if bad else "")))

    # base-peak chromatograms of QCs
    if len(qidx):
        fig, ax = plt.subplots(figsize=(10, 3.2))
        cmap = plt.get_cmap("Greys")
        for j, k in enumerate(qidx):
            ax.plot(tt[k] / 60, bpc[k], lw=0.8, color=cmap(0.45 + 0.55 * j / max(len(qidx) - 1, 1)),
                    label=sheet["sample"][k])
        ax.set_xlabel("retention time (min)")
        ax.set_ylabel("base-peak intensity")
        ax.legend(fontsize=7, ncol=min(4, len(qidx)))
        job.figure("qc_bpc", "Base-peak chromatograms of the QC injections", fig, wide=True,
                   how="The base-peak chromatogram plots the single most intense ion of every scan, which shows individual "
                       "compounds as sharp peaks. Pooled QCs are the same sample injected repeatedly, so peaks should line up "
                       "in time and height; a sideways shift is retention drift (fixed by alignment in step 3), a height change is sensitivity drift (step 8).",
                   yours=f"{len(qidx)} QC injections, shaded light→dark in injection order.")
    for b in bad:
        k = int(np.flatnonzero(sheet["sample"].values == b)[0])
        job.flag("warn", f"<b>{b}</b> has {2 ** abs(dev[k]):.1f}× {'less' if dev[k] < 0 else 'more'} total signal than the median "
                         "injection — check it for a failed or partial injection before trusting its values.")


# ====================================================================== step 2 — feature detection

def _detect(ctx):
    job, sheet, runs, pol, col, gc, params = ctx["job"], ctx["sheet"], ctx["runs"], ctx["pol"], ctx["col"], ctx["gc"], ctx["params"]
    job.step(30, "Estimating noise level and peak width")
    _pg(job, "Feature detection (OpenMS FeatureFindingMetabo)",
                    "Mass traces → chromatographic peaks → isotope-grouped features, run on every file separately.")
    ppm, ppm_src = _f(params, "mass_error_ppm", 10.0)
    job.param("mass_error_ppm", "Mass tolerance (ppm)", ppm,
              "How far the m/z of one ion may wander from scan to scan and between files. 10 ppm suits Orbitrap and modern "
              "Q-TOF data; use 20–30 for older TOFs, 3–5 for well-calibrated Orbitraps if you see neighbouring ions merge.", ppm_src)
    # noise: a low percentile of the MS1 centroid intensities. Centroided spectra rarely contain anything below the
    # instrument's noise floor, so the 10th percentile sits at that floor.
    rng = np.random.default_rng(0)
    p10 = []
    for r in runs:
        sel = _ms1_sel(r, pol)
        take = sel[:: max(1, len(sel) // 150)]
        v = np.concatenate([r.intensity[r.offsets[i]:r.offsets[i + 1]] for i in take]) if len(take) else np.array([])
        v = v[v > 0]
        if len(v) > 200000:
            v = rng.choice(v, 200000, replace=False)
        if len(v):
            p10.append(float(np.percentile(v, 10)))
    auto_noise = float(f"{np.median(p10):.2g}") if p10 else 1000.0
    noise, noise_src = _f(params, "noise_threshold_int", 1000.0, auto_noise)
    job.param("noise_threshold_int", "Noise threshold (intensity)", noise,
              f"Peaks below this are ignored when tracing ions. Derived from the data: the median over files of the 10th "
              f"percentile of all MS1 peak intensities ({fmt_n(min(p10)) if p10 else '?'}–{fmt_n(max(p10)) if p10 else '?'} per file) — the floor "
              "below which a centroided spectrum holds almost nothing but noise. Raise it (e.g. ×3) if detection is slow "
              "or finds thousands of tiny features; lower it to catch weaker compounds.", noise_src)
    # peak width from the most intense traces of a representative run (the median-signal QC, else sample)
    tot = ctx["tic_total"]
    cand = np.flatnonzero(sheet.type.values == "qc")
    if not len(cand):
        cand = np.flatnonzero(sheet.type.values == "sample")
    if not len(cand):
        cand = np.arange(len(runs))
    ref_k = int(cand[np.argsort(tot[cand])[len(cand) // 2]])
    auto_fwhm, n_fw = None, 0
    try:
        auto_fwhm, n_fw = _estimate_fwhm(runs[ref_k], pol, ppm, noise)
    except Exception:  # noqa: BLE001
        job.flag("info", "Peak width could not be estimated from the data (" + log_exc("fwhm") + "); 5 s was used.")
    fwhm, fwhm_src = _f(params, "chrom_fwhm", 5.0, round(float(np.clip(auto_fwhm, 1.0, 60.0)), 1) if auto_fwhm else None)
    job.param("chrom_fwhm", "Typical peak width, FWHM (s)", fwhm,
              (f"Median full width at half maximum of the {n_fw} most intense ion traces in {sheet['sample'][ref_k]}. " if fwhm_src == "auto" else
               "OpenMS default (not enough strong traces to measure it). " if fwhm_src == "default" else "Set by you. ")
              + "It sets the smoothing and how far apart two peaks must be to count as separate — ~2–5 s for UHPLC, 10–30 s for HPLC.", fwhm_src)
    min_trace, mt_src = _f(params, "min_trace_length", 5.0, round(float(np.clip(0.75 * fwhm, 1.5, 5.0)), 1))
    job.param("min_trace_length", "Minimum trace length (s)", min_trace,
              "An ion must be seen continuously for at least this long to count. Real chromatographic peaks last at least "
              "about their width; much shorter traces are electronic spikes. Derived as ¾ of the peak width, capped at the OpenMS default of 5 s.", mt_src)
    wmin, wmin_src = _f(params, "peak_width_min", 1.0, round(max(0.5, 0.25 * fwhm), 1))
    wmax, wmax_src = _f(params, "peak_width_max", 60.0, round(max(60.0, 8 * fwhm), 0))
    job.param("peak_width_min", "Narrowest accepted peak (FWHM, s)", wmin,
              "A quarter of the typical width: narrower 'peaks' are noise spikes, not compounds eluting from a column.", wmin_src)
    job.param("peak_width_max", "Widest accepted peak (FWHM, s)", wmax,
              "Eight times the typical width (at least 60 s): broader humps are column bleed or a rising baseline rather than one compound.", wmax_src)
    iso, iso_src = _b(params, "isotope_filtering", True)
    job.param("isotope_filtering", "Isotope pattern check", iso,
              "Only group a trace as the 13C isotope of another if the intensity ratio fits what organic molecules produce "
              "(OpenMS metabolite model, 5% RMS). Prevents unrelated co-eluting ions from being merged into one feature.", iso_src, "bool")
    single, single_src = _b(params, "remove_single_traces", False)
    job.param("remove_single_traces", "Drop features without a visible isotope", single,
              "Off by default: small or low-abundance metabolites often show no detectable 13C isotope, and dropping them "
              "loses real compounds. Turn on for a cleaner, more conservative feature list.", single_src, "bool")
    cfg = {"ppm": ppm, "noise": noise, "fwhm": fwhm, "min_trace": min_trace, "min_fwhm": wmin, "max_fwhm": wmax,
           "iso": iso, "single": single}
    ctx["cfg"] = cfg
    fmaps, fdfs, ntr = [], [], []
    for k, r in enumerate(runs):
        job.step(31 + int(12 * k / len(runs)), f"Detecting features in {sheet['sample'][k]} ({k + 1}/{len(runs)})")
        try:
            fm, df, nt = _find_features(r, pol, cfg)
        except Exception as e:  # noqa: BLE001
            raise UserFacingError(f"Feature detection failed on {sheet['sample'][k]}: {log_exc('ffm')}") from e
        fmaps.append(fm)
        fdfs.append(df)
        ntr.append(nt)
    nf = np.array([len(d) for d in fdfs])
    if nf.sum() == 0:
        raise UserFacingError("No features were detected in any file. The noise threshold "
                              f"({fmt_n(noise)}) may be above your signal — lower it and re-analyse.")
    ctx["fmaps"], ctx["fdfs"] = fmaps, fdfs

    job.section("features", "Step 2", "Feature detection",
                "A feature is one ion (m/z) eluting as one chromatographic peak, with its 13C isotopes attached. "
                "Every file is processed on its own here; step 4 matches the same feature across files.")
    o = np.argsort(sheet.order.values, kind="stable")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(np.arange(len(o)), nf[o], color=[col[k] for k in o])
    ax.set_xticks(np.arange(len(o)))
    ax.set_xticklabels([sheet["sample"][k] for k in o], rotation=90, fontsize=7)
    ax.set_ylabel("features detected")
    ax.set_xlabel("injection order →")
    _legend(ax, sheet, gc, loc="lower right", ncol=3)
    bio = sheet.type.isin(["sample", "qc"]).values
    med = float(np.median(nf[bio])) if bio.any() else float(np.median(nf))
    low = [sheet["sample"][k] for k in range(len(nf)) if bio[k] and nf[k] < 0.5 * med]
    blank_n = nf[sheet.type.values == "blank"]
    job.figure("features_per_sample", "Features per injection", fig, wide=True,
               how="How many features OpenMS found in each file. Samples and QCs should give similar counts; blanks far "
                   "fewer (only solvent, plasticiser and carry-over ions). A sample with far fewer features probably had a poor injection.",
               yours=f"Median <b>{commas(med)}</b> features per sample/QC (range {commas(nf[bio].min() if bio.any() else nf.min())}–{commas(nf[bio].max() if bio.any() else nf.max())})"
               + (f"; blanks {commas(blank_n.min())}–{commas(blank_n.max())}" if len(blank_n) else "") + "."
               + (f" Fewer than half the median: <b>{', '.join(low)}</b>." if low else ""))
    for s in low:
        job.flag("warn", f"<b>{s}</b> yielded fewer than half the median number of features — a weak or failed injection is likely.")
    # feature map of the reference run
    d = fdfs[ref_k]
    if len(d):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sc = ax.scatter(d.rt / 60, d.mz, c=np.log10(np.maximum(d.intensity, 1)), s=6, cmap="viridis", lw=0)
        plt.colorbar(sc, ax=ax, label="log10 intensity")
        ax.set_xlabel("retention time (min)")
        ax.set_ylabel("m/z")
        iso_frac = float((d.n_traces > 1).mean())
        job.figure("feature_map", f"Feature map of {sheet['sample'][ref_k]}", fig,
                   how="Every dot is one detected feature at its retention time and m/z, coloured by abundance. Real "
                       "metabolites spread over the whole gradient; a dense horizontal line at one m/z is a background ion or polymer series.",
                   yours=f"{commas(len(d))} features; {100 * iso_frac:.0f}% carry at least one 13C isotope trace "
                         f"(their charge is known: {100 * float((d.charge == 1).mean()):.0f}% singly, {100 * float((d.charge > 1).mean()):.0f}% multiply charged).")
    job.method("Feature detection", f"pyOpenMS {versions('pyopenms').get('pyopenms', '')}: MassTraceDetection (mass error {ppm:g} ppm, "
               f"noise threshold {fmt_n(noise)}, min trace {min_trace:g} s) → ElutionPeakDetection (FWHM {fwhm:g} s, width "
               f"{wmin:g}–{wmax:g} s) → FeatureFindingMetabo (isotope model {'metabolites 5% RMS, 13C spacing' if iso else 'none'}, "
               f"charges 1–3, single-trace features {'removed' if single else 'kept'}) — the OpenMS small-molecule workflow "
               "(Kenar et al. 2014, Mol Cell Proteomics 13:348; Röst et al. 2016, Nat Methods 13:741; UmetaFlow, Kontou et al. 2023, J Cheminform 15:52).")


# ====================================================================== step 3 — retention-time alignment

class _RTMap:
    """Original → aligned retention time for one run (and back), from an OpenMS TransformationDescription."""

    def __init__(self, tr=None, rt_lo=0.0, rt_hi=3600.0):
        self.grid = np.linspace(rt_lo, rt_hi, 400)
        if tr is None:
            self.fwd = self.grid.copy()
        else:
            self.fwd = np.array([tr.apply(float(x)) for x in self.grid])
        if np.any(np.diff(self.fwd) <= 0):   # a non-monotone warp cannot be inverted; keep order by sorting
            self.fwd = np.maximum.accumulate(self.fwd + np.arange(len(self.fwd)) * 1e-9)

    def to_aligned(self, x):
        return _lin_extrap(x, self.grid, self.fwd)

    def to_original(self, y):
        return _lin_extrap(y, self.fwd, self.grid)


def _lin_extrap(x, xp, fp):
    x = np.asarray(x, float)
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    if lo.any():
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    if hi.any():
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def _align(ctx):
    import pyopenms as oms
    job, sheet, runs, params, cfg = ctx["job"], ctx["sheet"], ctx["runs"], ctx["params"], ctx["cfg"]
    fmaps, fdfs = ctx["fmaps"], ctx["fdfs"]
    rt_lo = min(float(r.rt.min()) for r in runs)
    rt_hi = max(float(r.rt.max()) for r in runs)
    maps = [_RTMap(None, rt_lo, rt_hi) for _ in runs]
    ctx["rtmaps"] = maps
    ctx["align_resid"] = np.array([])
    _pg(job, "Alignment & linking", "Matching the same feature across files.")
    do, do_src = _b(params, "align", True)
    max_shift, ms_src = _f(params, "align_max_rt_diff", 100.0)
    nf = np.array([len(d) for d in fdfs])
    if not do or len(runs) < 2:
        job.param("align", "Retention-time alignment", do, "Switched off by you." if not do else "Only one file — nothing to align.", do_src, "bool")
        return
    job.step(44, "Aligning retention times")
    # reference: the QC (else sample) with the most features — the richest map gives the most anchor points
    cand = np.flatnonzero(sheet.type.values == "qc")
    if not len(cand):
        cand = np.flatnonzero(sheet.type.values == "sample")
    if not len(cand):
        cand = np.arange(len(runs))
    ref_user = str(params.get("align_reference") or "").strip()
    if ref_user and ref_user in list(sheet["sample"]):
        ref = int(np.flatnonzero(sheet["sample"].values == ref_user)[0])
        ref_src, ref_why = "user", "Chosen by you."
    else:
        ref = int(cand[np.argmax(nf[cand])])
        ref_src = "auto"
        ref_why = (f"The {'QC' if sheet.type[ref] == 'qc' else 'sample'} with the most features ({commas(nf[ref])}); every other "
                   "file is warped onto its time axis. A pooled QC is the natural reference because it contains every compound.")
    job.param("align", "Retention-time alignment", do, "OpenMS pose clustering, then a smooth (LOWESS) curve through the matched features.", do_src, "bool")
    job.param("align_reference", "Alignment reference", sheet["sample"][ref], ref_why, ref_src, "text")
    job.param("align_max_rt_diff", "Largest retention shift considered (s)", max_shift,
              "Pairs of features further apart than this are never matched while estimating the shift. The OpenMS default "
              "(100 s) covers ordinary column ageing; raise it only if chromatograms are visibly shifted by more.", ms_src)
    ctx["align_ref"] = ref
    aligner = oms.MapAlignmentAlgorithmPoseClustering()
    _oms_param(aligner, {"max_num_peaks_considered": -1,
                         "pairfinder:distance_MZ:max_difference": float(cfg["ppm"]),
                         "pairfinder:distance_MZ:unit": "ppm",
                         "pairfinder:distance_RT:max_difference": float(max_shift)})
    aligner.setReference(fmaps[ref])
    transformer = oms.MapAlignmentTransformer()
    before, after, failed = {}, {}, []
    for k, fm in enumerate(fmaps):
        if k == ref:
            continue
        if fm.size() < 10:
            failed.append(sheet["sample"][k])
            continue
        try:
            tr = oms.TransformationDescription()
            aligner.align(fm, tr)
            dp = tr.getDataPoints()
            x = np.array([d.first for d in dp])
            y = np.array([d.second for d in dp])
            if len(x) >= 30:
                lp = oms.Param()
                oms.TransformationModelLowess.getDefaultParameters(lp)
                lp.setValue("span", 0.5)
                tr.fitModel("lowess", lp)
            elif len(x) >= 3:
                tr.fitModel("linear", oms.Param())
            else:
                failed.append(sheet["sample"][k])
                continue
            m = _RTMap(tr, rt_lo, rt_hi)
            if not np.all(np.isfinite(m.fwd)):
                failed.append(sheet["sample"][k])
                continue
            maps[k] = m
            before[k] = (y, x - y)
            after[k] = (y, m.to_aligned(x) - y)
            transformer.transformRetentionTimes(fm, tr, True)
        except Exception:  # noqa: BLE001
            failed.append(sheet["sample"][k] + " (" + log_exc("align") + ")")
    if failed:
        blanks = set(sheet["sample"][sheet.type == "blank"])
        others = [f for f in failed if f.split(" (")[0] not in blanks]
        job.flag("warn" if others else "info",
                 "Retention times were left unaligned for " + ", ".join(failed) + " — too few features to match against the "
                 "reference" + (" (expected for blanks, which contain few compounds)." if not others else
                                "; their features may fail to link to the same compound in other files."))
    if not before:
        return
    res = np.concatenate([np.abs(a[1]) for a in after.values()])
    raw = np.concatenate([np.abs(b[1]) for b in before.values()])
    ctx["align_resid"] = res
    job.section("align", "Step 3", "Retention-time alignment",
                "The same compound elutes a little earlier or later in each injection. Alignment warps every file's time "
                "axis onto the reference so that the same compound can be recognised across files in step 4.")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    col = ctx["col"]
    for k in before:
        yb, db = before[k]
        ya, da = after[k]
        o = np.argsort(yb)
        axs[0].plot(yb[o] / 60, db[o], ".", ms=2, color=col[k], alpha=0.6)
        grid = np.linspace(yb.min(), yb.max(), 100)
        axs[0].plot(grid / 60, maps[k].to_original(grid) - grid, lw=1, color=col[k])
        axs[1].plot(ya[o] / 60, da[o], ".", ms=2, color=col[k], alpha=0.6)
    for ax, t in zip(axs, ("before alignment", "after alignment")):
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlabel("retention time in reference (min)")
        ax.set_title(t, fontsize=10)
    axs[0].set_ylabel("RT difference to reference (s)")
    lim = np.percentile(raw, 99.5) * 1.3 + 1
    axs[0].set_ylim(-lim, lim)
    _legend(axs[1], sheet, ctx["gc"], loc="upper right", ncol=2)
    job.figure("rt_alignment", "Retention-time deviation before and after alignment", fig, wide=True,
               how="Each dot is a feature found in both a file and the reference; its height is how many seconds earlier or "
                   "later it eluted. Left: raw deviations with the fitted correction curve per file. Right: what is left after "
                   "warping — it should be a flat band around zero, narrower than a peak width.",
               yours=f"Median shift {np.median(raw):.1f} s before (95th percentile {np.percentile(raw, 95):.1f} s), "
                     f"<b>{np.median(res):.1f} s</b> after (95th percentile {np.percentile(res, 95):.1f} s) — "
                     + (f"well below the {cfg['fwhm']:g} s peak width, so features can be matched reliably." if np.percentile(res, 95) < cfg["fwhm"]
                        else f"not negligible against the {cfg['fwhm']:g} s peak width; the linking tolerance below is widened to match."))
    job.method("Alignment", f"OpenMS MapAlignmentAlgorithmPoseClustering against {sheet['sample'][ref]} (m/z tolerance "
               f"{cfg['ppm']:g} ppm, max RT difference {max_shift:g} s), LOWESS transformation (span 0.5) fitted through the "
               "matched feature pairs (linear when fewer than 30 pairs); feature RTs transformed with MapAlignmentTransformer.")


# ====================================================================== step 4 — correspondence / linking

def _link(ctx):
    import pyopenms as oms
    job, sheet, params, cfg = ctx["job"], ctx["sheet"], ctx["params"], ctx["cfg"]
    fmaps, fdfs = ctx["fmaps"], ctx["fdfs"]
    job.step(48, "Linking features across files")
    res = ctx["align_resid"]
    auto_tol = max(1.5 * cfg["fwhm"], 3 * float(np.percentile(res, 95)) if len(res) else 0.0, 5.0)
    auto_tol = float(np.clip(round(auto_tol, 1), 5.0, 90.0))
    rt_tol, rt_src = _f(params, "link_rt_tol", 15.0, auto_tol)
    job.param("link_rt_tol", "Linking RT tolerance (s)", rt_tol,
              f"Features in different files are the same compound only if their aligned RTs differ by less than this. "
              f"Derived as the larger of 1.5 × peak width ({1.5 * cfg['fwhm']:.1f} s) and 3 × the 95th-percentile "
              f"deviation left after alignment ({3 * float(np.percentile(res, 95)) if len(res) else 0:.1f} s), at least 5 s.", rt_src)
    job.param("_link_mz", "Linking m/z tolerance (ppm)", cfg["ppm"], "Same as the mass tolerance above.", "fixed", "fixed")
    ctx["rt_tol"] = rt_tol
    grouper = oms.FeatureGroupingAlgorithmKD()
    _oms_param(grouper, {"warp:enabled": "false", "link:rt_tol": float(rt_tol), "link:mz_tol": float(cfg["ppm"]),
                         "mz_unit": "ppm"})
    cmap = oms.ConsensusMap()
    heads = {}
    for k, fm in enumerate(fmaps):
        h = oms.ColumnHeader()
        h.filename = sheet["file"][k]
        h.label = sheet["sample"][k]
        h.size = fm.size()
        heads[k] = h
    cmap.setColumnHeaders(heads)
    grouper.group(fmaps, cmap)
    # original (unaligned) RTs and hull bounds of every feature, keyed by (map, unique id)
    lut = {}
    for k, d in enumerate(fdfs):
        for u, rt, lo, hi, fw, ch, it in zip(d.uid.values, d.rt.values, d.rt_start.values, d.rt_end.values, d.fwhm.values,
                                             d.charge.values, d.intensity.values):
            lut[(k, int(u))] = (rt, lo, hi, fw, ch)
    n = len(fmaps)
    rows, I, RT0, LO, HI = [], [], [], [], []
    for cf in cmap:
        vals = np.full(n, np.nan)
        rt0 = np.full(n, np.nan)
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)
        chs = []
        fws = []
        for h in cf.getFeatureList():
            k = h.getMapIndex()
            if not np.isfinite(vals[k]) or h.getIntensity() > vals[k]:
                vals[k] = h.getIntensity()
                info = lut.get((k, int(h.getUniqueId())))
                if info:
                    rt0[k], lo[k], hi[k], fw, ch = info
                    fws.append(fw)
                    chs.append(ch)
        chs = [c for c in chs if c > 0]
        rows.append({"mz": cf.getMZ(), "rt": cf.getRT(), "charge": int(np.bincount(chs).argmax()) if chs else 0,
                     "fwhm": float(np.nanmedian(fws)) if len(fws) and np.isfinite(fws).any() else cfg["fwhm"]})
        I.append(vals)
        RT0.append(rt0)
        LO.append(lo)
        HI.append(hi)
    if not rows:
        raise UserFacingError("No features could be linked across files.")
    feat = pd.DataFrame(rows)
    I = np.array(I)
    order = np.lexsort((feat.mz.values, feat.rt.values))
    feat = feat.iloc[order].reset_index(drop=True)
    I, RT0, LO, HI = I[order], np.array(RT0)[order], np.array(LO)[order], np.array(HI)[order]
    ids, seen = [], {}
    for mz, rt in zip(feat.mz, feat.rt):
        b = _fid(mz, rt)
        seen[b] = seen.get(b, 0) + 1
        ids.append(b if seen[b] == 1 else f"{b}_{seen[b]}")
    feat.insert(0, "id", ids)
    det = np.isfinite(I)
    feat["n_detected"] = det.sum(1)
    ctx.update(feat=feat, I_det=I.copy(), det=det, RT0=RT0, LO=LO, HI=HI)

    job.section("linking", "Step 4", "Matching features across files",
                "Features from different files with the same m/z (within the mass tolerance) and aligned retention time "
                "(within the RT tolerance) are joined into one row of the feature table.")
    bio = sheet.type.isin(["sample", "qc"]).values
    frac = det[:, bio].mean(1) if bio.any() else det.mean(1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    nb = int(bio.sum()) if bio.any() else n
    cnt = det[:, bio].sum(1) if bio.any() else det.sum(1)
    axs[0].hist(cnt, bins=np.arange(0.5, nb + 1.5), color="#6b8fb3")
    axs[0].set_xlabel("found in how many samples/QCs (before gap filling)")
    axs[0].set_ylabel("features")
    sc = axs[1].scatter(feat.rt / 60, feat.mz, c=frac, s=5, cmap="viridis", vmin=0, vmax=1, lw=0)
    plt.colorbar(sc, ax=axs[1], label="fraction of samples detected")
    axs[1].set_xlabel("retention time (min)")
    axs[1].set_ylabel("m/z")
    full = float((frac >= 0.999).mean())
    job.figure("linking", "Consensus features", fig, wide=True,
               how="Left: for every feature, in how many samples/QCs it was detected. A peak on the right means compounds "
                   "found everywhere; features seen in only one or two files are often noise or near the detection limit. "
                   "Right: the consensus features on the m/z–RT plane, coloured by how consistently they were found.",
               yours=f"<b>{commas(len(feat))}</b> consensus features; {100 * full:.0f}% were detected in every sample and QC, "
                     f"{100 * float((cnt <= 1).mean()):.0f}% in only one. {100 * (1 - det[:, bio].mean()) if bio.any() else 0:.0f}% of "
                     "sample/QC values are missing before gap filling.")
    job.method("Linking", f"OpenMS FeatureGroupingAlgorithmKD (RT tolerance {rt_tol:g} s, m/z tolerance {cfg['ppm']:g} ppm, "
               "no additional warping) → consensus map; features named M<m/z>T<RT in min>.")


# ====================================================================== step 5 — gap filling

def _gap_fill(ctx):
    job, sheet, runs, params, cfg, pol = ctx["job"], ctx["sheet"], ctx["runs"], ctx["params"], ctx["cfg"], ctx["pol"]
    feat, I, det = ctx["feat"], ctx["I_det"].copy(), ctx["det"]
    RT0 = ctx["RT0"].copy()
    _pg(job, "Gap filling")
    do, src = _b(params, "gap_fill", True)
    job.param("gap_fill", "Re-integrate missing values from the raw data", do,
              "A feature missing from a file is usually there but was too weak for the feature finder. Integrating the raw "
              "ion chromatogram at the expected m/z and RT recovers its real (small) value, which is far better than "
              "guessing one later. When no peak rises above 3× noise the value stays missing.", src, "bool")
    filled = np.zeros_like(det)
    ctx["filled"] = filled
    ctx["gap_ratio"] = None
    if not do:
        ctx["I_raw"] = I
        ctx["RT_obs"] = RT0
        return
    rt_tol = ctx["rt_tol"]
    maps = ctx["rtmaps"]
    mzs, rts = feat.mz.values, feat.rt.values
    fw = np.clip(np.nan_to_num(feat.fwhm.values, nan=cfg["fwhm"]), 1.0, None)
    n_missing = int((~det).sum())
    comp = []   # (feature, run, FFM value, re-integrated value) for detected values: calibrates the two scales
    rng = np.random.default_rng(1)
    t0 = time.time()
    for k, r in enumerate(runs):
        job.step(52 + int(8 * k / len(runs)), f"Gap filling {sheet['sample'][k]} ({k + 1}/{len(runs)})")
        idx = _MS1Index(r, pol)
        miss = np.flatnonzero(~det[:, k])
        exp_rt = maps[k].to_original(rts)
        for i in miss:
            w = max(3 * fw[i], rt_tol) + 2 * fw[i]
            t, y = idx.xic(mzs[i], cfg["ppm"], exp_rt[i] - w, exp_rt[i] + w)
            if len(t) < 5 or not np.any(y > 0):
                continue
            pk = peaks.integrate(t, y, rt=float(exp_rt[i]), window=float(rt_tol), smooth_window=5, min_snr=3.0)
            if pk and pk["points"] >= 3 and abs(pk["rt"] - exp_rt[i]) <= rt_tol:
                I[i, k] = pk["area"]
                RT0[i, k] = pk["rt"]
                filled[i, k] = True
        have = np.flatnonzero(det[:, k])
        for i in rng.choice(have, min(len(have), 40), replace=False) if len(have) else []:
            w = 3 * fw[i] + 2 * fw[i]
            t, y = idx.xic(mzs[i], cfg["ppm"], RT0[i, k] - w, RT0[i, k] + w)
            pk = peaks.integrate(t, y, rt=float(RT0[i, k]), window=float(fw[i]), smooth_window=5, min_snr=3.0) if len(t) >= 5 else None
            if pk:
                comp.append((I[i, k], pk["area"]))
        del idx
    # the feature finder reports the area of the smoothed trace; a re-integrated raw area is on almost the same
    # scale, but not exactly — the median ratio over detected values puts filled values on the finder's scale
    ratio = 1.0
    if len(comp) >= 20:
        c = np.array(comp)
        ok = (c[:, 0] > 0) & (c[:, 1] > 0)
        ratio = float(np.median(c[ok, 0] / c[ok, 1])) if ok.sum() >= 20 else 1.0
        I[filled] *= ratio
        ctx["gap_ratio"] = ratio
    n_filled = int(filled.sum())
    ctx["I_raw"] = I
    ctx["RT_obs"] = RT0
    feat["n_filled"] = filled.sum(1)
    job.section("gapfill", "Step 5", "Gap filling",
                "Missing values are re-measured from the raw data instead of being guessed.")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    if len(comp) >= 5:
        c = np.array(comp)
        ok = (c[:, 0] > 0) & (c[:, 1] > 0)
        axs[0].loglog(c[ok, 1] * ratio, c[ok, 0], ".", ms=4, color="#6b8fb3")
        lo_, hi_ = np.nanmin(c[ok]) * 0.5, np.nanmax(c[ok]) * 2
        axs[0].plot([lo_, hi_], [lo_, hi_], color="#888", lw=0.8, ls="--")
        rr = float(np.corrcoef(np.log(c[ok, 0]), np.log(c[ok, 1] * ratio))[0, 1]) if ok.sum() > 2 else np.nan
        axs[0].set_xlabel("re-integrated from raw XIC (scaled)")
        axs[0].set_ylabel("feature finder area")
        axs[0].set_title("detected values measured both ways", fontsize=10)
    else:
        rr = np.nan
        axs[0].axis("off")
    bio = sheet.type.isin(["sample", "qc"]).values
    vals_det = np.log10(I[det & bio[None, :]]) if (det & bio[None, :]).any() else np.array([])
    vals_fill = np.log10(I[filled & bio[None, :]]) if (filled & bio[None, :]).any() else np.array([])
    bins = np.linspace(min(vals_det.min() if len(vals_det) else 3, vals_fill.min() if len(vals_fill) else 3),
                       max(vals_det.max() if len(vals_det) else 7, vals_fill.max() if len(vals_fill) else 7), 50)
    axs[1].hist(vals_det, bins=bins, color="#6b8fb3", alpha=0.8, label="detected")
    if len(vals_fill):
        axs[1].hist(vals_fill, bins=bins, color="#e39c3a", alpha=0.8, label="gap-filled")
    axs[1].set_xlabel("log10 intensity")
    axs[1].set_ylabel("values")
    axs[1].legend(fontsize=8)
    still = int((~np.isfinite(I[:, bio])).sum()) if bio.any() else int((~np.isfinite(I)).sum())
    job.figure("gap_filling", "Re-integrated missing values", fig, wide=True,
               how="Left: for features that were detected, the feature finder's area against a fresh integration of the raw "
                   "ion chromatogram — agreement along the diagonal shows the filled values are on the same scale. Right: filled "
                   "values (orange) should sit at the low end of the intensity range, because what the finder missed is mostly weak signal.",
               yours=f"<b>{commas(n_filled)}</b> of {commas(n_missing)} missing values ({100 * n_filled / max(n_missing, 1):.0f}%) were "
                     f"recovered from the raw data; {commas(still)} sample/QC values remain truly missing (no peak above 3× noise)."
                     + (f" Detected values agree with re-integration at r = {rr:.2f} on the log scale" if np.isfinite(rr) else "")
                     + (f"; filled areas were multiplied by {ratio:.2f} to match the finder's scale." if ctx['gap_ratio'] else "."))
    job.method("Gap filling", f"For each missing value the extracted-ion chromatogram at the consensus m/z ± {cfg['ppm']:g} ppm was "
               "integrated around the expected retention time (the aligned consensus RT mapped back through the file's alignment "
               f"curve, ± {rt_tol:g} s) with the app's peak integrator (Savitzky–Golay smoothing, drop-to-baseline trapezoid area, "
               "S/N ≥ 3); values with no peak stay missing. Filled areas were scaled by the median ratio of feature-finder to "
               f"re-integrated area over detected values ({ratio:.3f}). {time.time() - t0:.0f} s.")


# ====================================================================== MS2 spectra ↔ features

def _ms2_link(ctx):
    """Attach to each feature the best MS2 spectrum whose precursor m/z and retention time fall on it."""
    runs, feat, cfg, pol = ctx["runs"], ctx["feat"], ctx["cfg"], ctx["pol"]
    RT0, LO, HI = ctx["RT_obs"], ctx["LO"], ctx["HI"]
    best = {}   # feature → (score, run, spectrum index)
    n_ms2 = 0
    fw = np.nan_to_num(feat.fwhm.values, nan=cfg["fwhm"])
    for k, r in enumerate(runs):
        sel = np.flatnonzero((r.ms_level == 2) & np.isfinite(r.precursor_mz))
        if pol and (r.polarity[sel] == pol).any():
            sel = sel[(r.polarity[sel] == pol) | (r.polarity[sel] == 0)]
        if not len(sel):
            continue
        n_ms2 += len(sel)
        o = np.argsort(r.precursor_mz[sel])
        sel = sel[o]
        pmz = r.precursor_mz[sel]
        tic = np.array([float(r.intensity[r.offsets[i]:r.offsets[i + 1]].sum()) for i in sel])
        pint = np.nan_to_num(r.precursor_intensity[sel], nan=0.0)
        for i, mz in enumerate(feat.mz.values):
            tol = mz * cfg["ppm"] * 1e-6
            a, b = np.searchsorted(pmz, mz - tol), np.searchsorted(pmz, mz + tol, side="right")
            if b <= a:
                continue
            rt = RT0[i, k] if np.isfinite(RT0[i, k]) else ctx["rtmaps"][k].to_original(np.array([feat.rt.values[i]]))[0]
            lo = LO[i, k] if np.isfinite(LO[i, k]) else rt - fw[i]
            hi = HI[i, k] if np.isfinite(HI[i, k]) else rt + fw[i]
            lo, hi = min(lo, rt - 0.5 * fw[i]), max(hi, rt + 0.5 * fw[i])
            for j in range(a, b):
                t = r.rt[sel[j]]
                if lo <= t <= hi:
                    score = pint[j] if pint[j] > 0 else tic[j]
                    if i not in best or score > best[i][0]:
                        best[i] = (score, k, int(sel[j]))
    ctx["ms2"] = {i: (k, s) for i, (_, k, s) in best.items()}
    ctx["n_ms2_spectra"] = n_ms2


def _read_spectral_library(path: Path):
    """.msp (NIST/MoNA/MS-DIAL style) or .mgf → list of {name, precursor, mz, int, adduct, formula, polarity}."""
    path = Path(path)
    out = []
    if path.suffix.lower() == ".mgf":
        from pyteomics import mgf
        with mgf.read(str(path), use_index=False, convert_arrays=1) as rd:
            for s in rd:
                p = s["params"]
                pm = p.get("pepmass") or (np.nan,)
                name = p.get("name") or p.get("compound_name") or p.get("title") or f"spectrum {len(out) + 1}"
                ion = str(p.get("ionmode") or p.get("ion_mode") or "").lower()
                out.append({"name": str(name), "precursor": float(pm[0]) if pm[0] is not None else np.nan,
                            "mz": np.asarray(s["m/z array"], float), "int": np.asarray(s["intensity array"], float),
                            "adduct": str(p.get("adduct") or p.get("precursor_type") or ""),
                            "formula": str(p.get("formula") or ""),
                            "polarity": 1 if ion.startswith("pos") else -1 if ion.startswith("neg") else 0})
        return out
    num = re.compile(r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)[\s,:]+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")
    cur = None

    def close():
        if cur and cur["mz"]:
            cur["mz"], cur["int"] = np.asarray(cur["mz"], float), np.asarray(cur["int"], float)
            out.append(cur)
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                close()
                cur = None
                continue
            if cur is None:
                cur = {"name": "", "precursor": np.nan, "mz": [], "int": [], "adduct": "", "formula": "", "polarity": 0}
            if re.match(r"^[A-Za-z][A-Za-z0-9_ /\-\.]*:", s) and not re.match(r"^\d", s):
                key, val = s.split(":", 1)
                key, val = key.strip().lower().replace(" ", "_"), val.strip()
                if key == "name" and not cur["name"]:
                    cur["name"] = val
                elif key in ("precursormz", "precursor_mz", "pepmass", "precursor_m/z", "exactmass_precursor"):
                    try:
                        cur["precursor"] = float(val.split()[0])
                    except ValueError:
                        pass
                elif key in ("precursor_type", "precursortype", "adduct"):
                    cur["adduct"] = val
                elif key in ("ion_mode", "ionmode", "polarity"):
                    v = val.lower()
                    cur["polarity"] = 1 if v.startswith("p") else -1 if v.startswith("n") else 0
                elif key == "formula":
                    cur["formula"] = val
                continue
            for a, b in num.findall(s):
                cur["mz"].append(float(a))
                cur["int"].append(float(b))
    close()
    return out


def cosine(mz1, i1, mz2, i2, tol=0.02, power=0.5):
    """Matched-peaks cosine similarity of two centroided spectra.

    Intensities are raised to `power` (square root by default, as GNPS does, so a few dominant fragments do not
    decide the score alone). Every pair of peaks within `tol` Da is a candidate; pairs are then assigned greedily
    from the largest product down so that each peak is used at most once — the standard 'greedy cosine'.
    Returns (score 0–1, number of matched peaks).
    """
    mz1, mz2 = np.asarray(mz1, float), np.asarray(mz2, float)
    a, b = np.asarray(i1, float) ** power, np.asarray(i2, float) ** power
    if not len(a) or not len(b) or a.sum() <= 0 or b.sum() <= 0:
        return 0.0, 0
    o2 = np.argsort(mz2)
    mz2, b = mz2[o2], b[o2]
    cand = []
    for i, m in enumerate(mz1):
        lo, hi = np.searchsorted(mz2, m - tol), np.searchsorted(mz2, m + tol, side="right")
        for j in range(lo, hi):
            cand.append((a[i] * b[j], i, j))
    if not cand:
        return 0.0, 0
    cand.sort(reverse=True)
    used1, used2 = set(), set()
    s = 0.0
    n = 0
    for p, i, j in cand:
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        s += p
        n += 1
    return float(s / (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()))), n


# ====================================================================== step 6 — adducts, annotation

def _group_adducts(feat, X, pol, ppm, rt_tol):
    """Group co-eluting features that are different adducts of one neutral molecule.

    Every feature gets one hypothesis per possible adduct (neutral mass = (m/z·|z| − shift)/n). Two co-eluting
    features support each other when their neutral masses agree within the ppm tolerance for some pair of
    different adducts, and — when there are enough samples — their intensities rise and fall together. The
    hypothesis with most partners is accepted first (ties: the primary [M+H]+/[M−H]− ion, then intensity),
    its partners are removed, and so on.
    """
    adducts = chem.DEFAULT_ADDUCTS[1 if pol > 0 else -1]
    n = len(feat)
    mz, rt, z = feat.mz.values, feat.rt.values, feat.charge.values
    neutral = {}
    for a in adducts:
        za = abs(chem.ADDUCTS[a][0])
        m = np.array([chem.neutral_from_mz(x, a) for x in mz])
        ok = (z == za) | ((z == 0) & (za == 1))
        neutral[a] = np.where(ok, m, np.nan)
    logX = np.log(np.where(X > 0, X, np.nan))
    o = np.argsort(rt)
    hyp = {}   # (i, adduct) → {j: adduct_j}
    for p in range(n):
        i = o[p]
        q = p + 1
        while q < n and rt[o[q]] - rt[i] <= rt_tol:
            j = o[q]
            q += 1
            pairs = []
            for a in adducts:
                mi = neutral[a][i]
                if not np.isfinite(mi):
                    continue
                for b in adducts:
                    if a == b:
                        continue
                    mj = neutral[b][j]
                    if np.isfinite(mj) and abs(mi - mj) <= ppm * 1e-6 * mi:
                        pairs.append((a, b))
            if not pairs:
                continue
            both = np.isfinite(logX[i]) & np.isfinite(logX[j])
            if both.sum() >= 4:
                r = np.corrcoef(logX[i, both], logX[j, both])[0, 1]
                if not np.isfinite(r) or r < 0.5:
                    continue
            for a, b in pairs:
                hyp.setdefault((i, a), {})[j] = b
                hyp.setdefault((j, b), {})[i] = a
    group = np.full(n, -1)
    adduct = np.array([""] * n, dtype=object)
    gmass = np.full(n, np.nan)
    inten = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=1) if X.size else np.zeros(n)
    prim = PRIMARY[1 if pol > 0 else -1]
    gid = 0
    while True:
        best, bkey = None, None
        for (i, a), partners in hyp.items():
            if group[i] >= 0:
                continue
            free = {j: b for j, b in partners.items() if group[j] < 0}
            if not free:
                continue
            key = (len(set(free)), a == prim, np.nan_to_num(inten[i]))
            if bkey is None or key > bkey:
                best, bkey = (i, a, free), key
        if best is None:
            break
        i, a, free = best
        members = {i: a}
        used = {a}
        for j in sorted(free, key=lambda j: -np.nan_to_num(inten[j])):
            if free[j] not in used:
                members[j] = free[j]
                used.add(free[j])
        m = np.array([neutral[ad][j] for j, ad in members.items()])
        w = np.array([np.nan_to_num(inten[j], nan=1.0) for j in members])
        M = float(np.sum(m * w) / np.sum(w))
        for j, ad in members.items():
            group[j] = gid
            adduct[j] = ad
            gmass[j] = M
        gid += 1
    return group, adduct, gmass


def _load_library(path=None):
    lib = pd.read_csv(path or LIB_PATH)
    lib.columns = [c.strip().lower() for c in lib.columns]
    return lib


def _user_library(path):
    """User CSV: name + formula (or neutral mass / mz), optional rt (minutes; rt_s for seconds), optional adduct."""
    df = msio.read_table(Path(path))
    df.columns = [re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in df.columns]
    name = next((c for c in df.columns if c in ("name", "compound", "compound_name", "metabolite", "id")), None)
    if name is None:
        raise UserFacingError(f"The library {Path(path).name} needs a 'name' column.")
    out = pd.DataFrame({"name": df[name].astype(str)})
    fcol = next((c for c in df.columns if c in ("formula", "molecular_formula")), None)
    mcol = next((c for c in df.columns if c in ("mass", "monoisotopic_mass", "exact_mass", "neutral_mass", "monoisotopic")), None)
    zcol = next((c for c in df.columns if c in ("mz", "m/z", "precursor_mz", "precursormz")), None)
    out["formula"] = df[fcol].astype(str) if fcol else ""
    mass = np.full(len(df), np.nan)
    if mcol:
        mass = pd.to_numeric(df[mcol], errors="coerce").values
    if fcol:
        for k, f in enumerate(out["formula"]):
            if not np.isfinite(mass[k]) and f and f.lower() != "nan":
                try:
                    mass[k] = chem.formula_mass(f)
                except Exception:  # noqa: BLE001
                    pass
    out["mass"] = mass
    out["mz"] = pd.to_numeric(df[zcol], errors="coerce").values if zcol else np.nan
    acol = next((c for c in df.columns if c in ("adduct", "ion", "precursor_type")), None)
    out["adduct"] = df[acol].astype(str) if acol else ""
    rt = np.full(len(df), np.nan)
    for c, scale in (("rt_min", 60), ("rt", 60), ("retention_time", 60), ("rt_s", 1), ("rt_sec", 1)):
        if c in df.columns:
            rt = pd.to_numeric(df[c], errors="coerce").values * scale
            break
    out["rt"] = rt
    out = out[np.isfinite(out["mass"]) | np.isfinite(out["mz"])].reset_index(drop=True)
    return out


def _annotate(ctx):
    job, feat, params, cfg, pol = ctx["job"], ctx["feat"], ctx["params"], ctx["cfg"], ctx["pol"]
    sheet, runs = ctx["sheet"], ctx["runs"]
    job.step(61, "Grouping adducts and annotating")
    _pg(job, "Adducts & annotation")
    fw = cfg["fwhm"]
    art, art_src = _f(params, "adduct_rt_tol", round(max(1.0, 0.5 * fw), 1), round(max(1.0, 0.5 * fw), 1))
    job.param("adduct_rt_tol", "Adducts must co-elute within (s)", art,
              "Different ions of one molecule ([M+H]+, [M+Na]+, …) form in the ion source from the same eluting "
              "compound, so their apexes coincide. Half the typical peak width is tight enough to avoid chance pairings.", art_src)
    ann_ppm, ap_src = _f(params, "annotation_ppm", 5.0)
    job.param("annotation_ppm", "Annotation m/z tolerance (ppm)", ann_ppm,
              "A library compound is a candidate only if its calculated m/z is this close. 5 ppm is typical for calibrated "
              "Orbitrap/Q-TOF data; tighter gives fewer false candidates, wider suits older instruments.", ap_src)
    bio = sheet.type.isin(["sample", "qc"]).values
    X = ctx["I_raw"][:, bio] if bio.any() else ctx["I_raw"]
    try:
        grp, add, gmass = _group_adducts(feat, X, pol, cfg["ppm"], art)
    except Exception:  # noqa: BLE001
        job.flag("warn", "Adduct grouping failed: " + log_exc("adducts"))
        grp, add, gmass = np.full(len(feat), -1), np.array([""] * len(feat), dtype=object), np.full(len(feat), np.nan)
    feat["compound_group"] = [f"C{g + 1}" if g >= 0 else "" for g in grp]
    feat["adduct"] = add
    feat["neutral_mass"] = gmass

    # ---- library candidates (m/z only → MSI level 3)
    lib = _load_library()
    lib["mass"] = [chem.formula_mass(f) for f in lib["formula"]]
    adducts = chem.DEFAULT_ADDUCTS[1 if pol > 0 else -1]
    cands = {i: [] for i in range(len(feat))}
    lm = lib["mass"].values
    o = np.argsort(lm)
    lm_s = lm[o]

    def lib_hits(M):
        tol = M * ann_ppm * 1e-6
        a, b = np.searchsorted(lm_s, M - tol), np.searchsorted(lm_s, M + tol, side="right")
        return o[a:b]
    for i, (mz, z) in enumerate(zip(feat.mz.values, feat.charge.values)):
        if grp[i] >= 0:
            tries = [(add[i], gmass[i], 0)]
        else:
            tries = []
            for rank, a in enumerate(adducts):
                za = abs(chem.ADDUCTS[a][0])
                if z and z != za:
                    continue
                if not z and za != 1:
                    continue
                tries.append((a, chem.neutral_from_mz(mz, a), rank + 1))
        for a, M, rank in tries:
            for h in lib_hits(M):
                th = chem.adduct_mz(lm[h], a)
                cands[i].append({"name": lib["name"][h], "formula": lib["formula"][h], "class": lib["class"][h],
                                 "kegg": lib["kegg"][h] if isinstance(lib["kegg"][h], str) else "", "adduct": a,
                                 "ppm": float((mz - th) / th * 1e6), "rank": rank, "source": "built-in library",
                                 "level": "3", "evidence": "m/z" + (" + adduct group" if rank == 0 else "")})

    # ---- user library (m/z, optionally RT)
    lib_path = str(params.get("library") or "").strip()
    rt_win, rtw_src = _f(params, "library_rt_tol", 0.3)
    n_user = 0
    if lib_path:
        job.param("library", "Your compound library (CSV)", lib_path, "Columns: name, formula (or mass / mz), optional rt in minutes. "
                  "Matches with a retention time within the window below are reported as level 2 (see methods).", "user", "text")
        job.param("library_rt_tol", "Library RT window (min)", rt_win,
                  "How close the observed RT must be to your library RT. Only meaningful if the library RTs were measured "
                  "on the same column and gradient.", rtw_src)
        try:
            ul = _user_library(lib_path)
            n_user = len(ul)
            for i, (mz, rt, z) in enumerate(zip(feat.mz.values, feat.rt.values, feat.charge.values)):
                for _, u in ul.iterrows():
                    hit = None
                    if np.isfinite(u["mz"]):
                        if abs(mz - u["mz"]) <= u["mz"] * ann_ppm * 1e-6:
                            hit = (u["adduct"] or "as listed", float((mz - u["mz"]) / u["mz"] * 1e6))
                    else:
                        alist = [u["adduct"]] if u["adduct"] in chem.ADDUCTS else ([add[i]] if grp[i] >= 0 else adducts)
                        for a in alist:
                            if abs(chem.ADDUCTS[a][0]) != (z or 1) and not (z == 0 and abs(chem.ADDUCTS[a][0]) == 1):
                                continue
                            th = chem.adduct_mz(u["mass"], a)
                            if abs(mz - th) <= th * ann_ppm * 1e-6:
                                hit = (a, float((mz - th) / th * 1e6))
                                break
                    if hit is None:
                        continue
                    rt_ok = np.isfinite(u["rt"]) and abs(rt - u["rt"]) <= rt_win * 60
                    if np.isfinite(u["rt"]) and not rt_ok:
                        continue   # the library says it elutes elsewhere: not this compound
                    cands[i].append({"name": u["name"], "formula": u["formula"] if u["formula"] != "nan" else "", "class": "user library",
                                     "kegg": "", "adduct": hit[0], "ppm": hit[1], "rank": -2 if rt_ok else 0,
                                     "source": "your library", "level": "2" if rt_ok else "3",
                                     "evidence": "m/z + RT" if rt_ok else "m/z"})
        except UserFacingError:
            raise
        except Exception:  # noqa: BLE001
            job.flag("warn", f"Your library {Path(lib_path).name} could not be read: " + log_exc("user library"))
    else:
        job.param("library", "Your compound library (CSV)", "", "None given. Add a CSV (name, formula or mass, optional rt in "
                  "minutes) of standards run on your method to get retention-time-confirmed annotations.", "default", "text")
        job.param("library_rt_tol", "Library RT window (min)", rt_win, "Used only with your own library.", rtw_src)

    # ---- MS2 spectral library (cosine → level 2)
    ms2 = ctx.get("ms2", {})
    msp = str(params.get("msp_library") or "").strip()
    min_cos, mc_src = _f(params, "ms2_min_cosine", 0.7)
    min_match, mm_src = _f(params, "ms2_min_matched", 3)
    frag_tol, ft_src = _f(params, "ms2_tol_da", 0.02)
    n_ms2_hits = 0
    job.param("msp_library", "MS/MS spectral library (.msp / .mgf)", msp,
              ("Fragment spectra of features are compared with this library." if msp else
               "None given. With a spectral library (e.g. MoNA, MassBank or GNPS export as .msp/.mgf) features that have "
               "MS/MS spectra can be annotated at MSI level 2."), "user" if msp else "default", "text")
    job.param("ms2_min_cosine", "Minimum MS/MS cosine", min_cos,
              "0.7 is the usual threshold for a spectral match (GNPS default); above 0.9 is a very close match.", mc_src)
    job.param("ms2_min_matched", "Minimum matched fragments", int(min_match),
              "A high cosine from one or two shared peaks is easily chance; at least 3 matched fragments is a common minimum.", mm_src, "int")
    job.param("ms2_tol_da", "Fragment m/z tolerance (Da)", frag_tol,
              "0.02 Da suits high-resolution MS/MS; use 0.3–0.5 for ion-trap spectra.", ft_src)
    if msp and ms2:
        try:
            speclib = _read_spectral_library(Path(msp))
            pre = np.array([s["precursor"] for s in speclib])
            for i, (k, si) in ms2.items():
                r = runs[k]
                qmz, qint = r.spectrum(si)
                mz = feat.mz.values[i]
                okp = np.flatnonzero(np.isfinite(pre) & (np.abs(pre - mz) <= mz * cfg["ppm"] * 1e-6))
                bestc = None
                for e in okp:
                    s = speclib[e]
                    if s["polarity"] and s["polarity"] != pol:
                        continue
                    c, nm = cosine(qmz, qint, s["mz"], s["int"], tol=frag_tol)
                    if c >= min_cos and nm >= min_match and (bestc is None or c > bestc[0]):
                        bestc = (c, nm, s)
                if bestc:
                    c, nm, s = bestc
                    n_ms2_hits += 1
                    cands[i].append({"name": s["name"], "formula": s["formula"], "class": "spectral library", "kegg": "",
                                     "adduct": s["adduct"] or "", "ppm": float((mz - s["precursor"]) / s["precursor"] * 1e6),
                                     "rank": -3, "source": Path(msp).name, "level": "2",
                                     "evidence": f"MS/MS cosine {c:.2f} ({nm} peaks)", "cosine": c, "matched": nm})
        except Exception:  # noqa: BLE001
            job.flag("warn", f"The spectral library {Path(msp).name} could not be used: " + log_exc("msp"))
    elif msp and not ms2:
        job.flag("info", "A spectral library was given, but the runs contain no MS/MS spectra that fall on a feature, so no spectral matching was possible.")

    # ---- best annotation per feature: MS2 > m/z+RT > adduct-group-consistent > primary ion > other adducts
    names, levels, ads, ppms, forms, classes, evid, ncand = [], [], [], [], [], [], [], []
    rows_all = []
    for i in range(len(feat)):
        c = cands[i]
        for x in c:
            rows_all.append({"feature": feat.id[i], "mz": feat.mz[i], "rt_min": feat.rt[i] / 60, **{k: v for k, v in x.items() if k != "rank"}})
        if not c:
            names.append(""); levels.append("4"); ads.append(feat.adduct[i]); ppms.append(np.nan); forms.append("")
            classes.append(""); evid.append(""); ncand.append(0)
            continue
        top = min(x["rank"] for x in c)
        tier = [x for x in c if x["rank"] == top]
        if top > 0:
            # prefer the primary ion; among equals, the smallest mass error
            tier.sort(key=lambda x: abs(x["ppm"]))
        best_formula = tier[0]["formula"]
        same = [x for x in tier if x["formula"] == best_formula and x["adduct"] == tier[0]["adduct"]] if best_formula else tier[:1]
        uniq = list(dict.fromkeys(x["name"] for x in same))
        names.append(" / ".join(uniq[:4]) + (f" (+{len(uniq) - 4})" if len(uniq) > 4 else ""))
        levels.append(tier[0]["level"])
        ads.append(tier[0]["adduct"])
        ppms.append(tier[0]["ppm"])
        forms.append(best_formula)
        classes.append(tier[0]["class"])
        evid.append(tier[0]["evidence"])
        ncand.append(len({x["name"] for x in c}))
    feat["annotation"] = names
    feat["msi_level"] = levels
    feat["annotation_adduct"] = ads
    feat["annotation_ppm"] = ppms
    feat["formula"] = forms
    feat["class"] = classes
    feat["evidence"] = evid
    feat["n_candidates"] = ncand
    feat["has_ms2"] = [i in ms2 for i in range(len(feat))]
    ctx["annotations"] = pd.DataFrame(rows_all)
    ctx["n_user_lib"] = n_user

    job.section("annotation", "Step 6", "Adducts and annotation",
                "One molecule gives several ions ([M+H]+, [M+Na]+, [M+NH4]+ …). Grouping them tells you which features "
                "are the same compound and what its neutral mass is; that mass is then compared with a library.")
    ng = int(feat.compound_group.astype(bool).sum())
    n_groups = len({g for g in feat.compound_group if g})
    ann = feat.annotation.astype(bool).values
    lvl2 = (feat.msi_level == "2").values
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    vc = pd.Series([a for a in feat.adduct if a]).value_counts()
    if len(vc):
        axs[0].barh(vc.index[::-1], vc.values[::-1], color="#6b8fb3")
        axs[0].set_xlabel("features assigned")
        axs[0].set_title("adducts in multi-ion groups", fontsize=10)
    else:
        axs[0].text(0.5, 0.5, "no co-eluting adduct pairs found", ha="center", va="center", transform=axs[0].transAxes)
        axs[0].axis("off")
    cls = pd.Series([c for c in feat["class"] if c]).value_counts().head(12)
    if len(cls):
        axs[1].barh(cls.index[::-1], cls.values[::-1], color="#e39c3a")
        axs[1].set_xlabel("annotated features")
        axs[1].set_title("compound classes of best candidates", fontsize=10)
    else:
        axs[1].text(0.5, 0.5, "no library matches", ha="center", va="center", transform=axs[1].transAxes)
        axs[1].axis("off")
    job.figure("annotation", "Adduct groups and library candidates", fig, wide=True,
               how="Left: which ion types were recognised among co-eluting features whose masses fit a common molecule. "
                   "Right: compound classes of the best library candidates. m/z-only candidates are <b>putative</b> (MSI level 3): "
                   "many compounds share a formula, and a random m/z matches something in a large library surprisingly often.",
               yours=f"{commas(ng)} features fall into <b>{commas(n_groups)}</b> multi-ion compound groups. "
                     f"{commas(int(ann.sum()))} of {commas(len(feat))} features ({100 * ann.mean():.0f}%) have at least one library candidate"
                     + (f"; <b>{commas(int(lvl2.sum()))}</b> at level 2 (MS/MS or RT-confirmed)" if lvl2.any() else "")
                     + f". {commas(len(ms2))} features have an MS/MS spectrum.")
    top_ann = feat[ann].copy()
    top_ann["_i"] = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=1)[ann] if X.size else 0
    top_ann = top_ann.sort_values(["msi_level", "_i"], ascending=[True, False]).head(200)
    job.table("annotation_table", "Annotated features (best candidate; most intense first)",
              ["feature", "m/z", "RT (min)", "candidate(s)", "formula", "ion", "error (ppm)", "MSI level", "evidence", "other candidates"],
              [[r.id, f"{r.mz:.4f}", f"{r.rt / 60:.2f}", _short(r.annotation, 48), r.formula, r.annotation_adduct,
                f"{r.annotation_ppm:+.1f}" if np.isfinite(r.annotation_ppm) else "", r.msi_level, r.evidence, max(r.n_candidates - 1, 0)]
               for r in top_ann.itertuples()],
              note="MSI levels (Sumner et al. 2007): 1 = confirmed with an authentic standard on your method; 2 = probable "
                   "structure (MS/MS library match, or m/z + RT against your own library); 3 = putative compound class/formula "
                   "(m/z only); 4 = unknown. Isomers (e.g. leucine / isoleucine) cannot be told apart by mass and are listed together.",
              csv="annotations.csv")
    job.method("Adducts & annotation",
               f"Co-eluting features (within {art:g} s) were grouped when their neutral masses agreed within {cfg['ppm']:g} ppm for a pair "
               f"of {'positive' if pol > 0 else 'negative'}-mode adducts ({', '.join(adducts)}) and their intensities correlated across "
               f"samples (r ≥ 0.5 when ≥ 4 shared values); greedy assignment by number of partners. Annotation: m/z within {ann_ppm:g} ppm "
               f"of the built-in library ({len(lib)} endogenous metabolites and lipids; masses from formulas via pyteomics), using the "
               "group's neutral mass when grouped, else every default adduct (ranked [M+H]+/[M−H]− first)"
               + (f"; your library ({n_user} entries, RT window ±{rt_win:g} min)" if lib_path else "")
               + (f"; MS/MS against {Path(msp).name} by greedy matched-peak cosine on square-root intensities (fragment tolerance "
                  f"{frag_tol:g} Da, cosine ≥ {min_cos:g}, ≥ {int(min_match)} matched peaks)" if msp else "")
               + ". Confidence follows the Metabolomics Standards Initiative levels (Sumner et al. 2007, Metabolomics 3:211). "
                 "m/z + RT against your library is reported as level 2; if the library RTs are authentic standards run on this "
                 "same method, MSI would call it level 1.")


# ====================================================================== step 7 — blank and detection-rate filters

def _stat_groups(sheet):
    """Sample groups usable for group-wise work; samples without a group form one pseudo-group."""
    s = sheet[sheet.type == "sample"]
    g = s.group.replace("", "(no group)")
    return g


def _filters(ctx):
    job, sheet, feat, params = ctx["job"], ctx["sheet"], ctx["feat"], ctx["params"]
    job.step(64, "Filtering features")
    I = ctx["I_raw"]
    typ = sheet.type.values
    S, Q, B = typ == "sample", typ == "qc", typ == "blank"
    _pg(job, "Filtering", "Which features are kept for statistics.")
    ratio, r_src = _f(params, "blank_ratio", 3.0)
    mind, md_src = _f(params, "min_detection", 0.5)
    keep = np.ones(len(feat), bool)
    bio = S if S.any() else Q
    removed_blank = np.zeros(len(feat), bool)
    if B.any() and bio.any():
        mb = np.nanmean(np.where(np.isfinite(I[:, B]), I[:, B], 0.0), axis=1)
        ms = np.nanmean(np.where(np.isfinite(I[:, bio]), I[:, bio], 0.0), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fold = np.where(mb > 0, ms / mb, np.inf)
        removed_blank = fold < ratio
        feat["blank_fold"] = fold
        job.param("blank_ratio", "Blank filter: sample/blank ratio at least", ratio,
                  f"A feature is kept only if its mean in the samples is at least {ratio:g}× its mean in the {int(B.sum())} blank(s) "
                  "(missing counted as zero). Ions that are as strong in a blank come from solvent, tubes or carry-over, "
                  "not from the biology. 3× is a common choice; 5–10× is stricter.", r_src)
    else:
        feat["blank_fold"] = np.nan
        job.param("blank_ratio", "Blank filter: sample/blank ratio at least", ratio,
                  "No blank injections in the sample sheet, so contaminants could not be removed. Include a process/solvent blank next time.", r_src)
        job.flag("info", "No blank was injected (or none is marked type = blank), so solvent and plastic contaminants could not be filtered out.")
    keep &= ~removed_blank
    # detection-rate filter: present (detected or gap-filled) in ≥ min_detection of at least one group
    groups = _stat_groups(sheet)
    present = np.isfinite(I)
    rate = np.zeros(len(feat))
    if S.any():
        for g in groups.unique():
            cols = groups.index[groups == g].values
            rate = np.maximum(rate, present[:, cols].mean(1))
    elif Q.any():
        rate = present[:, Q].mean(1)
    else:
        rate = present.mean(1)
    removed_det = (rate < mind) & keep
    job.param("min_detection", "Keep features present in at least (fraction of a group)", mind,
              f"A feature must have a value (detected or recovered by gap filling) in at least {100 * mind:.0f}% of the samples "
              "of at least one group. This keeps compounds that are switched on in only one group while removing ions seen "
              "sporadically, whose statistics would rest on imputed numbers.", md_src)
    keep &= ~removed_det
    feat["detection_rate"] = rate
    ctx["removed_blank"], ctx["removed_det"] = removed_blank, removed_det
    ctx["keep"] = keep
    ctx["n_before_filter"] = len(feat)
    job.section("filtering", "Step 7", "Blank and detection filters",
                "Removing features that are contamination or too sporadic to measure. The QC-precision filter follows drift correction (step 8).")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    if B.any() and bio.any():
        fl = np.log2(np.clip(np.where(np.isfinite(feat.blank_fold), feat.blank_fold, 2 ** 12), 2 ** -4, 2 ** 12))
        axs[0].hist(fl, bins=50, color="#6b8fb3")
        axs[0].axvline(np.log2(ratio), color=UP, lw=1.2, ls="--")
        axs[0].set_xlabel("log2 (sample mean / blank mean)   (right edge = absent from blank)")
        axs[0].set_ylabel("features")
    else:
        axs[0].text(0.5, 0.5, "no blanks", ha="center", va="center", transform=axs[0].transAxes)
        axs[0].axis("off")
    axs[1].hist(rate, bins=np.linspace(0, 1, 21), color="#6b8fb3")
    axs[1].axvline(mind, color=UP, lw=1.2, ls="--")
    axs[1].set_xlabel("best detection rate in any group")
    axs[1].set_ylabel("features")
    job.figure("filters", "Blank ratio and detection rate", fig, wide=True,
               how="Left: how much stronger each feature is in the samples than in the blank; everything left of the red line "
                   "is removed as background. Right: the highest fraction of samples in any group where the feature has a value; "
                   "left of the red line is removed as too sporadic.",
               yours=f"The blank filter removed <b>{commas(int(removed_blank.sum()))}</b> features and the detection filter "
                     f"<b>{commas(int(removed_det.sum()))}</b> more; {commas(int(keep.sum()))} of {commas(len(feat))} remain.")


# ====================================================================== step 8 — QC-based drift correction

def _drift(ctx):
    import warnings
    from statsmodels.nonparametric.smoothers_lowess import lowess
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _drift_impl(ctx, lowess)


def _drift_impl(ctx, lowess):
    job, sheet, feat, params = ctx["job"], ctx["sheet"], ctx["feat"], ctx["params"]
    job.step(68, "Correcting signal drift with the QCs")
    I = ctx["I_raw"].copy()
    keep = ctx["keep"]
    typ = sheet.type.values
    Q = typ == "qc"
    order = sheet.order.values.astype(float)
    batches = sheet.batch.values
    _pg(job, "Drift correction (QC-RLSC)", "Uses the repeated pooled-QC injections to remove intensity drift over the run.")
    do, d_src = _b(params, "drift_correction", True)
    span_user, sp_src = _f(params, "loess_span", 0.75)
    ctx["I_corr"] = I
    ctx["drift_done"] = False
    qc_per_batch = {b: int((Q & (batches == b)).sum()) for b in np.unique(batches)}
    if not do:
        job.param("drift_correction", "QC-based drift correction", do, "Switched off by you.", d_src, "bool")
        return
    if max(qc_per_batch.values()) < 3:
        job.param("drift_correction", "QC-based drift correction", do,
                  f"Needs at least 3 pooled-QC injections per batch; this study has {max(qc_per_batch.values())}. Skipped.", d_src, "bool")
        job.flag("warn", f"Drift correction was skipped: it needs at least 3 pooled-QC injections spread through the run "
                         f"(this study has {int(Q.sum())}). Any intensity drift over the run remains in the data.")
        return
    nq_min = min(v for v in qc_per_batch.values() if v >= 3)
    use_linear = nq_min < 6
    job.param("drift_correction", "QC-based drift correction", do,
              "Fits each feature's QC intensity against injection order and divides the trend out of every injection (QC-RLSC, Dunn et al. 2011).", d_src, "bool")
    # choose the LOESS span by leave-one-QC-out cross-validation when there are enough QCs
    span, span_src = span_user, sp_src
    cv_note = ""
    logI = np.log2(np.where(I > 0, I, np.nan))
    if not use_linear and sp_src != "user":
        spans = [0.5, 0.75, 1.0]
        err = {s: [] for s in spans}
        rows = np.flatnonzero(keep & (np.isfinite(logI[:, Q]).sum(1) >= 6))
        rows = rows[np.argsort(-np.nanmean(logI[rows][:, Q], 1))][:200]
        for b in np.unique(batches):
            qb = np.flatnonzero(Q & (batches == b))
            if len(qb) < 6:
                continue
            for i in rows:
                y = logI[i, qb]
                x = order[qb]
                ok = np.isfinite(y)
                if ok.sum() < 6:
                    continue
                for s in spans:
                    e = []
                    for l in np.flatnonzero(ok):
                        m = ok.copy(); m[l] = False
                        f = lowess(y[m], x[m], frac=s, it=0, xvals=np.array([x[l]]))
                        e.append(abs(f[0] - y[l]))
                    err[s].append(np.mean(e))
        med = {s: np.median(v) for s, v in err.items() if v}
        if med:
            span = float(min(med, key=med.get))
            span_src = "auto"
            cv_note = " Chosen by leave-one-QC-out cross-validation among 0.5 / 0.75 / 1.0 (median error " + ", ".join(f"{s:g}: {v:.3f}" for s, v in med.items()) + " log2 units)."
    job.param("loess_span", "LOESS span", span,
              ("With fewer than 6 QCs per batch a LOESS curve would bend through every QC point and remove real signal, so a "
               "straight line (in log intensity) is fitted instead; the span is not used." if use_linear else
               "Fraction of QC injections used for each local fit: smaller follows faster changes but also QC noise." + cv_note), span_src)
    fits = np.full_like(I, np.nan)
    corrected = np.zeros(len(feat), bool)
    too_steep = np.zeros(len(feat), bool)
    max_trend = 2.0   # log2 units: the fitted trend may change intensity at most 4-fold across a batch
    gmed = np.nanmedian(logI[:, Q], axis=1)
    out = logI.copy()
    for b in np.unique(batches):
        inb = batches == b
        qb = np.flatnonzero(Q & inb)
        if len(qb) < 3:
            job.flag("warn", f"Batch {b} has only {len(qb)} QC injection(s); its drift could not be corrected.")
            continue
        xq = order[qb]
        xs = np.clip(order[inb], xq.min(), xq.max())   # no extrapolation beyond the first/last QC
        for i in range(len(feat)):
            y = logI[i, qb]
            ok = np.isfinite(y)
            if ok.sum() < 3:
                continue
            if use_linear or ok.sum() < 6:
                c = np.polyfit(xq[ok], y[ok], 1)
                f = np.polyval(c, xs)
            else:
                f = lowess(y[ok], xq[ok], frac=span, it=1, xvals=xs)
            if not np.all(np.isfinite(f)):
                continue
            if np.ptp(f) > max_trend:
                too_steep[i] = True      # a >4-fold 'trend' is noise in a few QC values, not instrument drift
                continue
            fits[i, inb] = f
            # divide out the trend and bring every batch to the overall QC median
            out[i, inb] = logI[i, inb] - f + gmed[i]
            corrected[i] = True
    Icorr = np.where(np.isfinite(out), 2 ** out, np.nan)
    ctx["I_corr"] = Icorr
    ctx["drift_done"] = True
    ctx["drift_fits"] = fits
    feat["drift_corrected"] = corrected
    cv0 = _cv(I[:, Q])
    cv1 = _cv(Icorr[:, Q])
    ctx["cv_before"], ctx["cv_after"] = cv0, cv1
    k = keep & corrected & np.isfinite(cv0) & np.isfinite(cv1)
    job.section("drift", "Step 8", "Drift correction with pooled QCs",
                "The mass spectrometer's sensitivity changes over a long sequence. Pooled QCs injected throughout show that "
                "change directly, feature by feature, so it can be removed from every sample.")
    # examples: well-measured features (QC in every injection, precise after correction) with the steepest trend
    full = np.isfinite(logI[:, Q]).all(1)
    qmean = np.nanmean(logI[:, Q], 1)
    strong = qmean >= np.nanmedian(qmean[k]) if k.any() else np.ones(len(feat), bool)
    cand = np.flatnonzero(k & full & strong & (cv1 < 20))
    if not len(cand):
        cand = np.flatnonzero(k)
    if len(cand):
        steep = np.nanmax(fits[cand], 1) - np.nanmin(fits[cand], 1)
        ex = list(cand[np.argsort(-steep)][:3])
        mid = cand[np.argsort(np.abs(steep - np.median(steep)))][:1]
        ex = list(dict.fromkeys(ex + list(mid)))[:4]
        fig, axs = plt.subplots(2, len(ex), figsize=(3.2 * len(ex), 5.8), squeeze=False)
        for c, i in enumerate(ex):
            for rrow, (M, title) in enumerate(((I, "before"), (Icorr, "after"))):
                ax = axs[rrow, c]
                ax.scatter(order, M[i], c=ctx["col"], s=16, lw=0, zorder=3)
                if rrow == 0:
                    oo = np.argsort(order)
                    ax.plot(order[oo], 2 ** fits[i, oo], color=UP, lw=1)
                ax.set_yscale("log")
                ax.set_title(f"{feat.id[i]}\n{title}" if rrow == 0 else title, fontsize=8)
                if rrow == 1:
                    ax.set_xlabel("injection order")
                if c == 0:
                    ax.set_ylabel("intensity")
                ax.text(0.02, 0.03, f"QC CV {(cv0 if rrow == 0 else cv1)[i]:.0f}%", transform=ax.transAxes, fontsize=8)
        fig.tight_layout()
        job.figure("drift_examples", "Examples: intensity against injection order, before and after", fig, wide=True,
                   how="Each dot is one injection (QCs black). The red curve is the trend fitted through the QCs; dividing it out "
                       "flattens the QCs (bottom row) and removes the same trend from the samples. The QC CV printed in each panel "
                       "should drop.",
                   yours="The three precisely measured features with the steepest drift, plus one with a typical drift. "
                         f"{commas(int(too_steep.sum()))} feature(s) whose QC 'trend' exceeded 4-fold were left uncorrected — "
                         "such a trend comes from a few unreliable QC values, not the instrument, and the QC CV filter judges them.")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, min(100, np.nanpercentile(np.concatenate([cv0[keep], cv1[keep]]), 99) + 5), 40)
    ax.hist(np.clip(cv0[keep & np.isfinite(cv0)], 0, bins[-1]), bins=bins, color="#9aa4ab", alpha=0.8, label="before")
    ax.hist(np.clip(cv1[keep & np.isfinite(cv1)], 0, bins[-1]), bins=bins, color="#2a64b0", alpha=0.6, label="after")
    ax.set_xlabel("QC coefficient of variation (%)")
    ax.set_ylabel("features")
    ax.legend()
    mb, ma = np.nanmedian(cv0[k]) if k.any() else np.nan, np.nanmedian(cv1[k]) if k.any() else np.nan
    ctx["cv_med"] = (mb, ma)
    job.figure("qc_cv", "QC precision before and after drift correction", fig,
               how="The coefficient of variation (CV = standard deviation / mean) of each feature across the pooled QC "
                   "injections is its analytical noise. Lower is better; < 20–30% is the usual standard for untargeted work.",
               yours=f"Median QC CV <b>{mb:.1f}% → {ma:.1f}%</b> over {commas(int(k.sum()))} corrected features; "
                     f"{100 * np.mean(cv1[k] < cv0[k]) if k.any() else 0:.0f}% of features improved."
                     + (" The QCs contain almost no drift, so little changed." if np.isfinite(mb) and mb - ma < 1 else ""))
    job.method("Drift correction", "QC-RLSC (Dunn et al. 2011, Nat Protoc 6:1060): per feature and batch, log2 QC intensity was "
               + ("regressed linearly on injection order (fewer than 6 QCs per batch)" if use_linear else f"smoothed against injection order by LOWESS (statsmodels, span {span:g})")
               + "; features whose fitted trend exceeded 4-fold were left uncorrected (such trends come from unreliable QC values); "
                 "each injection's value was divided by the fitted trend at its position (clamped to the first/last QC, "
                 "no extrapolation) and rescaled to the feature's median QC intensity, which also removes between-batch offsets.")


def _qc_cv_filter(ctx):
    job, sheet, feat, params = ctx["job"], ctx["sheet"], ctx["feat"], ctx["params"]
    Q = sheet.type.values == "qc"
    maxcv, mc_src = _f(params, "qc_cv_max", 30.0)
    keep = ctx["keep"]
    removed = np.zeros(len(feat), bool)
    cv = _cv(ctx["I_corr"][:, Q]) if Q.sum() >= 2 else np.full(len(feat), np.nan)
    feat["qc_cv"] = cv
    if Q.sum() >= 3:
        qpres = np.isfinite(ctx["I_corr"][:, Q]).mean(1)
        removed = keep & ((qpres < 0.5) | ~(cv <= maxcv))
        _pg(job, "Filtering")
        job.param("qc_cv_max", "QC filter: maximum CV (%)", maxcv,
                  f"Features whose (drift-corrected) intensity varies more than {maxcv:g}% across the {int(Q.sum())} pooled QCs — "
                  "or that are missing from over half of them — are measured too imprecisely to compare groups. 30% is the "
                  "common standard for untargeted metabolomics (20% for stricter, lipid or targeted-like work).", mc_src)
    else:
        _pg(job, "Filtering")
        job.param("qc_cv_max", "QC filter: maximum CV (%)", maxcv,
                  f"Needs at least 3 pooled QCs; this study has {int(Q.sum())}, so feature precision could not be checked.", mc_src)
        job.flag("warn", "Fewer than 3 pooled QC injections: the precision of each feature could not be measured and imprecise features could not be removed.")
    ctx["removed_cv"] = removed
    ctx["keep"] = keep & ~removed
    # summary of all filters, shown in the current section (drift correction, or filtering when drift was skipped)
    n0 = ctx["n_before_filter"]
    nb, nd = int(ctx["removed_blank"].sum()), int(ctx["removed_det"].sum())
    job.table("filter_counts", "Features removed by each filter", ["filter", "removed", "remaining"],
              [["linked features", "", commas(n0)],
               ["blank filter", commas(nb), commas(n0 - nb)],
               ["detection rate", commas(nd), commas(n0 - nb - nd)],
               [f"QC CV > {maxcv:g}% (after drift correction)" if Q.sum() >= 3 else "QC CV (skipped: < 3 QCs)", commas(int(removed.sum())),
                commas(int(ctx["keep"].sum()))]])
    feat["kept"] = ctx["keep"]


# ====================================================================== step 9 — normalisation, log, imputation

def _normalise(ctx):
    job, sheet, feat, params = ctx["job"], ctx["sheet"], ctx["feat"], ctx["params"]
    job.step(74, "Normalising")
    keep = ctx["keep"]
    if keep.sum() < 2:
        raise UserFacingError(f"Only {int(keep.sum())} feature(s) survived filtering — too few for statistics. Relax the "
                              "blank ratio, detection rate or QC CV thresholds.")
    typ = sheet.type.values
    use = np.isin(typ, ["sample", "qc"])
    X = ctx["I_corr"][keep][:, use]                     # features × injections (samples + QCs)
    cols = sheet[use].reset_index(drop=True)
    _pg(job, "Normalisation & scaling")
    method, m_src = _s(params, "normalisation", "PQN", ["PQN", "median", "TIC", "none"])
    Q = cols.type.values == "qc"
    if method == "PQN":
        ref = np.nanmedian(X[:, Q], axis=1) if Q.any() else np.nanmedian(X, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            quot = X / ref[:, None]
        fac = np.nanmedian(np.where(np.isfinite(quot) & (quot > 0), quot, np.nan), axis=0)
        why = ("Probabilistic quotient normalisation (Dieterle et al. 2006): each injection is divided by its median ratio to a "
               f"reference ({'the median QC' if Q.any() else 'the median sample'}). Robust to a few very changed compounds, "
               "and corrects for differing sample amount/dilution (e.g. urine, tissue weight).")
    elif method == "median":
        fac = np.nanmedian(X, axis=0) / np.nanmedian(X)
        why = "Each injection scaled so its median feature intensity equals the overall median."
    elif method == "TIC":
        full = np.all(np.isfinite(X), axis=1)
        s = np.nansum(X[full], axis=0) if full.sum() >= 10 else np.nansum(X, axis=0)
        fac = s / np.median(s)
        why = "Each injection scaled to the same total signal (sum of features present in every injection). Sensitive to a few very abundant compounds changing."
    else:
        fac = np.ones(X.shape[1])
        why = "No sample-amount normalisation — right when samples were already matched exactly (e.g. by cell count, volume or weight)."
    fac = np.where(np.isfinite(fac) & (fac > 0), fac, 1.0)
    Xn = X / fac[None, :]
    job.param("normalisation", "Normalisation", method, why, m_src, "select", options=["PQN", "median", "TIC", "none"])
    L = np.log2(np.where(Xn > 0, Xn, np.nan))
    imp, i_src = _s(params, "imputation", "half-min", ["half-min", "none", "knn"])
    nmiss = int(np.isnan(L).sum())
    Li = L.copy()
    if imp == "half-min" and nmiss:
        fmin = np.nanmin(L, axis=1)
        rows, cc = np.where(np.isnan(Li))
        Li[rows, cc] = fmin[rows] - 1.0      # half the minimum = minimum − 1 on the log2 scale
        why_i = ("Values still missing after gap filling had no peak above noise, i.e. are below the detection limit; they "
                 "are replaced by half the feature's lowest measured value.")
    elif imp == "knn" and nmiss:
        from sklearn.impute import KNNImputer
        kn = KNNImputer(n_neighbors=int(min(5, max(1, X.shape[1] - 1))))
        Li = kn.fit_transform(L.T).T
        why_i = "k-nearest-neighbour imputation (5 most similar injections). Suits values missing at random, not below the detection limit."
    else:
        why_i = "No imputation: missing values stay missing; PCA uses only complete features and tests use the available values."
    job.param("imputation", "Missing values", imp, why_i + f" {commas(nmiss)} of {commas(L.size)} values ({100 * nmiss / max(L.size, 1):.1f}%) were missing.",
              i_src, "select", options=["half-min", "none", "knn"])
    ctx.update(norm_cols=cols, norm_factors=fac, L=Li, L_raw=L, imputed=np.isnan(L), feat_kept=feat[keep].reset_index(drop=True),
               X_norm=Xn)
    job.section("normalisation", "Step 9", "Normalisation",
                "Correcting for how much material went into each injection, then putting intensities on a log scale.")
    before = np.log2(np.where(X > 0, X, np.nan))
    col = [c for c, u in zip(ctx["col"], use) if u]
    o = np.argsort(cols.order.values, kind="stable")
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    for ax, M, t in ((axs[0], before, "before"), (axs[1], Li, "after normalisation + imputation")):
        data = [M[np.isfinite(M[:, j]), j] for j in o]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for p, j in zip(bp["boxes"], o):
            p.set_facecolor(col[j]); p.set_alpha(0.7)
        ax.set_ylabel("log2 intensity")
        ax.set_title(t, fontsize=10)
    axs[1].set_xticks(np.arange(1, len(o) + 1))
    axs[1].set_xticklabels([cols["sample"][j] for j in o], rotation=90, fontsize=7)
    meds = np.nanmedian(Li, axis=0)
    job.figure("normalisation", "Intensity distributions per injection", fig, wide=True,
               how="Each box is one injection's feature intensities (log2) in run order. After normalisation the boxes should "
                   "line up; one that stays shifted is a sample with a genuinely different composition — or a problem.",
               yours=f"{method} factors ranged {fac.min():.2f}–{fac.max():.2f}"
                     + (f" (a {fac.max() / fac.min():.1f}-fold spread in effective sample amount)" if fac.max() / fac.min() > 1.3 else " — samples were already well matched")
                     + f". After normalisation the medians span {np.ptp(meds):.2f} log2 units.")
    job.method("Normalisation", {"PQN": "Probabilistic quotient normalisation (Dieterle et al. 2006, Anal Chem 78:4281) against the median QC "
                                 "(median sample if no QC), without a prior integral step",
                                 "median": "Median normalisation", "TIC": "Total-signal normalisation over features present in every injection",
                                 "none": "No normalisation"}[method]
               + f"; log2 transform; missing values: {imp}" + (" (feature minimum / 2)" if imp == "half-min" else "") + ".")


# ====================================================================== step 11 (numbers) — univariate statistics

def _stats_compute(ctx):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # scipy warns on near-constant features; their p is NaN/1 anyway
        return _stats_compute_impl(ctx)


def _stats_compute_impl(ctx):
    from scipy import stats as st
    job, params = ctx["job"], ctx["params"]
    cols, L, fk = ctx["norm_cols"], ctx["L"], ctx["feat_kept"]
    S = cols.type.values == "sample"
    grp = cols.group.values
    counts = pd.Series(grp[S & (grp != "")]).value_counts()
    order = [g for g in ctx["groups_all"] if g in counts.index and counts[g] >= 2]
    ctx["stat_groups"] = order
    ctx["stats"] = None
    _pg(job, "Statistics")
    alpha, a_src = _f(params, "alpha", 0.05)
    lfc, l_src = _f(params, "min_log2fc", 0.585)
    ctx["alpha"], ctx["min_lfc"] = alpha, lfc
    if len(order) < 2:
        reason = ("no sample has a group in the sample sheet" if not counts.size else
                  f"only {len(order)} group has at least 2 samples ({', '.join(f'{g}: {n}' for g, n in counts.items())})")
        job.param("group_a", "Reference group", "", f"Statistics need ≥ 2 groups with ≥ 2 samples each; {reason}.", "default", "text")
        job.param("group_b", "Comparison group", "", "—", "default", "text")
        job.flag("warn", f"<b>No group comparison was possible</b>: {reason}. Fill in the group column of the sample sheet "
                         "(QCs and blanks need no group) and re-analyse to get fold changes, p-values and a volcano plot. "
                         "Features, QC metrics and PCA below are still valid.")
        return
    ga_u, gb_u = str(params.get("group_a") or "").strip(), str(params.get("group_b") or "").strip()
    ctrl_like = [g for g in order if re.search(r"(^|[^a-z])(control|ctrl|ctl|wt|wild.?type|vehicle|veh|untreated|mock|sham|"
                                                r"placebo|healthy|normal|baseline|naive|dmso|pbs)([^a-z]|$)", g.lower())]
    if ga_u in order:
        ga, why_a = ga_u, "Set by you."
    elif len(ctrl_like) == 1:
        ga = ctrl_like[0]
        why_a = (f"'{ga}' looks like a control group by its name, so it is the reference (groups in sheet order: {', '.join(order)}). "
                 "Change it here if that guess is wrong.")
    else:
        ga = order[0]
        why_a = (f"The first group in your sample sheet ({', '.join(order)} in sheet order); no group name looks like a control. "
                 "Change it here if it is not your reference.")
    gb = gb_u if gb_u in order and gb_u != ga else next(g for g in order if g != ga)
    why_a += f" Fold changes are {gb} relative to {ga}."
    job.param("group_a", "Reference group (denominator)", ga, why_a, "user" if ga == ga_u else "auto", "select", options=order)
    job.param("group_b", "Comparison group", gb, "Set by you." if gb == gb_u else f"The next group in the sheet after {ga}.",
              "user" if gb == gb_u else "auto", "select", options=order)
    job.param("alpha", "FDR threshold", alpha, "Features with a Benjamini–Hochberg adjusted p-value below this are called significant: "
              f"about {100 * alpha:g}% of the called features are expected to be false positives.", a_src)
    job.param("min_log2fc", "Minimum |log2 fold change|", lfc,
              f"Also require at least a {2 ** lfc:.2g}-fold change. With analytical CVs of 10–30%, smaller changes are hard to "
              "confirm by any follow-up measurement even when statistically significant.", l_src)
    out = pd.DataFrame({"id": fk.id, "mz": fk.mz, "rt_min": fk.rt / 60, "annotation": fk.annotation, "msi_level": fk.msi_level})
    imputed = ctx["imputed"]

    def pair(a, b):
        A = L[:, (grp == a) & S]
        B = L[:, (grp == b) & S]
        with np.errstate(all="ignore"):
            t, p = st.ttest_ind(B, A, axis=1, equal_var=False, nan_policy="omit")
        p = np.asarray(p, float)
        l2 = np.nanmean(B, 1) - np.nanmean(A, 1)
        fa = imputed[:, (grp == a) & S].mean(1)
        fb = imputed[:, (grp == b) & S].mean(1)
        return l2, p, _bh(p), np.maximum(fa, fb)
    l2, p, q, impf = pair(ga, gb)
    out["log2fc"], out["p"], out["padj"], out["imputed_frac_max"] = l2, p, q, impf
    for g in order:
        out[f"mean_log2_{g}"] = np.nanmean(L[:, (grp == g) & S], 1)
    if len(order) > 2:
        arrs = [L[:, (grp == g) & S] for g in order]
        with np.errstate(all="ignore"):
            F, pa = st.f_oneway(*arrs, axis=1)
        out["p_anova"], out["padj_anova"] = pa, _bh(pa)
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                if {a, b} == {ga, gb}:
                    continue
                l2x, px, qx, _ = pair(a, b)
                out[f"log2fc_{b}_vs_{a}"], out[f"p_{b}_vs_{a}"], out[f"padj_{b}_vs_{a}"] = l2x, px, qx
    out["significant"] = (out.padj < alpha) & (out.log2fc.abs() >= lfc)
    out["direction"] = np.where(~out.significant, "", np.where(out.log2fc > 0, "up", "down"))
    ctx["stats"] = out
    ctx["contrast"] = (ga, gb)


# ====================================================================== step 10 — multivariate

def _scale(M, how):
    mu = np.nanmean(M, 0)
    sd = np.nanstd(M, 0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    return (M - mu) / (np.sqrt(sd) if how == "pareto" else sd)


def _pca(Z, k=5):
    Z = Z - Z.mean(0)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    k = min(k, len(s))
    return U[:, :k] * s[:k], Vt[:k].T, (s ** 2 / (s ** 2).sum())[:k]


def _multivariate(ctx):
    job, params, sheet = ctx["job"], ctx["params"], ctx["sheet"]
    cols, L, fk = ctx["norm_cols"], ctx["L"], ctx["feat_kept"]
    job.step(80, "PCA and PLS-DA")
    scaling, s_src = _s(params, "scaling", "autoscale", ["autoscale", "pareto"])
    _pg(job, "Normalisation & scaling")
    job.param("scaling", "Scaling for PCA / PLS-DA / heatmap", scaling,
              "Autoscaling (unit variance) gives every feature the same weight, so abundant compounds do not dominate; Pareto "
              "scaling (÷√SD) is a compromise that keeps some weight on large, reliable signals and down-weights noisy small ones.",
              s_src, "select", options=["autoscale", "pareto"])
    col = [c for c, u in zip(ctx["col"], np.isin(sheet.type.values, ["sample", "qc"])) if u]
    complete = np.all(np.isfinite(L), axis=1)
    M = L[complete].T                                  # injections × features
    Q = cols.type.values == "qc"
    S = cols.type.values == "sample"
    job.section("multivariate", "Step 10", "Multivariate overview",
                "How the samples relate to each other using all features at once, without (PCA) and with (PLS-DA) knowledge of the groups.")
    if M.shape[1] < 3 or M.shape[0] < 3:
        job.flag("info", "Too few samples or complete features for PCA.")
        return
    Z = _scale(M, scaling)
    sc, ld, ev = _pca(Z, 5)
    ctx["pca"] = (sc, ev)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axs[0]
    ax.scatter(sc[:, 0], sc[:, 1], c=col, s=46, edgecolor="white", lw=0.6, zorder=3)
    for j in range(len(sc)):
        ax.annotate(cols["sample"][j], (sc[j, 0], sc[j, 1]), fontsize=6, xytext=(3, 3), textcoords="offset points", color="#555")
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.0f}%)" if len(ev) > 1 else "PC2")
    ax.axhline(0, color="#ddd", lw=0.6, zorder=0); ax.axvline(0, color="#ddd", lw=0.6, zorder=0)
    _legend(ax, cols, ctx["gc"], loc="best")
    ax.set_title("scores", fontsize=10)
    ax = axs[1]
    lab = fk[complete].reset_index(drop=True)
    ax.scatter(ld[:, 0], ld[:, 1] if ld.shape[1] > 1 else np.zeros(len(ld)), s=6, color="#9aa4ab", lw=0)
    mag = np.hypot(ld[:, 0], ld[:, 1] if ld.shape[1] > 1 else 0)
    for j in np.argsort(-mag)[:10]:
        name = lab.annotation[j].split(" / ")[0] if lab.annotation[j] else lab.id[j]
        ax.scatter(ld[j, 0], ld[j, 1], s=14, color=UP)
        ax.annotate(_short(name, 22), (ld[j, 0], ld[j, 1]), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("PC1 loading"); ax.set_ylabel("PC2 loading")
    ax.set_title("loadings (top 10 labelled)", fontsize=10)
    # QC tightness: spread of QCs relative to the samples in the first components
    kpc = min(len(ev), max(2, int(np.searchsorted(np.cumsum(ev), 0.5) + 1)))
    qc_ratio = None
    if Q.sum() >= 2 and S.sum() >= 2:
        def spread(X):
            return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1)).mean())
        qc_ratio = spread(sc[Q, :kpc]) / max(spread(sc[S, :kpc]), 1e-12)
        ctx["qc_ratio"] = qc_ratio
    gtext = ""
    groups = ctx["stat_groups"]
    if len(groups) >= 2:
        yv = sc[S, 0]
        gv = cols.group.values[S]
        ok = np.isin(gv, groups)
        tot = ((yv[ok] - yv[ok].mean()) ** 2).sum()
        within = sum(((yv[ok][gv[ok] == g] - yv[ok][gv[ok] == g].mean()) ** 2).sum() for g in groups)
        r2 = 1 - within / tot if tot > 0 else 0
        r2b = 0.0
        if sc.shape[1] > 1:
            yv2 = sc[S, 1]
            tot2 = ((yv2[ok] - yv2[ok].mean()) ** 2).sum()
            w2 = sum(((yv2[ok][gv[ok] == g] - yv2[ok][gv[ok] == g].mean()) ** 2).sum() for g in groups)
            r2b = 1 - w2 / tot2 if tot2 > 0 else 0
        gtext = f" Group explains {100 * r2:.0f}% of PC1 and {100 * r2b:.0f}% of PC2" + (" — the groups separate without being told apart." if max(r2, r2b) > 0.5 else " — group differences are not the main source of variation.")
        ctx["pc_group_r2"] = max(r2, r2b)
    job.figure("pca", "PCA", fig, wide=True,
               how="Scores (left): each point is an injection; nearby points have similar metabolite profiles. Pooled QCs (black) "
                   "are the same sample measured repeatedly, so they must form a tight cluster in the middle of the samples — if "
                   "they spread as much as the samples, analytical noise swamps biology. Loadings (right): features far from the "
                   "centre drive the separation along that axis.",
               yours=f"PC1 {100 * ev[0]:.0f}%, PC2 {100 * ev[1]:.0f}% of the variance ({commas(M.shape[1])} complete features, {scaling})."
                     + (f" The QCs spread <b>{qc_ratio:.2f}×</b> as widely as the samples" + (" — tight, as they should be." if qc_ratio < 0.35 else
                        " — noticeably spread; check drift and the QC CVs." if qc_ratio < 0.7 else " — as wide as the biology: analytical variation dominates.") if qc_ratio is not None else "")
                     + gtext)
    if qc_ratio is not None:
        job.flag("ok" if qc_ratio < 0.35 else "warn", f"Pooled QCs cluster {'tightly' if qc_ratio < 0.35 else 'loosely'} in the PCA "
                 f"(spread {qc_ratio:.2f}× that of the samples)" + (" — the measurement is reproducible relative to the biological differences." if qc_ratio < 0.35 else
                                                                   " — analytical variation is large relative to the biology; interpret group differences cautiously."))
    _plsda(ctx, M, cols, scaling)
    _heatmap(ctx, complete)


def _plsda(ctx, M, cols, scaling):
    job, params = ctx["job"], ctx["params"]
    groups = ctx["stat_groups"]
    do, d_src = _b(params, "plsda", True)
    S = cols.type.values == "sample"
    g = cols.group.values
    ok = S & np.isin(g, groups)
    counts = pd.Series(g[ok]).value_counts()
    eligible = [x for x in groups if counts.get(x, 0) >= 3]
    _pg(job, "Statistics")
    if not do or len(eligible) < 2:
        job.param("plsda", "PLS-DA", do, ("Switched off by you." if not do else
                  "Needs at least 2 groups with ≥ 3 samples each."), d_src, "bool")
        return
    job.param("plsda", "PLS-DA", do, "Supervised projection that looks for the direction separating the groups. It will find "
              "separation even in random data, so it is only reported with a cross-validated Q² and a permutation test.", d_src, "bool")
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import StratifiedKFold, LeaveOneOut
        sel = S & np.isin(g, eligible)
        X = M[sel]
        lab = g[sel]
        Y = np.array([[1.0 if l == e else 0.0 for e in eligible] for l in lab])
        if len(eligible) == 2:
            Y = Y[:, 1:]
        Xs = _scale(X, scaling) if scaling == "pareto" else X
        ncomp = int(min(2, X.shape[0] - 2, X.shape[1]))
        if ncomp < 1:
            return
        n = len(X)
        folds = LeaveOneOut() if n < 14 else StratifiedKFold(n_splits=7, shuffle=True, random_state=0)

        def q2(Xm, Ym, lab_):
            press = 0.0
            for tr, te in (folds.split(Xm) if n < 14 else folds.split(Xm, lab_)):
                m = PLSRegression(n_components=ncomp, scale=(scaling == "autoscale")).fit(Xm[tr], Ym[tr])
                press += ((m.predict(Xm[te]) - Ym[te]) ** 2).sum()
            tss = ((Ym - Ym.mean(0)) ** 2).sum()
            return 1 - press / tss
        q2_obs = q2(Xs, Y, lab)
        model = PLSRegression(n_components=ncomp, scale=(scaling == "autoscale")).fit(Xs, Y)
        r2y = 1 - ((model.predict(Xs) - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
        rng = np.random.default_rng(0)
        nperm = 100 if n * X.shape[1] < 2e6 else 30
        perm = []
        tries = 0
        while len(perm) < nperm and tries < 20 * nperm:
            tries += 1
            pi = rng.permutation(n)
            ct = pd.crosstab(lab, lab[pi]).values
            if ((ct > 0).sum(1) == 1).all():
                continue   # the same partition with the names swapped is not a shuffle (PLS fits it equally well)
            perm.append(q2(Xs, Y[pi], lab[pi]))
        nperm = len(perm)
        perm = np.array(perm)
        pval = (1 + np.sum(perm >= q2_obs)) / (1 + nperm)
        T = model.x_scores_
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for e in eligible:
            m = lab == e
            axs[0].scatter(T[m, 0], T[m, 1] if T.shape[1] > 1 else np.zeros(m.sum()), s=46, color=ctx["gc"][e], label=e,
                           edgecolor="white", lw=0.6)
        axs[0].set_xlabel("latent variable 1"); axs[0].set_ylabel("latent variable 2" if T.shape[1] > 1 else "")
        axs[0].legend(fontsize=8)
        axs[0].set_title(f"PLS-DA scores (R²Y {r2y:.2f}, Q² {q2_obs:.2f})", fontsize=10)
        axs[1].hist(perm, bins=20, color="#9aa4ab")
        axs[1].axvline(q2_obs, color=UP, lw=1.5)
        axs[1].set_xlabel("Q² with group labels shuffled")
        axs[1].set_ylabel("permutations")
        axs[1].set_title(f"permutation test: p = {pval:.3f}", fontsize=10)
        good = q2_obs > 0.5 and pval < 0.05
        ctx["plsda"] = (q2_obs, pval, r2y)
        job.figure("plsda", "PLS-DA with cross-validation", fig, wide=True,
                   how="Left: samples projected onto the directions that best separate the groups. Separation here is <b>not</b> evidence "
                       "by itself — with more features than samples PLS-DA separates random labels perfectly. What counts is Q² "
                       f"(prediction of left-out samples, {'leave-one-out' if n < 14 else '7-fold'} cross-validation; > 0.5 is good, ≤ 0 means "
                       "no predictive value) and the permutation test (right): the real Q² (red) should beat nearly all shuffled-label Q² values.",
                   yours=f"R²Y {r2y:.2f} (fit), <b>Q² {q2_obs:.2f}</b> (prediction), permutation p = {pval:.3f} ({nperm} permutations)"
                         + (" — the groups are predictably different." if good else
                            " — weak but non-random group structure." if pval < 0.05 else
                            " — no evidence the groups differ in a predictable way; the score-plot separation is overfitting."))
        job.method("PLS-DA", f"scikit-learn PLSRegression ({ncomp} components, {scaling}) on groups {', '.join(eligible)}; Q² by "
                   f"{'leave-one-out' if n < 14 else '7-fold stratified'} cross-validation, significance by {nperm} label permutations "
                   "(Westerhuis et al. 2008, Metabolomics 4:81).")
    except Exception:  # noqa: BLE001
        job.flag("info", "PLS-DA could not be computed: " + log_exc("plsda"))


def _heatmap(ctx, complete):
    from scipy.cluster.hierarchy import dendrogram, linkage
    job, cols, L, fk = ctx["job"], ctx["norm_cols"], ctx["L"], ctx["feat_kept"]
    stats = ctx.get("stats")
    idx = np.flatnonzero(complete)
    if len(idx) < 3:
        return
    if stats is not None and stats.p.notna().sum() >= 3:
        pv = stats.p.values[idx]
        top = idx[np.argsort(np.nan_to_num(pv, nan=2.0))][:50]
        basis = "most significant (Welch t-test)"
    else:
        v = np.nanvar(L[idx], axis=1)
        top = idx[np.argsort(-v)][:50]
        basis = "most variable"
    use = np.flatnonzero(cols.type.values != "blank")
    H = L[np.ix_(top, use)]
    Hz = (H - H.mean(1, keepdims=True)) / np.where(H.std(1, keepdims=True) > 0, H.std(1, keepdims=True), 1)
    rl = linkage(Hz, "average", metric="correlation") if len(top) > 2 else None
    cl = linkage(Hz.T, "average", metric="euclidean")
    fig = plt.figure(figsize=(10, max(5, 0.16 * len(top) + 2)))
    axd = fig.add_axes([0.2, 0.86, 0.62, 0.1])
    dc = dendrogram(cl, ax=axd, no_labels=True, color_threshold=0, above_threshold_color="#888")
    axd.axis("off")
    axr = fig.add_axes([0.02, 0.1, 0.1, 0.74])
    dr = dendrogram(rl, ax=axr, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="#888") if rl is not None else {"leaves": list(range(len(top)))}
    axr.axis("off")
    ro = dr["leaves"][::-1] if rl is not None else list(range(len(top)))
    co = dc["leaves"]
    axc = fig.add_axes([0.2, 0.845, 0.62, 0.012])
    col = [c for c, u in zip(ctx["col"], np.isin(ctx["sheet"].type.values, ["sample", "qc"])) if u]
    axc.imshow(np.array([[matplotlib_rgb(col[use[j]]) for j in co]]), aspect="auto")
    axc.axis("off")
    ax = fig.add_axes([0.2, 0.1, 0.62, 0.74])
    im = ax.imshow(np.clip(Hz[np.ix_(ro, co)], -2.5, 2.5), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(co)))
    ax.set_xticklabels([cols["sample"][use[j]] for j in co], rotation=90, fontsize=6.5)
    names = [(_short(fk.annotation[top[i]].split(" / ")[0], 26) + " · " if fk.annotation[top[i]] else "") + fk.id[top[i]] for i in ro]
    ax.set_yticks(range(len(ro)))
    ax.set_yticklabels(names, fontsize=6)
    ax.yaxis.tick_right()
    cax = fig.add_axes([0.93, 0.1, 0.015, 0.2])
    plt.colorbar(im, cax=cax, label="row z-score")
    job.figure("heatmap", f"Top {len(top)} features ({basis}), clustered", fig, wide=True,
               how="Rows are features scaled to their own mean (red above, blue below), columns are injections; both are "
                   "ordered by hierarchical clustering so similar ones sit together. The colour strip on top marks group (QCs black). "
                   "Blocks of colour that switch at a group boundary are consistent group differences.",
               yours=f"The {len(top)} {basis} features. " + ("Samples cluster by group." if _clusters_by_group(cl, cols, use, ctx) else
                                                             "Samples do not cluster cleanly by group."))


def matplotlib_rgb(c):
    from matplotlib.colors import to_rgb
    return to_rgb(c)


def _clusters_by_group(cl, cols, use, ctx):
    from scipy.cluster.hierarchy import fcluster
    groups = ctx["stat_groups"]
    if len(groups) < 2:
        return False
    lab = fcluster(cl, len(groups) + (1 if (cols.type.values[use] == "qc").any() else 0), "maxclust")
    g = cols.group.values[use]
    s = np.isin(g, groups)
    if s.sum() < 4:
        return False
    # purity: fraction of samples whose cluster's majority group is their own
    ok = 0
    for c in np.unique(lab[s]):
        m = lab[s] == c
        ok += pd.Series(g[s][m]).value_counts().iloc[0]
    return ok / s.sum() >= 0.9


# ====================================================================== step 11 (figures) — univariate

def _univariate(ctx):
    job, cols, L = ctx["job"], ctx["norm_cols"], ctx["L"]
    stats = ctx.get("stats")
    if stats is None:
        return
    job.step(88, "Univariate statistics")
    ga, gb = ctx["contrast"]
    alpha, lfc = ctx["alpha"], ctx["min_lfc"]
    order = ctx["stat_groups"]
    S = cols.type.values == "sample"
    grp = cols.group.values
    sig = stats[stats.significant]
    up, dn = sig[sig.log2fc > 0], sig[sig.log2fc < 0]
    ctx["n_sig"] = (len(up), len(dn))
    job.section("stats", "Step 11", f"Group comparison: {gb} vs {ga}",
                "Welch's t-test on log2 intensities for every feature, Benjamini–Hochberg false-discovery-rate correction"
                + (f"; one-way ANOVA across all {len(order)} groups in the statistics table" if len(order) > 2 else "") + ".")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = -np.log10(np.clip(stats.p.values, 1e-300, 1))
    colr = np.where(stats.significant & (stats.log2fc > 0), UP, np.where(stats.significant & (stats.log2fc < 0), DOWN, NS))
    ax.scatter(stats.log2fc, y, c=colr, s=10, lw=0)
    ax.axvline(lfc, color="#999", lw=0.7, ls="--"); ax.axvline(-lfc, color="#999", lw=0.7, ls="--")
    if len(sig):
        pthr = sig.p.max()
        ax.axhline(-np.log10(pthr), color="#999", lw=0.7, ls="--")
    lab = stats.nsmallest(8, "p")
    lab = lab[lab.significant] if lab.significant.any() else lab.head(6)
    for r in lab.itertuples():
        name = r.annotation.split(" / ")[0] if r.annotation else r.id
        ax.annotate(_short(name, 24), (r.log2fc, -np.log10(max(r.p, 1e-300))), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel(f"log2 fold change ({gb} / {ga})")
    ax.set_ylabel("−log10 p")
    ax.set_ylim(top=float(np.nanmax(y)) * 1.12 + 0.2)
    ax.text(0.99, 0.99, f"{len(up)} up", color=UP, ha="right", va="top", transform=ax.transAxes, fontsize=9)
    ax.text(0.01, 0.99, f"{len(dn)} down", color=DOWN, ha="left", va="top", transform=ax.transAxes, fontsize=9)
    imp_sig = int((sig.imputed_frac_max >= 0.5).sum()) if len(sig) else 0
    lead = sig.reindex(sig.log2fc.abs().sort_values(ascending=False).index).head(1)
    job.figure("volcano", "Volcano plot", fig,
               how=f"Right = higher in {gb}, left = lower; higher = more significant. Dashed lines: |log2FC| = {lfc:g} "
                   f"({2 ** lfc:.2g}-fold) and the p-value where the FDR reaches {alpha:g}. Labels use the best library candidate "
                   "(putative unless marked level 2) or the feature ID.",
               yours=(f"<b>{len(up)}</b> features up and <b>{len(dn)}</b> down in {gb} (FDR < {alpha:g}, ≥ {2 ** lfc:.2g}-fold) out of {commas(len(stats))} tested."
                      + (f" Largest change: {lead.annotation.iloc[0].split(' / ')[0] or lead.id.iloc[0]} ({2 ** abs(lead.log2fc.iloc[0]):.1f}× {'up' if lead.log2fc.iloc[0] > 0 else 'down'})." if len(lead) else "")
                      + (f" {imp_sig} of the significant features rely on imputed values for at least half of one group — look at their raw data before believing them." if imp_sig else "")))
    # p-value histogram (one small check of the test)
    p05 = float((stats.p < 0.05).mean())
    # table of significant features
    rows = []
    for r in sig.sort_values("p").head(200).itertuples():
        rows.append([r.id, f"{r.mz:.4f}", f"{r.rt_min:.2f}", _short(r.annotation, 40), r.msi_level, f"{r.log2fc:+.2f}",
                     f"{r.p:.2e}", f"{r.padj:.2e}", f"{100 * r.imputed_frac_max:.0f}%"])
    job.table("significant", f"Significant features ({gb} vs {ga})",
              ["feature", "m/z", "RT (min)", "best candidate", "MSI", "log2FC", "p", "FDR", "imputed (max group)"], rows,
              note="Full results for every feature (and every group pair) are in statistics.csv.", csv="statistics.csv")
    # box/strip of the top 6
    top = stats.nsmallest(6, "p")
    if len(top):
        fig, axs = plt.subplots(1, len(top), figsize=(2.3 * len(top), 3.4), squeeze=False)
        Q = cols.type.values == "qc"
        for c, (i, r) in enumerate(top.iterrows()):
            ax = axs[0, c]
            data = [L[i, S & (grp == g)] for g in order]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55)
            for pch, g in zip(bp["boxes"], order):
                pch.set_facecolor(ctx["gc"][g]); pch.set_alpha(0.35)
            rng = np.random.default_rng(c)
            for k, g in enumerate(order):
                v = L[i, S & (grp == g)]
                imp = ctx["imputed"][i, S & (grp == g)]
                ax.scatter(k + 1 + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=ctx["gc"][g],
                           edgecolor=np.where(imp, "black", "none"), lw=0.8, zorder=3)
            if Q.any():
                ax.scatter(np.full(Q.sum(), len(order) + 1) + rng.uniform(-0.12, 0.12, Q.sum()), L[i, Q], s=12, color=QC_COLOR, zorder=3)
            ax.set_xticks(range(1, len(order) + 1 + (1 if Q.any() else 0)))
            ax.set_xticklabels(order + (["QC"] if Q.any() else []), rotation=45, fontsize=7)
            nm = r.annotation.split(" / ")[0] if r.annotation else r.id
            ax.set_title(_short(nm, 20), fontsize=8)
            ax.text(0.5, -0.32, f"FDR {r.padj:.1e}", transform=ax.transAxes, ha="center", fontsize=7)
            if c == 0:
                ax.set_ylabel("log2 normalised intensity")
        job.figure("top_features", "The six most significant features", fig, wide=True,
                   how="Every dot is a sample (QCs at the right). Always look at the points behind a small p-value: one extreme "
                       "sample, or a group made of imputed values (dots with a black ring), makes a finding fragile.",
                   yours=", ".join(f"{(r.annotation.split(' / ')[0] if r.annotation else r.id)} ({2 ** abs(r.log2fc):.1f}× {'up' if r.log2fc > 0 else 'down'})" for r in top.itertuples()) + ".")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(stats.p.dropna(), bins=20, color="#6b8fb3")
    ax.set_xlabel("p-value"); ax.set_ylabel("features")
    job.figure("pvalues", "p-value histogram", fig,
               how="With no real differences, p-values are spread evenly (flat histogram). Real effects add a peak near 0. A "
                   "U-shape or a hump in the middle hints at a problem with the test (outliers, batch confounded with group).",
               yours=f"{100 * p05:.0f}% of features have p < 0.05 (5% expected by chance)" + (" — a clear differential signal." if p05 > 0.15 else " — a modest signal." if p05 > 0.08 else " — little evidence of group differences."))
    if len(order) > 2 and "padj_anova" in stats:
        job.flag("info", f"{len(order)} groups: {int((stats.padj_anova < alpha).sum())} features differ somewhere among them (ANOVA FDR < {alpha:g}). "
                         f"The volcano shows {gb} vs {ga}; every pair is in statistics.csv.")
    job.method("Statistics", f"Welch's two-sample t-test (unequal variances) on log2 normalised intensities of {gb} vs {ga}"
               + (", one-way ANOVA across all groups and Welch tests for every other pair" if len(order) > 2 else "")
               + f"; Benjamini–Hochberg FDR per comparison; significant = FDR < {alpha:g} and |log2FC| ≥ {lfc:g}. QC and blank "
                 "injections never enter a comparison.")


# ====================================================================== outputs: tables, tiles, flags, widget

def _outputs(ctx):
    job, sheet, feat, cfg = ctx["job"], ctx["sheet"], ctx["feat"], ctx["cfg"]
    job.step(94, "Writing tables")
    root = job.root
    names = list(sheet["sample"])
    # raw table: every linked feature, gap-filled areas, detection status
    raw = feat.copy()
    raw.insert(3, "rt_min", raw.pop("rt") / 60)
    raw["fwhm_s"] = raw.pop("fwhm")
    I, det, filled = ctx["I_raw"], ctx["det"], ctx["filled"]
    for k, s in enumerate(names):
        raw[s] = I[:, k]
    for k, s in enumerate(names):
        raw[f"status_{s}"] = np.where(det[:, k], "detected", np.where(filled[:, k], "filled", "missing"))
    raw.to_csv(root / "feature_table_raw.csv", index=False)
    # processed: kept features, log2 normalised (+ imputed) values of samples and QCs
    cols, L, fk = ctx["norm_cols"], ctx["L"], ctx["feat_kept"]
    proc = fk[["id", "mz", "rt", "charge", "compound_group", "adduct", "neutral_mass", "annotation", "formula",
               "msi_level", "annotation_adduct", "qc_cv"]].copy()
    proc.insert(3, "rt_min", proc.pop("rt") / 60)
    for j, s in enumerate(cols["sample"]):
        proc[s] = L[:, j]
    proc.to_csv(root / "feature_table_processed.csv", index=False)
    stats = ctx.get("stats")
    if stats is not None:
        stats.sort_values("p").to_csv(root / "statistics.csv", index=False)
    else:
        pd.DataFrame({"note": ["No group comparison: the sample sheet has fewer than 2 groups with 2+ samples."]}).to_csv(root / "statistics.csv", index=False)
    ann = ctx.get("annotations")
    (ann if ann is not None and len(ann) else pd.DataFrame(columns=["feature", "mz", "rt_min", "name", "formula", "adduct", "ppm",
                                                                    "level", "source", "evidence"])).to_csv(root / "annotations.csv", index=False)
    sheet.drop(columns=["run"]).to_csv(root / "sample_sheet_used.csv", index=False)
    job.download("Feature table — raw areas, all features (.csv)", "feature_table_raw.csv")
    job.download("Feature table — filtered, normalised log2 (.csv)", "feature_table_processed.csv")
    job.download("Statistics (.csv)", "statistics.csv")
    job.download("Annotation candidates (.csv)", "annotations.csv")
    job.download("Sample sheet used (.csv)", "sample_sheet_used.csv")

    # ---- interactive feature table: top 500 kept features (significant first, then by intensity)
    job.section("feature_table", "Results", "Feature table",
                "Click a feature to see its extracted-ion chromatogram in every sample.")
    kept_idx = np.flatnonzero(ctx["keep"])
    pos = {int(i): j for j, i in enumerate(kept_idx)}
    mean_int = np.nanmean(np.where(np.isfinite(I), I, np.nan), axis=1)
    if stats is not None:
        rank_key = np.array([stats.p.values[pos[i]] if i in pos else 2.0 for i in range(len(feat))])
    else:
        rank_key = np.array([-(mean_int[i] if np.isfinite(mean_int[i]) else 0) if i in pos else 1e30 for i in range(len(feat))])
    order = [int(i) for i in np.argsort(np.nan_to_num(rank_key, nan=2.0), kind="stable") if int(i) in pos][:500]
    rows = []
    RT0, fi = ctx["RT_obs"], ctx["file_index"]
    ntot = ctx["n_files_total"]
    for i in order:
        j = pos[i]
        rts = [None] * ntot
        for k in range(len(names)):
            if np.isfinite(RT0[i, k]):
                rts[fi[k]] = round(float(RT0[i, k]) / 60, 3)
        rows.append({
            "id": feat.id[i], "mz": round(float(feat.mz[i]), 5), "rt_min": round(float(feat.rt[i]) / 60, 3),
            "charge": int(feat.charge[i]), "annotation": feat.annotation[i], "msi_level": feat.msi_level[i],
            "adduct": feat.adduct[i] or feat.annotation_adduct[i], "group": feat.compound_group[i],
            "log2fc": _jsonable(stats.log2fc.values[j], 3) if stats is not None else None,
            "padj": _jsonable(stats.padj.values[j], 6) if stats is not None else None,
            "significant": bool(stats.significant.values[j]) if stats is not None else False,
            "cv_qc": _jsonable(feat.qc_cv[i], 1), "mean_intensity": _jsonable(mean_int[i], 0),
            "detected": int(det[i].sum()), "filled": int(filled[i].sum()), "has_ms2": bool(feat.has_ms2[i]),
            "rt_by_run": rts,
        })
    job.widget("feature_table", rows=rows, runs=True, ppm=cfg["ppm"], rt_window_min=round(max(4 * cfg["fwhm"], ctx["rt_tol"]) / 60, 3),
               contrast=list(ctx["contrast"]) if stats is not None else None,
               columns=["id", "mz", "rt_min", "annotation", "msi_level", "log2fc", "padj", "cv_qc"])
    job.table("feature_table_static", "Top features", ["feature", "m/z", "RT (min)", "annotation", "MSI", "log2FC", "FDR", "QC CV %"],
              [[r["id"], f"{r['mz']:.4f}", f"{r['rt_min']:.2f}", _short(r["annotation"], 40), r["msi_level"],
                f"{r['log2fc']:+.2f}" if r["log2fc"] is not None else "", f"{r['padj']:.1e}" if r["padj"] is not None else "",
                f"{r['cv_qc']:.0f}" if r["cv_qc"] is not None else ""] for r in rows[:200]],
              note="The same features as the interactive table (≤ 200 shown here); all of them are in the CSV files.",
              csv="feature_table_processed.csv")

    # ---- tiles
    n_kept = int(ctx["keep"].sum())
    n_ann = int(feat.annotation.astype(bool)[ctx["keep"]].sum())
    job.result["tiles"] = []
    job.tile(commas(len(feat)), "features detected")
    job.tile(commas(n_kept), "after filtering")
    job.tile(commas(n_ann), "annotated (kept)")
    if stats is not None:
        up, dn = ctx["n_sig"]
        job.tile(commas(up + dn), f"significant ({ctx['contrast'][1]} vs {ctx['contrast'][0]})")
    else:
        job.tile("—", "significant (needs groups)")
    if ctx.get("cv_med") and np.isfinite(ctx["cv_med"][1]):
        job.tile(f"{ctx['cv_med'][1]:.0f}%", "median QC CV")

    # ---- summary flags
    job.flag("ok", f"<b>{commas(len(feat))}</b> features linked across {len(names)} files; "
                   f"{commas(int(ctx['filled'].sum()))} missing values recovered by gap filling. Filters removed "
                   f"{commas(int(ctx['removed_blank'].sum()))} (blank), {commas(int(ctx['removed_det'].sum()))} (detection rate) and "
                   f"{commas(int(ctx['removed_cv'].sum()))} (QC CV), leaving <b>{commas(n_kept)}</b>.")
    if ctx.get("drift_done"):
        mb, ma = ctx["cv_med"]
        job.flag("ok" if ma < 20 else "warn", f"QC-based drift correction brought the median QC CV from {mb:.1f}% to <b>{ma:.1f}%</b>"
                 + (" — excellent precision." if ma < 10 else " — good." if ma < 20 else " — still high; results carry substantial analytical noise."))
    lvl2 = int((feat.msi_level[ctx["keep"]] == "2").sum())
    job.flag("info", f"{commas(n_ann)} kept features have a library candidate"
             + (f", {lvl2} at MSI level 2" if lvl2 else ", all by m/z only (MSI level 3, putative)")
             + ". Mass alone cannot tell isomers apart; confirm important hits with MS/MS or an authentic standard.")
    if ctx["params"].get("example"):
        job.flag("info", "This example is simulated: its compounds have random masses, not those of real metabolites, so "
                         "<b>no library match is expected</b> and an annotated count near zero is the correct result. "
                         "Your own data are matched against 406 real metabolites and lipids.")
    if stats is not None:
        up, dn = ctx["n_sig"]
        ga, gb = ctx["contrast"]
        sig = stats[stats.significant].sort_values("p")
        top_names = [(r.annotation.split(" / ")[0] if r.annotation else r.id) for r in sig.head(5).itertuples()]
        job.flag("ok" if up + dn else "info",
                 f"<b>{up}</b> features higher and <b>{dn}</b> lower in {gb} than in {ga} (FDR < {ctx['alpha']:g}, ≥ {2 ** ctx['min_lfc']:.2g}-fold)"
                 + (f". Top: {', '.join(top_names)}." if top_names else "."))
        nmin = min(int(((cols.type == 'sample') & (cols.group == g)).sum()) for g in (ga, gb))
        if nmin < 3:
            job.flag("warn", f"Only {nmin} samples in a compared group: p-values from so few replicates are unreliable; treat the list as exploratory.")
    if ctx.get("plsda"):
        q2, pv, _ = ctx["plsda"]
        job.flag("ok" if pv < 0.05 and q2 > 0.3 else "info", f"PLS-DA cross-validated Q² = {q2:.2f}, permutation p = {pv:.3f}"
                 + (" — the groups are predictably different." if pv < 0.05 and q2 > 0.3 else " — group separation in PLS-DA is not convincingly better than chance."))

    job.method("Software", "pyOpenMS (Röst et al. 2016), NumPy/SciPy, statsmodels, scikit-learn, matplotlib. "
               "Workflow modelled on UmetaFlow (Kontou et al. 2023) and common metabolomics QC practice (Broadhurst et al. 2018, Metabolomics 14:72).")
