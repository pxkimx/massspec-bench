"""Raw-data overview: what is in each file, how the acquisition went, and whether the instrument was clean
and calibrated. The first thing to run on new data, and the entry point to the interactive explorer."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from . import chem, msio, peaks
from .common import BLANK_COLOR, PALETTE, QC_COLOR, commas, fmt_n, log_exc, versions

MAX_DETAIL = 4          # per-file maps for the first few files; the explorer covers the rest


def _contaminants():
    """Background ions every LC-MS lab sees (Keller et al. 2008). Masses computed from formulas."""
    pos = [
        ("Polysiloxane (C2H6OSi)5", "C10H30O5Si5", "[M+H]+"),
        ("Polysiloxane (C2H6OSi)6", "C12H36O6Si6", "[M+H]+"),
        ("Polysiloxane (C2H6OSi)7", "C14H42O7Si7", "[M+H]+"),
        ("Polysiloxane (C2H6OSi)8", "C16H48O8Si8", "[M+H]+"),
        ("DEHP phthalate", "C24H38O4", "[M+H]+"),
        ("DEHP phthalate", "C24H38O4", "[M+Na]+"),
        ("Dibutyl phthalate", "C16H22O4", "[M+H]+"),
        ("Erucamide (slip agent)", "C22H43NO", "[M+H]+"),
        ("Oleamide (slip agent)", "C18H35NO", "[M+H]+"),
        ("Tributyl phosphate", "C12H27O4P", "[M+H]+"),
        ("Triethylamine", "C6H15N", "[M+H]+"),
    ]
    neg = [
        ("Palmitic acid", "C16H32O2", "[M-H]-"),
        ("Stearic acid", "C18H36O2", "[M-H]-"),
        ("Sodium dodecyl sulfate", "C12H26O4S", "[M-H]-"),
        ("Trifluoroacetic acid", "C2HF3O2", "[M-H]-"),
        ("Perfluorooctanoic acid", "C8HF15O2", "[M-H]-"),
    ]
    out = []
    for pol, lst in ((1, pos), (-1, neg)):
        for name, f, a in lst:
            out.append({"name": name, "formula": f, "adduct": a, "polarity": pol,
                        "mz": chem.adduct_mz(chem.formula_mass(f), a)})
    return out


def _mins(x):
    return np.asarray(x, float) / 60.0


def _colors(runs, sheet):
    typ = {Path(r["file"]).name: r for r in (sheet or [])}
    groups = sorted({r.get("group") for r in (sheet or []) if r.get("group")})
    out = []
    for k, run in enumerate(runs):
        row = typ.get(Path(run.source).name, {})
        if row.get("type") == "qc":
            out.append(QC_COLOR)
        elif row.get("type") == "blank":
            out.append(BLANK_COLOR)
        elif row.get("group") in groups:
            out.append(PALETTE[groups.index(row["group"]) % len(PALETTE)])
        else:
            out.append(PALETTE[k % len(PALETTE)])
    return out


def run(job, files, params):
    files = [Path(f) for f in files]
    sheet = params.get("sheet") or []
    runs = msio.load_runs(job, files, 2, 45, centroid_profile=False)
    job.result["versions"] = versions("pyopenms", "pyteomics", "numpy", "scipy", "matplotlib", "rainbow-api", "timsrust_pyo3")
    cols = _colors(runs, sheet)
    summ = [r.summary() for r in runs]

    # ------------------------------------------------------------ tiles + flags
    job.tile(len(runs), "files" if len(runs) != 1 else "file")
    job.tile(commas(sum(s["spectra"] for s in summ)), "spectra")
    job.tile(commas(sum(s["ms2"] for s in summ)), "MS/MS spectra")
    job.tile(f"{min(s['rt_min'] for s in summ):.1f}–{max(s['rt_max'] for s in summ):.1f}", "retention time (min)")
    job.tile(f"{min(s['mz_min'] for s in summ):.0f}–{max(s['mz_max'] for s in summ):.0f}", "m/z range")
    fmts = sorted({s["format"] for s in summ})
    job.flag("ok", f"Read {len(runs)} file{'s' if len(runs) != 1 else ''} ({', '.join(fmts)}). "
                   f"{'All spectra are' if all(s['centroided'] for s in summ) else 'Some files hold'} "
                   f"{'centroided' if all(s['centroided'] for s in summ) else '<b>profile</b> (raw peak-shape) data; the analyses centroid it automatically'}.")
    pols = sorted({p for s in summ for p in s["polarity"].split("/") if p != "?"})
    if len(pols) > 1:
        job.flag("warn", f"Files contain more than one polarity ({', '.join(pols)}). Positive and negative ions are different "
                         "measurements — the untargeted and targeted analyses treat each polarity separately.")
    has_ms2 = any(s["ms2"] for s in summ)
    has_srm = any(s["chromatograms"] for s in summ)
    if has_srm:
        job.flag("info", f"{sum(s['chromatograms'] for s in summ)} chromatogram traces found (SRM/MRM transitions, SIM ions or "
                         "UV/other detectors). SRM data are quantified with <b>Targeted quantification</b> and a transition list.")
    if not any(s["spectra"] for s in summ):
        job.flag("warn", "These files contain no mass spectra — only detector traces — so only the traces are shown.")

    # ------------------------------------------------------------ overview section
    job.section("overview", "Step 1", "What is in these files",
                "One row per file, read directly from the raw data. Retention times are in minutes.")
    job.table("files", "Files", ["File", "Format", "Instrument", "Spectra", "MS1", "MS/MS", "Polarity", "Mode", "RT (min)", "m/z"],
              [[s["name"], s["format"], s["instrument"] or "—", commas(s["spectra"]), commas(s["ms1"]), commas(s["ms2"]),
                s["polarity"], "centroid" if s["centroided"] else "profile", f"{s['rt_min']:.1f}–{s['rt_max']:.1f}",
                f"{s['mz_min']:.1f}–{s['mz_max']:.1f}"] for s in summ])

    if any(s["ms1"] for s in summ):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
        for r, c in zip(runs, cols):
            i1 = r.idx(1)
            if not len(i1):
                continue
            axes[0].plot(_mins(r.rt[i1]), r.tic[i1], lw=0.8, color=c, label=r.name, alpha=0.85)
            axes[1].plot(_mins(r.rt[i1]), r.bpi[i1], lw=0.8, color=c, alpha=0.85)
        axes[0].set_ylabel("TIC")
        axes[1].set_ylabel("Base peak intensity")
        axes[1].set_xlabel("Retention time (min)")
        axes[0].set_title("Total ion and base peak chromatograms (MS1)")
        if len(runs) <= 12:
            axes[0].legend(fontsize=7, ncol=3, loc="upper right")
        tics = np.array([r.tic[r.idx(1)].sum() for r in runs])
        med = np.median(tics[tics > 0]) if (tics > 0).any() else 0
        odd = [r.name for r, t in zip(runs, tics) if med and (t < med / 3 or t > med * 3)]
        yours = (f"Total signal differs more than threefold from the median in <b>{', '.join(odd)}</b> — check injection volume, "
                 "spray stability or sample amount before comparing it with the others." if odd else
                 (f"The {len(runs)} files have similar total signal (within threefold of the median), so no injection looks failed."
                  if len(runs) > 1 else "A single run: the TIC shows where compounds elute; the base-peak trace shows the tallest ion at each moment."))
        if odd:
            job.flag("warn", f"Unusual total ion current in {', '.join(odd)} (more than 3× from the median).")
        job.figure("tic_overlay", "Chromatograms", fig, wide=True,
                   how="The total ion chromatogram (TIC) adds up every ion in each MS1 scan; the base peak chromatogram (BPC) plots "
                       "only the tallest ion. The BPC is cleaner because it ignores the chemical background that inflates the TIC. "
                       "Replicate injections should overlay closely — a flat trace, a late shift or a missing region points to an "
                       "injection, spray or column problem.",
                   yours=yours)

    # ------------------------------------------------------------ per-file maps
    job.section("maps", "Step 2", "LC-MS maps", "Where the signal is, in retention time and m/z at once.")
    for k, r in enumerate(runs[:MAX_DETAIL]):
        try:
            _run_map(job, r, k)
        except Exception:  # noqa: BLE001
            job.flag("info", f"Could not draw the map for {r.name}: " + log_exc("run map"))
    if len(runs) > MAX_DETAIL:
        job.flag("info", f"Maps are drawn for the first {MAX_DETAIL} files; open any file in the interactive explorer below.")

    # ------------------------------------------------------------ MS/MS acquisition
    if has_ms2:
        try:
            _acquisition(job, runs, cols)
        except Exception:  # noqa: BLE001
            job.flag("info", "Could not summarise the MS/MS acquisition: " + log_exc("acquisition"))

    # ------------------------------------------------------------ TIC peaks & a spectrum
    try:
        _tic_peaks(job, runs[0])
    except Exception:  # noqa: BLE001
        job.flag("info", "Could not list chromatographic peaks: " + log_exc("tic peaks"))

    # ------------------------------------------------------------ contaminants & mass accuracy
    try:
        _contaminant_check(job, runs, cols)
    except Exception:  # noqa: BLE001
        job.flag("info", "Contaminant check failed: " + log_exc("contaminants"))

    # ------------------------------------------------------------ SRM traces
    if has_srm:
        _srm(job, runs)

    # ------------------------------------------------------------ explorer
    job.section("explorer", "Interactive", "Explore the raw data",
                "Click the chromatogram to see the spectrum at that time; click a peak in the spectrum (or type an m/z) "
                "to extract its ion chromatogram across all files and integrate it. Drag to zoom, double-click to reset.")
    job.widget("ms_explorer", title="Raw data explorer")

    job.method("Data import", f"Raw files ({', '.join(fmts)}) were read with pyOpenMS {job.result['versions'].get('pyopenms', '')}"
               " (mzML/mzXML), pyteomics (MGF), rainbow (Agilent, Waters), timsrust (Bruker timsTOF) or ThermoRawFileParser (Thermo). "
               "Spectra were kept as acquired (profile or centroid).")
    job.method("Chromatograms", "TIC = sum of MS1 intensities per scan; BPC = maximum MS1 intensity per scan. Extracted-ion chromatograms "
               "sum the intensity within ±ppm of the target m/z; peaks are smoothed (Savitzky–Golay, 2nd order), baseline-corrected "
               "(rolling 10th percentile) and integrated by the trapezoid rule above a straight baseline between the peak boundaries.")
    job.method("Contaminants", "Common background ions (Keller et al., Anal Chim Acta 2008) were searched in the run-averaged MS1 spectrum "
               "at ±10 ppm; their measured m/z gives an estimate of mass accuracy without an internal calibrant.")
    job.result["summary"] = {"files": len(runs), "spectra": int(sum(s["spectra"] for s in summ)), "ms2": int(sum(s["ms2"] for s in summ))}
    job.param_group("Overview", "The overview has no analysis parameters; it describes the files as recorded.")
    job.param("_files", "Files", len(runs), "Every file you selected.", "data", "fixed")


def _run_map(job, r, k):
    i1 = r.idx(1)
    if not len(i1):
        return
    has_ms2 = (r.ms_level == 2).any()
    fig, axes = plt.subplots(1, 2 if has_ms2 else 1, figsize=(11 if has_ms2 else 7.5, 4.6), squeeze=False)
    ax = axes[0, 0]
    # a 2-D histogram of MS1 peaks, weighted by intensity; subsample spectra for very long runs
    step = max(1, len(i1) // 1500)
    sel = i1[::step]
    mz = np.concatenate([r.spectrum(i)[0] for i in sel])
    it = np.concatenate([r.spectrum(i)[1] for i in sel]).astype(float)
    rt = np.concatenate([np.full(r.offsets[i + 1] - r.offsets[i], r.rt[i]) for i in sel]) / 60
    if len(mz) and it.sum() > 0 and np.ptp(rt) > 0:
        lo, hi = np.percentile(mz, [0.1, 99.9])
        H, xe, ye = np.histogram2d(rt, mz, bins=[300, 300], range=[[rt.min(), rt.max()], [lo, hi]], weights=it)
        H = np.where(H > 0, H, np.nan)
        vmin = np.nanpercentile(H, 60) if np.isfinite(H).any() else 1   # fade the chemical background
        ax.imshow(H.T, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="magma_r",
                  norm=LogNorm(vmin=max(vmin, 1e-9), vmax=np.nanmax(H)), interpolation="nearest")
    ax.set_xlabel("Retention time (min)")
    ax.set_ylabel("m/z")
    ax.set_title(f"{r.name}: MS1 ion map")
    if has_ms2:
        ax2 = axes[0, 1]
        i2 = r.idx(2)
        z = r.precursor_charge[i2]
        for zz, c in zip([0, 1, 2, 3, 4], ["#9aa4ab", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]):
            m = (z == zz) if zz < 4 else (z >= 4)
            if m.any():
                ax2.scatter(_mins(r.rt[i2][m]), r.precursor_mz[i2][m], s=2, c=c, label=("unknown" if zz == 0 else f"{zz}+" if zz < 4 else "4+ and up"),
                            rasterized=True, linewidths=0)
        ax2.set_xlabel("Retention time (min)")
        ax2.set_ylabel("Precursor m/z")
        ax2.set_title("MS/MS precursors, by charge")
        ax2.legend(markerscale=4, fontsize=8, loc="upper left")
    job.figure(f"map_{k}", f"Ion map — {r.name}", fig, wide=True,
               how="Each dot is ion signal at a retention time (x) and m/z (y); darker means more intense. Compounds appear as short "
                   "horizontal streaks (one m/z eluting for a few seconds). Long horizontal lines across the whole run are chemical "
                   "background or contaminants; diagonal bands are homologous series (polymers, lipids) or, in peptide data, "
                   "the charge-state trend with hydrophobicity. The right panel shows which ions were picked for fragmentation.",
               yours=f"{commas(len(i1))} MS1 scans between {r.rt[i1].min()/60:.1f} and {r.rt[i1].max()/60:.1f} min, m/z "
                     f"{r.mz.min():.0f}–{r.mz.max():.0f}."
                     + (f" {commas((r.ms_level == 2).sum())} MS/MS spectra; {np.mean(r.precursor_charge[r.idx(2)] >= 2)*100:.0f}% of precursors "
                        "carry 2+ or more charges (typical of peptides; small molecules are almost all 1+)." if has_ms2 else ""))


def _acquisition(job, runs, cols):
    job.section("acquisition", "Step 3", "MS/MS acquisition",
                "How the instrument spent its time: how many precursors it fragmented, and which ones.")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    rates, med_cycle = [], []
    for r, c in zip(runs, cols):
        i2 = r.idx(2)
        if not len(i2):
            continue
        t = _mins(r.rt[i2])
        bins = np.arange(0, max(t.max(), 1) + 1, 1.0)
        h, e = np.histogram(t, bins)
        axes[0].plot(e[:-1] + 0.5, h, color=c, lw=1, label=r.name)
        rates.append(len(i2) / max((r.rt.max() - r.rt.min()) / 60, 1e-9))
        i1 = r.idx(1)
        if len(i1) > 2:
            med_cycle.append(float(np.median(np.diff(r.rt[i1]))))
    axes[0].set_xlabel("Retention time (min)")
    axes[0].set_ylabel("MS/MS per minute")
    axes[0].set_title("Fragmentation rate")
    allz = np.concatenate([r.precursor_charge[r.idx(2)] for r in runs])
    zs = np.arange(0, 7)
    cnt = [(allz == z).sum() if z < 6 else (allz >= 6).sum() for z in zs]
    axes[1].bar([("?" if z == 0 else f"{z}+" if z < 6 else "6+") for z in zs], cnt, color="#5b8def")
    axes[1].set_title("Precursor charge")
    axes[1].set_ylabel("MS/MS spectra")
    allmz = np.concatenate([r.precursor_mz[r.idx(2)] for r in runs])
    axes[2].hist(allmz[np.isfinite(allmz)], bins=60, color="#8c6bb1")
    axes[2].set_xlabel("Precursor m/z")
    axes[2].set_title("Precursor m/z")
    frac_multi = float(np.mean(allz >= 2)) if len(allz) else 0
    kind = ("peptides (most precursors are 2+ or higher)" if frac_multi > 0.6 else
            "small molecules (almost all precursors are singly charged)" if frac_multi < 0.15 else "a mixture of charge states")
    job.figure("acquisition", "MS/MS acquisition", fig, wide=True,
               how="Left: how many fragmentation spectra were recorded each minute — it should rise where compounds elute and never sit "
                   "flat at its maximum for long (that means the instrument had more candidates than time: undersampling). "
                   "Middle: precursor charge states; unknown (?) means the instrument could not assign one. Right: which m/z values were picked.",
               yours=f"About {np.mean(rates):.0f} MS/MS spectra per minute on average"
                     + (f", MS1 cycle time ≈ {np.median(med_cycle):.2f} s" if med_cycle else "")
                     + f". The charge distribution looks like {kind}.")


def _tic_peaks(job, r):
    i1 = r.idx(1)
    if len(i1) < 20:
        return
    t, y = r.rt[i1], r.tic[i1]
    pk = peaks.detect_all(t, y, min_snr=8)
    pk = sorted(pk, key=lambda p: -p["area"])[:15]
    if not pk:
        return
    job.section("peaks", "Step 4", f"Largest peaks in {r.name}",
                "The biggest chromatographic peaks in the total ion current, and the spectrum under the tallest one.")
    top = pk[0]
    j = i1[np.argmin(np.abs(t - top["rt"]))]
    mz, it = r.average_spectrum(top["start"], top["end"]) if top["end"] > top["start"] else r.spectrum(j)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].plot(_mins(t), y, color="#33454f", lw=0.8)
    for n, p in enumerate(pk):
        s = (t >= p["start"]) & (t <= p["end"])
        base = np.interp(t[s], [p["start"], p["end"]], [y[s][0], y[s][-1]])
        axes[0].fill_between(_mins(t[s]), np.minimum(base, y[s]), y[s], color=PALETTE[n % 10], alpha=0.45, lw=0)
        axes[0].text(p["rt"] / 60, p["height"], str(n + 1), fontsize=7, ha="center", va="bottom")
    axes[0].set_xlabel("Retention time (min)")
    axes[0].set_ylabel("TIC")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
    axes[0].set_title("TIC peaks (numbered by area)")
    _stick(axes[1], mz, it, label_top=8)
    axes[1].set_title(f"Spectrum under peak 1 ({top['rt']/60:.2f} min)")
    top_ion = mz[np.argmax(it)] if len(it) else float("nan")
    job.figure("tic_peaks", "Chromatographic peaks", fig, wide=True,
               how="Peaks were found in the smoothed, baseline-corrected TIC and integrated between the points where they return to "
                   "baseline. The right panel averages every MS1 scan across peak 1: the labelled m/z values are its most intense ions — "
                   "usually the protonated/deprotonated molecule, its adducts and isotopes.",
               yours=f"{len(pk)} peaks above 8× the noise; the largest elutes at {top['rt']/60:.2f} min (FWHM {top['fwhm']:.1f} s) and "
                     f"its most intense ion is m/z {top_ion:.4f}.")
    job.table("tic_peak_table", "TIC peaks", ["#", "RT (min)", "Start–end (min)", "Area", "Height", "FWHM (s)", "S/N"],
              [[n + 1, f"{p['rt']/60:.2f}", f"{p['start']/60:.2f}–{p['end']/60:.2f}", fmt_n(p["area"]), fmt_n(p["height"]),
                f"{p['fwhm']:.1f}", f"{p['snr']:.0f}"] for n, p in enumerate(pk)])


def _stick(ax, mz, it, label_top=8, color="#1f3b57"):
    mz = np.asarray(mz)
    it = np.asarray(it, float)
    if not len(mz):
        return
    ax.vlines(mz, 0, it, color=color, lw=0.8)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_ylim(0, it.max() * 1.15)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
    order = np.argsort(it)[::-1]
    shown = []
    for i in order:
        if len(shown) >= label_top:
            break
        if it[i] < 0.05 * it.max():
            break                                   # small peaks stay unlabelled: they are mostly noise
        if all(abs(mz[i] - s) > (mz.max() - mz.min()) * 0.04 for s in shown):
            ax.text(mz[i], it[i], f"{mz[i]:.4f}", fontsize=7, ha="center", va="bottom", rotation=0)
            shown.append(mz[i])


def _contaminant_check(job, runs, cols):
    cont = _contaminants()
    rows, errs = [], []
    for r in runs:
        i1 = r.idx(1)
        if not len(i1):
            continue
        sel = i1[:: max(1, len(i1) // 200)]
        mz = np.concatenate([r.spectrum(i)[0] for i in sel])
        it = np.concatenate([r.spectrum(i)[1] for i in sel]).astype(float)
        if not r.is_centroided:
            continue
        amz, ait = msio.merge_peaks(mz, it, 5.0, n_spectra=len(sel))
        if not len(amz):
            continue
        pol = 1 if (r.polarity[i1] >= 0).mean() >= 0.5 else -1
        floor = np.percentile(ait, 99)            # only report contaminants that stand out of the background
        for c in cont:
            if c["polarity"] != pol:
                continue
            tol = c["mz"] * 10e-6
            m = (amz > c["mz"] - tol) & (amz < c["mz"] + tol)
            if m.any():
                k = np.flatnonzero(m)[np.argmax(ait[m])]
                if ait[k] >= floor:
                    e = (amz[k] - c["mz"]) / c["mz"] * 1e6
                    rows.append([r.name, c["name"], c["adduct"], f"{c['mz']:.4f}", f"{amz[k]:.4f}", f"{e:+.1f}", fmt_n(ait[k]),
                                 f"{ait[k] / ait.max() * 100:.1f}%"])
                    errs.append(e)
        # PEG ladder: ≥ 5 consecutive peaks 44.0262 apart among the tallest ions
        tall = amz[ait >= np.percentile(ait, 99.5)]
        if len(tall) > 5:
            d = np.abs(tall[:, None] + 44.02621 - tall[None, :]) < 0.005
            best = 0
            for s in range(len(tall)):
                n, cur = 1, s
                while True:
                    nxt = np.flatnonzero(d[cur])
                    if not len(nxt):
                        break
                    cur = nxt[0]
                    n += 1
                best = max(best, n)
            if best >= 5:
                rows.append([r.name, f"PEG-like ladder (44.026 spacing, {best} members)", "—", "—", "—", "—", "—", "—"])
    job.section("contaminants", "Step 5", "Background and mass accuracy",
                "Common laboratory contaminants that stand out in the run-averaged spectrum, and what their m/z says about calibration.")
    if not rows:
        job.flag("ok", "No common contaminant (polysiloxanes, phthalates, slip agents, PEG, fatty acids) stands out of the background.")
        job.table("contaminants", "Contaminants found", ["File", "Contaminant"], [["—", "none above the 99th intensity percentile"]])
        return
    job.table("contaminants", "Contaminants found", ["File", "Contaminant", "Ion", "Expected m/z", "Measured m/z", "Error (ppm)", "Intensity", "% of base peak"],
              rows, note="Only ions in the top 1% of intensities are reported, so trace amounts are not flagged.")
    names = sorted({r[1].split(" (")[0] for r in rows})
    job.flag("info", f"Background ions present: {', '.join(names)}. Polysiloxanes come from air/septa, phthalates and slip "
                     "agents from plastics, PEG from detergents or some filters; heavy contamination suppresses real signal.")
    if errs:
        e = np.array(errs)
        med = float(np.median(e))
        lvl = "ok" if abs(med) < 5 else "warn"
        job.flag(lvl, f"Mass accuracy estimated from {len(e)} background ions: median error <b>{med:+.1f} ppm</b> "
                      + ("— within the ±5 ppm expected for an Orbitrap or Q-TOF." if lvl == "ok" else
                         "— larger than expected for high-resolution instruments; widen the ppm tolerance in the analyses or recalibrate."))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.axhspan(-5, 5, color="#e0f4f1", zorder=0)
        ax.scatter(range(len(e)), e, color="#0f766e", zorder=2)
        ax.axhline(0, color="#33454f", lw=0.8)
        ax.set_xticks(range(len(e)))
        ax.set_xticklabels([r[1][:18] + f" ({r[0][:10]})" for r in rows if r[5] != "—"][:len(e)], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Mass error (ppm)")
        ax.set_title("Mass error of background ions")
        job.figure("mass_accuracy", "Mass accuracy from background ions", fig,
                   how="Background ions have exactly known formulas, so the difference between their measured and theoretical m/z is a "
                       "free calibration check. The shaded band is ±5 ppm; points consistently on one side mean a calibration offset.",
                   yours=f"Median {med:+.1f} ppm, spread {np.std(e):.1f} ppm across {len(e)} measurements.")


def _srm(job, runs):
    job.section("srm", "Step 6", "Other traces",
                "Chromatograms stored as traces rather than spectra: SRM/MRM transitions (precursor → product ion), "
                "SIM ions, and non-MS detectors such as UV/DAD, ELSD, CAD or FID.")
    r = next(x for x in runs if x.chromatograms)
    ch = r.chromatograms[:24]
    n = len(ch)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.2 * nrow), squeeze=False)
    for a, c in zip(axes.flat, ch):
        a.plot(_mins(c["time"]), c["intensity"], lw=0.9, color="#1f3b57")
        a.set_title(f"{c['q1']:.1f} → {c['q3']:.1f}" if c.get("q1") and c.get("q3") else str(c["id"])[:30], fontsize=8)
        a.tick_params(labelsize=7)
    for a in list(axes.flat)[n:]:
        a.axis("off")
    fig.supxlabel("Retention time (min)", fontsize=9)
    fig.tight_layout()
    job.figure("srm_grid", f"Traces in {r.name}", fig, wide=True,
               how="For SRM/MRM, each panel is one transition: the instrument let through only one precursor m/z (Q1) and recorded one of its "
                   "fragments (Q3), so a peak is highly specific to that compound. Qualifier transitions of the same compound should "
                   "peak at exactly the same time. UV, ELSD, CAD or FID traces come from detectors in line with the mass spectrometer and show "
                   "compounds whether or not they ionise.",
               yours=f"{len(r.chromatograms)} traces in this file" + (" (first 24 shown)." if len(r.chromatograms) > 24 else "."))
