"""Synthetic LC-MS data with a known answer, for the self-test, the examples and the test harness.

`write_lcms` turns a list of ions (m/z, retention time, peak width, height) into an mzML file.
`metabolomics_study` builds a small untargeted study on top of it — two groups, pooled QCs, a blank,
injection-order drift and RT shifts — and writes the truth next to it, so a pipeline can be scored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROTON = 1.007276467
NA_H = 21.981943  # Na+ minus H+ (the [M+Na]+ − [M+H]+ spacing)
C13 = 1.0033548


def write_lcms(path: Path, ions: list[tuple], rt_max: float = 900.0, scan_dt: float = 0.5, ppm_err: float = 2.0,
               noise_peaks: int = 150, noise_level: float = 2e3, polarity: int = 1, profile: bool = False,
               resolution: float = 60000.0, seed: int = 0, ms2: list[dict] | None = None):
    """Write one LC-MS run.

    ions: (mz, rt_apex_s, sigma_s, height) — every isotope/adduct is its own ion.
    ms2:  optional [{rt, precursor_mz, charge, mz: [...], intensity: [...]}] fragment spectra to interleave.
    profile=True writes Gaussian peak shapes on a fine m/z grid instead of centroids.
    """
    import pyopenms as oms
    rng = np.random.default_rng(seed)
    ions = np.asarray(ions, float) if len(ions) else np.zeros((0, 4))
    exp = oms.MSExperiment()
    times = np.arange(scan_dt, rt_max, scan_dt)
    ms2 = sorted(ms2 or [], key=lambda d: d["rt"])
    j = 0
    for t in times:
        w = ions[:, 3] * np.exp(-0.5 * ((t - ions[:, 1]) / ions[:, 2]) ** 2) if len(ions) else np.zeros(0)
        on = w > noise_level * 0.5
        mz = ions[on, 0] * (1 + rng.normal(0, ppm_err * 1e-6, on.sum()))
        it = w[on] * rng.lognormal(0, 0.05, on.sum())
        nz = rng.uniform(80, 1000, noise_peaks)
        mz = np.concatenate([mz, nz])
        it = np.concatenate([it, rng.exponential(noise_level, noise_peaks)])
        o = np.argsort(mz)
        mz, it = mz[o], it[o]
        s = oms.MSSpectrum()
        if profile:
            pm, pi = [], []
            for m, v in zip(mz, it):
                sd = m / resolution / 2.355
                g = np.linspace(-3 * sd, 3 * sd, 9)
                pm.append(m + g)
                pi.append(v * np.exp(-0.5 * (g / sd) ** 2))
            mz, it = np.concatenate(pm), np.concatenate(pi)
            o = np.argsort(mz)
            mz, it = mz[o], it[o]
            s.setType(oms.SpectrumSettings.SpectrumType.PROFILE)
        else:
            s.setType(oms.SpectrumSettings.SpectrumType.CENTROID)
        s.set_peaks((mz, it.astype(np.float32)))
        s.setRT(float(t))
        s.setMSLevel(1)
        s.setNativeID(f"scan={exp.getNrSpectra() + 1}")
        ins = s.getInstrumentSettings()
        ins.setPolarity(oms.IonSource.Polarity.POSITIVE if polarity > 0 else oms.IonSource.Polarity.NEGATIVE)
        s.setInstrumentSettings(ins)
        exp.addSpectrum(s)
        while j < len(ms2) and ms2[j]["rt"] <= t + scan_dt:
            d = ms2[j]
            s2 = oms.MSSpectrum()
            s2.set_peaks((np.asarray(d["mz"], float), np.asarray(d["intensity"], np.float32)))
            s2.setRT(float(d["rt"]))
            s2.setMSLevel(2)
            s2.setType(oms.SpectrumSettings.SpectrumType.CENTROID)
            s2.setNativeID(f"scan={exp.getNrSpectra() + 1}")
            p = oms.Precursor()
            p.setMZ(float(d["precursor_mz"]))
            p.setCharge(int(d.get("charge", 1)))
            s2.setPrecursors([p])
            ins2 = s2.getInstrumentSettings()
            ins2.setPolarity(oms.IonSource.Polarity.POSITIVE if polarity > 0 else oms.IonSource.Polarity.NEGATIVE)
            s2.setInstrumentSettings(ins2)
            exp.addSpectrum(s2)
            j += 1
    exp.sortSpectra(True)
    oms.MzMLFile().store(str(path), exp)
    return path


def compound_ions(neutral_mass: float, rt: float, height: float, sigma: float = 3.0, na_frac: float = 0.15,
                  n_carbon: int | None = None, polarity: int = 1):
    """[M+H]+ (or [M−H]−) with its 13C isotopes, plus a smaller [M+Na]+ in positive mode."""
    nc = n_carbon or max(2, int(round(neutral_mass / 14.5)))
    iso = [1.0, 0.0107 * nc, (0.0107 * nc) ** 2 / 2]
    base = neutral_mass + PROTON if polarity > 0 else neutral_mass - PROTON
    out = [(base + k * C13, rt, sigma, height * r) for k, r in enumerate(iso)]
    if polarity > 0 and na_frac > 0:
        out += [(base + NA_H + k * C13, rt, sigma, height * na_frac * r) for k, r in enumerate(iso[:2])]
    return out


def metabolomics_study(out: Path, n_per_group: int = 5, n_compounds: int = 150, n_qc: int = 4, frac_diff: float = 0.2,
                       drift: float = 0.25, rt_shift: float = 5.0, seed: int = 7, rt_max: float = 720.0,
                       profile: bool = False) -> dict:
    """A two-group untargeted study. Writes mzML files, sample_sheet.csv and truth.csv into `out`.

    Truth: each compound's neutral mass, RT, true log2 fold change (treated/control), and whether it is
    a blank contaminant. Intensities fall linearly with injection order by `drift` (a fraction), which
    the QC-based drift correction should undo.
    """
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mass = np.sort(rng.uniform(110, 850, n_compounds))
    rt = rng.uniform(40, rt_max - 60, n_compounds)
    base = 10 ** rng.uniform(4.3, 6.8, n_compounds)
    lfc = np.zeros(n_compounds)
    diff = rng.choice(n_compounds, int(frac_diff * n_compounds), replace=False)
    lfc[diff] = rng.choice([-1, 1], len(diff)) * rng.uniform(1.0, 2.5, len(diff))
    contaminant = np.zeros(n_compounds, bool)
    contaminant[rng.choice(np.setdiff1d(np.arange(n_compounds), diff), max(3, n_compounds // 25), replace=False)] = True

    samples = [("control", i) for i in range(n_per_group)] + [("treated", i) for i in range(n_per_group)]
    rng.shuffle(samples)
    order = []
    qc_every = max(1, len(samples) // max(n_qc - 1, 1))
    q = 0
    order.append(("blank", "blank", 0))
    order.append(("qc", "", q)); q += 1
    for k, (g, i) in enumerate(samples):
        order.append(("sample", g, i))
        if (k + 1) % qc_every == 0 and q < n_qc:
            order.append(("qc", "", q)); q += 1
    while q < n_qc:
        order.append(("qc", "", q)); q += 1

    rows = []
    n_inj = len(order)
    for inj, (typ, g, i) in enumerate(order):
        f = drift_factor = 1 - drift * inj / max(n_inj - 1, 1)
        if typ == "blank":
            name = "Blank_01"
            h = np.where(contaminant, base * 0.8, 0.0)
        elif typ == "qc":
            name = f"QC_{i+1:02d}"
            h = base * 2 ** (lfc / 2) * f          # pool = geometric middle of the groups
        else:
            name = f"{g.capitalize()}_{i+1:02d}"
            eff = lfc if g == "treated" else 0
            h = base * 2 ** eff * rng.lognormal(0, 0.18, n_compounds) * f
        shift = rng.normal(0, rt_shift) + 0.004 * rt * rng.normal(0, 1)   # offset + slight stretch
        ions = []
        for c in range(n_compounds):
            if h[c] > 0:
                ions += compound_ions(mass[c], rt[c] + (shift[c] if np.ndim(shift) else shift), h[c], sigma=rng.uniform(2.5, 4))
        fn = out / f"{name}.mzML"
        write_lcms(fn, ions, rt_max=rt_max, seed=seed * 1000 + inj, profile=profile)
        rows.append({"file": fn.name, "sample": name, "group": g if typ == "sample" else "",
                     "type": typ, "order": inj + 1, "batch": 1, "_drift": drift_factor})
    sheet = pd.DataFrame(rows)
    sheet.drop(columns=["_drift"]).to_csv(out / "sample_sheet.csv", index=False)
    truth = pd.DataFrame({"neutral_mass": mass, "mz_MH": mass + PROTON, "rt": rt, "log2fc": lfc,
                          "differential": lfc != 0, "contaminant": contaminant, "base_intensity": base})
    truth.to_csv(out / "truth.csv", index=False)
    return {"files": [out / r["file"] for r in rows], "sheet": out / "sample_sheet.csv", "truth": out / "truth.csv"}


if __name__ == "__main__":
    import sys
    d = metabolomics_study(Path(sys.argv[1] if len(sys.argv) > 1 else "sim_study"))
    print("wrote", len(d["files"]), "files")
