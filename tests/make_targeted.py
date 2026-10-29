"""Synthetic targeted-quantification studies with a known answer.

1. `fullscan_study(out)` — high-resolution full-scan LC-MS (mzML, centroided MS1): 7 metabolites with real
   formulas, 5 stable-isotope-labelled internal standards (one analyte uses a surrogate ISTD, one has none and
   no expected RT in the list), an 8-level calibration series (1–1000 ng/mL), 3 blanks (one right after the top
   standard, with 0.01 % carry-over; the last one without ISTD), low/mid/high QCs in duplicate, and 10 unknowns
   in two groups (one above the top standard, one below the LLOQ). Per-injection signal variation and a mild
   matrix suppression in real samples are shared by analyte and ISTD, as they are in real data.
2. `srm_study(out)` — triple-quad style SRM/MRM: 4 analytes (quantifier + qualifier transitions) + 2 ISTDs,
   written twice from the same numbers: as mzML chromatograms (the normal SRM export) and as MS2 spectra
   (what some converters produce), so both extraction paths can be compared.

Each writes sample_sheet.csv, a target list and truth.csv (true concentration per sample × analyte).
Run: python tests/make_targeted.py [out_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import chem  # noqa: E402
from server.simulate import compound_ions, write_lcms  # noqa: E402

# name, formula (neutral), RT (min), height per ng/mL, ISTD, write RT in list?
ANALYTES = [
    ("Creatinine", "C4H7N3O", 1.10, 900, "Creatinine-d3", True),
    ("Serotonin", "C10H12N2O", 2.20, 1500, "Serotonin-d4", True),
    ("Kynurenine", "C10H12N2O3", 2.90, 1200, "Kynurenine-d4", True),
    ("Tryptophan", "C11H12N2O2", 3.60, 2500, "Tryptophan-d5", True),
    ("Paraxanthine", "C7H8N4O2", 4.30, 1800, "Caffeine-13C3", True),      # surrogate ISTD (different RT)
    ("Caffeine", "C8H10N4O2", 5.20, 3000, "Caffeine-13C3", True),
    ("Melatonin", "C13H16N2O2", 6.40, 4000, None, False),                  # external calibration, RT not given
]
ISTDS = [  # name, formula as a user would type it (D for deuterium), RT offset (min) vs the analyte, height
    ("Creatinine-d3", "C4H4D3N3O", "Creatinine", -0.01, 2.5e5),
    ("Serotonin-d4", "C10H8D4N2O", "Serotonin", -0.02, 2.0e5),
    ("Kynurenine-d4", "C10H8D4N2O3", "Kynurenine", -0.02, 2.0e5),
    ("Tryptophan-d5", "C11H7D5N2O2", "Tryptophan", -0.02, 3.0e5),
    ("Caffeine-13C3", "[13C]3C5H10N4O2", "Caffeine", 0.0, 3.0e5),
]
STD_LEVELS = [1, 2, 5, 10, 50, 100, 500, 1000]
QC_LEVELS = {"L": 3, "M": 80, "H": 800}


def _mass(f):
    import re
    return chem.formula_mass(re.sub(r"D(?![a-z])", "[2H]", f))


def _carbons(f):
    comp = chem.parse_formula(f)
    return sum(v for k, v in comp.items() if k == "C")          # unlabelled carbons carry the natural 13C


def _injections(rng, n_unknown=10):
    """Injection sequence: blank, standards low→high, blank (carry-over), QCs, unknowns, QCs, unknowns, blank."""
    seq = [("blank", "Blank_01", None, "")]
    seq += [("standard", f"STD_{k + 1:02d}_{lv}ngmL", lv, "") for k, lv in enumerate(STD_LEVELS)]
    seq += [("blank", "Blank_02_after_ULOQ", None, "")]
    seq += [("qc", f"QC_{k}_1", v, "") for k, v in QC_LEVELS.items()]
    groups = ["control"] * (n_unknown // 2) + ["treated"] * (n_unknown - n_unknown // 2)
    unk = [("sample", f"U{k + 1:02d}", None, g) for k, g in enumerate(groups)]
    order = rng.permutation(len(unk))
    unk = [unk[k] for k in order]
    seq += unk[: len(unk) // 2] + [("qc", f"QC_{k}_2", v, "") for k, v in QC_LEVELS.items()] + unk[len(unk) // 2:]
    seq += [("blank", "Blank_03_noIS", None, "")]
    return seq


def _true_conc(rng, names, seq):
    """True concentrations of the unknowns: group effects on three analytes, one >ULOQ and one <LLOQ value."""
    base = {n: 10 ** rng.uniform(1.0, 2.3) for n in names}          # 10–200 ng/mL
    effect = {"Caffeine": 2.5, "Paraxanthine": 2.0, "Kynurenine": 1.8}
    truth = {}
    for typ, s, lv, g in seq:
        if typ != "sample":
            continue
        truth[s] = {n: base[n] * (effect.get(n, 1.0) if g == "treated" else 1.0) * rng.lognormal(0, 0.2) for n in names}
    # one sample over the top standard and one below the LLOQ, to exercise the reporting
    truth["U01"]["Caffeine"] = 1500.0
    truth["U02"]["Serotonin"] = 0.4
    return truth


def fullscan_study(out: Path, seed: int = 11) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    seq = _injections(rng)
    names = [a[0] for a in ANALYTES]
    truth = _true_conc(rng, names, seq)
    kyn_mz = _mass("C10H12N2O3")
    rows, trows = [], []
    top = max(STD_LEVELS)
    for inj, (typ, s, lv, g) in enumerate(seq):
        shift = rng.normal(0, 1.5)                                   # s, whole-run RT offset
        inj_f = rng.lognormal(0, 0.06)                               # injection volume / spray variation
        matrix = 0.92 if typ in ("sample", "qc") else 1.0            # ion suppression in real matrix
        width = 2.9 * rng.lognormal(0, 0.04)                         # s; column state sets every peak's width alike
        ions = []
        for name, f, rt, rf, istd, _ in ANALYTES:
            if typ == "standard" or typ == "qc":
                c = lv
            elif typ == "sample":
                c = truth[s][name]
            elif s == "Blank_02_after_ULOQ":
                c = top * 1e-4                                       # carry-over: 0.01 % of the top standard
            else:
                c = 0.0
            if c > 0:
                h = rf * c * inj_f * matrix * rng.lognormal(0, 0.015)
                ions += compound_ions(_mass(f), rt * 60 + shift + rng.normal(0, 0.4), h, sigma=width * rng.lognormal(0, 0.01),
                                      n_carbon=_carbons(f))
        # an isobaric interference 0.7 min after kynurenine, in every injection (outside its RT window)
        ions += compound_ions(kyn_mz, 2.9 * 60 + 42 + shift, 4e4 * inj_f, sigma=3.0, n_carbon=10, na_frac=0)
        if s != "Blank_03_noIS":
            for iname, f, parent, drt, h in ISTDS:
                prt = next(a[2] for a in ANALYTES if a[0] == parent)
                ions += compound_ions(_mass(f), (prt + drt) * 60 + shift + rng.normal(0, 0.4), h * inj_f * matrix * rng.lognormal(0, 0.015),
                                      sigma=width * rng.lognormal(0, 0.01), n_carbon=_carbons(f))
        fn = out / f"{s}.mzML"
        write_lcms(fn, ions, rt_max=480.0, scan_dt=0.5, noise_level=50.0, noise_peaks=150, seed=seed * 100 + inj)
        rows.append({"file": fn.name, "sample": s, "group": g, "type": typ, "order": inj + 1, "batch": 1,
                     "concentration": lv if lv is not None else ""})
    pd.DataFrame(rows).to_csv(out / "sample_sheet.csv", index=False)
    # target list: formula for most, m/z for two, qualifiers for caffeine (13C isotope) and tryptophan (Na adduct)
    for name, f, rt, rf, istd, give_rt in ANALYTES:
        row = {"Compound": name, "Formula": f, "Adduct": "M+H", "m/z": "", "RT (min)": rt if give_rt else "",
               "Internal Standard": istd or "", "Role": "analyte", "Quantifier": "quantifier", "Units": "ng/mL"}
        if name in ("Serotonin", "Melatonin"):
            row["Formula"] = ""
            row["m/z"] = round(_mass(f) + chem.PROTON, 5)
        trows.append(row)
        if name == "Caffeine":
            trows.append({**row, "Formula": "", "m/z": round(_mass(f) + chem.PROTON + chem.C13_DIFF, 5), "Quantifier": "qualifier"})
        if name == "Tryptophan":
            trows.append({**row, "Formula": f, "Adduct": "[M+Na]+", "Quantifier": "qualifier"})
    for iname, f, parent, drt, h in ISTDS:
        prt = next(a[2] for a in ANALYTES if a[0] == parent)
        trows.append({"Compound": iname, "Formula": f, "Adduct": "", "m/z": "", "RT (min)": round(prt + drt, 3),
                      "Internal Standard": "", "Role": "ISTD", "Quantifier": "", "Units": ""})
    pd.DataFrame(trows).to_csv(out / "targets.csv", index=False)
    tr = pd.DataFrame(truth).T
    tr.index.name = "sample"
    tr.to_csv(out / "truth.csv")
    return {"files": [out / r["file"] for r in rows], "sheet": out / "sample_sheet.csv", "targets": out / "targets.csv",
            "truth": out / "truth.csv"}


# ------------------------------------------------------------------ SRM / MRM

SRM = [  # name, RT (min), [(q1, q3, relative intensity)], response per ng/mL, ISTD
    ("Caffeine", 3.10, [(195.1, 138.1, 1.0), (195.1, 110.1, 0.45)], 2500, "Caffeine-13C3"),
    ("Paraxanthine", 2.60, [(181.1, 124.1, 1.0), (181.1, 96.1, 0.60)], 1500, "Caffeine-13C3"),
    ("Tryptophan", 2.20, [(205.1, 188.1, 1.0), (205.1, 146.1, 0.35)], 2000, "Tryptophan-d5"),
    ("Kynurenine", 1.70, [(209.1, 192.1, 1.0), (209.1, 94.1, 0.25)], 1000, "Tryptophan-d5"),
]
SRM_IS = [("Caffeine-13C3", 3.10, [(198.1, 140.1, 1.0)], 2e5), ("Tryptophan-d5", 2.18, [(210.1, 192.1, 1.0)], 2e5)]
SRM_LEVELS = [1, 3, 10, 30, 100, 300, 1000]


def srm_study(out: Path, seed: int = 5) -> dict:
    import pyopenms as oms
    out = Path(out)
    (out / "chrom").mkdir(parents=True, exist_ok=True)
    (out / "spectra").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    seq = [("blank", "SRM_Blank_01", None, "")]
    seq += [("standard", f"SRM_Cal{k + 1}", lv, "") for k, lv in enumerate(SRM_LEVELS)]
    seq += [("blank", "SRM_Blank_02", None, "")]
    seq += [("qc", f"SRM_QC_{k}", v, "") for k, v in QC_LEVELS.items()]
    seq += [("sample", f"SRM_S{k + 1:02d}", None, "A" if k < 3 else "B") for k in range(6)]
    truth = {s: {n: 10 ** rng.uniform(0.8, 2.6) for n, *_ in SRM} for t, s, lv, g in seq if t == "sample"}
    rows = []
    groups = [(n, rt, trs, rf, False) for n, rt, trs, rf, _ in SRM] + [(n, rt, trs, h, True) for n, rt, trs, h in SRM_IS]
    for inj, (typ, s, lv, g) in enumerate(seq):
        shift = rng.normal(0, 1.5)
        inj_f = rng.lognormal(0, 0.06)
        matrix = 0.9 if typ in ("sample", "qc") else 1.0
        chrom_exp, spec_exp = oms.MSExperiment(), oms.MSExperiment()
        spectra = []
        for gi, (n, rt, trs, rf, is_is) in enumerate(groups):
            if is_is:
                h = rf * inj_f * matrix
            else:
                c = lv if typ in ("standard", "qc") else truth[s][n] if typ == "sample" else (1000 * 5e-5 if s == "SRM_Blank_02" else 0.0)
                h = rf * c * inj_f * matrix
            apex = rt * 60 + shift
            t = np.arange(apex - 90, apex + 90, 0.5) + 0.05 * gi      # scheduled window, staggered dwell
            sigma = 2.9 * rng.lognormal(0, 0.04) if gi == 0 else sigma
            ys = []
            for k, (q1, q3, rel) in enumerate(trs):
                hh = h * rel
                if s == "SRM_S04" and n == "Caffeine" and k == 1:
                    hh *= 1.8                                          # co-eluting interference on the qualifier
                sig = hh * np.exp(-0.5 * ((t - apex) / sigma) ** 2) * rng.lognormal(0, 0.03, len(t))
                y = np.clip(150 + rng.normal(0, 25, len(t)) + sig, 0, None)
                ys.append(y)
                ch = oms.MSChromatogram()
                ch.set_peaks((t, y.astype(np.float32)))
                p = oms.Precursor(); p.setMZ(q1); ch.setPrecursor(p)
                pr = oms.Product(); pr.setMZ(q3); ch.setProduct(pr)
                ch.setNativeID(f"SRM SIC Q1={q1} Q3={q3} {n}")
                ch.setChromatogramType(oms.ChromatogramSettings.ChromatogramType.SELECTED_REACTION_MONITORING_CHROMATOGRAM)
                chrom_exp.addChromatogram(ch)
            q1 = trs[0][0]
            for j, tj in enumerate(t):
                spectra.append((tj, q1, [q3 for _, q3, _ in trs], [y[j] for y in ys]))
        for tj, q1, mzs, its in sorted(spectra, key=lambda v: v[0]):
            sp = oms.MSSpectrum()
            o = np.argsort(mzs)
            sp.set_peaks((np.asarray(mzs, float)[o], np.asarray(its, np.float32)[o]))
            sp.setRT(float(tj))
            sp.setMSLevel(2)
            sp.setType(oms.SpectrumSettings.SpectrumType.CENTROID)
            p = oms.Precursor(); p.setMZ(q1)
            sp.setPrecursors([p])
            sp.setNativeID(f"scan={spec_exp.getNrSpectra() + 1}")
            spec_exp.addSpectrum(sp)
        oms.MzMLFile().store(str(out / "chrom" / f"{s}.mzML"), chrom_exp)
        oms.MzMLFile().store(str(out / "spectra" / f"{s}.mzML"), spec_exp)
        rows.append({"file": f"{s}.mzML", "sample": s, "group": g, "type": typ, "order": inj + 1, "concentration": lv if lv is not None else ""})
    pd.DataFrame(rows).to_csv(out / "sample_sheet.csv", index=False)
    lines = ["Compound\tQ1\tQ3\tRT\tQuantifier\tISTD\tIS_ISTD\tUnits"]
    for n, rt, trs, rf, istd in SRM:
        for k, (q1, q3, rel) in enumerate(trs):
            lines.append(f"{n}\t{q1}\t{q3}\t{rt}\t{'TRUE' if k == 0 else 'FALSE'}\t{istd}\tFALSE\tng/mL")
    for n, rt, trs, h in SRM_IS:
        for q1, q3, rel in trs:
            lines.append(f"{n}\t{q1}\t{q3}\t{rt}\tTRUE\t\tTRUE\t")
    (out / "targets_srm.tsv").write_text("\n".join(lines) + "\n")
    pd.DataFrame(truth).T.rename_axis("sample").to_csv(out / "truth.csv")
    return {"chrom": [out / "chrom" / r["file"] for r in rows], "spectra": [out / "spectra" / r["file"] for r in rows],
            "sheet": out / "sample_sheet.csv", "targets": out / "targets_srm.tsv", "truth": out / "truth.csv"}


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "_out" / "targeted"
    a = fullscan_study(base / "fullscan")
    print("full-scan:", len(a["files"]), "files ->", a["files"][0].parent)
    b = srm_study(base / "srm")
    print("SRM:", len(b["chrom"]), "files × 2 layouts ->", base / "srm")
