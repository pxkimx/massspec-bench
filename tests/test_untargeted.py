"""End-to-end test of server/untargeted.py on synthetic studies with a known answer (server/simulate.py).

Scores, per study:
  main       5 control + 5 treated + 4 pooled QCs + 1 blank, centroid, intensity drift over injection order.
             Fraction of true compounds recovered; precision / recall of the significant calls against the true
             differential compounds (FDR-controlled calls should be mostly right); direction and size of fold
             changes; contaminants removed by the blank filter; QC CV before/after drift correction; adduct groups
             consistent with the truth; result.json + PDF report written.
  profile    the same design in profile mode (centroided by msio.load_runs) — compounds recovered.
  nogroups   the main files with the group column emptied — features, QC metrics and PCA still produced, and a
             flag says statistics need groups.
  annot      a small study built from real library compounds, with MS/MS spectra, an .msp spectral library and a
             user RT library — checks MSI level 2/3 annotations, cosine matching and adduct grouping.
  negative   negative mode, three groups (ANOVA path), two batches with a 2-fold sensitivity step that the
             per-batch QC correction must remove.

Run: "$HOME/Library/Application Support/MassSpecBench/venv/bin/python" tests/test_untargeted.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "_out"
os.environ.setdefault("MB_HOME", str(OUT / "mbhome"))      # keep the parse cache and jobs out of the real app folder
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from server import chem, msio, simulate, untargeted  # noqa: E402
from server.common import Job, run_safely  # noqa: E402

DATA = OUT / "untargeted"
NA_H = simulate.NA_H
C13 = simulate.C13


# ---------------------------------------------------------------------------------------------- helpers

def _run(name, files, sheet, **params):
    root = OUT / "jobs" / name
    shutil.rmtree(root, ignore_errors=True)
    job = Job(root, "untargeted", name)
    t = time.time()
    run_safely(job, untargeted.run, files, {"sheet": sheet, **params})
    dt = time.time() - t
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "done", status
    result = json.loads((root / "result.json").read_text())
    return root, result, dt


def _sheet(d, files):
    return msio.match_sheet(files, msio.read_table(d / "sample_sheet.csv"))


def _match_truth(raw, truth, ppm=10, rt_tol=30):
    """For each feature, the (compound, ion) pairs of the truth it corresponds to."""
    ions = []
    for c, t in truth.iterrows():
        for ion, m in (("MH", t.mz_MH), ("MH+1", t.mz_MH + C13), ("MH+2", t.mz_MH + 2 * C13),
                       ("Na", t.mz_MH + NA_H), ("Na+1", t.mz_MH + NA_H + C13)):
            ions.append((c, ion, m, t.rt))
    ions = pd.DataFrame(ions, columns=["c", "ion", "mz", "rt"])
    out = []
    for mz, rt in zip(raw.mz.values, raw.rt_min.values * 60):
        hit = ions[(np.abs(ions.mz - mz) / ions.mz * 1e6 <= ppm) & (np.abs(ions.rt - rt) <= rt_tol)]
        out.append(list(zip(hit.c, hit.ion)))
    return out


def score_main(root, truth_csv):
    truth = pd.read_csv(truth_csv)
    raw = pd.read_csv(root / "feature_table_raw.csv")
    st = pd.read_csv(root / "statistics.csv")
    raw["match"] = _match_truth(raw, truth)
    s = {}
    rec = {c for m in raw.match for c, ion in m if ion == "MH"}
    s["compounds_recovered"] = len(rec) / len(truth)
    s["features"] = len(raw)
    s["features_unexplained"] = int((raw.match.str.len() == 0).sum())
    # adduct groups: members of one compound group should all belong to one true compound
    g = raw[raw.compound_group.notna() & (raw.compound_group.astype(str) != "")]
    cons = [bool(set.intersection(*[{c for c, _ in m} for m in sub.match])) for _, sub in g.groupby("compound_group")]
    s["adduct_groups"] = len(cons)
    s["adduct_groups_consistent"] = float(np.mean(cons)) if cons else np.nan
    # blank filter
    cont = set(truth.index[truth.contaminant])
    kept = raw.kept.astype(bool)
    is_cont = np.array([any(c in cont for c, _ in m) for m in raw.match])
    is_real = np.array([len(m) > 0 and not any(c in cont for c, _ in m) for m in raw.match])
    s["contaminant_features_removed"] = float((~kept[is_cont]).mean()) if is_cont.any() else np.nan
    s["real_features_kept"] = float(kept[is_real].mean())
    # QC CV
    result = json.loads((root / "result.json").read_text())
    s["tiles"] = {t["label"]: t["value"] for t in result["tiles"]}
    # statistics against the truth
    st = st.merge(raw[["id", "match"]], on="id")
    diff = set(truth.index[truth.differential])
    sig = st[st.significant.astype(bool)]
    tp = np.array([any(c in diff for c, _ in m) for m in sig.match])
    s["significant_features"] = len(sig)
    s["precision"] = float(tp.mean()) if len(sig) else np.nan
    called = {c for m in sig.match for c, _ in m if c in diff}
    s["recall_compounds"] = len(called) / len(diff)
    wrong = sum(1 for m, l in zip(sig.match, sig.log2fc) for c, _ in m if c in diff and np.sign(truth.log2fc[c]) != np.sign(l))
    s["wrong_direction"] = wrong
    x, y = [], []
    for m, l in zip(st.match, st.log2fc):
        for c, ion in m:
            if ion == "MH":
                x.append(truth.log2fc[c]); y.append(l)
    s["log2fc_r"] = float(np.corrcoef(x, y)[0, 1])
    s["log2fc_slope"] = float(np.polyfit(x, y, 1)[0])
    return s


def _qc_cv(root):
    """Median QC CV of kept features before (raw, gap-filled) and after drift correction."""
    raw = pd.read_csv(root / "feature_table_raw.csv")
    k = raw[raw.kept.astype(bool)]
    qc = [c for c in raw.columns if c.startswith("QC_") and not c.startswith("status_")]
    x = k[qc].values
    before = np.nanmedian(100 * np.nanstd(x, 1, ddof=1) / np.nanmean(x, 1))
    after = np.nanmedian(k.qc_cv)
    return float(before), float(after)


# ---------------------------------------------------------------------------------------------- studies

def test_main():
    d = DATA / "main"
    if not (d / "truth.csv").exists():
        simulate.metabolomics_study(d, n_per_group=5, n_qc=4)
    files = sorted(d.glob("*.mzML"))
    root, result, dt = _run("untargeted_main", files, _sheet(d, files))
    s = score_main(root, d / "truth.csv")
    s["qc_cv_before"], s["qc_cv_after"] = _qc_cv(root)
    s["runtime_s"] = round(dt, 1)
    print("main:", json.dumps(s, indent=1, default=str))
    assert s["compounds_recovered"] >= 0.85, s
    assert s["precision"] >= 0.85, s
    assert s["recall_compounds"] >= 0.7, s
    assert s["wrong_direction"] == 0, s
    assert s["contaminant_features_removed"] >= 0.9, s
    assert s["real_features_kept"] >= 0.9, s
    assert s["adduct_groups_consistent"] >= 0.95, s
    assert s["qc_cv_after"] < s["qc_cv_before"], s
    assert (root / "report.pdf").exists() or any("PDF" in f["text"] for f in result["flags"])
    for f in ("feature_table_raw.csv", "feature_table_processed.csv", "statistics.csv", "annotations.csv"):
        assert (root / f).exists(), f
    w = [it for sec in result["sections"] for it in sec["items"] if it["type"] == "feature_table"]
    assert w and w[0]["runs"] and len(w[0]["rows"]) > 100, "feature_table widget missing"
    ids = {sec["id"] for sec in result["sections"]}
    assert {"rawqc", "features", "align", "linking", "gapfill", "annotation", "filtering", "drift", "normalisation",
            "multivariate", "stats", "feature_table"} <= ids, ids
    keys = set(result["params"])
    assert {"mass_error_ppm", "noise_threshold_int", "chrom_fwhm", "blank_ratio", "qc_cv_max", "normalisation",
            "group_a", "group_b"} <= keys, keys
    assert result["params"]["group_a"] == "control", result["params"]["group_a"]   # first group in the sheet
    return s


def test_profile():
    d = DATA / "profile"
    if not (d / "truth.csv").exists():
        simulate.metabolomics_study(d, n_per_group=3, n_qc=3, n_compounds=80, profile=True, seed=11)
    files = sorted(d.glob("*.mzML"))
    root, result, dt = _run("untargeted_profile", files, _sheet(d, files))
    truth = pd.read_csv(d / "truth.csv")
    raw = pd.read_csv(root / "feature_table_raw.csv")
    raw["match"] = _match_truth(raw, truth)
    rec = len({c for m in raw.match for c, ion in m if ion == "MH"}) / len(truth)
    print(f"profile: {rec:.3f} of compounds recovered, {len(raw)} features, {dt:.1f} s")
    assert rec >= 0.85, rec
    return rec


def test_nogroups():
    d = DATA / "main"
    if not (d / "truth.csv").exists():
        simulate.metabolomics_study(d, n_per_group=5, n_qc=4)
    files = sorted(d.glob("*.mzML"))
    sheet = _sheet(d, files)
    for r in sheet:
        r["group"] = ""
    root, result, dt = _run("untargeted_nogroups", files, sheet)
    ids = {sec["id"] for sec in result["sections"]}
    assert {"features", "multivariate", "drift"} <= ids, ids
    assert "stats" not in ids
    assert any("No group comparison" in f["text"] for f in result["flags"]), [f["text"] for f in result["flags"]]
    raw = pd.read_csv(root / "feature_table_raw.csv")
    print(f"nogroups: {len(raw)} features, sections {sorted(ids)}, {dt:.1f} s")
    assert len(raw) > 100


# ---- annotation study: real metabolites, MS/MS and libraries

ANNOT = {   # name → (formula, rt s, MS2 fragments or None)
    "L-Tryptophan": ("C11H12N2O2", 210.0, [188.0706, 146.0600, 118.0651, 91.0542]),
    "L-Phenylalanine": ("C9H11NO2", 160.0, [120.0808, 103.0542, 91.0542, 77.0386]),
    "Hypoxanthine": ("C5H4N4O", 75.0, None),
    "L-Carnitine": ("C7H15NO3", 50.0, None),
    "LPC 16:0": ("C24H50NO7P", 480.0, None),
    "Creatinine": ("C4H7N3O", 60.0, None),
    "Glutathione": ("C10H17N3O6S", 95.0, None),
}


def _annot_study(d):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    rows = []
    layout = [("blank", ""), ("qc", ""), ("sample", "ctrl"), ("sample", "case"), ("sample", "ctrl"), ("qc", ""),
              ("sample", "case"), ("sample", "ctrl"), ("sample", "case"), ("qc", "")]
    nq = ns = 0
    for inj, (typ, g) in enumerate(layout):
        ions, ms2 = [], []
        if typ != "blank":
            for name, (formula, rt, frags) in ANNOT.items():
                M = chem.formula_mass(formula)
                h = 2e6 * (4 if (name == "L-Tryptophan" and g == "case") else 1) * rng.lognormal(0, 0.1)
                nc = chem.parse_formula(formula).get("C", 6)
                ions += simulate.compound_ions(M, rt + rng.normal(0, 1), h, sigma=3.0, na_frac=0.3, n_carbon=nc)
                if frags:
                    inten = [100, 60, 35, 20] * rng.lognormal(0, 0.1, 4)
                    ms2.append({"rt": rt + 0.25, "precursor_mz": M + chem.PROTON, "charge": 1, "mz": frags, "intensity": inten})
        if typ == "qc":
            nq += 1; name = f"QC_{nq:02d}"
        elif typ == "blank":
            name = "Blank_01"
        else:
            ns += 1; name = f"{g.capitalize()}_{ns:02d}"
        simulate.write_lcms(d / f"{name}.mzML", ions, rt_max=600, seed=100 + inj, ms2=ms2)
        rows.append({"file": f"{name}.mzML", "sample": name, "group": g, "type": typ, "order": inj + 1, "batch": 1})
    pd.DataFrame(rows).to_csv(d / "sample_sheet.csv", index=False)
    # spectral library: the two true spectra (slightly different intensities) and a decoy at a different precursor
    msp = []
    for name in ("L-Tryptophan", "L-Phenylalanine"):
        formula, rt, frags = ANNOT[name]
        pm = chem.formula_mass(formula) + chem.PROTON
        msp.append(f"Name: {name}\nPrecursor_type: [M+H]+\nIon_mode: P\nPrecursorMZ: {pm:.5f}\nNum Peaks: {len(frags)}\n"
                   + "\n".join(f"{m:.4f} {v}" for m, v in zip(frags, (100, 50, 40, 15))) + "\n")
    msp.append("Name: Decoy compound\nPrecursorMZ: 205.0972\nIon_mode: P\nNum Peaks: 3\n50.0 100\n60.0 80\n70.0 10\n")
    (d / "library.msp").write_text("\n".join(msp))
    # user RT library: two right RTs, one deliberately wrong RT (must not be confirmed)
    pd.DataFrame({"name": ["Hypoxanthine", "L-Carnitine", "Creatinine"], "formula": ["C5H4N4O", "C7H15NO3", "C4H7N3O"],
                  "rt_min": [75 / 60, 50 / 60, 5.0]}).to_csv(d / "my_standards.csv", index=False)


def test_annotation():
    d = DATA / "annot"
    if not (d / "library.msp").exists():
        _annot_study(d)
    files = sorted(d.glob("*.mzML"))
    root, result, dt = _run("untargeted_annot", files, _sheet(d, files), msp_library=str(d / "library.msp"),
                            library=str(d / "my_standards.csv"))
    raw = pd.read_csv(root / "feature_table_raw.csv")
    ok = {}
    for name, (formula, rt, frags) in ANNOT.items():
        mh = chem.formula_mass(formula) + chem.PROTON
        f = raw[(np.abs(raw.mz - mh) / mh * 1e6 < 10) & (np.abs(raw.rt_min * 60 - rt) < 20)]
        assert len(f), f"{name} [M+H]+ not detected"
        r = f.iloc[0]
        ok[name] = (str(r.annotation), str(r.msi_level), str(r.evidence), str(r.compound_group), r.neutral_mass)
    print("annot:", json.dumps({k: v[:3] for k, v in ok.items()}, indent=1))
    for name in ("L-Tryptophan", "L-Phenylalanine"):
        assert name in ok[name][0] and ok[name][1] == "2" and "cosine" in ok[name][2], ok[name]
    for name in ("Hypoxanthine", "L-Carnitine"):
        assert name in ok[name][0] and ok[name][1] == "2" and "RT" in ok[name][2], ok[name]
    assert "Creatinine" in ok["Creatinine"][0] and ok["Creatinine"][1] == "3", ok["Creatinine"]   # wrong library RT → m/z only
    assert "LPC 16:0" in ok["LPC 16:0"][0] and ok["LPC 16:0"][1] == "3", ok["LPC 16:0"]
    # [M+H]+ and [M+Na]+ of LPC 16:0 grouped with the right neutral mass
    M = chem.formula_mass("C24H50NO7P")
    assert ok["LPC 16:0"][3] not in ("", "nan") and abs(ok["LPC 16:0"][4] - M) / M * 1e6 < 5, ok["LPC 16:0"]
    # MS/MS cosine sanity
    a = ANNOT["L-Tryptophan"][2]
    c, n = untargeted.cosine(a, [100, 60, 35, 20], a, [100, 60, 35, 20])
    assert abs(c - 1) < 1e-9 and n == 4
    c2, _ = untargeted.cosine(a, [100, 60, 35, 20], [50.0, 60.0], [1, 1])
    assert c2 == 0
    st = pd.read_csv(root / "statistics.csv")
    trp = st[st.annotation.astype(str).str.contains("Tryptophan")]
    print("annot: tryptophan log2FC", trp.log2fc.round(2).tolist())
    assert (trp.log2fc > 1.4).any(), trp


def _neg_study(d, n_comp=60):
    """Negative mode, three groups, two batches (3 QCs each) with a 2-fold sensitivity step between them."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    mass = np.sort(rng.uniform(120, 700, n_comp))
    rt = rng.uniform(40, 540, n_comp)
    base = 10 ** rng.uniform(4.5, 6.5, n_comp)
    eff = {"A": np.zeros(n_comp), "B": np.zeros(n_comp), "C": np.zeros(n_comp)}
    eff["B"][:10] = 1.5
    eff["C"][10:20] = -1.5
    rows = []
    inj = 0
    for batch, scale in ((1, 1.0), (2, 0.5)):
        layout = ["qc", "A", "B", "C", "qc", "A", "B", "C", "qc"] if batch == 1 else ["qc", "C", "B", "A", "qc", "C", "A", "B", "qc"]
        for k, g in enumerate(layout):
            inj += 1
            name = f"QC_b{batch}_{k}" if g == "qc" else f"{g}_b{batch}_{k}"
            h = base * scale * (2 ** (sum(eff.values()) / 3) if g == "qc" else 2 ** eff[g] * rng.lognormal(0, 0.15, n_comp))
            ions = []
            for c in range(n_comp):
                ions += simulate.compound_ions(mass[c], rt[c] + rng.normal(0, 1.5), h[c], sigma=3.0, polarity=-1)
            simulate.write_lcms(d / f"{name}.mzML", ions, rt_max=600, polarity=-1, seed=500 + inj)
            rows.append({"file": f"{name}.mzML", "sample": name, "group": "" if g == "qc" else g,
                         "type": "qc" if g == "qc" else "sample", "order": inj, "batch": batch})
    pd.DataFrame(rows).to_csv(d / "sample_sheet.csv", index=False)
    pd.DataFrame({"neutral_mass": mass, "rt": rt, "B": eff["B"], "C": eff["C"]}).to_csv(d / "truth.csv", index=False)


def test_negative_batches():
    d = DATA / "negative"
    if not (d / "truth.csv").exists():
        _neg_study(d)
    files = sorted(d.glob("*.mzML"))
    root, result, dt = _run("untargeted_negative", files, _sheet(d, files))
    truth = pd.read_csv(d / "truth.csv")
    raw = pd.read_csv(root / "feature_table_raw.csv")
    mh = truth.neutral_mass - chem.PROTON
    found = sum(((np.abs(raw.mz - m) / m * 1e6 < 10) & (np.abs(raw.rt_min * 60 - r) < 20)).any() for m, r in zip(mh, truth.rt))
    st = pd.read_csv(root / "statistics.csv")
    assert "p_anova" in st.columns
    pol = next(f for g in result["params_panel"] for f in g["fields"] if f["key"] == "polarity")
    # batch step: after correction the two batches' QCs should agree (median |log2 batch2/batch1| of QC means)
    k = raw[raw.kept.astype(bool)]
    q1 = [c for c in raw.columns if c.startswith("QC_b1") and not c.startswith("status_")]
    q2 = [c for c in raw.columns if c.startswith("QC_b2") and not c.startswith("status_")]
    proc = pd.read_csv(root / "feature_table_processed.csv")
    step_raw = float(np.nanmedian(np.log2(k[q2].mean(1) / k[q1].mean(1))))
    step_after = float(np.nanmedian(proc[[c for c in proc.columns if c.startswith("QC_b2")]].mean(1)
                                    - proc[[c for c in proc.columns if c.startswith("QC_b1")]].mean(1)))
    print(f"negative: {found}/{len(truth)} compounds, polarity {pol['resolved']}, ANOVA-significant "
          f"{int((st.padj_anova < 0.05).sum())}, batch step {step_raw:+.2f} → {step_after:+.2f} log2, {dt:.1f} s")
    assert pol["resolved"] == "negative"
    assert found / len(truth) >= 0.85
    assert abs(step_raw) > 0.7 and abs(step_after) < 0.15


if __name__ == "__main__":
    t0 = time.time()
    test_annotation()
    test_main()
    test_profile()
    test_nogroups()
    test_negative_batches()
    print(f"all untargeted tests passed in {time.time() - t0:.0f} s")
