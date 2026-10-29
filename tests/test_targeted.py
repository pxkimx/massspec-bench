"""End-to-end test of server/targeted.py on the synthetic studies from make_targeted.py.

Scores: R² per analyte, back-calculated accuracy of the unknowns against the true concentrations (goal ±15 %),
QC accuracy, carry-over detection, and that SRM stored as chromatograms and SRM stored as MS2 spectra give the
same concentrations. Also exercises targets_text, quadratic + 1/x² + auto-exclude, the no-standards path and
reintegrate().

Run: "$HOME/Library/Application Support/MassSpecBench/venv/bin/python" tests/test_targeted.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "_out"
os.environ.setdefault("MB_HOME", str(OUT / "mbhome"))      # keep the parse cache and jobs out of the real app folder
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from server import msio, targeted  # noqa: E402
from server.common import Job, run_safely  # noqa: E402

DATA = OUT / "targeted"


def _data():
    import make_targeted as mt  # noqa: PLC0415
    fs, srm = DATA / "fullscan", DATA / "srm"
    if not (fs / "truth.csv").exists():
        mt.fullscan_study(fs)
    if not (srm / "truth.csv").exists():
        mt.srm_study(srm)
    return fs, srm


def _run(name, files, sheet_csv, **params):
    root = OUT / "jobs" / name
    shutil.rmtree(root, ignore_errors=True)
    job = Job(root, "targeted", name)
    sheet = msio.match_sheet(files, msio.read_table(sheet_csv)) if sheet_csv else msio.match_sheet(files, None)
    run_safely(job, targeted.run, files, {"sheet": sheet, **params})
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "done", status
    result = json.loads((root / "result.json").read_text())
    return root, result


def _accuracy(root, truth_csv):
    res = pd.read_csv(root / "targeted_results.csv")
    truth = pd.read_csv(truth_csv, index_col=0)
    out = []
    for s, row in truth.iterrows():
        for a, v in row.items():
            r = res[(res["sample"] == s) & (res["target"] == a)]
            if not len(r):
                continue
            r = r.iloc[0]
            out.append({"sample": s, "target": a, "true": v, "measured": r["concentration"], "reportable": r["reportable"],
                        "acc": 100 * r["concentration"] / v})
    return pd.DataFrame(out)


def test_fullscan():
    fs, _ = _data()
    files = sorted(fs.glob("*.mzML"))
    root, result = _run("targeted_fullscan", files, fs / "sample_sheet.csv", targets=str(fs / "targets.csv"))
    cal = pd.read_csv(root / "calibration.csv")
    print("\n== Full-scan HRMS (7 analytes, 5 ISTDs, 8-level curve) ==")
    print(cal[["target", "istd", "r2", "lloq", "uloq", "lod", "loq", "n_included", "status"]].to_string(index=False))
    acc = _accuracy(root, fs / "truth.csv")
    inr = acc[~acc.reportable.isin(["<LLOQ", ">ULOQ", "n.d."])]
    print(f"\nUnknowns in range: {len(inr)}/{len(acc)}; |bias| median {np.median(np.abs(inr.acc - 100)):.1f} %, "
          f"max {np.max(np.abs(inr.acc - 100)):.1f} %; within ±15 %: {(np.abs(inr.acc - 100) <= 15).mean() * 100:.0f} %")
    print(inr.groupby("target").acc.agg(lambda a: f"mean {a.mean():.1f}%  range {a.min():.0f}–{a.max():.0f}%").to_string())
    oor = acc[acc.reportable.isin(["<LLOQ", ">ULOQ"])]
    print("Out-of-range calls:", ", ".join(f"{r.sample}/{r.target}={r.reportable} (true {r.true:.3g})" for r in oor.itertuples()))
    res = pd.read_csv(root / "targeted_results.csv")
    qc = res[(res.type == "qc") & (res.role == "analyte")]
    qacc = 100 * qc.concentration / qc.nominal
    print(f"QC accuracy: median {qacc.median():.1f} %, range {qacc.min():.1f}–{qacc.max():.1f} %, within ±15 %: {(np.abs(qacc - 100) <= 15).mean() * 100:.0f} %")
    mel = res[(res.target == "Melatonin") & (res.type == "standard")]
    print(f"Melatonin (no RT in list) found at {mel.rt_min.median():.2f} min (true 6.40)")
    flags = [f["text"] for f in result["flags"]]
    print("Flags:\n  " + "\n  ".join(flags))
    assert (cal.r2 > 0.99).all(), cal
    assert (np.abs(inr.acc - 100) <= 15).mean() >= 0.9
    assert set(zip(oor["sample"], oor.target, oor.reportable)) >= {("U01", "Caffeine", ">ULOQ"), ("U02", "Serotonin", "<LLOQ")}
    assert (np.abs(qacc - 100) <= 15).mean() >= 2 / 3
    assert abs(mel.rt_min.median() - 6.40) < 0.05
    assert any(w["type"] == "target_review" for s in result["sections"] for w in s["items"])
    # reintegrate one cell with the automatic boundaries: must reproduce the automatic area closely
    wr = next(w for s in result["sections"] for w in s["items"] if w["type"] == "target_review")["rows"]
    row = next(r for r in wr if r["target"] == "Caffeine" and r["sample"].startswith("QC_M"))
    run = msio.load_cached(files[row["file_index"]])
    out = targeted.reintegrate(run, row, row["start"] * 60, row["end"] * 60)
    print(f"reintegrate Caffeine/{row['sample']}: auto area {row['area']:.4g}, manual {out['area']:.4g}; "
          f"conc auto {row['concentration']:.4g}, manual {out['concentration']:.4g}")
    assert abs(out["area"] / row["area"] - 1) < 0.03
    return root


def test_srm():
    _, srm = _data()
    chrom = sorted((srm / "chrom").glob("*.mzML"))
    spec = sorted((srm / "spectra").glob("*.mzML"))
    root_c, res_c = _run("targeted_srm_chrom", chrom, srm / "sample_sheet.csv", targets=str(srm / "targets_srm.tsv"))
    # the same list pasted as text, on the SRM-as-spectra files
    root_s, res_s = _run("targeted_srm_spectra", spec, srm / "sample_sheet.csv", targets_text=(srm / "targets_srm.tsv").read_text())
    print("\n== SRM/MRM (4 analytes, 2 ISTDs, qualifier transitions) ==")
    for root, lab in ((root_c, "chromatograms"), (root_s, "MS2 spectra")):
        cal = pd.read_csv(root / "calibration.csv")
        acc = _accuracy(root, srm / "truth.csv")
        inr = acc[~acc.reportable.isin(["<LLOQ", ">ULOQ", "n.d."])]
        res = pd.read_csv(root / "targeted_results.csv")
        qc = res[(res.type == "qc") & (res.role == "analyte")]
        qacc = 100 * qc.concentration / qc.nominal
        mode = res[res.role == "analyte"].extraction.iloc[0]
        print(f"[{lab}: {mode}] R² {', '.join(f'{t} {r:.4f}' for t, r in zip(cal.target, cal.r2))}")
        print(f"   unknowns within ±15 %: {(np.abs(inr.acc - 100) <= 15).mean() * 100:.0f} % ({len(inr)}), median |bias| "
              f"{np.median(np.abs(inr.acc - 100)):.1f} %, max {np.max(np.abs(inr.acc - 100)):.1f} %; QC {qacc.min():.1f}–{qacc.max():.1f} %")
        assert (cal.r2 > 0.99).all()
        assert (np.abs(inr.acc - 100) <= 15).mean() >= 0.9
    a = pd.read_csv(root_c / "targeted_results.csv")
    b = pd.read_csv(root_s / "targeted_results.csv")
    m = a.merge(b, on=["sample", "target"], suffixes=("_c", "_s"))
    m = m[np.isfinite(m.concentration_c) & (m.concentration_c > 0)]
    d = np.abs(m.concentration_s / m.concentration_c - 1) * 100
    print(f"   chromatogram vs spectra concentrations: median diff {d.median():.2f} %, max {d.max():.2f} %")
    assert d.max() < 2
    fl = a[(a["sample"] == "SRM_S04") & (a.target == "Caffeine")]["flags"].iloc[0]
    print(f"   SRM_S04 caffeine (qualifier interference): flags = {fl}")
    assert "ion ratio" in str(fl)
    return root_c


def test_options():
    """Quadratic, 1/x², auto-exclude on the SRM set; then the full-scan unknowns alone (no standards)."""
    fs, srm = _data()
    chrom = sorted((srm / "chrom").glob("*.mzML"))
    root, result = _run("targeted_srm_quad", chrom, srm / "sample_sheet.csv", targets=str(srm / "targets_srm.tsv"),
                        model="quadratic", weighting="1/x²", auto_exclude=True, min_snr=5, smooth=7, ppm=5, rt_window=0.4)
    cal = pd.read_csv(root / "calibration.csv")
    print("\n== options: quadratic, 1/x², auto-exclude ==")
    print(cal[["target", "model", "weighting", "r2", "n_included", "n_standards", "excluded"]].to_string(index=False))
    assert (cal.model == "quadratic").all() and (cal.weighting == "1/x2").all()
    assert result["params"]["auto_exclude"] is True
    files = [f for f in sorted(fs.glob("*.mzML")) if f.name.startswith("U")]
    root, result = _run("targeted_nostd", files, fs / "sample_sheet.csv", targets=str(fs / "targets.csv"))
    gt = pd.read_csv(root / "group_tests.csv")
    print("\n== no standards: group comparison on ISTD-normalised responses ==")
    print(gt[["target", "group_a", "group_b", "log2fc", "q"]].round(4).to_string(index=False))
    sig = set(gt[gt.q < 0.05].target)
    # true effects: paraxanthine ×2, kynurenine ×1.8 (caffeine ×2.5 is masked by U01, a control spiked to 1500 ng/mL)
    assert {"Paraxanthine", "Kynurenine"} <= sig, sig
    assert not sig & {"Creatinine", "Tryptophan", "Melatonin"}, sig


if __name__ == "__main__":
    test_fullscan()
    test_srm()
    test_options()
    print("\nall targeted tests passed")
