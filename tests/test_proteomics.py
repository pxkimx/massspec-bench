"""End-to-end test of server/proteomics.py on the simulated DDA study (tests/make_dda.py), scored against the truth.

    "$HOME/Library/Application Support/MassSpecBench/venv/bin/python" tests/test_proteomics.py [--regen]

Scores printed:
  * identification: share of simulated peptides identified at 1% peptide FDR; empirical false discovery
    proportion from entrapment peptides and noise spectra (anything identified that was not simulated)
  * quantification: correlation of estimated vs true log2 fold changes; protein intensity vs true abundance
  * differential: recall of the changed proteins and the false discovery proportion at q < 0.05
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import common, proteomics  # noqa: E402

OUT = ROOT / "tests" / "_out"


def main(regen: bool = False, extra: dict | None = None, tag: str = "proteomics_sim"):
    data = OUT / "dda"
    if regen or not (data / "truth_proteins.csv").exists():
        from make_dda import dda_study
        dda_study(data)
    files = sorted(data.glob("*.mzML"))
    sheet = pd.read_csv(data / "sample_sheet.csv")
    rows = [{"file": str(data / r.file), "sample": r.sample, "group": r.group, "type": r.type, "batch": str(r.batch),
             "order": float(r.order)} for r in sheet.itertuples()]
    fasta = (data / "fasta.txt").read_text().strip()
    jdir = OUT / "jobs" / tag
    shutil.rmtree(jdir, ignore_errors=True)
    job = common.Job(jdir, "proteomics", "simulated DDA")
    t0 = time.time()
    common.run_safely(job, proteomics.run, files, {"sheet": rows, "fasta": fasta, **(extra or {})})
    dt = time.time() - t0
    st = json.loads((jdir / "status.json").read_text())
    print("status:", st["state"], st.get("error", ""), f"({dt:.1f} s)")
    if st["state"] != "done":
        print(st.get("trace", ""))
        sys.exit(1)
    res = json.loads((jdir / "result.json").read_text())
    return score(data, jdir, res, dt)


def score(data: Path, jdir: Path, res: dict, dt: float) -> dict:
    tp = pd.read_csv(data / "truth_peptides.csv")
    tprot = pd.read_csv(data / "truth_proteins.csv").set_index("protein")
    IL = lambda s: s.replace("I", "L")  # noqa: E731
    sim = set(tp.loc[~tp.entrap, "seq"].map(IL))
    ent = set(tp.loc[tp.entrap, "seq"].map(IL))
    psm = pd.read_csv(jdir / "psms.csv")
    ok = psm[(~psm.decoy) & (psm.peptide_q <= 0.01)]
    ids = set(ok.sequence.map(IL))
    found = ids & sim
    false = ids - sim
    print("\n== identification ==")
    print(f"simulated peptides: {len(sim)} ; identified at 1% peptide FDR: {len(ids)}")
    print(f"recall of simulated peptides: {len(found)}/{len(sim)} = {len(found)/len(sim):.1%}")
    print(f"identified but not simulated: {len(false)} (of which entrapment: {len(false & ent)}) → empirical FDP {len(false)/max(len(ids),1):.2%}")
    psm_ok = psm[(~psm.decoy) & (psm.spectrum_q <= 0.01)]
    print(f"PSMs at 1%: {len(psm_ok)}; empirical PSM FDP {(~psm_ok.sequence.map(IL).isin(sim)).mean():.2%}; "
          f"median |ppm| {psm_ok.ppm.abs().median():.2f}, median ppm {psm_ok.ppm.median():+.2f} (simulated offset +1.5)")
    s = res.get("summary", {})
    print("summary:", {k: v for k, v in s.items() if k != "timings_s"}, "timings:", s.get("timings_s"))

    print("\n== quantification ==")
    prot = pd.read_csv(jdir / "proteins.csv").set_index("protein")
    common_p = prot.index.intersection(tprot.index)
    print(f"true proteins: {len(tprot)} ; quantified: {len(prot)} (in truth: {len(common_p)}; not in truth: {len(prot.index.difference(tprot.index))})")
    A = [c for c in prot.columns if c.startswith("log2_A")]
    B = [c for c in prot.columns if c.startswith("log2_B")]
    est = prot.loc[common_p, B].mean(axis=1) - prot.loc[common_p, A].mean(axis=1)
    tru = tprot.loc[common_p, "log2fc"]
    m = est.notna() & np.isfinite(tru)   # on/off proteins have no finite true fold change
    if "on_off" in tprot:
        oo = tprot.index[tprot.on_off]
        print("on/off proteins (absent in A):", {p: ("quantified" if p in prot.index else "not quantified") for p in oo})
    r_fc = np.corrcoef(est[m], tru[m])[0, 1]
    print(f"log2FC estimated vs true: Pearson r = {r_fc:.3f}, median abs error {np.median(np.abs(est[m]-tru[m])):.3f} (n={m.sum()})")
    from scipy.stats import spearmanr
    a_est = prot.loc[common_p, A].mean(axis=1)
    rho = spearmanr(a_est, np.log2(tprot.loc[common_p, "abundance"]), nan_policy="omit")[0]
    print(f"protein intensity vs true abundance (group A): Spearman rho = {rho:.3f}")

    out = {"recall_peptides": len(found) / len(sim), "fdp_peptides": len(false) / max(len(ids), 1), "r_log2fc": r_fc,
           "rho_abundance": rho, "runtime_s": dt}
    dpath = jdir / "differential.csv"
    if dpath.exists():
        print("\n== differential abundance ==")
        de = pd.read_csv(dpath).set_index("protein")
        called = de.index[(de.qvalue < 0.05) & (de.log2fc.abs() >= 1)]
        called_q = de.index[de.qvalue < 0.05]
        truly = set(tprot.index[tprot.differential])
        tested_true = truly & set(de.index)
        tp_ = len(set(called) & truly)
        print(f"true changed proteins: {len(truly)} (tested: {len(tested_true)})")
        print(f"called (q<0.05 & |LFC|≥1): {len(called)}; true positives {tp_}; false {len(called) - tp_}; "
              f"recall {tp_/max(len(truly),1):.1%}; FDP {(len(called)-tp_)/max(len(called),1):.1%}")
        tpq = len(set(called_q) & truly)
        print(f"called (q<0.05 only): {len(called_q)}; FDP {(len(called_q)-tpq)/max(len(called_q),1):.1%}; recall {tpq/max(len(truly),1):.1%}")
        out.update(recall_de=tp_ / max(len(truly), 1), fdp_de=(len(called) - tp_) / max(len(called), 1))
    print("\nflags:")
    for f in res["flags"]:
        print(f"  [{f['level']}] {f['text'][:220]}")
    print("figures:", [it["id"] for sec in res["sections"] for it in sec["items"] if it["type"] == "figure"])
    print("\nscores:", {k: round(float(v), 3) for k, v in out.items()})
    return out


def tmt_test():
    """Simulated TMT6 plex (one file; channels A,A,A,B,B,B) — reporter-ion quantification path."""
    data = OUT / "dda_tmt"
    if not (data / "channel_groups.txt").exists():
        from make_dda import dda_study
        dda_study(data, tmt=True)
    jdir = OUT / "jobs" / "proteomics_tmt"
    shutil.rmtree(jdir, ignore_errors=True)
    job = common.Job(jdir, "proteomics", "simulated TMT6")
    t0 = time.time()
    common.run_safely(job, proteomics.run, sorted(data.glob("*.mzML")),
                      {"fasta": (data / "fasta.txt").read_text().strip(), "tmt": "tmt6", "tmt_level": 2,
                       "channel_groups": (data / "channel_groups.txt").read_text().strip()})
    st = json.loads((jdir / "status.json").read_text())
    print("status:", st["state"], st.get("error", ""), f"({time.time() - t0:.1f} s)")
    if st["state"] != "done":
        print(st.get("trace", ""))
        return
    res = json.loads((jdir / "result.json").read_text())
    print("summary:", res.get("summary"))
    tprot = pd.read_csv(data / "truth_proteins.csv").set_index("protein")
    prot = pd.read_csv(jdir / "proteins.csv").set_index("protein")
    cols = [c for c in prot.columns if c.startswith("log2_")]
    print("channels:", cols)
    cp = prot.index.intersection(tprot.index)
    est = prot.loc[cp, cols[3:]].mean(axis=1) - prot.loc[cp, cols[:3]].mean(axis=1)
    tru = tprot.loc[cp, "log2fc"]
    m = np.isfinite(tru) & est.notna()
    print(f"TMT log2FC vs truth: r = {np.corrcoef(est[m], tru[m])[0, 1]:.3f}, slope = {np.polyfit(tru[m], est[m], 1)[0]:.2f} (n={m.sum()})")
    if (jdir / "differential.csv").exists():
        de = pd.read_csv(jdir / "differential.csv").set_index("protein")
        called = set(de.index[(de.qvalue < 0.05) & (de.log2fc.abs() >= 1)])
        truly = set(tprot.index[tprot.differential])
        print(f"DA: called {len(called)}, true positives {len(called & truly)}, recall {len(called & truly)/len(truly):.1%}, FDP {len(called - truly)/max(len(called),1):.1%}")
    for f in res["flags"]:
        print(f"  [{f['level']}] {f['text'][:200]}")


def real_raw():
    """Real Thermo .raw: one gel fraction of human retina from the draft human proteome (PXD000561, Orbitrap Velos,
    69 MB), searched against the reviewed human proteome. Exercises vendor conversion (ThermoRawFileParser) → mzML."""
    import urllib.request
    d = OUT / "real"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "Adult_Retina_Gel_Velos_5_f30.raw"
    if not f.exists():
        urllib.request.urlretrieve("https://ftp.pride.ebi.ac.uk/pride/data/archive/2014/04/PXD000561/" + f.name, f)
    jdir = OUT / "jobs" / "proteomics_retina"
    shutil.rmtree(jdir, ignore_errors=True)
    job = common.Job(jdir, "proteomics", "Human retina gel fraction (PXD000561)")
    t0 = time.time()
    common.run_safely(job, proteomics.run, [f], {"organism": "human"})
    st = json.loads((jdir / "status.json").read_text())
    print("status:", st["state"], st.get("error", ""), f"({time.time() - t0:.1f} s)")
    if st["state"] != "done":
        print(st.get("trace", ""))
        return
    res = json.loads((jdir / "result.json").read_text())
    print("summary:", res.get("summary"))
    for fl in res["flags"]:
        print(f"  [{fl['level']}] {fl['text'][:300]}")
    for sec in res["sections"]:
        for it in sec["items"]:
            if it["type"] == "figure" and it["id"] in ("mass_error", "id_rate", "pep_props", "target_decoy", "ids_rt"):
                print(f"- {it['id']}: {it['yours']}")
    fr = [p for g in res["params_panel"] for p in g["fields"] if p["key"] in ("fragment_tol", "precursor_tol")]
    for p in fr:
        print(p["key"], p["value"], "|", p["why"])
    psm = pd.read_csv(jdir / "psms.csv")
    ok = psm[(~psm.decoy) & (psm.spectrum_q <= 0.01)]
    print("top genes by PSMs:", ok.gene.value_counts().head(12).to_dict())


def real():
    """Real data: OpenMS example BSA digest (3 LTQ-Orbitrap DDA runs, ~11-14 MB each) searched against BSA + the
    reviewed E. coli proteome as background. No known answer beyond "BSA should dominate"."""
    import urllib.request
    d = OUT / "real"
    d.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/OpenMS/OpenMS/develop/share/OpenMS/examples/BSA/"
    for i in (1, 2, 3):
        if not (d / f"BSA{i}.mzML").exists():
            urllib.request.urlretrieve(base + f"BSA{i}.mzML", d / f"BSA{i}.mzML")
    fa = d / "bsa_plus_ecoli.fasta"
    if not fa.exists():
        bsa = urllib.request.urlopen("https://rest.uniprot.org/uniprotkb/P02769.fasta").read().decode()
        fa.write_text(bsa + proteomics.fetch_proteome("ecoli").read_text())
    files = [d / f"BSA{i}.mzML" for i in (1, 2, 3)]
    jdir = OUT / "jobs" / "proteomics_bsa"
    shutil.rmtree(jdir, ignore_errors=True)
    job = common.Job(jdir, "proteomics", "BSA (OpenMS example)")
    t0 = time.time()
    common.run_safely(job, proteomics.run, files, {"fasta": str(fa)})
    st = json.loads((jdir / "status.json").read_text())
    print("status:", st["state"], st.get("error", ""), f"({time.time() - t0:.1f} s)")
    res = json.loads((jdir / "result.json").read_text())
    print("summary:", res.get("summary"))
    for f in res["flags"]:
        print(f"  [{f['level']}] {f['text'][:300]}")
    for sec in res["sections"]:
        for it in sec["items"]:
            if it["type"] == "figure" and it["id"] in ("mass_error", "id_rate", "pep_props", "target_decoy"):
                print(f"- {it['id']}: {it['yours']}")
    fr = [p for g in res["params_panel"] for p in g["fields"] if p["key"] == "fragment_tol"]
    print("fragment tolerance:", fr[0]["value"], "|", fr[0]["why"])
    prot = pd.read_csv(jdir / "proteins.csv")
    print(prot[["protein", "gene", "n_peptides"]].sort_values("n_peptides", ascending=False).head(8).to_string())


if __name__ == "__main__":
    if "--real" in sys.argv:
        real()
        sys.exit()
    if "--real-raw" in sys.argv:
        real_raw()
        sys.exit()
    if "--tmt" in sys.argv:
        tmt_test()
        sys.exit()
    print("########## default settings (match-between-runs on)")
    a = main(regen="--regen" in sys.argv)
    print("\n########## match-between-runs off (exercises missing values + imputation)")
    b = main(extra={"match_between_runs": False}, tag="proteomics_sim_nombr")
