"""Simulate a label-free DDA proteomics experiment with a known answer (for tests/test_proteomics.py).

Two groups x three replicates. ~80 real E. coli proteins are "in the sample"; ~15 % of them change 2-4x
between groups. Their tryptic peptides get realistic MS1 isotope envelopes (from the elemental formula),
retention times from a hydrophobicity model, per-peptide ionisation efficiencies, and MS2 spectra with b/y
ions, missing fragments and noise peaks. DDA sampling is intensity-dependent, so weak peptides are sometimes
not fragmented (the source of missing values in real data).

Three kinds of spectra carry no identifiable peptide, so the identification rate and the FDR are testable:
  * entrapment peptides — shuffled sequences that are NOT in the database (any ID of one is a false positive)
  * pure noise MS2 spectra
The search database is the whole reviewed E. coli proteome (~4,400 proteins), so the search space is realistic.

Writes into `out`: A_1..3.mzML, B_1..3.mzML, sample_sheet.csv, truth_proteins.csv, truth_peptides.csv and the
FASTA path in fasta.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import chem, simulate  # noqa: E402
from server.proteomics import fetch_proteome, read_fasta  # noqa: E402

PROTON = 1.007276467
CAM = 57.021464
OX = 15.994915
TMT6 = 229.162932
TMT6_REPORTERS = [126.127726, 127.124761, 128.134436, 129.131471, 130.141145, 131.138180]


def tryptic(seq: str, missed: int = 1):
    """Tryptic peptides (cleave after K/R, not before P) with up to `missed` missed cleavages."""
    sites = [i + 1 for i in range(len(seq) - 1) if seq[i] in "KR" and seq[i + 1] != "P"]
    b = [0] + sites + [len(seq)]
    out = []
    for i in range(len(b) - 1):
        for m in range(missed + 1):
            if i + 1 + m < len(b):
                out.append((seq[b[i]:b[i + 1 + m]], m))
    return out


def oms_seq(pep: str, ox: bool) -> str:
    s = pep.replace("C", "C(Carbamidomethyl)")
    if ox:
        k = pep.index("M")
        # the first M is oxidised; rebuild with the tag after that residue
        parts = []
        for i, a in enumerate(pep):
            parts.append(a + ("(Carbamidomethyl)" if a == "C" else "") + ("(Oxidation)" if (a == "M" and i == k) else ""))
        s = "".join(parts)
    return s


def chem_seq(pep: str, ox: bool, tmt: bool = False) -> str:
    """Sequence in chem.fragment_ions notation (fixed CAM is added by chem; oxidation/TMT as bracket deltas)."""
    s = pep
    if ox:
        k = pep.index("M")
        s = pep[:k + 1] + f"[+{OX}]" + pep[k + 1:]
    if tmt:
        s = f"[+{TMT6}]" + s.replace("K", f"K[+{TMT6}]")
    return s


def isotopes(pep: str, ox: bool, n: int = 4):
    import pyopenms as oms
    s = oms.AASequence.fromString(oms_seq(pep, ox))
    d = s.getFormula().getIsotopeDistribution(oms.CoarseIsotopePatternGenerator(n))
    ab = np.array([p.getIntensity() for p in d.getContainer()], float)
    return s.getMonoWeight(), ab / ab.max()


def dda_study(out: Path, n_proteins: int = 80, frac_diff: float = 0.15, n_per_group: int = 3, n_entrap: int = 150,
              n_noise_ms2: int = 1200, n_onoff: int = 3, tmt: bool = False, rt_max: float = 2400.0, seed: int = 11, ppm_offset: float = 1.5) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fasta = fetch_proteome("ecoli")
    prots = read_fasta(fasta)
    # every peptide in the database (I and L are isobaric, so compare with I→L) — to keep simulated peptides unique
    counts: dict[str, int] = {}
    for p in prots:
        for pep, _ in tryptic(p["seq"], 2):
            k = pep.replace("I", "L")
            counts[k] = counts.get(k, 0) + 1

    def good(pep):
        return 7 <= len(pep) <= 25 and not set(pep) & set("XUBZO") and counts.get(pep.replace("I", "L"), 0) == 1

    cand = [p for p in prots if 150 <= len(p["seq"]) <= 900]
    order = rng.permutation(len(cand))
    chosen = []
    for i in order:
        p = cand[i]
        peps = [(s, m) for s, m in tryptic(p["seq"], 1) if good(s) and (m == 0 or rng.random() < 0.12)]
        peps = list({s: m for s, m in peps}.items())
        if len(peps) >= 4:
            rng.shuffle(peps)
            chosen.append((p, peps[: int(rng.integers(4, 12))]))
        if len(chosen) >= n_proteins:
            break

    n = len(chosen)
    abund = 10 ** rng.uniform(4.6, 7.4, n)
    lfc = np.zeros(n)
    diff = rng.choice(n, int(round(frac_diff * n)), replace=False)
    lfc[diff] = rng.choice([-1, 1], len(diff)) * rng.uniform(1.0, 2.0, len(diff))
    # a few proteins only present in group B (switched on) — the case imputation exists for
    onoff = np.zeros(n, bool)
    onoff[rng.choice(np.setdiff1d(np.arange(n), diff), n_onoff, replace=False)] = True
    fac = {"A": np.where(onoff, 0.0, 1.0), "B": np.where(onoff, 1.0, 2.0 ** lfc)}

    from pyteomics import achrom
    peptides = []
    for pi, (p, peps) in enumerate(chosen):
        for s, m in peps:
            ox = "M" in s and rng.random() < 0.15
            peptides.append({"protein": p["id"], "pi": pi, "seq": s, "missed": m, "ox": ox, "entrap": False,
                             "eff": rng.lognormal(0, 0.7)})
    # entrapment: shuffled real peptides (C-terminal K/R kept) that are absent from the database
    pool = [q["seq"] for q in peptides]
    k = 0
    while k < n_entrap:
        s = pool[int(rng.integers(len(pool)))]
        body = list(s[:-1])
        rng.shuffle(body)
        t = "".join(body) + s[-1]
        if t.replace("I", "L") in counts or "P" == t[1:2]:
            continue
        peptides.append({"protein": "ENTRAPMENT", "pi": -1, "seq": t, "missed": 0, "ox": False, "entrap": True,
                         "eff": rng.lognormal(0, 0.7)})
        k += 1
    hyd = np.array([achrom.calculate_RT(q["seq"], achrom.RCs_guo_ph2_0) for q in peptides])
    hyd = (hyd - np.percentile(hyd, 1)) / (np.percentile(hyd, 99) - np.percentile(hyd, 1))
    rt = 240 + np.clip(hyd, -0.02, 1.02) * (rt_max - 540)
    for q, r in zip(peptides, rt):
        mono, iso = isotopes(q["seq"], q["ox"])
        if tmt:
            mono += TMT6 * (q["seq"].count("K") + 1)
        q["tmt"] = tmt
        q.update(rt=float(r), mono=mono, iso=iso, sigma=float(rng.uniform(4, 7)))
        nb = sum(q["seq"][:-1].count(a) for a in "KRH")
        z1 = 3 if (len(q["seq"]) > 16 or nb >= 1) and rng.random() < 0.6 else 2
        q["charges"] = [(z1, 1.0)] + ([(5 - z1, 0.3)] if rng.random() < 0.35 else [])
        q["frags"] = None

    groups = ["A"] * n_per_group + ["B"] * n_per_group
    samples = [f"{g}_{i % n_per_group + 1}" for i, g in enumerate(groups)]
    rows = []
    if tmt:  # one plex: the six channels are the samples, carried by reporter ions in every MS2 spectrum
        assert len(groups) == 6, "TMT6 simulation needs 2 x 3 samples"
        ch_noise = rng.lognormal(0, 0.12, (6, n))
        ch_load = rng.lognormal(0, 0.1, 6)
        ch_fac = np.array([fac[g] for g in groups]) * ch_noise * ch_load[:, None]    # channels x proteins
        loop = [("plex", "TMT6_plex1")]
    else:
        loop = list(zip(groups, samples))
    for si, (g, name) in enumerate(loop):
        load = rng.lognormal(0, 0.12)
        prot_noise = rng.lognormal(0, 0.12, n)
        shift = rng.normal(0, 6)
        stretch = 1 + rng.normal(0, 0.004)
        ions, ms2 = [], []
        for q in peptides:
            if q["entrap"]:
                h0 = 3e5 * q["eff"]
            else:
                pi = q["pi"]
                h0 = abund[pi] * (ch_fac[:, pi].mean() if tmt else fac[g][pi] * prot_noise[pi]) * q["eff"] * 0.4
                if h0 <= 0:
                    continue
            h0 *= load * rng.lognormal(0, 0.1)
            t0 = q["rt"] * stretch + shift + rng.normal(0, 1.5)
            for z, frac in q["charges"]:
                h = h0 * frac
                for kk, a in enumerate(q["iso"]):
                    ions.append(((q["mono"] + kk * chem.C13_DIFF + z * PROTON) / z, t0, q["sigma"], h * a))
                # DDA: the chance of being picked for MS2 rises with intensity
                p_pick = 1 / (1 + (6e4 / max(h, 1)) ** 1.5)
                n_ms2 = int(rng.random() < p_pick) + int(rng.random() < p_pick * 0.4)
                for _ in range(n_ms2):
                    tt = t0 + rng.normal(0, q["sigma"] * 0.6)
                    sp = _ms2(q, z, tt, h, rng, ppm_offset, keep=rng.uniform(0.2, 0.85))
                    if tmt:
                        share = np.ones(6) / 6 if q["entrap"] else ch_fac[:, q["pi"]] / ch_fac[:, q["pi"]].sum()
                        rep_it = h * 0.03 * share * rng.lognormal(0, 0.05, 6)
                        m = np.concatenate([np.array(TMT6_REPORTERS) * (1 + rng.normal(0, 2e-6, 6)), sp["mz"]])
                        v = np.concatenate([rep_it, sp["intensity"]])
                        o = np.argsort(m)
                        sp["mz"], sp["intensity"] = m[o], v[o]
                    ms2.append(sp)
        for _ in range(n_noise_ms2):
            pm = rng.uniform(400, 1300)
            mz = np.sort(rng.uniform(150, min(2000, pm * 2), int(rng.integers(20, 80))))
            ms2.append({"rt": float(rng.uniform(200, rt_max - 100)), "precursor_mz": pm, "charge": int(rng.choice([2, 3])),
                        "mz": mz, "intensity": rng.exponential(2e3, len(mz))})
        fn = out / f"{name}.mzML"
        simulate.write_lcms(fn, ions, rt_max=rt_max, scan_dt=1.0, ppm_err=2.0, noise_peaks=120, noise_level=3e3,
                            seed=seed * 100 + si, ms2=ms2)
        rows.append({"file": fn.name, "sample": name, "group": "" if tmt else g, "type": "sample", "batch": 1, "order": si + 1})
        print(f"wrote {fn.name}: {len(ms2)} MS2 spectra", flush=True)

    pd.DataFrame(rows).to_csv(out / "sample_sheet.csv", index=False)
    gene = {p["id"]: p["gene"] for p in prots}
    pd.DataFrame({"protein": [c[0]["id"] for c in chosen], "gene": [gene[c[0]["id"]] for c in chosen],
                  "log2fc": np.where(onoff, np.inf, lfc), "differential": (lfc != 0) | onoff, "on_off": onoff, "abundance": abund,
                  "n_peptides": [len(c[1]) for c in chosen]}).to_csv(out / "truth_proteins.csv", index=False)
    pd.DataFrame([{k: v for k, v in q.items() if k in ("protein", "seq", "missed", "ox", "entrap", "rt")} for q in peptides]
                 ).to_csv(out / "truth_peptides.csv", index=False)
    (out / "fasta.txt").write_text(str(fasta))
    if tmt:
        (out / "channel_groups.txt").write_text(",".join(groups))
    return {"files": [out / r["file"] for r in rows], "sheet": out / "sample_sheet.csv", "fasta": fasta,
            "truth_proteins": out / "truth_proteins.csv", "truth_peptides": out / "truth_peptides.csv"}


def _ms2(q, z, rt, h, rng, ppm_offset, keep=0.72):
    """One fragment spectrum: b/y ions (1+, and 2+ for 3+ precursors), dropouts, intensity scatter, noise peaks."""
    if q["frags"] is None:
        q["frags"] = {}
    key = z
    if key not in q["frags"]:
        fr = chem.fragment_ions(chem_seq(q["seq"], q["ox"], q.get("tmt", False)), charges=(1, 2) if z >= 3 else (1,), types=("b", "y"))
        q["frags"][key] = fr
    fr = q["frags"][key]
    mz, it = [], []
    L = len(q["seq"])
    for f in fr:
        if not (150 <= f["mz"] <= 2000):
            continue
        idx = int(f["ion"][1:])
        base = 1.0 if f["ion"][0] == "y" else 0.45
        if f["z"] == 2:
            base *= 0.4
        # middle fragments are the strongest; very short/long ones weaker
        base *= 0.4 + np.sin(np.pi * idx / L)
        if rng.random() < keep:
            v = base * rng.lognormal(0, 0.6)
            m0 = f["mz"] * (1 + rng.normal(0, 3e-6))
            mz.append(m0)
            it.append(v)
            # 13C isotope peak of the fragment (resolved at Orbitrap/TOF resolution)
            r1 = f["mz"] * f["z"] * 0.00055
            mz.append(m0 + chem.C13_DIFF / f["z"] * (1 + rng.normal(0, 3e-6)))
            it.append(v * r1 * rng.lognormal(0, 0.15))
    it = np.asarray(it) * h * 0.05
    nn = int(rng.integers(25, 60))
    pm_neutral = q["mono"]
    nmz = rng.uniform(150, min(2000, pm_neutral), nn)
    nit = rng.exponential(np.median(it) * 0.15 if len(it) else 1e3, nn)
    allmz = np.concatenate([np.asarray(mz), nmz])
    allit = np.concatenate([it, nit])
    o = np.argsort(allmz)
    prec = (q["mono"] + z * PROTON) / z * (1 + (ppm_offset + rng.normal(0, 1.5)) * 1e-6)
    return {"rt": float(rt), "precursor_mz": float(prec), "charge": int(z), "mz": allmz[o], "intensity": allit[o]}


if __name__ == "__main__":
    tmt = "--tmt" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = dda_study(Path(args[0] if args else ROOT / "tests" / "_out" / ("dda_tmt" if tmt else "dda")), tmt=tmt)
    print("wrote", len(d["files"]), "files; FASTA", d["fasta"])
