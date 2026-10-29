"""Chemistry helpers: formula masses, adducts, isotope patterns, peptide masses and fragment ions.

Masses come from pyteomics (IUPAC monoisotopic masses), isotope distributions from OpenMS's
CoarseIsotopePatternGenerator, so nothing here is a hand-typed constant except the adduct table.
"""
from __future__ import annotations

import re

import numpy as np

PROTON = 1.007276467
ELECTRON = 0.000548580
C13_DIFF = 1.0033548378

# name, charge, number of M, mass shift added to n*M (Da). Shift = sum of adduct atoms − electrons.
ADDUCTS = {
    "[M+H]+": (1, 1, 1.007276467),
    "[M+NH4]+": (1, 1, 18.033823),
    "[M+Na]+": (1, 1, 22.989218),
    "[M+K]+": (1, 1, 38.963158),
    "[M+H-H2O]+": (1, 1, -17.003289),
    "[M+2H]2+": (2, 1, 2 * 1.007276467),
    "[2M+H]+": (1, 2, 1.007276467),
    "[2M+Na]+": (1, 2, 22.989218),
    "[M]+": (1, 1, -ELECTRON),
    "[M-H]-": (-1, 1, -1.007276467),
    "[M+Cl]-": (-1, 1, 34.969402),
    "[M+HCOO]-": (-1, 1, 44.998201),
    "[M+CH3COO]-": (-1, 1, 59.013851),
    "[M-H-H2O]-": (-1, 1, -19.01839),
    "[M-2H]2-": (-2, 1, -2 * 1.007276467),
    "[2M-H]-": (-1, 2, -1.007276467),
}
DEFAULT_ADDUCTS = {1: ["[M+H]+", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+H-H2O]+", "[M+2H]2+", "[2M+H]+"],
                   -1: ["[M-H]-", "[M+Cl]-", "[M+HCOO]-", "[M-H-H2O]-", "[M-2H]2-", "[2M-H]-"]}


def adduct_mz(neutral: float, adduct: str) -> float:
    z, n, shift = ADDUCTS[adduct]
    return (n * neutral + shift) / abs(z)


def neutral_from_mz(mz: float, adduct: str) -> float:
    z, n, shift = ADDUCTS[adduct]
    return (mz * abs(z) - shift) / n


def ppm(observed, theoretical):
    return (np.asarray(observed) - theoretical) / theoretical * 1e6


# ---------------------------------------------------------------- formulas

def parse_formula(formula: str) -> dict:
    """'C6H12O6' → {'C': 6, 'H': 12, 'O': 6}. Accepts isotopes like [13C]6 and parentheses."""
    f = formula.replace(" ", "")
    f = re.sub(r"\[(\d+)([A-Z][a-z]?)\]", r"\2[\1]", f)   # [13C] → C[13] (pyteomics notation)

    def expand(s):
        while "(" in s:
            s = re.sub(r"\(([^()]*)\)(\d*)", lambda m: _mul(m.group(1), int(m.group(2) or 1)), s)
        return s
    f = expand(f)
    out: dict[str, int] = {}
    for el, iso, n in re.findall(r"([A-Z][a-z]?)(\[\d+\])?(\d*)", f):
        key = el + (iso or "")
        out[key] = out.get(key, 0) + int(n or 1)
    if not out:
        raise ValueError(f"not a formula: {formula}")
    return out


def _mul(inner, k):
    return "".join(f"{el}{iso}{int(n or 1) * k}" for el, iso, n in re.findall(r"([A-Z][a-z]?)(\[\d+\])?(\d*)", inner))


def formula_mass(formula: str, average: bool = False) -> float:
    from pyteomics import mass
    comp = parse_formula(formula)
    return float(mass.calculate_mass(composition=comp, average=average))


def isotope_pattern(formula: str, n_peaks: int = 6, charge: int = 1, adduct: str | None = None):
    """Theoretical isotope envelope (m/z, relative abundance with the tallest = 100) for a formula/adduct."""
    import pyopenms as oms
    comp = parse_formula(formula)
    f = "".join(f"{k.split('[')[0]}{'(' + k.split('[')[1][:-1] + ')' if '[' in k else ''}{v}" for k, v in comp.items())
    ef = oms.EmpiricalFormula(f)
    gen = oms.CoarseIsotopePatternGenerator(n_peaks)
    iso = ef.getIsotopeDistribution(gen)
    mono = formula_mass(formula)
    ab = np.array([p.getIntensity() for p in iso.getContainer()], float)
    k = np.arange(len(ab))
    neutral = mono + k * C13_DIFF
    if adduct:
        mz = np.array([adduct_mz(m, adduct) for m in neutral])
    else:
        mz = (neutral + charge * PROTON) / abs(charge) if charge else neutral
    ab = ab / ab.max() * 100 if ab.max() > 0 else ab
    return mz, ab


# ---------------------------------------------------------------- peptides

def peptide_mass(seq: str, charge: int = 0) -> float:
    """Monoisotopic mass (charge 0) or m/z. Modifications in ProForma-ish brackets: PEPM[+15.995]TIDE."""
    from pyteomics import mass
    plain, delta = _strip_mods(seq)
    m = mass.fast_mass(plain) + delta
    return (m + charge * PROTON) / charge if charge else m


def _strip_mods(seq: str):
    delta = sum(float(x) for x in re.findall(r"\[([+-]?\d+\.?\d*)\]", seq))
    return re.sub(r"\[[^\]]*\]", "", seq).upper().replace("C", "C"), delta


def fragment_ions(seq: str, charges=(1,), types=("b", "y"), fixed_cam: bool = True):
    """b/y (and a/c/z) fragment m/z for a peptide. Carbamidomethyl C (+57.021) when fixed_cam.

    Returns a list of dicts {ion: 'b3', z, mz} ordered by ion type then position.
    """
    from pyteomics import mass
    mods = _per_residue_mods(seq)
    plain = re.sub(r"\[[^\]]*\]", "", seq).upper()
    aa = mass.std_aa_mass
    res = np.array([aa[a] + (57.021464 if (fixed_cam and a == "C") else 0.0) + mods.get(i, 0.0) for i, a in enumerate(plain)])
    H2O = 18.0105646863
    NH3 = 17.0265491
    CO = 27.9949146
    out = []
    n = len(plain)
    for t in types:
        for i in range(1, n):
            if t in ("a", "b", "c"):
                m = res[:i].sum()
                m += {"a": -CO, "b": 0.0, "c": NH3}[t]
                label = f"{t}{i}"
            else:
                m = res[n - i:].sum() + H2O
                m += {"x": CO - 2 * 1.007825032, "y": 0.0, "z": -NH3 + 1.007825032}[t]
                label = f"{t}{i}"
            for z in charges:
                out.append({"ion": label, "z": z, "mz": (m + z * PROTON) / z})
    return out


def _per_residue_mods(seq: str) -> dict:
    out = {}
    pos = -1
    for tok in re.findall(r"[A-Za-z]|\[[^\]]*\]", seq):
        if tok.startswith("["):
            try:
                out[max(pos, 0)] = out.get(max(pos, 0), 0.0) + float(tok[1:-1])
            except ValueError:
                pass
        else:
            pos += 1
    return out


def annotate_spectrum(mz, intensity, seq: str, charges=(1, 2), tol_ppm: float = 20.0, types=("b", "y")):
    """Match observed fragment peaks to a peptide's theoretical ions (closest peak within tolerance)."""
    mz = np.asarray(mz, float)
    theo = fragment_ions(seq, charges=charges, types=types)
    hits = []
    if not len(mz):
        return hits, theo
    for t in theo:
        k = int(np.argmin(np.abs(mz - t["mz"])))
        e = (mz[k] - t["mz"]) / t["mz"] * 1e6
        if abs(e) <= tol_ppm:
            hits.append({**t, "obs_mz": float(mz[k]), "intensity": float(intensity[k]), "ppm": float(e), "peak": k})
    return hits, theo


# ---------------------------------------------------------------- charge-state deconvolution

def deconvolute(mz, intensity, z_range=(5, 60), mass_range=(5000, 80000), step: float = 1.0, tol_ppm: float = 50.0,
                polarity: int = 1):
    """Zero-charge mass spectrum from a multiply-charged envelope (intact proteins, antibodies).

    For every candidate neutral mass M on a grid, sum the signal found at (M + z·H⁺)/z across the charge
    range, and keep only masses supported by at least three consecutive charge states — a single
    peak can fit many (M, z) pairs by chance, a ladder of charges cannot. This is the same idea as the
    scoring step in UniDec/MaxEnt-style tools, without their iterative deconvolution of overlaps.
    """
    mz = np.asarray(mz, float)
    it = np.asarray(intensity, float)
    if not len(mz):
        return np.array([]), np.array([])
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    cum = np.concatenate([[0], np.cumsum(it)])
    masses = np.arange(mass_range[0], mass_range[1], step)
    zs = np.arange(z_range[0], z_range[1] + 1)
    sig = np.zeros((len(zs), len(masses)))
    for k, z in enumerate(zs):
        target = (masses + polarity * z * PROTON) / z
        tol = target * tol_ppm * 1e-6
        lo = np.searchsorted(mz, target - tol)
        hi = np.searchsorted(mz, target + tol, side="right")
        sig[k] = cum[hi] - cum[lo]
    present = sig > 0
    # consecutive-charge support: at least 3 neighbouring charge states carry signal
    run3 = present[:-2] & present[1:-1] & present[2:]
    support = run3.any(0)
    score = np.where(support, sig.sum(0), 0.0)
    return masses, score


def pick_masses(masses, score, min_rel: float = 0.05, min_sep: float = 50.0, top: int = 20):
    """Local maxima of a deconvoluted spectrum, at least min_sep Da apart (removes harmonic ghosts nearby)."""
    if not len(score) or score.max() <= 0:
        return []
    from scipy.signal import find_peaks
    pk, _ = find_peaks(score, height=score.max() * min_rel, distance=max(1, int(min_sep / max(masses[1] - masses[0], 1e-9))))
    pk = pk[np.argsort(score[pk])[::-1]][:top]
    return [{"mass": float(masses[p]), "score": float(score[p]), "rel": float(score[p] / score.max() * 100)} for p in pk]
