"""Targeted quantification: known compounds, calibration curves, concentrations.

For every target in a target list (small molecules or peptides; full-scan m/z, PRM, or SRM/MRM transitions)
and every injection, the quantifier trace is extracted and one peak integrated with `peaks.integrate` — the
same integration the explorer uses — near the expected retention time. Qualifier ions are integrated inside
the quantifier's boundaries so their ratio can confirm identity. Responses (area, or area ÷ internal-standard
area) of the calibration standards are fitted with a weighted linear or quadratic curve, and every other
injection is back-calculated against it, with the acceptance criteria of the FDA 2018 Bioanalytical Method
Validation guidance / ICH M10 (±15 %, ±20 % at the LLOQ). Without standards the pipeline still reports
areas/ISTD-normalised responses and compares groups.

`reintegrate(run, target_row, start_s, end_s)` recomputes one peak between user-chosen boundaries for the
UI's manual-integration tool.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import chem, msio, peaks
from .common import (BLANK_COLOR, PALETTE, QC_COLOR, UserFacingError, commas, fmt_n, log_exc, versions)

TYPE_COLOR = {"standard": "#2a64b0", "qc": QC_COLOR, "blank": BLANK_COLOR, "sample": "#e8743b"}
TYPE_LABEL = {"standard": "calibration standard", "qc": "QC", "blank": "blank", "sample": "sample"}
SRM_TOL = 0.3            # Da: Q1/Q3 matching for triple-quad transitions (unit resolution)
PAGE = 12                # targets per figure page

# ------------------------------------------------------------------ target list

TARGET_ALIASES = {
    "name": ["name", "compound", "compound_name", "analyte", "analyte_name", "target", "target_name", "molecule",
             "molecule_name", "metabolite", "peptide", "peptide_sequence", "peptide_modified_sequence", "id",
             "compound_id", "component", "component_name"],
    "mz": ["mz", "m_z", "target_mz", "ion_mz", "exact_mz", "theoretical_mz", "mz_theoretical", "monoisotopic_mz",
           "mass_to_charge", "precursor_ion_mz", "ms1_mz", "quant_mz"],
    "mass": ["mass", "neutral_mass", "monoisotopic_mass", "exact_mass", "mono_mass", "mw_monoisotopic"],
    "formula": ["formula", "molecular_formula", "chemical_formula", "sum_formula", "elemental_formula",
                "neutral_formula", "molecule_formula"],
    "adduct": ["adduct", "adduct_type", "ion", "ion_type", "precursor_adduct", "precursor_type", "ionization",
               "ion_form", "species"],
    "q1": ["q1", "q1_mz", "precursor", "precursor_mz", "parent", "parent_mz", "parent_ion", "prec_mz", "q1_mass",
           "precursor_m_z"],
    "q3": ["q3", "q3_mz", "product", "product_mz", "fragment", "fragment_mz", "daughter", "daughter_ion",
           "product_ion", "q3_mass", "product_m_z", "transition_mz"],
    "rt": ["rt", "retention_time", "expected_rt", "explicit_retention_time", "rt_expected", "tr", "ret_time",
           "rt_min", "retention_time_min", "rt_s", "rt_sec", "retention_time_s", "retention_time_sec", "apex_rt"],
    "rt_window": ["rt_window", "window", "rt_tolerance", "rt_tol", "delta_rt", "rt_delta", "rt_win",
                  "explicit_retention_time_window", "retention_time_window", "rt_window_min", "rt_window_s",
                  "window_min"],
    "ppm": ["ppm", "mz_tol_ppm", "tolerance_ppm", "mass_tolerance", "mz_tolerance", "ppm_tolerance", "tol_ppm",
            "mass_tolerance_ppm", "mass_error_ppm"],
    "istd": ["istd", "internal_standard", "is", "is_name", "internal_std", "istd_name", "reference_is",
             "is_compound", "internal_standard_name", "istd_compound"],
    "is_istd": ["is_istd", "is_internal_standard", "istd_flag", "internal_standard_flag", "is_is", "isistd",
                "role", "compound_type", "compound_role", "standard_type", "category"],
    "units": ["units", "unit", "conc_units", "concentration_units", "concentration_unit"],
    "quantifier": ["quantifier", "quant", "quan", "is_quantifier", "quantitative", "quant_ion", "quantifier_ion",
                   "ion_role", "transition_type", "transition_role", "qual_quant"],
    "polarity": ["polarity", "ion_mode", "ionization_mode", "ion_polarity", "mode", "esi_mode"],
}


def _norm_col(c: str) -> str:
    n = str(c).strip().lower().replace("m/z", "mz")
    return re.sub(r"[^a-z0-9]+", "_", n).strip("_")


def _truthy(v) -> bool | None:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    s = str(v).strip().lower()
    if s in ("", "nan", "none"):
        return None
    if s in ("1", "true", "t", "yes", "y", "x", "1.0"):
        return True
    if s in ("0", "false", "f", "no", "n", "0.0"):
        return False
    return None


def _num(v):
    try:
        f = float(str(v).strip().replace(",", "."))
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def read_target_table(params: dict) -> pd.DataFrame:
    """The target list as a DataFrame, from params['targets'] (CSV/TSV/XLSX path) or params['targets_text']."""
    path, text = params.get("targets"), params.get("targets_text")
    if path and str(path).strip():
        p = Path(str(path).strip())
        if not p.exists():
            raise UserFacingError(f"The target list {p.name} was not found at {p}.")
        if p.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
            try:
                return pd.read_excel(p)
            except ImportError as e:
                raise UserFacingError("Reading Excel target lists needs the openpyxl package; save the list as CSV "
                                      "or paste it instead.") from e
        text = p.read_text(errors="replace")
    if not text or not str(text).strip():
        raise UserFacingError("No target list. Give a CSV/TSV/Excel file (or paste a table) with at least a name and "
                              "either m/z, formula (+ adduct), or Q1/Q3 for each compound.")
    txt = str(text).strip("﻿\n\r ")
    first = txt.splitlines()[0]
    counts = {"\t": first.count("\t"), ",": first.count(","), ";": first.count(";")}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        sep = r"\s{2,}|\s*\|\s*"
        return pd.read_csv(io.StringIO(txt), sep=sep, engine="python", dtype=str)
    return pd.read_csv(io.StringIO(txt), sep=sep, dtype=str, skipinitialspace=True)


def _canon_adduct(a: str | None, polarity: int) -> str | None:
    if a is None:
        return None
    s = str(a).strip().replace(" ", "").replace("−", "-")
    if not s or s.lower() == "nan":
        return None
    if s in chem.ADDUCTS:
        return s
    core = re.sub(r"^\[|\](\d*[+-])?$|(\d*[+-])$", "", s)
    core = core.strip("[]")
    for ch in (["+", "-"] if polarity >= 0 else ["-", "+"]) + ["2+", "2-"]:
        k = f"[{core}]{ch}"
        if k in chem.ADDUCTS:
            return k
    if core.upper() in ("MH", "M+H"):
        return "[M+H]+"
    if core.upper() in ("M-H",):
        return "[M-H]-"
    raise ValueError(f"unknown adduct '{a}' (known: {', '.join(chem.ADDUCTS)})")


def _formula_mass(f: str) -> float:
    # deuterium written as D (common in internal-standard names/formulas) → [2H], which pyteomics understands
    f = re.sub(r"D(?![a-z])", "[2H]", str(f).strip())
    return chem.formula_mass(f)


def parse_targets(df: pd.DataFrame, default_polarity: int = 1, units_default: str = "") -> tuple[list[dict], list[str]]:
    """Group target-list rows into compounds with a quantifier and optional qualifier transitions.

    Returns (compounds, notes). rt/rt_window stay in the list's units here (minutes unless the column
    name says seconds); run() converts to seconds after checking them against the data.
    """
    notes: list[str] = []
    low = {c: _norm_col(c) for c in df.columns}
    ren = {}
    for canon, al in TARGET_ALIASES.items():
        for c, l in low.items():
            if c in ren:
                continue
            if l in al or (canon == "rt" and re.fullmatch(r"(rt|retention_time)_?(min|minutes|s|sec|seconds)?", l)):
                ren[c] = canon
                break
    rt_seconds = any(canon == "rt" and re.search(r"(^|_)(s|sec|seconds)$", low[c]) for c, canon in ren.items())
    win_seconds = any(canon == "rt_window" and re.search(r"(^|_)(s|sec|seconds)$", low[c]) for c, canon in ren.items())
    df = df.rename(columns=ren)
    if "q1" in df.columns and "q3" not in df.columns and "mz" not in df.columns:
        df = df.rename(columns={"q1": "mz"})      # a lone "precursor m/z" column is an MS1 target m/z
    if not any(c in df.columns for c in ("mz", "formula", "q1", "mass")):
        raise UserFacingError("The target list needs an m/z column, a formula column, or Q1/Q3 columns. Columns found: "
                              + ", ".join(map(str, low)) + ".")
    get = lambda r, k: (r.get(k) if k in r.index else None)  # noqa: E731
    groups: dict[str, list] = {}
    order = []
    for i, r in df.iterrows():
        name = get(r, "name")
        name = str(name).strip() if name is not None and str(name).strip() not in ("", "nan") else None
        if name is None:
            name = str(get(r, "formula") or f"target_{i + 1}").strip()
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(r)
    compounds = []
    for name in order:
        rows = groups[name]
        qflags = [_quant_flag(get(r, "quantifier")) for r in rows]
        qi = next((k for k, q in enumerate(qflags) if q is True), 0)
        rows = [rows[qi]] + [r for k, r in enumerate(rows) if k != qi]

        def first(key):
            for r in rows:
                v = get(r, key)
                if v is not None and str(v).strip() not in ("", "nan", "None"):
                    return v
            return None
        pol_s = str(first("polarity") or "").strip().lower()
        pol = 1 if pol_s in ("+", "pos", "positive", "p", "1", "+1") else -1 if pol_s in ("-", "−", "neg", "negative", "n", "-1") else 0
        role = str(first("is_istd") or "").strip().lower()
        is_istd = _truthy(role)
        if is_istd is None and role:
            is_istd = bool(re.search(r"istd|internal|^is$|^i\.s\.", role))
        rt = _num(first("rt"))
        win = _num(first("rt_window"))
        c = {"name": name, "is_istd": bool(is_istd), "istd": (str(first("istd")).strip() if first("istd") is not None else None),
             "rt_list": rt, "rt_s": (rt if rt_seconds else rt * 60) if rt is not None else None,
             "win_list": win, "win_s": (win if win_seconds else win * 60) if win is not None else None,
             "ppm": _num(first("ppm")), "units": str(first("units") or units_default or "").strip(),
             "polarity": pol, "transitions": []}
        for k, r in enumerate(rows):
            try:
                tr = _row_transition(r, get, pol or default_polarity)
            except ValueError as e:
                notes.append(f"{name}: {e}")
                continue
            if tr is None:
                notes.append(f"{name}: a row has no m/z, formula or Q1/Q3 and was skipped.")
                continue
            tr["quant"] = (k == 0)
            c["transitions"].append(tr)
        if not c["transitions"]:
            notes.append(f"{name} has no usable m/z and was dropped.")
            continue
        if c["transitions"][0]["kind"] == "ms1" and c["polarity"] == 0 and c["transitions"][0].get("adduct"):
            c["polarity"] = 1 if chem.ADDUCTS[c["transitions"][0]["adduct"]][0] > 0 else -1
        compounds.append(c)
    # a compound other targets name as their ISTD is an ISTD even when not flagged
    lut = {c["name"].lower(): c["name"] for c in compounds}
    for c in compounds:
        if c["istd"] and c["istd"].lower() in ("nan", "none", "", "-"):
            c["istd"] = None
        if c["istd"]:
            key = lut.get(c["istd"].lower())
            if key is None:
                notes.append(f"{c['name']}: internal standard '{c['istd']}' is not in the target list, so it is quantified without one.")
                c["istd"] = None
            elif key == c["name"]:
                c["istd"] = None
            else:
                c["istd"] = key
    for c in compounds:
        if c["istd"]:
            next(x for x in compounds if x["name"] == c["istd"])["is_istd"] = True
    for c in compounds:
        if c["is_istd"]:
            c["istd"] = None
    return compounds, notes


def _quant_flag(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if "qual" in s:
        return False
    if "quan" in s:
        return True
    return _truthy(v)


def _row_transition(r, get, polarity):
    q1, q3 = _num(get(r, "q1")), _num(get(r, "q3"))
    if q1 is not None and q3 is not None:
        return {"kind": "transition", "q1": q1, "q3": q3, "label": f"{q1:.4g} → {q3:.4g}"}
    mz = _num(get(r, "mz"))
    adduct = _canon_adduct(get(r, "adduct"), polarity)
    if mz is not None:
        return {"kind": "ms1", "mz": mz, "label": f"m/z {mz:.4f}", "adduct": adduct}
    f = get(r, "formula")
    mass = _num(get(r, "mass"))
    if f is not None and str(f).strip() not in ("", "nan"):
        try:
            mass = _formula_mass(str(f))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"formula '{f}' could not be read ({e})") from e
    if mass is not None:
        a = adduct or ("[M+H]+" if polarity >= 0 else "[M-H]-")
        mzv = chem.adduct_mz(mass, a)
        return {"kind": "ms1", "mz": mzv, "label": f"{a} {mzv:.4f}", "adduct": a, "formula": f,
                "default_adduct": adduct is None}
    if q1 is not None:
        return {"kind": "ms1", "mz": q1, "label": f"m/z {q1:.4f}"}
    return None


# ------------------------------------------------------------------ extraction

def _run_polarity(run) -> int:
    p = run.polarity[run.polarity != 0]
    return int(np.sign(p.sum())) if len(p) and p.sum() != 0 else (int(p[0]) if len(p) else 0)


def trace(run, tr: dict, ppm: float, rt_range=None, polarity: int = 0):
    """(time s, intensity, mode, how) of one transition in one run.

    mode: 'ms1' full-scan XIC · 'srm' stored SRM/MRM chromatogram · 'srm-spectra' SRM written as MS2 spectra
    (unit-resolution product window) · 'prm' high-resolution MS2 product XIC · None when the file has no
    matching data.
    """
    if tr["kind"] == "ms1":
        pol = polarity if polarity and len(set(run.polarity[run.ms_level == 1].tolist()) - {0}) > 1 else None
        t, y = run.xic(tr["mz"], ppm, level=1, rt_range=rt_range, polarity=pol)
        if not len(t):
            return t, y, None, "no MS1 spectra"
        return t, y, "ms1", f"±{ppm:g} ppm"
    q1, q3 = tr["q1"], tr["q3"]
    cands = [c for c in run.chromatograms if abs(c.get("q1", np.nan) - q1) <= SRM_TOL and abs(c.get("q3", np.nan) - q3) <= SRM_TOL]
    if cands:
        if rt_range is not None and len(cands) > 1:
            mid = 0.5 * (rt_range[0] + rt_range[1])
            cands.sort(key=lambda c: (not (len(c["time"]) and c["time"][0] <= mid <= c["time"][-1]),
                                      abs(c["q1"] - q1) + abs(c["q3"] - q3)))
        else:
            cands.sort(key=lambda c: abs(c["q1"] - q1) + abs(c["q3"] - q3))
        c = cands[0]
        t, y = np.asarray(c["time"], float), np.asarray(c["intensity"], float)
        if rt_range is not None:
            m = (t >= rt_range[0]) & (t <= rt_range[1])
            t, y = t[m], y[m]
        return t, y, "srm", f"chromatogram '{c.get('id', '')}'"
    ms2 = run.idx(2)
    if len(ms2):
        pm = run.precursor_mz[ms2]
        sel = ms2[np.abs(pm - q1) <= SRM_TOL]
        if len(sel):
            npk = np.diff(run.offsets)[sel]
            if np.median(npk) <= 20:     # a handful of product m/z per scan: triple-quad SRM stored as spectra
                tol_ppm = SRM_TOL / q3 * 1e6
                t, y = run.xic(q3, tol_ppm, level=2, rt_range=rt_range, precursor=q1, precursor_tol=SRM_TOL)
                return t, y, "srm-spectra", f"MS2 of {q1:.4g}, product ±{SRM_TOL} Da"
            t, y = run.xic(q3, ppm, level=2, rt_range=rt_range, precursor=q1, precursor_tol=SRM_TOL)
            return t, y, "prm", f"MS2 of {q1:.4g}, product ±{ppm:g} ppm"
    return np.zeros(0), np.zeros(0), None, "no matching SRM chromatogram or MS2 precursor"


def _smooth3(y):
    if len(y) < 3:
        return np.asarray(y, float)
    return np.convolve(np.pad(np.asarray(y, float), 1, mode="edge"), np.ones(3) / 3, mode="valid")


def area_between(t, y, start_s: float, end_s: float) -> dict:
    """Area above a straight baseline joining the trace at two chosen times (manual / qualifier integration)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    if len(t) < 2 or end_s <= start_s:
        return {"area": 0.0, "height": 0.0, "rt": None, "start": start_s, "end": end_s, "points": 0, "base_start": 0.0, "base_end": 0.0}
    ys = _smooth3(y)
    b0, b1 = float(np.interp(start_s, t, ys)), float(np.interp(end_s, t, ys))
    inside = (t > start_s) & (t < end_s)
    st = np.concatenate([[start_s], t[inside], [end_s]])
    sy = np.concatenate([[np.interp(start_s, t, y)], y[inside], [np.interp(end_s, t, y)]])
    line = np.interp(st, [start_s, end_s], [b0, b1])
    net = np.clip(sy - line, 0, None)
    k = int(np.argmax(net))
    return {"area": float(np.trapezoid(net, st)), "height": float(net[k]), "rt": float(st[k]), "start": float(start_s),
            "end": float(end_s), "points": int(inside.sum() + 2), "base_start": b0, "base_end": b1}


def local_snr(t, y, pk) -> float:
    """Peak height above the local background ÷ background noise (robust SD) of the trace around the peak.

    The background is the trace within three peak widths either side of the integration boundaries. When the
    background is exactly flat (typical of high-resolution XICs, which are zero between peaks) the RMS of the
    few noise spikes is used, after clipping any neighbouring peak (> 3 RMS, iterated); a perfectly empty
    background gives the cap of 10 000.
    """
    y = np.asarray(y, float)
    lo, hi = pk["i_start"], pk["i_end"]
    w = max(hi - lo + 1, 5)
    idx = np.r_[max(0, lo - 3 * w):lo, hi + 1:min(len(y), hi + 1 + 3 * w)]
    bg = y[idx] if len(idx) >= 5 else y
    med = float(np.median(bg))
    sd = 1.4826 * float(np.median(np.abs(bg - med)))
    if sd <= 0:
        v = bg - med
        for _ in range(25):
            sd = float(np.sqrt(np.mean(v ** 2))) if len(v) else 0.0
            keep = np.abs(v) <= 3 * sd
            if sd <= 0 or keep.all():
                break
            v = v[keep]
    h = float(np.max(y[lo:hi + 1]) - med) if hi >= lo else 0.0
    if sd <= 0:
        return 1e4 if h > 0 else 0.0
    return float(min(max(h / sd, 0.0), 1e4))


def reintegrate(run, target_row: dict, start_s: float, end_s: float) -> dict:
    """Manual integration of one target in one run between two retention times (seconds).

    target_row is a row of the 'target_review' widget (needs mode + mz, or q1/q3; ppm). The area is the trace
    above a straight line joining the (lightly smoothed) trace at the two boundaries. When the row carries
    `istd_area` and `cal` (both are in the widget rows) the response and back-calculated concentration are
    returned too. Times in the result are seconds; `t`/`y` is the trace around the peak for redrawing.
    """
    tr = ({"kind": "transition", "q1": float(target_row["q1"]), "q3": float(target_row["q3"])}
          if target_row.get("q1") is not None and target_row.get("q3") is not None
          else {"kind": "ms1", "mz": float(target_row["mz"])})
    ppm = float(target_row.get("ppm") or 10.0)
    span = max(3 * (end_s - start_s), 60.0)
    t, y, mode, _ = trace(run, tr, ppm, rt_range=(start_s - span, end_s + span), polarity=int(target_row.get("polarity") or 0))
    out = area_between(t, y, start_s, end_s)
    ins = np.flatnonzero((t >= start_s) & (t <= end_s))
    if len(ins) >= 3:
        pk = {"i_start": int(ins[0]), "i_end": int(ins[-1])}
        out["snr"] = local_snr(t, y, pk)
        net = y[ins] - np.interp(t[ins], [start_s, end_s], [out["base_start"], out["base_end"]])
        h = net.max()
        above = t[ins][net >= h / 2]
        out["fwhm"] = float(above[-1] - above[0]) if len(above) > 1 else None
    else:
        out["snr"], out["fwhm"] = 0.0, None
    out["mode"] = mode
    out["t"] = [float(v) for v in t]
    out["y"] = [float(v) for v in y]
    ia = target_row.get("istd_area")
    resp = out["area"] / ia if ia else (out["area"] if not target_row.get("istd") else None)
    out["response"] = resp
    cal = target_row.get("cal")
    if cal and resp is not None:
        out["concentration"] = inverse(cal["coef"], resp)
    return out


# ------------------------------------------------------------------ calibration maths

def _weights(x, weighting):
    x = np.asarray(x, float)
    if weighting == "1/x":
        return 1 / np.maximum(x, 1e-12)
    if weighting == "1/x2":
        return 1 / np.maximum(x, 1e-12) ** 2
    return np.ones_like(x)


def fit_curve(x, y, model="linear", weighting="1/x") -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = _weights(x, weighting)
    X = np.column_stack([np.ones_like(x), x] + ([x ** 2] if model == "quadratic" else []))
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    yhat = X @ coef
    ybar = np.sum(w * y) / np.sum(w)
    ss_res = float(np.sum(w * (y - yhat) ** 2))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    return {"coef": [float(c) for c in coef], "model": model, "weighting": weighting,
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


def predict(coef, x):
    x = np.asarray(x, float)
    return sum(c * x ** k for k, c in enumerate(coef))


def inverse(coef, y):
    """Concentration for a response (the root of the calibration function nearest the calibrated range)."""
    if y is None or not np.isfinite(y):
        return float("nan")
    a, b = coef[0], coef[1]
    c = coef[2] if len(coef) > 2 else 0.0
    if abs(c) < 1e-300:
        return (y - a) / b if b else float("nan")
    disc = b * b - 4 * c * (a - y)
    if disc < 0:
        return float("nan")
    r = [(-b + s * np.sqrt(disc)) / (2 * c) for s in (1, -1)]
    # the root on the rising branch (where the derivative has the sign of the slope at zero)
    good = [v for v in r if (b + 2 * c * v) * np.sign(b or 1) > 0]
    return float(min(good, key=abs) if good else min(r, key=abs))


def _tol(x, lloq_x):
    return 20.0 if np.isclose(x, lloq_x) else 15.0


def calibrate(x, y, model, weighting, auto_exclude, ids=None) -> dict:
    """Fit a curve, optionally dropping the worst calibrator until all pass (±15 %, ±20 % at the LLOQ)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    ids = list(ids) if ids is not None else list(range(n))
    detected = np.isfinite(y) & (y > 0)
    incl = detected.copy()
    reason = ["" if d else "not detected" for d in detected]
    min_levels = 4 if model == "quadratic" else 3
    used_model = model
    if len(np.unique(x[incl])) < min_levels and model == "quadratic" and len(np.unique(x[incl])) >= 3:
        used_model = "linear"
    fit = None
    while True:
        if len(np.unique(x[incl])) < 3:
            return {"ok": False, "why": f"only {len(np.unique(x[incl]))} detected calibration level(s)", "x": x, "y": y,
                    "included": incl, "reason": reason, "ids": ids}
        fit = fit_curve(x[incl], y[incl], used_model, weighting)
        back = np.array([inverse(fit["coef"], v) if d else np.nan for v, d in zip(y, detected)])
        acc = 100 * back / x
        lloq_x = x[incl].min()
        dev = np.abs(acc - 100) - np.array([_tol(v, lloq_x) for v in x])
        if not auto_exclude:
            break
        cand = np.flatnonzero(incl & (dev > 0))
        if not len(cand):
            break
        worst = cand[np.argmax(dev[cand])]
        trial = incl.copy()
        trial[worst] = False
        if len(np.unique(x[trial])) < min_levels:
            break
        incl = trial
        reason[worst] = f"auto-excluded (accuracy {acc[worst]:.0f}%)"
    passed = incl & (np.abs(acc - 100) <= np.array([_tol(v, x[incl].min()) for v in x]))
    lv = np.unique(x[incl])
    lloq = next((v for v in lv if passed[(x == v) & incl].any()), lv.min())
    uloq = next((v for v in lv[::-1] if passed[(x == v) & incl].any()), lv.max())
    # LOD/LOQ from the low end of the curve (ICH Q2: residual SD of a regression through the low range ÷ slope)
    lod = loq = np.nan
    low_lv = lv[:4]
    m = incl & np.isin(x, low_lv)
    if m.sum() >= 3 and len(low_lv) >= 2:
        A = np.column_stack([np.ones(m.sum()), x[m]])
        cf, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        res = y[m] - A @ cf
        sd = float(np.sqrt(np.sum(res ** 2) / max(m.sum() - 2, 1)))
        if cf[1] > 0:
            lod, loq = 3.3 * sd / cf[1], 10 * sd / cf[1]
    return {"ok": True, **fit, "x": x, "y": y, "back": back, "acc": acc, "included": incl, "passed": passed,
            "reason": reason, "ids": ids, "lloq": float(lloq), "uloq": float(uloq), "lod": float(lod), "loq": float(loq),
            "n_levels": int(len(lv)), "model_used": used_model,
            "frac_pass": float(passed.sum() / max(detected.sum(), 1))}


# ------------------------------------------------------------------ helpers

def _fc(v) -> str:
    """Concentration with three significant figures, no scientific notation for everyday numbers."""
    if v is None or not np.isfinite(v):
        return "—"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(len(pv))
    out[o] = np.clip(r, 0, 1)
    q[ok] = out
    return q


def _opt(params, key, default):
    v = params.get(key)
    return default if v is None or (isinstance(v, str) and not v.strip()) else v


def _bool(v, default=False):
    if isinstance(v, bool):
        return v
    t = _truthy(v)
    return default if t is None else t


def _norm_weighting(w):
    s = str(w).strip().lower().replace(" ", "").replace("²", "2").replace("^", "")
    if s in ("1/x2", "1/xx", "x2", "1/x**2"):
        return "1/x2"
    if s in ("1/x", "x"):
        return "1/x"
    if s in ("none", "1", "", "no", "unweighted", "equal"):
        return "none"
    raise UserFacingError(f"weighting must be 1/x, 1/x² or none (got '{w}').")


def _ion_ratio_tol(rel_pct):
    """Commission Decision 2002/657/EC maximum permitted tolerance for relative ion intensities (LC-MS/MS)."""
    return 20.0 if rel_pct > 50 else 25.0 if rel_pct > 20 else 30.0 if rel_pct > 10 else 50.0


# ------------------------------------------------------------------ the pipeline

def run(job, files: list[Path], params: dict):
    files = [Path(f) for f in files]
    if not files:
        raise UserFacingError("No data files were given.")
    job.result["versions"] = versions("numpy", "scipy", "pandas", "matplotlib", "pyopenms", "pyteomics")

    # ---- parameters
    ppm_default = float(_opt(params, "ppm", 10.0))
    win_default_min = float(_opt(params, "rt_window", 0.5))
    weighting = _norm_weighting(_opt(params, "weighting", "1/x"))
    model = str(_opt(params, "model", "linear")).strip().lower()
    if model not in ("linear", "quadratic"):
        raise UserFacingError(f"model must be linear or quadratic (got '{model}').")
    auto_exclude = _bool(params.get("auto_exclude"), False)
    min_snr = float(_opt(params, "min_snr", 10.0))
    smooth_w = int(float(_opt(params, "smooth", 5)))
    units_default = str(params.get("units") or "").strip()

    # ---- sample sheet
    sheet = params.get("sheet") or msio.match_sheet(files, None)
    rows = _sheet_for(files, sheet)

    # ---- data
    job.step(2, "Reading the target list")
    tdf = read_target_table(params)
    runs = msio.load_runs(job, files, 3, 30)
    pols = [p for p in (_run_polarity(r) for r in runs) if p]
    default_pol = int(np.sign(sum(pols))) if pols and sum(pols) != 0 else 1
    comps, notes = parse_targets(tdf, default_pol, units_default)
    if not comps:
        raise UserFacingError("No usable targets in the target list. " + " ".join(notes[:5]))
    units = next((c["units"] for c in comps if c["units"]), units_default)

    # RT units sanity check: a list in seconds read as minutes would put every target beyond the run
    run_end = max((float(r.rt.max()) if r.n else 0.0) for r in runs)
    run_end = max([run_end] + [float(c["time"][-1]) for r in runs for c in r.chromatograms if len(c["time"])])
    given = [c for c in comps if c["rt_s"] is not None]
    if given and run_end > 0 and all(c["rt_s"] > 1.2 * run_end for c in given) and all(c["rt_list"] <= run_end for c in given):
        for c in comps:
            if c["rt_list"] is not None:
                c["rt_s"] = c["rt_list"]
            if c["win_list"] is not None:
                c["win_s"] = c["win_list"]
        job.flag("info", f"The expected retention times in the target list are larger than the runs are long "
                         f"({run_end / 60:.1f} min) when read as minutes, so they were read as <b>seconds</b>.")
    for c in comps:
        c["win_s"] = c["win_s"] if c["win_s"] else win_default_min * 60
        c["ppm_used"] = c["ppm"] or ppm_default
        c["pol_used"] = c["polarity"] or default_pol
        if c["transitions"][0]["kind"] == "ms1" and c["transitions"][0].get("default_adduct") and c["polarity"] == 0 and default_pol < 0:
            # formula with no adduct: the default follows the data's polarity
            for tr in c["transitions"]:
                if tr.get("default_adduct"):
                    m = _formula_mass(tr["formula"])
                    tr["mz"] = chem.adduct_mz(m, "[M-H]-")
                    tr["adduct"] = "[M-H]-"
                    tr["label"] = f"[M-H]- {tr['mz']:.4f}"

    for n_ in notes:
        job.flag("warn", n_)

    analytes = [c for c in comps if not c["is_istd"]]
    istds = [c for c in comps if c["is_istd"]]
    types = [r["type"] for r in rows]
    conc = [r["concentration"] for r in rows]
    std_idx = [i for i, r in enumerate(rows) if r["type"] == "standard" and r["concentration"] is not None and r["concentration"] > 0]
    qc_idx = [i for i, r in enumerate(rows) if r["type"] == "qc"]

    # ---- expected RT for targets without one
    job.step(32, "Locating targets without an expected retention time")
    inferred = {}
    for c in comps:
        if c["rt_s"] is not None:
            c["rt_source"] = "target list"
            continue
        ref = sorted(std_idx, key=lambda i: -conc[i]) + qc_idx
        if not ref:
            ref = [i for i, t in enumerate(types) if t != "blank"]
        found = []
        for i in ref[:12]:
            t, y, mode, _ = trace(runs[i], c["transitions"][0], c["ppm_used"], None, c["pol_used"])
            if mode is None or len(t) < 5:
                continue
            pk = peaks.integrate(t, y, smooth_window=smooth_w, min_snr=3)
            if pk:
                found.append((pk["rt"], pk["height"]))
        if not found:
            c["rt_s"] = None
            c["rt_source"] = "not found"
            job.flag("warn", f"<b>{c['name']}</b>: no expected RT was given and no peak was found in the standards/QCs, so it could not be integrated.")
            continue
        rts = np.array([f[0] for f in found])
        med = float(np.median(rts))
        cons = np.abs(rts - med) <= 9
        if cons.sum() >= max(2, 0.5 * len(rts)) or len(rts) == 1:
            c["rt_s"] = float(np.median(rts[cons]))
            c["rt_source"] = f"found in {int(cons.sum())}/{len(rts)} {'standards/QCs' if std_idx or qc_idx else 'injections'}"
            inferred[c["name"]] = (c["rt_s"], int(cons.sum()), len(rts), True)
        else:
            c["rt_s"] = float(found[int(np.argmax([f[1] for f in found]))][0])
            c["rt_source"] = "tallest peak (inconsistent across standards)"
            inferred[c["name"]] = (c["rt_s"], int(cons.sum()), len(rts), False)
    if inferred:
        txt = "; ".join(f"<b>{k}</b> {v[0] / 60:.2f} min ({v[1]}/{v[2]} agree)" for k, v in inferred.items())
        bad = [k for k, v in inferred.items() if not v[3]]
        job.flag("warn" if bad else "info",
                 f"No expected RT was given for {len(inferred)} target(s); it was taken from the tallest peak in the "
                 f"calibration standards/QCs: {txt}." + (f" The standards disagree for {', '.join(bad)} — check those chromatograms and add the RT to the target list." if bad else ""))

    # ---- extraction & integration (ISTDs first so analytes can follow their ISTD's retention time)
    job.step(38, "Extracting and integrating every target")
    res: dict[tuple[str, int], dict] = {}
    work = istds + analytes
    for k, c in enumerate(work):
        if k % 4 == 0:
            job.step(38 + int(27 * k / max(len(work), 1)), f"Integrating {c['name']} ({k + 1}/{len(work)})")
        for i, r in enumerate(runs):
            res[(c["name"], i)] = _integrate_one(c, r, i, res, comps, smooth_w)

    modes = {}
    for c in comps:
        ms = {res[(c["name"], i)]["mode"] for i in range(len(runs))} - {None}
        c["mode"] = "/".join(sorted(ms)) if ms else "not found"
        modes[c["name"]] = c["mode"]
        if not ms:
            job.flag("warn", f"<b>{c['name']}</b> ({c['transitions'][0]['label']}): no matching data in any file "
                             f"({res[(c['name'], 0)]['how']}).")

    # ---- responses
    for c in comps:
        for i in range(len(runs)):
            d = res[(c["name"], i)]
            if c["istd"]:
                ia = res[(c["istd"], i)]["area"]
                d["istd_area"] = ia
                d["response"] = d["area"] / ia if ia > 0 else np.nan
            else:
                d["istd_area"] = None
                d["response"] = d["area"]

    # ---- peak quality flags
    _flag_peaks(comps, res, rows, runs, min_snr)

    # ---- calibration
    job.step(68, "Fitting calibration curves")
    cals = {}
    n_std_levels = len({conc[i] for i in std_idx})
    for c in analytes:
        if n_std_levels < 3:
            break
        x = np.array([conc[i] for i in std_idx])
        y = np.array([res[(c["name"], i)]["response"] for i in std_idx], float)
        cal = calibrate(x, y, model, weighting, auto_exclude, ids=std_idx)
        cals[c["name"]] = cal
    # ---- back-calculation
    for c in analytes:
        cal = cals.get(c["name"])
        for i in range(len(runs)):
            d = res[(c["name"], i)]
            d["conc"] = np.nan
            d["report"] = ""
            if cal is None or not cal["ok"]:
                continue
            if d["area"] <= 0:
                d["report"] = "n.d."
                d["flags"].append("not detected")
                continue
            v = inverse(cal["coef"], d["response"])
            d["conc"] = v
            if not np.isfinite(v):
                d["report"] = "—"
            elif v < cal["lloq"] * (1 - 1e-9):
                d["report"] = "<LLOQ"
                d["flags"].append("below LLOQ")
            elif v > cal["uloq"] * (1 + 1e-9):
                d["report"] = ">ULOQ"
                d["flags"].append("above ULOQ (extrapolated)")
            else:
                d["report"] = _fc(v)

    job.step(76, "Writing figures and tables")
    _report(job, files, runs, rows, comps, analytes, istds, res, cals, params=dict(
        ppm=ppm_default, rt_window=win_default_min, weighting=weighting, model=model, auto_exclude=auto_exclude,
        min_snr=min_snr, smooth=smooth_w, units=units, std_idx=std_idx, qc_idx=qc_idx, modes=modes))


def _sheet_for(files, sheet) -> list[dict]:
    """One sheet row per file, in file order (sheet rows are matched by path, then by file stem)."""
    by_path, by_stem = {}, {}
    for r in sheet or []:
        f = str(r.get("file") or "")
        if f:
            by_path[str(Path(f))] = r
            by_stem[msio._stem(Path(f).name)] = r
        s = str(r.get("sample") or "")
        if s:
            by_stem.setdefault(msio._stem(s), r)
    out = []
    auto = msio.match_sheet(files, None)
    for f, a in zip(files, auto):
        r = by_path.get(str(f)) or by_stem.get(msio._stem(f.name)) or a
        typ = str(r.get("type") or a["type"]).strip().lower()
        if typ not in ("sample", "qc", "blank", "standard"):
            typ = a["type"]
        cv = r.get("concentration")
        cv = _num(cv) if cv is not None else None
        order = _num(r.get("order"))
        out.append({"file": str(f), "sample": str(r.get("sample") or a["sample"]), "group": str(r.get("group") or "").strip(),
                    "type": typ, "batch": str(r.get("batch") or "1"), "order": order if order is not None else a["order"],
                    "concentration": cv})
    return out


def _integrate_one(c, run, i, res, comps, smooth_w) -> dict:
    tq = c["transitions"][0]
    center, win = c["rt_s"], c["win_s"]
    d = {"mode": None, "how": "", "pk": None, "area": 0.0, "height": 0.0, "rt": np.nan, "snr": 0.0, "fwhm": np.nan,
         "points": 0, "t": np.zeros(0), "y": np.zeros(0), "quals": [], "flags": [], "center": center}
    if center is None:
        d["how"] = "no expected RT"
        return d
    # follow the internal standard: a labelled ISTD co-elutes, so its observed RT corrects run-to-run drift
    if c["istd"]:
        ist = next(x for x in comps if x["name"] == c["istd"])
        di = res.get((c["istd"], i))
        if di and di["pk"] and ist["rt_s"] is not None and abs(di["rt"] - ist["rt_s"]) <= ist["win_s"]:
            center = di["rt"] + (c["rt_s"] - ist["rt_s"])
    d["center"] = center
    span = max(2.5 * win, 75.0)
    rng = (center - span, center + span)
    t, y, mode, how = trace(run, tq, c["ppm_used"], rng, c["pol_used"])
    d.update(mode=mode, how=how, t=t, y=y)
    if mode is None or len(t) < 5:
        return d
    pk = peaks.integrate(t, y, rt=center, window=win, smooth_window=smooth_w, min_snr=3)
    if pk is None:
        return d
    d.update(pk=pk, area=pk["area"], height=pk["height"] - 0.5 * (pk["base_start"] + pk["base_end"]), rt=pk["rt"],
             fwhm=pk["fwhm"], points=pk["points"], snr=local_snr(t, y, pk))
    for tr in c["transitions"][1:]:
        t2, y2, m2, _ = trace(run, tr, c["ppm_used"], rng, c["pol_used"])
        a = area_between(t2, y2, pk["start"], pk["end"]) if m2 else {"area": np.nan}
        d["quals"].append(a["area"])
    return d


def _flag_peaks(comps, res, rows, runs, min_snr):
    types = [r["type"] for r in rows]
    std_i = [i for i, t in enumerate(types) if t == "standard"]
    qc_i = [i for i, t in enumerate(types) if t == "qc"]
    for c in comps:
        exp = c["rt_s"]
        for i in range(len(runs)):
            d = res[(c["name"], i)]
            if d["pk"] is None:
                if types[i] != "blank" and d["mode"] is not None:
                    d["flags"].append("no peak")
                continue
            if types[i] == "blank":
                continue          # a small peak in a blank is judged by the carry-over check, not peak quality
            if d["points"] < 6:
                d["flags"].append(f"{d['points']} points")
            if d["snr"] < min_snr:
                d["flags"].append(f"S/N {d['snr']:.0f}")
            if exp is not None and abs(d["rt"] - exp) > c["win_s"] / 2:
                d["flags"].append(f"RT shift {(d['rt'] - exp) / 60:+.2f} min")
        # qualifier / quantifier ion ratios against the mean of the standards (else QCs, else all detected)
        c["ion_ref"] = []
        for q in range(len(c["transitions"]) - 1):
            def ratio(i):
                d = res[(c["name"], i)]
                if d["pk"] is None or d["area"] <= 0 or len(d["quals"]) <= q or not np.isfinite(d["quals"][q]):
                    return np.nan
                return d["quals"][q] / d["area"]
            ref_i = [i for i in std_i if res[(c["name"], i)]["snr"] >= min_snr] or \
                    [i for i in qc_i if res[(c["name"], i)]["snr"] >= min_snr] or \
                    [i for i in range(len(runs)) if types[i] != "blank" and res[(c["name"], i)]["snr"] >= min_snr]
            rs = np.array([ratio(i) for i in ref_i])
            rs = rs[np.isfinite(rs)]
            if not len(rs):
                c["ion_ref"].append(None)
                continue
            ref = float(np.mean(rs))
            rel = 100 * min(ref, 1 / ref) if ref > 0 else 0.0
            tol = _ion_ratio_tol(rel)
            src = "standards" if any(i in std_i for i in ref_i) else "QCs" if any(i in qc_i for i in ref_i) else "all injections"
            c["ion_ref"].append({"ref": ref, "tol": tol, "source": src, "cv": float(np.std(rs, ddof=1) / ref * 100) if len(rs) > 1 and ref > 0 else np.nan})
            for i in range(len(runs)):
                d = res[(c["name"], i)]
                d.setdefault("ratios", [])
                r = ratio(i)
                d["ratios"].append(r)
                if d["pk"] is None or types[i] == "blank":
                    continue
                if ref > 0 and (not np.isfinite(r) or abs(r / ref - 1) * 100 > tol):
                    d["flags"].append(f"ion ratio {r / ref * 100 - 100:+.0f}%" if np.isfinite(r) else "qualifier missing")
    # ISTD response stability (common SOP: 50–150 % of the mean in standards and QCs)
    for c in comps:
        if not c["is_istd"]:
            continue
        ref_i = [i for i in std_i + qc_i if res[(c["name"], i)]["area"] > 0] or [i for i in range(len(runs)) if res[(c["name"], i)]["area"] > 0]
        m = np.mean([res[(c["name"], i)]["area"] for i in ref_i]) if ref_i else np.nan
        c["istd_mean"] = m
        for i in range(len(runs)):
            d = res[(c["name"], i)]
            if types[i] == "blank" or not np.isfinite(m) or m <= 0:
                continue
            f = d["area"] / m
            d["istd_rel"] = f
            if d["area"] > 0 and not 0.5 <= f <= 1.5:
                d["flags"].append(f"ISTD {100 * f:.0f}% of mean")
    for c in comps:
        if c["istd"]:
            for i in range(len(runs)):
                if types[i] != "blank" and res[(c["istd"], i)]["area"] <= 0 and res[(c["name"], i)]["pk"] is not None:
                    res[(c["name"], i)]["flags"].append("ISTD not found")


# ------------------------------------------------------------------ figures, tables, exports

def _report(job, files, runs, rows, comps, analytes, istds, res, cals, params):
    units = params["units"]
    u = f" ({units})" if units else ""
    names = [r["sample"] for r in rows]
    types = [r["type"] for r in rows]
    order = np.argsort([r["order"] for r in rows], kind="stable")
    nR = len(runs)
    std_idx, qc_idx = params["std_idx"], params["qc_idx"]
    good_cals = {k: v for k, v in cals.items() if v["ok"]}

    # ============ tiles
    n_flag = sum(1 for c in comps for i in range(nR) if types[i] != "blank" and
                 [f for f in res[(c["name"], i)]["flags"] if not f.startswith(("below LLOQ", "above ULOQ", "not detected"))])
    n_cells = sum(1 for c in comps for i in range(nR) if types[i] != "blank")
    job.tile(len(analytes), "analytes" + (f" + {len(istds)} ISTD" if istds else ""))
    job.tile(nR, "injections")
    if good_cals:
        r2 = np.median([v["r2"] for v in good_cals.values()])
        job.tile(f"{len(good_cals)}/{len(analytes)}", "calibrated")
        job.tile(f"{r2:.4f}", "median R²")
    job.tile(f"{100 * (1 - n_flag / max(n_cells, 1)):.0f}%", "peaks without flags")

    # ============ Step 1: targets
    job.section("targets", "Step 1", "Targets & extraction",
                "What was looked for in each file, and how: a full-scan extracted-ion chromatogram (XIC) at the target m/z, "
                "a stored SRM/MRM transition, or a product-ion trace from MS2 spectra (PRM, or SRM written as spectra).")
    trows = []
    for c in comps:
        tq = c["transitions"][0]
        trows.append([c["name"], "ISTD" if c["is_istd"] else "analyte", c["mode"], tq["label"],
                      str(len(c["transitions"]) - 1), f"{c['rt_s'] / 60:.2f}" if c["rt_s"] is not None else "—",
                      c.get("rt_source", ""), f"±{c['win_s'] / 60:.2f}", f"{c['ppm_used']:g}" if "ms1" in c["mode"] or "prm" in c["mode"] else "—",
                      c["istd"] or ""])
    job.table("target_list", "Targets", ["Target", "Role", "Extraction", "Quantifier", "Qualifiers", "Expected RT (min)",
                                          "RT from", "Window (min)", "ppm", "ISTD"], trows,
              note="Window = the ± search range around the expected RT (the ISTD's observed RT when the target has one). "
                   "Qualifiers are integrated between the quantifier's boundaries.")

    # ============ Step 2: integration review
    job.section("review", "Step 2", "Peak integration review",
                "Every integrated peak, overlaid across injections. This is where to check that the right peak was picked "
                "and the boundaries are sensible before trusting any number downstream.")
    pages = [comps[k:k + PAGE] for k in range(0, len(comps), PAGE)]
    for p, page in enumerate(pages):
        fig = _xic_page(page, res, rows, nR)
        flagged = [(c["name"], sum(1 for i in range(nR) if types[i] != "blank" and res[(c["name"], i)]["flags"] and
                                   any(not f.startswith(("below LLOQ", "above ULOQ", "not detected")) for f in res[(c["name"], i)]["flags"])))
                   for c in page]
        worst = sorted([f for f in flagged if f[1]], key=lambda f: -f[1])
        shifts = [abs(res[(c["name"], i)]["rt"] - c["rt_s"]) / 60 for c in page for i in range(nR)
                  if res[(c["name"], i)]["pk"] is not None and c["rt_s"] is not None]
        job.figure(f"xic_review_{p + 1}", "Extracted chromatograms" + (f" ({p + 1}/{len(pages)})" if len(pages) > 1 else ""), fig, wide=True,
                   how="One panel per target: the quantifier trace of every injection overlaid, coloured by sample type. The shaded "
                       "area under each trace is exactly what was integrated (above a straight baseline between the boundaries). "
                       "The dashed line is the expected RT and the grey band the search window. The intensity axis is square-root "
                       "scaled so that low calibrators and blanks stay visible next to the top standard. Good integration looks like "
                       "one peak per trace at the same RT, with shading that starts and ends where the peak meets the baseline.",
                   yours=(f"Median |RT − expected| across these targets is <b>{np.median(shifts):.2f} min</b>. " if shifts else "")
                         + ("Flagged injections: " + ", ".join(f"<b>{n}</b> {k}" for n, k in worst[:6]) + (" …" if len(worst) > 6 else "")
                            + ". Open those cells in the review table to inspect or re-integrate."
                            if worst else "No peak on this page carries a quality flag (points, S/N, RT shift, ion ratio, ISTD)."))

    # review widget: one row per target × injection
    wrows = []
    for c in comps:
        tq = c["transitions"][0]
        cal = cals.get(c["name"])
        for i in range(nR):
            d = res[(c["name"], i)]
            pk = d["pk"]
            wrows.append({k: _clean(v) for k, v in {
                "target": c["name"], "sample": names[i], "file_index": i, "type": types[i], "group": rows[i]["group"],
                "mode": d["mode"], "mz": tq.get("mz"), "q1": tq.get("q1"), "q3": tq.get("q3"), "ppm": c["ppm_used"],
                "polarity": c["pol_used"], "istd": c["istd"],
                "rt": d["rt"] / 60 if pk else None, "start": pk["start"] / 60 if pk else None, "end": pk["end"] / 60 if pk else None,
                "expected_rt": c["rt_s"] / 60 if c["rt_s"] is not None else None, "window": c["win_s"] / 60,
                "area": d["area"], "height": d["height"], "snr": d["snr"], "points": d["points"],
                "istd_area": d.get("istd_area"), "response": d.get("response"), "concentration": d.get("conc"),
                "report": d.get("report", ""), "flags": list(d["flags"]),
            }.items()})
            if cal and cal["ok"]:
                wrows[-1]["cal"] = {"coef": cal["coef"], "model": cal["model_used"], "weighting": cal["weighting"],
                                    "lloq": cal["lloq"], "uloq": cal["uloq"]}
    job.widget("target_review", title="Integration review", rows=wrows, units="min",
               note="rt/start/end/expected_rt/window in minutes; reintegrate() takes seconds.")

    srows = []
    for c in comps:
        ds = [res[(c["name"], i)] for i in range(nR) if types[i] != "blank"]
        det = [d for d in ds if d["pk"] is not None]
        fl = sum(1 for d in ds if any(not f.startswith(("below LLOQ", "above ULOQ", "not detected")) for f in d["flags"]))
        ir = c.get("ion_ref") or []
        irs = ", ".join(f"{x['ref']:.2f} ±{x['tol']:.0f}%" for x in ir if x) or "—"
        srows.append([c["name"], f"{len(det)}/{len(ds)}", f"{np.median([d['rt'] for d in det]) / 60:.2f}" if det else "—",
                      f"{np.median([d['fwhm'] for d in det]):.1f}" if det else "—", f"{np.median([d['points'] for d in det]):.0f}" if det else "—",
                      f"{np.median([d['snr'] for d in det]):.0f}" if det else "—", irs, str(fl)])
    job.table("integration_summary", "Integration summary (non-blank injections)",
              ["Target", "Peak found", "Median RT (min)", "Median FWHM (s)", "Median points", "Median S/N", "Ion ratio (ref ± tol)", "Flagged"],
              srows, note=f"S/N = peak height above the local background ÷ the background's robust SD (flag below {params['min_snr']:g}). "
                          "Points = scans between the integration boundaries (flag below 6). Ion-ratio tolerances follow "
                          "Commission Decision 2002/657/EC (±20–50 % depending on the relative intensity).")

    if istds:
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        xs = np.arange(nR)
        pos = {int(o): k for k, o in enumerate(order)}
        outs = []
        for j, c in enumerate(istds):
            m = c.get("istd_mean", np.nan)
            v = np.array([res[(c["name"], i)]["area"] for i in range(nR)])
            rel = v / m * 100 if np.isfinite(m) and m > 0 else v * np.nan
            ax.plot([pos[i] for i in order], rel[order], "-", color=PALETTE[j % len(PALETTE)], lw=0.8, alpha=0.7, label=c["name"])
            for i in range(nR):
                ax.scatter(pos[i], rel[i], s=14, color=TYPE_COLOR[types[i]], zorder=3, edgecolor="none")
                if types[i] != "blank" and np.isfinite(rel[i]) and not 50 <= rel[i] <= 150:
                    outs.append(f"{c['name']} in {names[i]} ({rel[i]:.0f}%)")
        ax.axhspan(50, 150, color="#2ca02c", alpha=0.06, lw=0)
        ax.axhline(100, color="#888", lw=0.6, ls=":")
        if nR <= 80:
            ax.set_xticks(xs)
            ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=6 if nR <= 40 else 4.5)
        ax.set_ylabel("ISTD area, % of mean")
        ax.set_xlabel("injection order")
        ax.legend(fontsize=7, ncol=min(4, len(istds)), loc="upper right")
        job.figure("istd_response", "Internal-standard response across the run", fig, wide=True,
                   how="Each internal standard's peak area in every injection, as a percentage of its mean in the standards and QCs, "
                       "in injection order (dots coloured by sample type). The ISTD is added at the same amount everywhere, so its area "
                       "should be flat; the green band is the common 50–150 % acceptance range. A drop in one sample means ion "
                       "suppression or a pipetting/injection problem there; a trend means the instrument drifted — which the "
                       "analyte/ISTD ratio corrects, as long as the ISTD is still well above its noise.",
                   yours=(f"Outside 50–150 %: {', '.join(outs[:8])}{' …' if len(outs) > 8 else ''}." if outs else
                          "Every non-blank injection is within 50–150 % of the mean — consistent sample preparation and injection."))

    # ============ Step 3: calibration
    cal_rows_csv, pt_rows = [], []
    if cals:
        job.section("calibration", "Step 3", "Calibration",
                    f"Standards' responses (area{' ÷ ISTD area' if any(c['istd'] for c in analytes) else ''}) against their nominal "
                    f"concentration, fitted with a {params['model']} curve weighted {_wlabel(params['weighting'])}. Every standard is then "
                    "back-calculated from the curve; FDA/ICH M10 accept a curve when ≥ 75 % of standards are within ±15 % "
                    "(±20 % at the lowest level).")
        cpages = [analytes[k:k + PAGE] for k in range(0, len(analytes), PAGE)]
        for p, page in enumerate(cpages):
            page = [c for c in page if c["name"] in cals]
            if not page:
                continue
            fig = _cal_page(page, cals, res, rows, units)
            poor = [c["name"] for c in page if cals[c["name"]]["ok"] and cals[c["name"]]["r2"] < 0.99]
            job.figure(f"calibration_{p + 1}", "Calibration curves" + (f" ({p + 1}/{len(cpages)})" if len(cpages) > 1 else ""), fig, wide=True,
                       how="Filled blue dots: calibration standards used in the fit; hollow dots: standards excluded (auto-excluded or not "
                           "detected); the line is the fitted curve, solid over the validated range (LLOQ–ULOQ) and dashed outside it. "
                           "Orange ticks are the unknown samples and black diamonds the QCs, placed on the curve at their back-calculated "
                           "concentration. Both axes are logarithmic so three orders of magnitude fit on one plot — a straight line with "
                           "slope 1 means response proportional to concentration.",
                       yours=("R² below 0.99 for " + ", ".join(f"<b>{n}</b>" for n in poor) + " — look at the accuracy plot for which levels bend the curve."
                              if poor else "Every curve on this page has R² ≥ 0.99."))

        # accuracy (residual) plot
        fig, ax = plt.subplots(figsize=(8, 4))
        worst = []
        for j, c in enumerate([c for c in analytes if c["name"] in good_cals]):
            cal = cals[c["name"]]
            col = PALETTE[j % len(PALETTE)]
            jit = 1 + 0.03 * (j - len(good_cals) / 2) / max(len(good_cals), 1)
            inc = cal["included"]
            ok = np.isfinite(cal["acc"])
            ax.scatter(cal["x"][inc & ok] * jit, cal["acc"][inc & ok], s=22, color=col, label=c["name"], zorder=3)
            ax.scatter(cal["x"][~inc & ok] * jit, cal["acc"][~inc & ok], s=22, facecolor="none", edgecolor=col, zorder=3)
            for xv, a, used in zip(cal["x"], cal["acc"], inc):
                if used and np.isfinite(a) and abs(a - 100) > _tol(xv, cal["x"][inc].min()):
                    worst.append(f"{c['name']} at {_fc(xv)} ({a:.0f}%)")
        if good_cals:
            xall = np.concatenate([v["x"] for v in good_cals.values()])
            ax.set_xscale("log")
            ax.axhspan(85, 115, color="#2ca02c", alpha=0.07, lw=0)
            ax.axhline(100, color="#888", lw=0.6)
            for yv in (80, 120):
                ax.axhline(yv, color="#2ca02c", lw=0.6, ls=":")
            ax.set_xlim(xall.min() / 1.6, xall.max() * 1.6)
            yl = ax.get_ylim()
            ax.set_ylim(min(yl[0], 70), max(yl[1], 130))
            ax.set_xlabel(f"nominal concentration{u}")
            ax.set_ylabel("back-calculated / nominal (%)")
            ax.legend(fontsize=7, ncol=2, loc="best")
            job.figure("calibration_accuracy", "Back-calculated accuracy of the standards", fig,
                       how="Each standard's concentration read back from its own curve, as a percentage of what was spiked. Points should "
                           "scatter evenly around 100 % inside the green ±15 % band (dotted lines: ±20 %, the limit at the LLOQ). A funnel "
                           "that widens at low concentration means the weighting is too weak (try 1/x²); a smile or frown shape means "
                           "the response is not linear (try quadratic, or narrow the range). Hollow points were excluded from the fit.",
                       yours=("Outside the limits (included in the fit): " + ", ".join(worst[:8]) + (" …" if len(worst) > 8 else "") + "."
                              if worst else "All included standards are within ±15 % (±20 % at the LLOQ)."))
        else:
            plt.close(fig)

        crow = []
        for c in analytes:
            cal = cals.get(c["name"])
            if cal is None:
                continue
            if not cal["ok"]:
                crow.append([c["name"], "—", "—", "—", "—", "—", "—", "—", "—", "—", cal["why"]])
                cal_rows_csv.append({"target": c["name"], "status": "failed: " + cal["why"]})
                continue
            coef = cal["coef"]
            excl = [f"{_fc(x)}" for x, inc, rs in zip(cal["x"], cal["included"], cal["reason"]) if not inc]
            nfail = int((cal["included"] & ~cal["passed"]).sum())
            status = []
            if cal["r2"] < 0.99:
                status.append("R² below 0.99")
            if cal["frac_pass"] < 0.75:
                status.append("under 75 % of standards pass")
            if cal["n_levels"] < 6:
                status.append(f"{cal['n_levels']} levels (FDA asks ≥ 6)")
            if cal["model_used"] != cal["model"]:
                status.append("fell back to linear (too few levels)")
            crow.append([c["name"], cal["model_used"], _wlabel(cal["weighting"]), f"{coef[1]:.4g}", f"{coef[0]:.3g}",
                         f"{coef[2]:.3g}" if len(coef) > 2 else "—", f"{cal['r2']:.4f}", f"{_fc(cal['lloq'])} – {_fc(cal['uloq'])}",
                         f"{_fc(cal['lod'])} / {_fc(cal['loq'])}", ", ".join(excl) or "none",
                         "; ".join(status) or "pass"])
            cal_rows_csv.append({"target": c["name"], "istd": c["istd"] or "", "model": cal["model_used"], "weighting": cal["weighting"],
                                 "slope": coef[1], "intercept": coef[0], "quadratic": coef[2] if len(coef) > 2 else 0.0,
                                 "r2": cal["r2"], "n_levels": cal["n_levels"], "n_standards": int(len(cal["x"])),
                                 "n_included": int(cal["included"].sum()), "n_within_limits": int(cal["passed"].sum()),
                                 "lloq": cal["lloq"], "uloq": cal["uloq"], "lod": cal["lod"], "loq": cal["loq"], "units": units,
                                 "excluded": "; ".join(f"{_fc(x)} ({rs})" for x, inc, rs in zip(cal["x"], cal["included"], cal["reason"]) if not inc),
                                 "status": "; ".join(status) or "pass"})
            serious = [s_ for s_ in status if s_.startswith(("R²", "under"))]
            if serious:
                job.flag("warn", f"<b>{c['name']}</b> calibration: {'; '.join(serious)}.")
            if nfail:
                job.flag("warn", f"<b>{c['name']}</b>: {nfail} standard(s) in the fit are outside ±15 % (±20 % at LLOQ)"
                                 + ("" if params["auto_exclude"] else " — turn on auto-exclude, or check those chromatograms") + ".")
            for k, (x, yv, b, a, inc, rs, pa) in enumerate(zip(cal["x"], cal["y"], cal["back"], cal["acc"], cal["included"], cal["reason"], cal["passed"])):
                pt_rows.append({"target": c["name"], "sample": names[cal["ids"][k]], "nominal": x, "response": yv,
                                "back_calculated": b, "accuracy_pct": a, "included": bool(inc), "within_limits": bool(pa), "note": rs})
        (job.root / "calibration.csv").write_text(pd.DataFrame(cal_rows_csv).to_csv(index=False))
        (job.root / "calibration_points.csv").write_text(pd.DataFrame(pt_rows).to_csv(index=False))
        job.table("calibration_table", "Calibration summary", ["Analyte", "Model", "Weighting", "Slope", "Intercept", "x²",
                                                               "R²", f"Range{u}", f"LOD / LOQ{u}", "Excluded", "Status"], crow,
                  note="Range = LLOQ–ULOQ, the lowest and highest standards that back-calculate within ±20 % / ±15 %. LOD = 3.3·σ/S "
                       "and LOQ = 10·σ/S (ICH Q2), with σ the residual SD of an unweighted straight line through the lowest four "
                       "levels and S its slope — the whole-curve residual SD would be dominated by the top standards. R² is the "
                       "weighted coefficient of determination.", csv="calibration.csv")
        tbl = [[r["target"], r["sample"], _fc(r["nominal"]), fmt_n(r["response"]) if np.isfinite(r["response"]) and r["response"] >= 10 else f"{r['response']:.4g}",
                _fc(r["back_calculated"]), f"{r['accuracy_pct']:.1f}" if np.isfinite(r["accuracy_pct"]) else "—",
                "yes" if r["included"] else "no", r["note"] or ("pass" if r["within_limits"] else "outside limits")] for r in pt_rows]
        job.table("calibration_points", "Calibration standards, back-calculated", ["Analyte", "Standard", f"Nominal{u}", "Response",
                                                                                   f"Back-calc.{u}", "Accuracy %", "In fit", "Note"],
                  tbl[:200], csv="calibration_points.csv")
        job.download("Calibration summary (CSV)", "calibration.csv")
        job.download("Calibration standards (CSV)", "calibration_points.csv")
    else:
        why = ("no injections are marked as calibration standards with a concentration in the sample sheet" if not std_idx
               else f"only {len({rows[i]['concentration'] for i in std_idx})} standard concentration level(s) (3 needed)")
        job.flag("info", f"No calibration curves: {why}. Results are reported as peak areas"
                         + (" and area ratios to the internal standard" if any(c["istd"] for c in analytes) else "") + " instead of concentrations.")

    # ============ Step 4: quantification
    job.section("quant", "Step 4", "Quantification" if good_cals else "Responses",
                "Every injection back-calculated against its analyte's curve. Values below the lowest valid standard are reported as "
                "&lt;LLOQ and above the highest as &gt;ULOQ: the curve says nothing reliable outside the range it was built on."
                if good_cals else "Peak areas, or area ÷ internal-standard area where the target list names an internal standard.")
    qc_rows, qc_flag = [], []
    if good_cals and qc_idx:
        nominal_qcs = [i for i in qc_idx if rows[i]["concentration"] is not None and rows[i]["concentration"] > 0]
        fig = None
        if nominal_qcs:
            fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(good_cals) + 2), 3.6))
            levels = sorted({rows[i]["concentration"] for i in nominal_qcs})
            mk = ["v", "o", "^", "s", "D", "P"]
            n_in = n_all = 0
            for j, c in enumerate([c for c in analytes if c["name"] in good_cals]):
                for li, lv in enumerate(levels):
                    ii = [i for i in nominal_qcs if rows[i]["concentration"] == lv]
                    v = np.array([res[(c["name"], i)]["conc"] for i in ii], float)
                    acc = 100 * v / lv
                    ok = np.isfinite(acc)
                    n_all += len(acc)
                    n_in += int((np.abs(acc[ok] - 100) <= 15).sum())
                    ax.scatter(j + (li - (len(levels) - 1) / 2) * 0.18 + np.zeros(ok.sum()), acc[ok], marker=mk[li % len(mk)], s=26,
                               color=QC_COLOR, facecolor="none" if li % 2 else QC_COLOR, label=f"QC {_fc(lv)}" if j == 0 else None, zorder=3)
                    mean = float(np.nanmean(acc)) if ok.any() else np.nan
                    cv = float(np.nanstd(v, ddof=1) / np.nanmean(v) * 100) if ok.sum() > 1 else np.nan
                    passed = np.isfinite(mean) and abs(mean - 100) <= 15 and (not np.isfinite(cv) or cv <= 15)
                    qc_rows.append([c["name"], _fc(lv), str(int(ok.sum())), _fc(np.nanmean(v)) if ok.any() else "—",
                                    f"{mean:.1f}" if np.isfinite(mean) else "—", f"{cv:.1f}" if np.isfinite(cv) else "—",
                                    "pass" if passed else "FAIL"])
                    if not passed:
                        qc_flag.append(f"{c['name']} QC {_fc(lv)}")
            ax.axhspan(85, 115, color="#2ca02c", alpha=0.07, lw=0)
            ax.axhline(100, color="#888", lw=0.6)
            gl = [c["name"] for c in analytes if c["name"] in good_cals]
            ax.set_xticks(range(len(gl)))
            ax.set_xticklabels(gl, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel("measured / nominal (%)")
            ax.legend(fontsize=7, loc="best")
            yl = ax.get_ylim()
            ax.set_ylim(min(yl[0], 75), max(yl[1], 125))
            job.figure("qc_accuracy", "QC accuracy", fig,
                       how="Quality-control samples of known concentration, measured as unknowns. Each marker is one QC injection's "
                           "back-calculated concentration as a percentage of its nominal value; the green band is ±15 %. FDA/ICH M10 accept "
                           "an analytical run when at least two-thirds of all QCs, and half at each level, are within ±15 %.",
                       yours=f"<b>{n_in}/{n_all}</b> QC measurements within ±15 % ({100 * n_in / max(n_all, 1):.0f} %)"
                             + (" — the run meets the 2/3 rule." if n_all and n_in / n_all >= 2 / 3 else " — below the 2/3 rule; results of this run are not reliable.")
                             + (f" Failing levels: {', '.join(qc_flag[:6])}." if qc_flag else ""))
            job.tile(f"{n_in}/{n_all}", "QCs within ±15%")
            if qc_flag:
                job.flag("warn", "QC levels outside ±15 % accuracy or 15 % CV: " + ", ".join(f"<b>{q}</b>" for q in qc_flag[:10]) + ".")
            elif n_all:
                job.flag("ok", f"All QC levels are within ±15 % of nominal with CV ≤ 15 % ({n_in}/{n_all} individual QCs pass).")
        pooled = [i for i in qc_idx if i not in nominal_qcs]
        if pooled and len(pooled) >= 2:
            for c in analytes:
                v = np.array([res[(c["name"], i)]["conc" if c["name"] in good_cals else "response"] for i in pooled], float)
                if np.isfinite(v).sum() > 1:
                    qc_rows.append([c["name"], "pooled QC", str(int(np.isfinite(v).sum())), _fc(np.nanmean(v)), "—",
                                    f"{np.nanstd(v, ddof=1) / np.nanmean(v) * 100:.1f}", "—"])
        if qc_rows:
            job.table("qc_table", "Quality controls", ["Analyte", f"Nominal{u}", "n", f"Mean measured{u}", "Mean accuracy %", "CV %", "Status"], qc_rows,
                      note="Acceptance (FDA 2018 / ICH M10): mean within ±15 % of nominal and CV ≤ 15 % at each level.")
    elif not good_cals and qc_idx and len(qc_idx) >= 2:
        prow = []
        for c in comps:
            v = np.array([res[(c["name"], i)]["response"] for i in qc_idx], float)
            if np.isfinite(v).sum() > 1 and np.nanmean(v) > 0:
                prow.append([c["name"], str(int(np.isfinite(v).sum())), f"{np.nanstd(v, ddof=1) / np.nanmean(v) * 100:.1f}"])
        if prow:
            job.table("qc_cv", "QC precision", ["Target", "QC injections", "Response CV %"], prow,
                      note="Repeat injections of the same QC should give the same response; CV ≤ 15 % is the usual target (≤ 20–30 % in untargeted work).")

    # carry-over / blank interference
    blanks = [i for i in range(nR) if types[i] == "blank"]
    if blanks and good_cals:
        top = max(rows[i]["concentration"] for i in std_idx)
        seq = list(order)
        post = [i for i in blanks if seq.index(i) > 0 and rows[seq[seq.index(i) - 1]]["type"] == "standard"
                and rows[seq[seq.index(i) - 1]]["concentration"] == top]
        use = post or blanks
        crow, bad = [], []
        for c in analytes:
            cal = good_cals.get(c["name"])
            if not cal:
                continue
            lloq_i = [i for i in std_idx if np.isclose(rows[i]["concentration"], cal["lloq"])]
            la = np.mean([res[(c["name"], i)]["area"] for i in lloq_i]) if lloq_i else np.nan
            vals = [res[(c["name"], i)]["area"] / la * 100 if la > 0 else np.nan for i in use]
            mx = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
            crow.append([c["name"], ", ".join(names[i] for i in use[:4]) + (" …" if len(use) > 4 else ""), f"{mx:.1f}" if np.isfinite(mx) else "—",
                         "pass" if np.isfinite(mx) and mx <= 20 else "FAIL" if np.isfinite(mx) else "—"])
            if np.isfinite(mx) and mx > 20:
                bad.append(f"{c['name']} ({mx:.0f}%)")
        kind = "carry-over (blank injected right after the top standard)" if post else "blank interference (no blank follows the top standard, so all blanks are used)"
        job.table("carryover", "Carry-over / blank check", ["Analyte", "Blank(s)", "Max analyte area, % of LLOQ", "≤ 20 %"], crow,
                  note=f"Checked: {kind}. FDA/ICH M10: the analyte response in a blank must not exceed 20 % of the LLOQ response.")
        if bad:
            job.flag("warn", f"{'Carry-over' if post else 'Blank signal'} above 20 % of the LLOQ response: " + ", ".join(f"<b>{b}</b>" for b in bad)
                             + ". Low samples injected after high ones may be overestimated — add wash injections or raise the LLOQ.")
        else:
            job.flag("ok", f"{'Carry-over' if post else 'Blank signal'} is ≤ 20 % of the LLOQ response for every calibrated analyte.")

    # results tables & exports
    long_rows = []
    for c in comps:
        for i in range(nR):
            d = res[(c["name"], i)]
            long_rows.append({
                "sample": names[i], "file": Path(rows[i]["file"]).name, "type": types[i], "group": rows[i]["group"],
                "order": rows[i]["order"], "target": c["name"], "role": "ISTD" if c["is_istd"] else "analyte", "extraction": d["mode"] or "",
                "rt_min": d["rt"] / 60 if d["pk"] else np.nan, "start_min": d["pk"]["start"] / 60 if d["pk"] else np.nan,
                "end_min": d["pk"]["end"] / 60 if d["pk"] else np.nan, "area": d["area"], "height": d["height"], "snr": d["snr"],
                "fwhm_s": d["fwhm"], "points": d["points"], "istd_area": d.get("istd_area"),
                "ion_ratio": d.get("ratios", [np.nan])[0] if d.get("ratios") else np.nan,
                "response": d.get("response"), "concentration": d.get("conc", np.nan), "reportable": d.get("report", ""),
                "nominal": rows[i]["concentration"] if types[i] in ("standard", "qc") else None,
                "units": units, "flags": "; ".join(d["flags"])})
    ldf = pd.DataFrame(long_rows)
    (job.root / "targeted_results.csv").write_text(ldf.to_csv(index=False))
    job.download("Results, long format (CSV)", "targeted_results.csv")
    an = [c["name"] for c in analytes]
    if good_cals:
        wide = pd.DataFrame({c: [res[(c, i)].get("report", "") if c in good_cals else "" for i in order] for c in an},
                            index=[names[i] for i in order])
        wide_num = pd.DataFrame({c: [res[(c, i)].get("conc", np.nan) for i in order] for c in an}, index=wide.index)
        wide.insert(0, "group", [rows[i]["group"] for i in order])
        wide.insert(0, "type", [types[i] for i in order])
        wide.index.name = "sample"
        (job.root / "concentrations.csv").write_text(wide.to_csv())
        wide_num.index.name = "sample"
        (job.root / "concentrations_numeric.csv").write_text(wide_num.to_csv())
        job.download("Concentration matrix (CSV)", "concentrations.csv")
        job.download("Concentration matrix, unrounded incl. out-of-range estimates (CSV)", "concentrations_numeric.csv")
        cols = ["Sample", "Type", "Group"] + [f"{c}{u}" for c in an]
        trow = [[names[i], types[i], rows[i]["group"]] + [_disp(res[(c, i)].get("report", "")) for c in an] for i in order if types[i] != "standard"]
        job.table("concentrations", "Concentrations", cols, trow[:200],
                  note="&lt;LLOQ / &gt;ULOQ: outside the validated range (the extrapolated estimate is in concentrations_numeric.csv and the "
                       "long table); n.d.: no peak. Calibration standards are listed in the calibration table.", csv="concentrations.csv")
        samp = [i for i in range(nR) if types[i] == "sample"]
        if samp:
            n_in = sum(1 for c in an if c in good_cals for i in samp if res[(c, i)].get("report") not in ("<LLOQ", ">ULOQ", "n.d.", "—", ""))
            n_tot = sum(1 for c in an if c in good_cals for i in samp)
            n_hi = sum(1 for c in an if c in good_cals for i in samp if res[(c, i)].get("report") == ">ULOQ")
            n_lo = sum(1 for c in an if c in good_cals for i in samp if res[(c, i)].get("report") in ("<LLOQ", "n.d."))
            job.tile(f"{100 * n_in / max(n_tot, 1):.0f}%", "sample values in range")
            if n_hi:
                job.flag("warn", f"{n_hi} sample value(s) are above the top standard (&gt;ULOQ). Their numbers are extrapolated — dilute and re-inject to report them.")
            if n_lo:
                job.flag("info", f"{n_lo} sample value(s) are below the LLOQ or not detected.")
    else:
        wide = pd.DataFrame({c: [res[(c, i)]["response"] for i in order] for c in an}, index=[names[i] for i in order])
        wide.insert(0, "group", [rows[i]["group"] for i in order])
        wide.insert(0, "type", [types[i] for i in order])
        wide.index.name = "sample"
        (job.root / "concentrations.csv").write_text(wide.to_csv())
        job.download("Response matrix (CSV)", "concentrations.csv")
        rl = "area ratio" if any(c["istd"] for c in analytes) else "area"
        trow = [[names[i], types[i], rows[i]["group"]] + [(fmt_n(res[(c, i)]["response"]) if res[(c, i)]["response"] >= 10 else f"{res[(c, i)]['response']:.3g}")
                                                          if np.isfinite(res[(c, i)]["response"]) else "—" for c in an] for i in order]
        job.table("responses", f"Responses ({rl})", ["Sample", "Type", "Group"] + an, trow[:200], csv="concentrations.csv")
    disp = ldf[ldf.role == "analyte"].head(200)
    job.table("results_long", "All results", ["Sample", "Type", "Target", "RT (min)", "Area", "S/N", "Response", f"Conc.{u}", "Reported", "Flags"],
              [[r.sample, r.type, r.target, f"{r.rt_min:.2f}" if np.isfinite(r.rt_min) else "—", fmt_n(r.area), f"{r.snr:.0f}",
                f"{r.response:.4g}" if r.response is not None and np.isfinite(r.response) else "—", _fc(r.concentration), _disp(r.reportable), r.flags or ""]
               for r in disp.itertuples()], csv="targeted_results.csv",
              note="First 200 analyte rows; the CSV has every target (ISTDs included) and every column (height, FWHM, points, boundaries, ion ratio).")

    # flags summary for peaks
    kinds = {}
    for c in comps:
        for i in range(nR):
            if types[i] == "blank":
                continue
            for f in res[(c["name"], i)]["flags"]:
                k = re.sub(r"[-+]?\d+(\.\d+)?%?|of mean", "", f).strip()
                kinds.setdefault(k, []).append(f"{c['name']}/{names[i]}")
    kinds = {k: v for k, v in kinds.items() if k not in ("below LLOQ", "above ULOQ (extrapolated)", "not detected")}
    if kinds:
        job.flag("warn", "Peaks needing a look: " + "; ".join(f"<b>{k}</b> ×{len(v)} (e.g. {v[0]})" for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1]))[:6])
                 + ". Review them in the chromatogram table.")
    else:
        job.flag("ok", "Every peak in the non-blank injections has ≥ 6 points, S/N above the threshold, the expected RT and a consistent ion ratio.")

    # ============ Step 5: group comparison
    samp = [i for i in range(nR) if types[i] == "sample" and rows[i]["group"]]
    groups = sorted({rows[i]["group"] for i in samp})
    groups = [g for g in groups if sum(rows[i]["group"] == g for i in samp) >= 2]
    if len(groups) >= 2 and analytes:
        _groups(job, analytes, res, rows, samp, groups, good_cals, units)

    # ============ params panel & methods
    _params_panel(job, comps, params, cals)
    _methods(job, comps, analytes, params, good_cals)


def _disp(rep) -> str:
    """Reportable value for display tables: words instead of < / > (the PDF renderer reads those as markup)."""
    if rep is None or (isinstance(rep, float) and not np.isfinite(rep)) or rep == "":
        return "—"
    return {"<LLOQ": "below LLOQ", ">ULOQ": "above ULOQ"}.get(str(rep), str(rep))


def _wlabel(w):
    return {"1/x": "1/x", "1/x2": "1/x²", "none": "none (equal)"}.get(w, w)


def _xic_page(page, res, rows, nR):
    ncol = min(4, len(page))
    nrow = int(np.ceil(len(page) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.5 * nrow + 0.5), squeeze=False)
    types = [r["type"] for r in rows]
    sq = (lambda v: np.sqrt(np.clip(v, 0, None)), lambda v: np.square(v))
    for ax, c in zip(axes.flat, page):
        ymax = 0
        for i in sorted(range(nR), key=lambda i: {"blank": 0, "sample": 1, "qc": 2, "standard": 3}[types[i]]):
            d = res[(c["name"], i)]
            if not len(d["t"]):
                continue
            col = TYPE_COLOR[types[i]]
            t, y = d["t"] / 60, d["y"]
            ax.plot(t, y, color=col, lw=0.6, alpha=0.75)
            pk = d["pk"]
            if pk:
                a, b = pk["i_start"], pk["i_end"] + 1
                line = np.interp(d["t"][a:b], [pk["start"], pk["end"]], [pk["base_start"], pk["base_end"]])
                ax.fill_between(t[a:b], line, np.maximum(y[a:b], line), color=col, alpha=0.10, lw=0)
            vis = y[np.abs(d["t"] - c["rt_s"]) <= max(1.6 * c["win_s"], 45.0)] if c["rt_s"] is not None else y
            ymax = max(ymax, float(np.max(vis)) if len(vis) else 0)
        if c["rt_s"] is not None:
            ax.axvline(c["rt_s"] / 60, color="#555", lw=0.7, ls="--")
            ax.axvspan((c["rt_s"] - c["win_s"]) / 60, (c["rt_s"] + c["win_s"]) / 60, color="#888", alpha=0.07, lw=0)
            half = max(1.6 * c["win_s"], 45.0)
            ax.set_xlim((c["rt_s"] - half) / 60, (c["rt_s"] + half) / 60)
        try:
            ax.set_yscale("function", functions=sq)
        except Exception:  # noqa: BLE001
            pass
        ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)
        nfl = sum(1 for i in range(nR) if types[i] != "blank" and
                  any(not f.startswith(("below LLOQ", "above ULOQ", "not detected")) for f in res[(c["name"], i)]["flags"]))
        ax.set_title(f"{c['name']}{' (ISTD)' if c['is_istd'] else ''}", fontsize=9, loc="left")
        ax.text(0.99, 0.97, (f"{nfl} flagged" if nfl else "") , transform=ax.transAxes, fontsize=7, color="#c9304f", ha="right", va="top")
        ax.text(0.01, 0.97, c["transitions"][0]["label"], transform=ax.transAxes, fontsize=6.5, color="#666", ha="left", va="top")
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_n(v)))
        ax.set_xlabel("RT (min)", fontsize=7)
    for ax in list(axes.flat)[len(page):]:
        ax.axis("off")
    present = sorted(set(types), key=["standard", "qc", "sample", "blank"].index)
    handles = [plt.Line2D([], [], color=TYPE_COLOR[t], lw=2) for t in present]
    fig.legend(handles, [TYPE_LABEL[t] for t in present], loc="upper center", ncol=len(present), fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _cal_page(page, cals, res, rows, units):
    ncol = min(4, len(page))
    nrow = int(np.ceil(len(page) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow + 0.3), squeeze=False)
    types = [r["type"] for r in rows]
    for ax, c in zip(axes.flat, page):
        cal = cals[c["name"]]
        x, y, inc = cal["x"], cal["y"], cal["included"]
        ok = np.isfinite(y) & (y > 0)
        if not cal["ok"]:
            ax.scatter(x[ok], y[ok], s=18, facecolor="none", edgecolor=TYPE_COLOR["standard"])
            ax.set_title(f"{c['name']}", fontsize=9, loc="left")
            ax.text(0.5, 0.5, "no curve:\n" + cal["why"], transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#c9304f")
            if ok.any():
                ax.set_xscale("log"); ax.set_yscale("log")
            continue
        xmin = x[inc].min()
        xmax = x[inc].max()
        g_in = np.geomspace(cal["lloq"], cal["uloq"], 100)
        ax.plot(g_in, predict(cal["coef"], g_in), color=TYPE_COLOR["standard"], lw=1.2)
        unk = [(res[(c["name"], i)]["conc"], res[(c["name"], i)]["response"], types[i]) for i in range(len(rows))
               if types[i] in ("sample", "qc") and np.isfinite(res[(c["name"], i)].get("conc", np.nan)) and res[(c["name"], i)]["conc"] > 0]
        lo_x = min([xmin / 1.5] + [u[0] for u in unk])
        hi_x = max([xmax * 1.5] + [u[0] for u in unk])
        for a_, b_ in ((lo_x, cal["lloq"]), (cal["uloq"], hi_x)):
            if b_ > a_:
                g = np.geomspace(a_, b_, 40)
                ax.plot(g, predict(cal["coef"], g), color=TYPE_COLOR["standard"], lw=0.8, ls="--")
        ax.scatter(x[inc & ok], y[inc & ok], s=20, color=TYPE_COLOR["standard"], zorder=3)
        ax.scatter(x[~inc & ok], y[~inc & ok], s=20, facecolor="white", edgecolor=TYPE_COLOR["standard"], zorder=3)
        for cx, ry, t in unk:
            if ry > 0:
                if t == "qc":
                    ax.scatter(cx, ry, marker="D", s=14, color=QC_COLOR, zorder=4)
                else:
                    ax.scatter(cx, ry, marker="|", s=90, color=TYPE_COLOR["sample"], zorder=4, linewidths=1.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(c["name"], fontsize=9, loc="left")
        ax.text(0.03, 0.97, f"R² {cal['r2']:.4f}\n{cal['model_used']}, {_wlabel(cal['weighting'])}", transform=ax.transAxes,
                fontsize=7, va="top", color="#c9304f" if cal["r2"] < 0.99 else "#333")
        ax.set_xlabel(f"concentration{' (' + units + ')' if units else ''}", fontsize=7)
        ax.set_ylabel("area ratio" if c["istd"] else "area", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(page):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _groups(job, analytes, res, rows, samp, groups, good_cals, units):
    from scipy import stats
    job.section("groups", "Step 5", "Group comparison",
                f"{'Concentrations' if good_cals else 'Responses'} of the study samples by group. Welch's t-test on log2 values "
                "(does not assume equal variances; the log makes fold changes symmetric), Benjamini–Hochberg FDR across all tests.")
    pairs = [(a, b) for k, a in enumerate(groups) for b in groups[k + 1:]]
    stat_rows = []
    for c in analytes:
        key = "conc" if c["name"] in good_cals else "response"
        for a, b in pairs:
            va = np.array([res[(c["name"], i)].get(key, np.nan) for i in samp if rows[i]["group"] == a], float)
            vb = np.array([res[(c["name"], i)].get(key, np.nan) for i in samp if rows[i]["group"] == b], float)
            va, vb = va[np.isfinite(va) & (va > 0)], vb[np.isfinite(vb) & (vb > 0)]
            if len(va) >= 2 and len(vb) >= 2:
                t, p = stats.ttest_ind(np.log2(vb), np.log2(va), equal_var=False)
                fc = float(np.mean(np.log2(vb)) - np.mean(np.log2(va)))
            else:
                p, fc = np.nan, np.nan
            stat_rows.append({"target": c["name"], "group_a": a, "group_b": b, "n_a": len(va), "n_b": len(vb),
                              "mean_a": float(np.mean(va)) if len(va) else np.nan, "mean_b": float(np.mean(vb)) if len(vb) else np.nan,
                              "log2fc": fc, "p": float(p) if np.isfinite(p) else np.nan})
    q = _bh([r["p"] for r in stat_rows])
    for r, qq in zip(stat_rows, q):
        r["q"] = qq
    sdf = pd.DataFrame(stat_rows)
    (job.root / "group_tests.csv").write_text(sdf.to_csv(index=False))
    job.download("Group comparisons (CSV)", "group_tests.csv")
    gcol = {g: PALETTE[k % len(PALETTE)] for k, g in enumerate(groups)}
    pages = [analytes[k:k + PAGE] for k in range(0, len(analytes), PAGE)]
    for p, page in enumerate(pages):
        ncol = min(4, len(page))
        nrow = int(np.ceil(len(page) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(max(2.4, 0.7 * len(groups) + 1.2) * ncol, 2.6 * nrow + 0.2), squeeze=False)
        rng = np.random.default_rng(0)
        for ax, c in zip(axes.flat, page):
            key = "conc" if c["name"] in good_cals else "response"
            allv = []
            for k, g in enumerate(groups):
                ii = [i for i in samp if rows[i]["group"] == g]
                v = np.array([res[(c["name"], i)].get(key, np.nan) for i in ii], float)
                rep = [res[(c["name"], i)].get("report", "") for i in ii]
                ok = np.isfinite(v)
                xs = k + rng.uniform(-0.12, 0.12, ok.sum())
                oor = np.array([r in ("<LLOQ", ">ULOQ", "n.d.") for r in rep])[ok]
                ax.scatter(xs[~oor], v[ok][~oor], s=18, color=gcol[g], zorder=3)
                ax.scatter(xs[oor], v[ok][oor], s=18, facecolor="white", edgecolor=gcol[g], zorder=3)
                if ok.any():
                    ax.hlines(np.mean(v[ok]), k - 0.25, k + 0.25, color="#333", lw=1.2)
                allv += list(v[ok][v[ok] > 0])
            if allv and max(allv) / max(min(allv), 1e-12) > 30:
                ax.set_yscale("log")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, fontsize=7, rotation=30 if len(groups) > 3 else 0)
            st = [r for r in stat_rows if r["target"] == c["name"] and np.isfinite(r["p"])]
            best = min(st, key=lambda r: r["p"]) if st else None
            ax.set_title(c["name"], fontsize=9, loc="left")
            if best:
                ax.text(0.98, 0.97, f"q = {best['q']:.2g}" + (f" ({best['group_b']} vs {best['group_a']})" if len(pairs) > 1 else ""),
                        transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#c9304f" if best["q"] < 0.05 else "#555")
            ax.set_ylabel((f"conc. ({units})" if units else "concentration") if key == "conc" else ("area ratio" if c["istd"] else "area"), fontsize=7)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(page):]:
            ax.axis("off")
        fig.tight_layout()
        sig = sdf[(sdf.q < 0.05) & sdf.target.isin([c["name"] for c in page])].sort_values("q")
        job.figure(f"groups_{p + 1}", "Targets by group" + (f" ({p + 1}/{len(pages)})" if len(pages) > 1 else ""), fig, wide=len(page) > 2,
                   how="Each dot is one study sample, the bar the group mean. Hollow dots are outside the calibrated range (&lt;LLOQ, &gt;ULOQ) "
                       "or not detected — their values are estimates. q is the Benjamini–Hochberg-adjusted p-value of Welch's t-test on "
                       "log2 values; q &lt; 0.05 (red) is the usual threshold. With few samples per group only large, consistent "
                       "differences reach significance.",
                   yours=("Different between groups (q &lt; 0.05): " + ", ".join(
                       f"<b>{r.target}</b> {2 ** abs(r.log2fc):.1f}× {'higher' if r.log2fc > 0 else 'lower'} in {r.group_b} (q = {r.q:.2g})"
                       for r in sig.itertuples()) + "." if len(sig) else "No target differs between groups at q &lt; 0.05."))
    tbl = [[r["target"], f"{r['group_b']} vs {r['group_a']}", f"{r['n_b']}/{r['n_a']}", _fc(r["mean_b"]), _fc(r["mean_a"]),
            f"{2 ** r['log2fc']:.2f}" if np.isfinite(r["log2fc"]) else "—", f"{r['p']:.2g}" if np.isfinite(r["p"]) else "—",
            f"{r['q']:.2g}" if np.isfinite(r["q"]) else "—"] for r in stat_rows]
    job.table("group_tests", "Group comparisons", ["Target", "Comparison", "n", "Mean (2nd)", "Mean (1st)", "Fold change", "p", "q (BH)"], tbl[:200],
              csv="group_tests.csv")
    nsig = int((sdf.q < 0.05).sum())
    if nsig:
        job.flag("info", f"{nsig} target/group comparison(s) differ at FDR 5 % (Welch t-test on log2 values).")


def _params_panel(job, comps, p, cals):
    n_ppm = sum(1 for c in comps if c["ppm"])
    n_win = sum(1 for c in comps if c["win_list"] is not None)
    any_ms1 = any("ms1" in c["mode"] or "prm" in c["mode"] for c in comps)
    job.param_group("Extraction & integration", "Values given per target in the target list override these defaults.")
    job.param("ppm", "m/z tolerance (ppm)", p["ppm"],
              (f"Full-scan/PRM traces sum every centroid within ±{p['ppm']:g} ppm of the target m/z. " if any_ms1 else
               "Not used: every target here is an SRM transition (matched within ±0.3 Da). ")
              + (f"{n_ppm} target(s) set their own ppm in the list." if n_ppm else "")
              + " Orbitrap/TOF data usually need 5–10 ppm; widen it if peaks look ragged or are missing in some files.",
              source="user" if p.get("ppm") != 10.0 else "default")
    job.param("rt_window", "RT search window (± min)", p["rt_window"],
              f"The peak nearest the expected RT within ±{p['rt_window']:g} min is integrated"
              + (f" ({n_win} target(s) carry their own window)" if n_win else "")
              + ". Peaks more than half a window from the expected RT are flagged. Narrow it if a neighbouring isomer is picked; "
                "widen it if the chromatography drifts.", source="user" if p["rt_window"] != 0.5 else "default")
    job.param("smooth", "Smoothing (points)", p["smooth"],
              f"Savitzky–Golay smoothing over {p['smooth']} points is used only to find the apex and boundaries; the area is always "
              "integrated on the raw trace. Use 0 to switch it off (very sparse SRM data), 7–11 for noisy traces.",
              source="user" if p["smooth"] != 5 else "default", type="int")
    job.param("min_snr", "Minimum S/N", p["min_snr"],
              f"Peaks with S/N below {p['min_snr']:g} are flagged (10 is the usual limit of quantification criterion). The value only "
              "flags — it never removes a peak.", source="user" if p["min_snr"] != 10 else "default")
    job.param_group("Calibration")
    job.param("model", "Curve model", p["model"],
              ("Straight line: response = intercept + slope × concentration. Use quadratic only when the accuracy plot shows a "
               "systematic bend at the top (detector saturation) and at least 4 levels." if p["model"] == "linear" else
               "Quadratic: accounts for curvature at high concentrations (saturation). Regulators expect the simplest model that "
               "fits, so check that linear really fails first."), source="user" if p["model"] != "linear" else "default",
              type="select", options=["linear", "quadratic"])
    job.param("weighting", "Weighting", {"1/x2": "1/x²"}.get(p["weighting"], p["weighting"]),
              f"Weighting {_wlabel(p['weighting'])}. Mass-spec error grows with concentration, so an unweighted fit is dominated by the "
              "top standards and misses the low end. 1/x (default) is the usual compromise; 1/x² gives the low standards more say "
              "when the curve spans more than ~3 orders of magnitude.",
              source="user" if p["weighting"] != "1/x" else "default", type="select", options=["1/x", "1/x²", "none"])
    nex = sum(int((~v["included"]).sum()) for v in cals.values() if v.get("ok")) if cals else 0
    job.param("auto_exclude", "Auto-exclude failing standards", bool(p["auto_exclude"]),
              (f"On: the standard furthest outside ±15 % (±20 % at LLOQ) is dropped and the curve refitted until all pass "
               f"({nex} standard(s) excluded or not detected across all curves). Document exclusions — FDA/ICH M10 allow at most 25 %."
               if p["auto_exclude"] else
               "Off: every detected standard is in the fit and failures are only flagged. Turn on to drop the worst standard "
               "repeatedly until the rest pass (FDA/ICH M10 allow excluding up to 25 % of standards with a documented reason)."),
              source="user" if p["auto_exclude"] else "default", type="bool")


def _methods(job, comps, analytes, p, good_cals):
    modes = sorted({m for c in comps for m in c["mode"].split("/") if m != "not found"})
    mode_txt = {"ms1": f"full-scan extracted-ion chromatograms (±{p['ppm']:g} ppm)", "srm": "stored SRM/MRM chromatograms (Q1/Q3 matched within ±0.3 Da)",
                "srm-spectra": "SRM transitions stored as MS2 spectra (precursor and product within ±0.3 Da)",
                "prm": f"PRM product-ion chromatograms from high-resolution MS2 spectra (±{p['ppm']:g} ppm)"}
    job.method("Extraction", f"{len(analytes)} analytes and {len(comps) - len(analytes)} internal standards were quantified from "
               + "; ".join(mode_txt[m] for m in modes if m in mode_txt) + ". Formula-defined targets were converted to m/z with monoisotopic "
               "masses (pyteomics) and the adduct given (default [M+H]+ / [M−H]− by polarity).")
    job.method("Integration", f"For each target the peak nearest the expected retention time (±{p['rt_window']:g} min, or the per-target window; "
               "for analytes with a co-eluting internal standard, centred on the ISTD's observed RT) was found on a Savitzky–Golay-smoothed "
               f"trace ({p['smooth']} points) and integrated on the raw trace by the trapezoid rule above a straight baseline between the peak "
               "boundaries (drop to baseline at valleys). Qualifier ions were integrated within the quantifier's boundaries and their ratio "
               "compared with the mean ratio in the calibration standards, using the tolerances of Commission Decision 2002/657/EC. "
               f"Peaks with fewer than 6 points, S/N &lt; {p['min_snr']:g} (height over the robust SD of the local background), or an RT shift "
               "over half the window were flagged.")
    if good_cals:
        job.method("Calibration", f"Responses (analyte area, or analyte/ISTD area ratio) of the calibration standards were fitted by weighted least "
                   f"squares ({p['model']}, weighting {_wlabel(p['weighting'])}). Standards were back-calculated and judged against the "
                   "acceptance criteria of the FDA Bioanalytical Method Validation Guidance for Industry (2018) and ICH M10 (2022): ±15 % of "
                   "nominal, ±20 % at the LLOQ, at least 75 % of standards passing"
                   + ("; failing standards were excluded one at a time, worst first, and the curve refitted" if p["auto_exclude"] else "")
                   + ". The reportable range runs from the lowest to the highest passing standard. LOD and LOQ were estimated as 3.3·σ/S and "
                   "10·σ/S (ICH Q2(R2)) from an unweighted regression through the four lowest levels. Samples outside the range are reported "
                   "as &lt;LLOQ or &gt;ULOQ. QCs were judged at ±15 % accuracy and ≤ 15 % CV; carry-over as the analyte area in the blank after "
                   "the top standard relative to the LLOQ area (limit 20 %).")
    job.method("Statistics", "Where study samples carry groups, targets were compared between groups with Welch's t-test on log2 values "
               "and Benjamini–Hochberg false-discovery-rate correction (scipy).")
