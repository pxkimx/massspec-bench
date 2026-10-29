"""Vendor raw formats → `msio.Run`, on macOS (Apple Silicon), with nothing but Python packages and one
downloaded helper.

    thermo_raw          Thermo .raw      ThermoRawFileParser (self-contained osx-arm64 build) → indexed mzML
                                         with Thermo's own centroiding → pyOpenMS
    bruker_tdf          timsTOF .d       timsrust (Rust, via timsrust_pyo3) + the analysis.tdf SQLite tables
    agilent_masshunter  Agilent .d       rainbow-api (MSPeak.bin centroids / MSProfile.bin profile, DAD, FID…)
    agilent_chemstation Agilent .D/.ms   rainbow-api (ChemStation MSD .ms, UV/DAD .ch/.uv, FID, ELSD, CAD)
    waters_raw          Waters .raw      rainbow-api (MassLynx quadrupole / SQ / TQ functions, PDA, ELSD, CAD)
    andi_cdf            ANDI-MS .cdf     scipy's netCDF3 reader (the universal GC-MS/LC-MS export)

Every reader returns a Run: spectra (retention time in seconds), plus any non-MS detector traces (UV, ELSD,
CAD, FID, SIM/MRM) in `run.chromatograms` so the explorer can show them next to the TIC. Failures that the
user can act on raise `UserFacingError` saying what is wrong and what to do (usually: convert to mzML with
msconvert, `msio.CONVERT_HELP`).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import msio, tools
from .common import UserFacingError, log_exc


def _say(progress, msg: str):
    if progress:
        try:
            progress(msg)
        except Exception:  # noqa: BLE001 - a broken progress callback must not stop the read
            pass


def _is_fatal(e: BaseException) -> bool:
    """Rust panics from timsrust arrive as pyo3 PanicException, a BaseException; only real interrupts are fatal."""
    return isinstance(e, (KeyboardInterrupt, SystemExit, MemoryError))


# ====================================================================== Thermo .raw

_TRFP_OK: set[str] = set()


def _runnable(exe: Path) -> Path:
    """Make sure a downloaded helper actually starts.

    Apple Silicon refuses to run an arm64 executable that carries no code signature at all (it is killed
    with SIGKILL, exit 137, before printing anything). ThermoRawFileParser's osx-arm64 release is such a
    binary, so it is given an ad-hoc signature once — the same thing Homebrew does for the binaries it
    builds. Nothing is signed with any identity and no system setting is changed.
    """
    key = str(exe)
    if key in _TRFP_OK:
        return exe
    def probe():
        return subprocess.run([str(exe), "--version"], capture_output=True, timeout=120)
    try:
        r = probe()
        if r.returncode in (-9, 137) and sys.platform == "darwin":
            subprocess.run(["codesign", "--force", "--sign", "-", str(exe)], capture_output=True, timeout=120)
            r = probe()
    except (OSError, subprocess.TimeoutExpired) as e:
        raise UserFacingError(f"ThermoRawFileParser ({exe}) could not be started: {e}. Delete the folder "
                              f"{exe.parent} so it is downloaded again, then retry.") from e
    if r.returncode in (-9, 137):
        raise UserFacingError(f"macOS stopped ThermoRawFileParser from running ({exe}). Allow it in System "
                              "Settings → Privacy & Security, or delete that folder so it is downloaded again.")
    _TRFP_OK.add(key)
    return exe


def _trfp_meta(js: Path) -> dict:
    """The useful bits of ThermoRawFileParser's metadata JSON (instrument, serial, date, method)."""
    out = {}
    try:
        d = json.loads(js.read_text(errors="replace"))
    except Exception:  # noqa: BLE001
        return out
    flat = {}
    for sect in d.values():
        if isinstance(sect, list):
            for p in sect:
                if isinstance(p, dict) and p.get("name") and p.get("value") not in (None, "", "None"):
                    flat.setdefault(p["name"], p["value"])
    for key, names in {
        "instrument": ["Thermo Scientific instrument model", "instrument model"],
        "serial": ["instrument serial number"],
        "date": ["Content Creation Date"],
        "method": ["device acquisition method"],
        "software": ["Software Version"],
        "n_ms1_vendor": ["Number of MS1 spectra"],
        "n_ms2_vendor": ["Number of MS2 spectra"],
    }.items():
        for n in names:
            if n in flat:
                out[key] = flat[n]
                break
    if out.get("instrument") and not out["instrument"].lower().startswith("thermo"):
        out["instrument"] = "Thermo " + out["instrument"]
    return out


def read_thermo(path: Path, progress=None, profile: bool = False) -> msio.Run:
    """Thermo .raw via ThermoRawFileParser → indexed mzML (temp) → pyOpenMS.

    profile=False (default) uses Thermo's own centroiding for every MS level (what every downstream tool
    expects); profile=True keeps the profile data (-p) and leaves centroiding to OpenMS later.
    """
    path = Path(path)
    try:
        exe = tools.ensure_tool("ThermoRawFileParser", progress=progress)
    except tools.ToolUnavailable as e:
        raise UserFacingError(f"{path.name}: Thermo .raw files are read with ThermoRawFileParser, which could not "
                              f"be installed. {e} Alternatively convert the file to mzML yourself. "
                              + msio.CONVERT_HELP) from e
    exe = _runnable(Path(exe))
    tmp = Path(tempfile.mkdtemp(prefix="msb_trfp_"))
    try:
        _say(progress, f"Converting {path.name} with ThermoRawFileParser")
        cmd = [str(exe), "-i", str(path), "-o", str(tmp), "-f", "2", "-m", "0", "-l", "3", "-e"]
        if profile:
            cmd.append("-p")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        except OSError as e:
            raise UserFacingError(f"ThermoRawFileParser could not be started: {e}") from e
        mzmls = sorted(tmp.glob("*.mzML"))
        if r.returncode != 0 or not mzmls:
            msg = " ".join((r.stderr or r.stdout or "").split())
            msg = re.split(r"\s*Stack ?trace", msg, flags=re.I)[0].replace(str(path), path.name)[-400:]
            hint = ""
            low = msg.lower()
            if any(k in low for k in ("not a valid", "corrupt", "unable to open", "crc failed")) or not msg:
                hint = " The file may be incomplete (still being copied / acquisition aborted) or not a Thermo .raw file."
            raise UserFacingError(f"ThermoRawFileParser could not convert {path.name} (exit {r.returncode}): "
                                  f"{msg or 'no output'}.{hint}")
        _say(progress, f"Reading converted {path.name}")
        run = msio.read_openms(mzmls[0], "mzML", name=path.stem, source=str(path))
        meta = dict(run.meta)
        js = sorted(tmp.glob("*metadata.json"))
        if js:
            meta.update(_trfp_meta(js[0]))
        meta.update(vendor="Thermo", reader="ThermoRawFileParser " + _trfp_version(exe),
                    centroiding="Thermo (vendor) peak picking" if not profile else "none (profile kept)")
        run.meta = meta
        run.fmt = "thermo_raw"
        # TRFP writes the run's TIC and base-peak trace as chromatograms; both are recomputed from the spectra
        run.chromatograms = [c for c in run.chromatograms if not re.match(r"(?i)(basepeak|bpc|tic)", str(c["id"]))]
        return run
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _trfp_version(exe) -> str:
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60)
        return (r.stdout or "").strip().splitlines()[0] if r.stdout.strip() else ""
    except Exception:  # noqa: BLE001
        return ""


# ====================================================================== Bruker timsTOF (TDF)

MAX_PEAKS = 2e8          # above this the MS1 frames are thinned (top-N centroids per frame)
MS1_BUDGET = 1.5e8       # peaks the MS1 frames may use in total once thinning kicks in


def _tdf_tables(tdf: Path):
    con = sqlite3.connect(f"file:{tdf}?mode=ro&immutable=1", uri=True)
    try:
        tables = {r[0].lower() for r in con.execute("select name from sqlite_master where type='table'")}
        cols = {r[1] for r in con.execute("pragma table_info(Frames)")}
        want = [c for c in ("Id", "Time", "Polarity", "ScanMode", "MsMsType", "NumPeaks") if c in cols]
        frames = con.execute(f"select {','.join(want)} from Frames order by Id").fetchall()
        frames = {c: np.array([f[k] for f in frames]) for k, c in enumerate(want)}
        gm = {}
        if "globalmetadata" in tables:
            gm = {str(k): str(v) for k, v in con.execute("select Key, Value from GlobalMetadata")}
        n_prec = con.execute("select count(*) from Precursors").fetchone()[0] if "precursors" in tables else 0
        return frames, gm, n_prec
    finally:
        con.close()


def _centroid_tof(tof: np.ndarray, inten: np.ndarray, a: float, b: float):
    """Sum one frame over ion mobility and centroid it along the TOF axis.

    Intensities of the same TOF index are added exactly (that is the collapse over mobility); a peak is then
    a run of neighbouring TOF bins, split at gaps and at valleys between two maxima, reported at its
    intensity-weighted m/z with its summed intensity. m/z = (a + b·tof)², timsrust's calibration.
    """
    if not len(tof):
        return np.array([]), np.array([], dtype=np.float32)
    dense = np.bincount(tof, weights=inten)
    u = np.flatnonzero(dense)
    v = dense[u]
    brk = np.zeros(len(u), dtype=bool)
    if len(u) > 1:
        brk[1:] = np.diff(u) > 1
    if len(u) > 2:
        valley = (v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:])
        brk[1:-1] |= valley
    seg = np.cumsum(brk)
    mz = (a + b * u.astype(np.float64)) ** 2
    s = np.bincount(seg, weights=v)
    w = np.bincount(seg, weights=v * mz)
    return w / s, s.astype(np.float32)


def read_bruker_tdf(path: Path, progress=None) -> msio.Run:
    """timsTOF .d (TDF): MS1 frames collapsed over ion mobility, MS2 per DDA precursor / DIA window.

    MS1: every MS1 frame is read with timsrust, summed over its ~700–900 mobility scans (same TOF index →
    same m/z) and centroided along m/z. MS2: timsrust's SpectrumReader — one spectrum per DDA-PASEF
    precursor (all its PASEF frames merged, with precursor m/z, charge, intensity and 1/K0), or one per
    diaPASEF isolation window per frame (precursor m/z = window centre).
    """
    path = Path(path)
    tdf = path / "analysis.tdf"
    if not (path / "analysis.tdf_bin").exists():
        raise UserFacingError(f"{path.name} has analysis.tdf but no analysis.tdf_bin (the spectra). Copy the whole "
                              ".d folder again — the acquisition may be incomplete.")
    try:
        import timsrust_pyo3 as tr
    except ImportError as e:
        raise UserFacingError("Reading timsTOF data needs the timsrust_pyo3 package, which is missing from this "
                              "installation. Reinstall MassSpec Bench, or convert the file to mzML. " + msio.CONVERT_HELP) from e
    try:
        frames, gm, n_prec = _tdf_tables(tdf)
    except sqlite3.DatabaseError as e:
        raise UserFacingError(f"{path.name}: analysis.tdf is not a readable SQLite database ({e}). The file is "
                              "damaged or was copied while the instrument was still writing it.") from e
    fid = frames["Id"].astype(np.int64)
    ftime = frames["Time"].astype(np.float64)
    fpol = np.array([-1 if str(p).strip() == "-" else 1 for p in frames.get("Polarity", ["+"] * len(fid))], dtype=np.int8)
    msms = frames.get("MsMsType", np.zeros(len(fid))).astype(int)
    npk = frames.get("NumPeaks", np.zeros(len(fid))).astype(np.float64)
    ms1 = np.flatnonzero(msms == 0)

    try:
        md = tr.Metadata(str(tdf))
        n_samples = int(float(gm.get("DigitizerNumSamples", 0))) or 400000
        lo, hi = md.resolve_mzs([0, n_samples])
        a = float(np.sqrt(lo))
        b = (float(np.sqrt(hi)) - a) / n_samples
        fr = tr.FrameReader(str(path))
    except BaseException as e:  # noqa: BLE001
        if _is_fatal(e):
            raise
        raise UserFacingError(f"{path.name}: timsrust could not open this timsTOF file ({type(e).__name__}: {e}). "
                              "It may be damaged or from a newer TDF schema. " + msio.CONVERT_HELP) from e

    notes = []
    raw_ms1 = float(npk[ms1].sum())
    cap = None
    if raw_ms1 * 0.6 + float(npk[msms != 0].sum()) * 0.3 > MAX_PEAKS and len(ms1):
        cap = max(3000, int(MS1_BUDGET / len(ms1)))
    thinned = 0
    spectra = []
    _say(progress, f"Reading {len(ms1)} MS1 frames of {path.name}")
    for k, i in enumerate(ms1):
        if k and k % 250 == 0:
            _say(progress, f"MS1 frame {k}/{len(ms1)}")
        try:
            f = fr.read_frame(int(fid[i]) - 1)
        except BaseException as e:  # noqa: BLE001
            if _is_fatal(e):
                raise
            notes.append(f"frame {fid[i]} unreadable ({type(e).__name__})")
            continue
        mz, it = _centroid_tof(np.asarray(f.tof_indices, dtype=np.int64), np.asarray(f.intensities, dtype=np.float64), a, b)
        if cap and len(mz) > cap:
            keep = np.sort(np.argpartition(it, -cap)[-cap:])
            mz, it = mz[keep], it[keep]
            thinned += 1
        spectra.append({"rt": float(ftime[i]), "ms_level": 1, "mz": mz, "intensity": it, "polarity": int(fpol[i]),
                        "centroided": True, "scan_id": f"frame={fid[i]}"})
    if thinned:
        notes.append(f"large file: each MS1 frame keeps its {cap:,} most intense centroids ({thinned} of {len(ms1)} "
                     "frames were thinned) to keep memory reasonable; low-abundance MS1 noise is dropped, MS2 is complete")

    n_ms2 = 0
    if (msms != 0).any():
        _say(progress, f"Reading MS2 spectra of {path.name}")
        pol_by_frame = dict(zip(fid.tolist(), fpol.tolist()))
        try:
            sr = tr.SpectrumReader(str(path))
            n = len(sr)
            for k in range(n):
                if k and k % 5000 == 0:
                    _say(progress, f"MS2 spectrum {k}/{n}")
                s = sr.get(k)
                p = s.precursor
                mz = np.asarray(s.mz_values, dtype=np.float64)
                it = np.asarray(s.intensities, dtype=np.float32)
                pmz = p.mz if p is not None and p.mz else s.isolation_mz
                spectra.append({
                    "rt": float(p.rt) if p is not None else np.nan, "ms_level": 2, "mz": mz, "intensity": it,
                    "polarity": pol_by_frame.get(int(p.frame_index) if p is not None and p.frame_index is not None else -1, int(fpol[0]) if len(fpol) else 0),
                    "precursor_mz": float(pmz) if pmz else np.nan,
                    "precursor_charge": int(p.charge) if p is not None and p.charge else 0,
                    "precursor_intensity": float(p.intensity) if p is not None and p.intensity else np.nan,
                    "ion_mobility": float(p.im) if p is not None and p.im else None,
                    "centroided": True, "scan_id": f"index={s.index}",
                    "isolation_width": float(s.isolation_width or 0),
                })
                n_ms2 += 1
        except BaseException as e:  # noqa: BLE001
            if _is_fatal(e):
                raise
            notes.append(f"MS2 spectra could not be read ({type(e).__name__}: {str(e)[:120]}); MS1 only")
            log_exc("bruker ms2")
    spectra = [s for s in spectra if np.isfinite(s["rt"])]
    if not spectra:
        raise UserFacingError(f"{path.name}: no spectra could be read from this timsTOF file. " + msio.CONVERT_HELP)

    acq = {8: "DDA-PASEF", 9: "diaPASEF", 2: "MRM/PRM", 10: "PRM-PASEF"}
    kinds = sorted({acq.get(int(x), f"MsMsType {x}") for x in np.unique(msms) if x != 0})
    meta = {
        "vendor": "Bruker", "instrument": ("Bruker " + gm.get("InstrumentName", "timsTOF")).strip(),
        "date": gm.get("AcquisitionDateTime", ""), "serial": gm.get("InstrumentSerialNumber", ""),
        "software": " ".join(x for x in (gm.get("AcquisitionSoftware", ""), gm.get("AcquisitionSoftwareVersion", "")) if x),
        "method": gm.get("MethodName", ""), "sample": gm.get("SampleName", ""),
        "acquisition": ", ".join(kinds) or "MS1 only", "n_frames": int(len(fid)), "n_ms1_frames": int(len(ms1)),
        "n_precursors": int(n_prec), "im_range": [gm.get("OneOverK0AcqRangeLower"), gm.get("OneOverK0AcqRangeUpper")],
        "reader": "timsrust_pyo3 " + _version("timsrust_pyo3"),
        "mz_calibration": "timsrust linear TOF→m/z (sqrt-linear between MzAcqRangeLower/Upper); not Bruker's "
                          "temperature-compensated SDK calibration, so m/z can differ by a few ppm from DataAnalysis",
        "ms1": "frames summed over ion mobility, centroided along m/z (TOF bins split at gaps/valleys)",
        "intensity": "raw detector counts (Frames.SummedIntensities is the same scaled to 100 ms accumulation)",
    }
    if notes:
        meta["note"] = "; ".join(notes)
    return msio.from_lists(path.stem, str(path), "bruker_tdf", spectra, meta=meta)


def _version(pkg: str) -> str:
    from importlib.metadata import version
    try:
        return version(pkg)
    except Exception:  # noqa: BLE001
        return ""


# ====================================================================== rainbow helpers (Agilent, Waters)

_LZF_LOCK = threading.Lock()


def _lzf_fallback():
    """rainbow needs python-lzf for LZF-compressed MSProfile.bin. That package ships no wheels (it is built
    from source, which needs Apple's command-line tools), so if it is missing a small pure-Python LZF
    decoder stands in for it — slower, but byte-identical."""
    try:
        import lzf  # noqa: F401
        return
    except ImportError:
        pass
    import types

    def decompress(data, out_len):
        data = bytes(data)
        out = bytearray(out_len)
        ip = op = 0
        n = len(data)
        while ip < n:
            ctrl = data[ip]
            ip += 1
            if ctrl < 32:
                ctrl += 1
                out[op:op + ctrl] = data[ip:ip + ctrl]
                ip += ctrl
                op += ctrl
            else:
                ln = ctrl >> 5
                ref = op - ((ctrl & 0x1F) << 8) - 1
                if ln == 7:
                    ln += data[ip]
                    ip += 1
                ref -= data[ip]
                ip += 1
                ln += 2
                if ref + ln <= op:
                    out[op:op + ln] = out[ref:ref + ln]
                else:
                    for q in range(ln):
                        out[op + q] = out[ref + q]
                op += ln
        return bytes(out[:op])

    with _LZF_LOCK:
        if "lzf" not in sys.modules:
            sys.modules["lzf"] = types.SimpleNamespace(decompress=decompress, __name__="lzf", _msb_fallback=True)


_CAPTURE_LOCK = threading.Lock()
_RAW_WIDTH = 977.0     # sentinel bin width: "do not bin, hand me the raw (m/z, intensity) pairs"


@contextmanager
def _raw_pairs():
    """Read ChemStation .ms and Waters MS functions without rainbow's m/z binning.

    rainbow bins every scan onto a shared m/z grid (1 Da by default), which throws away the 0.05–0.1 Da
    (Agilent MSD) or ~0.03 Da (Waters) precision the file records and costs scans × grid memory. Its
    parsers decode the raw pairs and pass them to one function, `bin_datapairs`; while this context is
    active that function, when called with the sentinel width, keeps the raw pairs and returns a one-column
    placeholder whose label (-1-k) says which captured block belongs to which DataFile.
    """
    import rainbow._binning as rbin
    from rainbow.agilent import chemstation
    from rainbow.waters import masslynx
    orig = rbin.bin_datapairs
    captured = []

    def capture(keys, values, pair_counts, bin_width, display_precision=None, data_dtype=np.int64, labels_only=False):
        if bin_width != _RAW_WIDTH:
            return orig(keys, values, pair_counts, bin_width, display_precision=display_precision,
                        data_dtype=data_dtype, labels_only=labels_only)
        captured.append((np.asarray(keys, dtype=np.float64), np.asarray(values, dtype=np.float64),
                         np.asarray(pair_counts, dtype=np.int64)))
        return np.array([-float(len(captured))]), np.zeros((len(pair_counts), 1), dtype=data_dtype)

    with _CAPTURE_LOCK:
        mods = [m for m in (chemstation, masslynx) if hasattr(m, "bin_datapairs")]
        saved = [m.bin_datapairs for m in mods]
        for m in mods:
            m.bin_datapairs = capture
        try:
            yield captured
        finally:
            for m, f in zip(mods, saved):
                m.bin_datapairs = f


def _captured_block(df, captured):
    """The raw pairs behind a DataFile read under `_raw_pairs`, or None if it was not captured."""
    try:
        y = np.asarray(df.ylabels, dtype=float)
    except Exception:  # noqa: BLE001
        return None
    if y.shape == (1,) and y[0] < 0 and float(y[0]).is_integer() and -int(y[0]) <= len(captured):
        return captured[-int(y[0]) - 1]
    return None


def _spectra_from_pairs(keys, values, counts, times_min, base: dict, sid: str):
    """Split flat raw pairs into per-scan spectra (zero intensities dropped)."""
    out = []
    off = np.concatenate([[0], np.cumsum(counts)])
    for i in range(len(counts)):
        m = keys[off[i]:off[i + 1]]
        v = values[off[i]:off[i + 1]]
        nz = v > 0
        m, v = m[nz], v[nz]
        if len(m) > 1 and (np.diff(m) <= 0).any():
            m, v = msio.merge_peaks(m, v, 0.5)        # sorts, and adds up duplicate m/z within a scan
        out.append({**base, "rt": float(times_min[i]) * 60.0, "mz": m, "intensity": v,
                    "centroided": msio._looks_centroided(m), "scan_id": f"{sid} scan={i + 1}"})
    return out


def _spectra_from_matrix(ylabels, data, times_min, base: dict, sid: str):
    """A binned (scans × m/z) DataFile → spectra (zero columns dropped per scan)."""
    y = np.asarray(ylabels, dtype=np.float64)
    out = []
    for i in range(data.shape[0]):
        row = np.asarray(data[i], dtype=np.float64)
        nz = row > 0
        out.append({**base, "rt": float(times_min[i]) * 60.0, "mz": y[nz], "intensity": row[nz],
                    "centroided": True, "scan_id": f"{sid} scan={i + 1}"})
    return out


def _trim_zeros(mz, it):
    """Profile scan without its long runs of zeros; zeros that flank a signal are kept (peak pickers need them)."""
    nz = it > 0
    keep = nz.copy()
    keep[1:] |= nz[:-1]
    keep[:-1] |= nz[1:]
    return mz[keep], it[keep]


_UV_DEFAULT_NM = (210, 254, 280)


def _trace_id(df) -> str:
    md = getattr(df, "metadata", {}) or {}
    det = df.detector or "Analog"
    sig = str(md.get("signal", "")).strip()
    wl = md.get("wavelength")
    if det == "UV" and wl:
        return f"UV {float(wl):g} nm"
    m = re.search(r"(\d{3})\s*nm", sig)
    if det == "UV" and m:
        return f"UV {m.group(1)} nm"
    if sig and det.lower() not in sig.lower():
        return f"{det} {sig}"
    return sig or f"{det} ({df.name})"


def _traces_from_datafile(df, seen: set) -> list[dict]:
    """Non-MS detector data → chromatogram entries (time in seconds, q1 = q3 = 0)."""
    try:
        data = np.asarray(df.data, dtype=np.float64)
        t = np.asarray(df.xlabels, dtype=np.float64) * 60.0
    except Exception:  # noqa: BLE001
        return []
    if data.ndim != 2 or not data.shape[0]:
        return []
    md = getattr(df, "metadata", {}) or {}
    base = {"q1": 0.0, "q3": 0.0, "detector": df.detector or "Analog", "unit": str(md.get("unit", "")), "file": df.name}
    out = []
    if data.shape[1] == 1:
        cid = _trace_id(df)
        k = 2
        while cid in seen:
            cid = f"{_trace_id(df)} ({k})"
            k += 1
        seen.add(cid)
        out.append({"id": cid, **base, "time": t, "intensity": data[:, 0]})
        return out
    # a diode-array spectrum block (time × wavelength): keep a few standard wavelengths that are not
    # already there as recorded single-wavelength signals, plus the max-absorbance plot
    try:
        wl = np.asarray(df.ylabels, dtype=np.float64)
    except Exception:  # noqa: BLE001
        return out
    for nm in _UV_DEFAULT_NM:
        j = int(np.argmin(np.abs(wl - nm)))
        if abs(wl[j] - nm) > 3:
            continue
        cid = f"UV {wl[j]:g} nm"
        if cid in seen:
            continue
        seen.add(cid)
        out.append({"id": cid, **base, "time": t, "intensity": data[:, j], "from": "DAD spectra"})
    cid = f"{df.detector or 'UV'} max plot ({wl.min():g}–{wl.max():g} nm)"
    if cid not in seen:
        seen.add(cid)
        out.append({"id": cid, **base, "time": t, "intensity": data.max(axis=1), "from": "DAD spectra"})
    return out


def _signals_first(datafiles):
    """Recorded single-wavelength signals before diode-array blocks, so a wavelength that was recorded as its
    own signal keeps that trace and is not extracted a second time from the DAD spectra."""
    def width(df):
        try:
            return 0 if np.asarray(df.data).shape[1] == 1 else 1
        except Exception:  # noqa: BLE001
            return 1
    return sorted(datafiles, key=width)


def _sim_to_traces(spectra: list[dict], chroms: list[dict]) -> tuple[list[dict], bool]:
    """In a run that has full scans, SIM scans (a few fixed ions) become one trace per ion instead of
    interleaving with the scans and making the TIC saw-toothed. A SIM-only run keeps them as spectra so
    extracted-ion chromatograms work on it as usual."""
    sim = [s for s in spectra if s.get("_sim")]
    full = [s for s in spectra if not s.get("_sim") and s["ms_level"] == 1]
    for s in spectra:
        s.pop("_sim", None)
    if not sim or not full:
        return spectra, False
    ions: dict[float, list] = {}
    for s in sim:
        for m, v in zip(np.round(s["mz"], 1), s["intensity"]):
            ions.setdefault(float(m), []).append((s["rt"], float(v)))
    for m, pts in sorted(ions.items()):
        pts.sort()
        chroms.append({"id": f"SIM m/z {m:.1f}", "q1": m, "q3": 0.0, "detector": "MS (SIM)",
                       "time": np.array([p[0] for p in pts]), "intensity": np.array([p[1] for p in pts])})
    keep = set(map(id, sim))
    return [s for s in spectra if id(s) not in keep], True


def _clean_meta(md: dict) -> dict:
    out = {}
    for k, v in (md or {}).items():
        if isinstance(v, (str, int, float, bool)) and v != "":
            out[k] = v
        elif k == "modules" and isinstance(v, list):
            out["modules"] = ", ".join(f"{m.get('name', '')} {m.get('model', '')}".strip() for m in v if isinstance(m, dict))
    return out


def _rainbow():
    try:
        import rainbow as rb
    except ImportError as e:
        raise UserFacingError("Reading Agilent/Waters files needs the rainbow-api package, which is missing from "
                              "this installation. Reinstall MassSpec Bench, or convert the file to mzML. "
                              + msio.CONVERT_HELP) from e
    return rb


# ====================================================================== Agilent (MassHunter .d, ChemStation .D / .ms)

# MassHunter enums (Agilent MHDAC): IonPolarity 0 = positive, 1 = negative; ScanType 1 = scan,
# 2 = selected ion (SIM), 16 = MRM, 32 = product ion, 64 = precursor ion, 128 = neutral loss.
_MH_SIM, _MH_MRM = 2, 16


def _masshunter_records(acq: Path) -> list[dict]:
    """Per-scan records of MSScan.bin (MS level, scan type, polarity, precursor m/z) via rainbow's reader."""
    from rainbow.agilent import masshunter as mh
    x = mh.parse_scan_xsd(str(acq / "MSScan.xsd"))
    n = mh.count_scans(str(acq / "MSTS.xml")) if (acq / "MSTS.xml").exists() else None
    return mh.read_scan_records(str(acq / "MSScan.bin"), x, n)


def _mh_spectra(df, records, profile: bool) -> list[dict]:
    """Spectra from a MassHunter per-scan DataFile (CentroidDataFile or ProfileDataFile)."""
    by_time = {}
    for r in records:
        by_time.setdefault(round(float(r.get("ScanTime", -1)), 5), r)
    out = []
    times = np.asarray(df.xlabels, dtype=np.float64)
    for i, tmin in enumerate(times):
        mz, it = df.scan(i)
        mz = np.asarray(mz, dtype=np.float64)
        it = np.asarray(it, dtype=np.float64)
        if profile:
            mz, it = _trim_zeros(mz, it)
        else:
            nz = it > 0
            mz, it = mz[nz], it[nz]
        r = by_time.get(round(float(tmin), 5), {})
        lvl = int(r.get("MSLevel", 1) or 1)
        st = int(r.get("ScanType", 1) or 1)
        pol = r.get("IonPolarity")
        prec = float(r.get("MzOfInterest", 0) or 0)
        d = {"rt": float(tmin) * 60.0, "ms_level": 2 if lvl == 2 or st == _MH_MRM else 1, "mz": mz, "intensity": it,
             "polarity": 1 if pol == 0 else -1 if pol == 1 else 0, "centroided": not profile,
             "scan_id": f"scanId={r.get('ScanID', i + 1)}", "_sim": st == _MH_SIM}
        if d["ms_level"] == 2 and prec > 0:
            d["precursor_mz"] = prec
        if "CollisionEnergy" in r:
            d["collision_energy"] = float(r["CollisionEnergy"] or 0)
        out.append(d)
    return out


def read_agilent(path: Path, progress=None, profile: bool = False) -> msio.Run:
    """Agilent MassHunter .d, ChemStation .D folders and single ChemStation .ms files (GC-MS/LC-MS MSD).

    MassHunter MS: MSPeak.bin (vendor centroids) is used when present; otherwise MSProfile.bin (profile,
    centroided later by OpenMS). profile=True prefers MSProfile.bin when both exist. ChemStation MSD .ms
    files are read at the m/z precision stored in the file (no 1-Da binning). UV/DAD, FID, ELSD, CAD and
    other detector signals become traces in run.chromatograms.
    """
    path = Path(path)
    rb = _rainbow()
    _lzf_fallback()
    name = path.name
    if path.is_file():
        return _read_chemstation_ms(path, progress)
    acq = path / "AcqData"
    acq_files = {p.name for p in acq.iterdir()} if acq.is_dir() else set()
    has_peak = "MSPeak.bin" in acq_files
    has_prof = "MSProfile.bin" in acq_files
    icp = "MSScan_XSpecific.bin" in acq_files
    mh_ms = {"MSScan.bin", "MSScan.xsd"} <= acq_files and (has_peak or has_prof)
    use_centroid = mh_ms and has_peak and not (profile and has_prof) and not icp
    use_hrms = mh_ms and (icp or not use_centroid)
    _say(progress, f"Reading {name} with rainbow")
    try:
        with _raw_pairs() as captured:
            d = rb.read(str(path), hrms=bool(use_hrms), centroid=bool(use_centroid), bin_width=_RAW_WIDTH if not mh_ms else None)
    except ModuleNotFoundError as e:
        raise UserFacingError(f"{name}: {e}") from e
    except MemoryError as e:
        raise UserFacingError(f"{name} is too large to read into memory here (the MassHunter profile data is loaded "
                              "whole). " + msio.CONVERT_HELP) from e
    except Exception as e:  # noqa: BLE001
        raise UserFacingError(f"{name}: the Agilent data could not be read ({type(e).__name__}: {str(e)[:300]}). "
                              + msio.CONVERT_HELP) from e

    datafiles = list(d.datafiles) + list(getattr(d, "analog", []) or [])
    meta = {**_clean_meta(d.metadata), "reader": "rainbow-api " + _version("rainbow-api")}
    spectra, chroms, seen, notes = [], [], set(), []
    records = None
    if mh_ms and not icp:
        try:
            records = _masshunter_records(acq)
        except Exception:  # noqa: BLE001 - scan records only add polarity / MS level / precursor
            notes.append("MSScan.bin records unreadable: polarity and MS level unknown")
            log_exc("masshunter records")
            records = []
    ms_from_acq = False
    for df in datafiles:
        cls = type(df).__name__
        if df.detector == "MS":
            if cls in ("CentroidDataFile", "ProfileDataFile"):
                spectra += _mh_spectra(df, records or [], profile=cls == "ProfileDataFile")
                ms_from_acq = True
                meta["ms_source"] = "MSProfile.bin (profile)" if cls == "ProfileDataFile" else "MSPeak.bin (vendor centroids)"
            elif icp and df.name.startswith("MSProfile"):
                # ICP-MS: element isotopes over time, not molecular spectra → one trace per isotope
                y = np.asarray(df.ylabels, dtype=float)
                for j, m in enumerate(y):
                    chroms.append({"id": f"ICP-MS m/z {m:g}", "q1": float(m), "q3": 0.0, "detector": "ICP-MS",
                                   "time": np.asarray(df.xlabels, float) * 60.0, "intensity": np.asarray(df.data[:, j], float)})
                meta["technique"] = "ICP-MS"
                notes.append("ICP-MS data: isotope traces only, no molecular spectra")
    # ChemStation-format MS (.ms files) — only when MassHunter AcqData did not already give the same scans
    for df in datafiles:
        if df.detector != "MS" or type(df).__name__ in ("CentroidDataFile", "ProfileDataFile") or (icp and df.name.startswith("MSProfile")):
            continue
        if ms_from_acq:
            notes.append(f"{df.name} skipped (same scans as AcqData)")
            continue
        md = df.metadata or {}
        sim = str(md.get("acquisition_mode", "")).upper() == "SIM"
        base = {"ms_level": 1, "polarity": 0, "_sim": sim}
        blk = _captured_block(df, captured)
        if blk is not None:
            spectra += _spectra_from_pairs(*blk, df.xlabels, base, df.name)
        else:
            spectra += _spectra_from_matrix(df.ylabels, np.asarray(df.data), df.xlabels, base, df.name)
        for k in ("method", "date", "instrument"):
            if md.get(k) and k not in meta:
                meta[k] = str(md[k])
    for df in _signals_first(datafiles):
        if df.detector != "MS":
            chroms += _traces_from_datafile(df, seen)
            md = df.metadata or {}
            if md.get("instrument") and "instrument" not in meta:
                meta["instrument"] = str(md["instrument"])
    spectra, split = _sim_to_traces(spectra, chroms)
    if split:
        notes.append("SIM scans are shown as per-ion traces; full scans as spectra")
    if mh_ms and not spectra and not icp:
        what = "MSPeak.bin" if use_centroid else "MSProfile.bin"
        raise UserFacingError(f"{name}: the MassHunter MS data ({what}) held no readable scans. "
                              + msio.CONVERT_HELP)
    if not spectra and not chroms:
        raise UserFacingError(f"{name}: no mass spectra and no detector signals could be read from this Agilent folder. "
                              "rainbow reads MassHunter MSPeak.bin/MSProfile.bin and ChemStation .ms/.ch/.uv; other "
                              "layouts are not supported. " + msio.CONVERT_HELP)
    if not spectra:
        notes.append("no MS data in this file — detector traces only")
        meta["no_ms"] = True
    meta.setdefault("vendor", "Agilent")
    if "instrument" not in meta or not meta["instrument"]:
        meta["instrument"] = "Agilent " + ("MassHunter" if acq.is_dir() else "ChemStation")
    if notes:
        meta["note"] = "; ".join(dict.fromkeys(notes))
    fmt = "agilent_masshunter" if acq.is_dir() else "agilent_chemstation"
    return msio.from_lists(path.stem, str(path), fmt, spectra, chroms, meta)


def _read_chemstation_ms(path: Path, progress=None) -> msio.Run:
    """A lone ChemStation .ms file (GC-MS/LC-MS MSD), outside its .D folder."""
    _rainbow()
    from rainbow.agilent import chemstation
    _say(progress, f"Reading {path.name} with rainbow")
    try:
        with _raw_pairs() as captured:
            df = chemstation.parse_file(str(path), bin_width=_RAW_WIDTH)
    except Exception as e:  # noqa: BLE001
        raise UserFacingError(f"{path.name}: not a readable ChemStation MS file ({type(e).__name__}: {str(e)[:200]}). "
                              + msio.CONVERT_HELP) from e
    if df is None or df.detector != "MS":
        raise UserFacingError(f"{path.name}: this .ms file holds no mass spectra. " + msio.CONVERT_HELP)
    md = df.metadata or {}
    base = {"ms_level": 1, "polarity": 0}
    blk = _captured_block(df, captured)
    spectra = (_spectra_from_pairs(*blk, df.xlabels, base, path.name) if blk is not None
               else _spectra_from_matrix(df.ylabels, np.asarray(df.data), df.xlabels, base, path.name))
    meta = {**_clean_meta(md), "vendor": "Agilent", "instrument": str(md.get("instrument") or "Agilent ChemStation MSD"),
            "reader": "rainbow-api " + _version("rainbow-api")}
    return msio.from_lists(path.stem, str(path), "agilent_chemstation", spectra, meta=meta)


# ====================================================================== Waters .raw (MassLynx)

def read_waters(path: Path, progress=None) -> msio.Run:
    """Waters MassLynx .raw folder: MS functions (quadrupole/SQ/TQ scan, SIR, MRM), PDA, ELSD, CAD.

    Every MS scan function becomes spectra with its polarity; MRM functions become one trace per
    transition (q1 → q3). High-resolution (Q-TOF/Synapt) .raw data is not supported by rainbow.
    """
    path = Path(path)
    rb = _rainbow()
    from rainbow.waters import masslynx
    _say(progress, f"Reading {path.name} with rainbow")
    try:
        with _raw_pairs() as captured:
            d = rb.read(str(path), bin_width=_RAW_WIDTH, format="waters")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "HRMS" in msg:
            raise UserFacingError(f"{path.name} is high-resolution Waters data (Q-TOF / Synapt / Xevo G2), which no "
                                  "macOS reader supports. " + msio.CONVERT_HELP) from e
        raise UserFacingError(f"{path.name}: the Waters data could not be read ({type(e).__name__}: {msg[:300]}). "
                              + msio.CONVERT_HELP) from e
    funcs = sorted(p for p in path.iterdir() if re.match(r"^_FUNC\d{3}\.DAT$", p.name, re.I))
    ftypes = []
    inf = next((p for p in path.iterdir() if p.name.upper() == "_FUNCTNS.INF"), None)
    q1s = q3s = None
    if inf is not None:
        try:
            ftypes = masslynx._function_types(str(inf), len(funcs))
            q1s = masslynx.parse_funcinf(str(inf))
            q3s = masslynx.parse_funcinf_q3(str(inf))
        except Exception:  # noqa: BLE001
            log_exc("waters functns")
    ftype = {f.name.upper(): (ftypes[k] if k < len(ftypes) else None) for k, f in enumerate(funcs)}
    spectra, chroms, seen, notes = [], [], set(), []
    meta = {**_clean_meta(d.metadata), "vendor": "Waters", "reader": "rainbow-api " + _version("rainbow-api")}
    for df in _signals_first(list(d.datafiles) + list(getattr(d, "analog", []) or [])):
        fname = os.path.basename(df.name).upper()
        ft = ftype.get(fname)
        md = df.metadata or {}
        if df.detector == "MS":
            pol = {"+": 1, "-": -1}.get(str(md.get("polarity", "")), 0)
            sim = str(md.get("acquisition_mode", "")) == "SIM"
            base = {"ms_level": 1, "polarity": pol, "_sim": sim}
            blk = _captured_block(df, captured)
            if blk is not None:
                spectra += _spectra_from_pairs(*blk, df.xlabels, base, fname)
            else:
                spectra += _spectra_from_matrix(df.ylabels, np.asarray(df.data), df.xlabels, base, fname)
        elif ft is not None and ft not in (12,) and fname.startswith("_FUNC") and q1s is not None:
            # an MS function rainbow could not name (no polarity in _extern.inf): MRM / parent / daughter scans
            k = int(re.findall(r"\d+", fname)[0]) - 1
            data = np.asarray(df.data, dtype=float)
            q1 = q1s[k][q1s[k] != 0]
            q3 = q3s[k][: len(q1)] if q3s is not None else np.zeros(len(q1))
            for j in range(min(data.shape[1], len(q1))):
                cid = f"MRM {q1[j]:.2f} > {q3[j]:.2f}" if q3[j] else f"MS {q1[j]:.2f} (function {k + 1})"
                n = 2
                while cid in seen:
                    cid = f"{cid.split(' #')[0]} #{n}"
                    n += 1
                seen.add(cid)
                chroms.append({"id": cid, "q1": float(q1[j]), "q3": float(q3[j]), "detector": "MS (MRM)",
                               "time": np.asarray(df.xlabels, float) * 60.0, "intensity": data[:, j], "function": k + 1})
            meta["acquisition"] = "MRM"
        else:
            chroms += _traces_from_datafile(df, seen)
    spectra, split = _sim_to_traces(spectra, chroms)
    if split:
        notes.append("SIR (SIM) functions are shown as per-ion traces; full scans as spectra")
    if not spectra and not chroms:
        raise UserFacingError(f"{path.name}: no MS functions or detector signals could be read. " + msio.CONVERT_HELP)
    if not spectra and not any(c.get("detector", "").startswith("MS") for c in chroms):
        notes.append("no MS data in this file — detector traces only")
        meta["no_ms"] = True
    notes.append("every MS function is treated as MS1 (MSe / DDA fragment functions are not distinguished)")
    meta["instrument"] = meta.get("instrument") or _waters_instrument(path)
    meta["note"] = "; ".join(dict.fromkeys(notes))
    return msio.from_lists(path.stem, str(path), "waters_raw", spectra, chroms, meta)


def _waters_instrument(path: Path) -> str:
    hdr = next((p for p in path.iterdir() if p.name.upper() == "_HEADER.TXT"), None)
    if hdr is not None:
        try:
            for line in hdr.read_text(errors="replace").splitlines():
                if line.startswith("$$ Instrument:"):
                    v = line.split(":", 1)[1].strip()
                    if v:
                        return "Waters " + v
        except OSError:
            pass
    return "Waters MassLynx"


# ====================================================================== ANDI-MS netCDF

def read_andi(path: Path, progress=None) -> msio.Run:
    """ANDI-MS / AIA netCDF (ASTM E2078): the vendor-neutral export of almost every GC-MS data system."""
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic.startswith(b"\x89HDF"):
        return _read_andi_hdf5(path)
    if not magic.startswith(b"CDF"):
        raise UserFacingError(f"{path.name} is not a netCDF file (ANDI-MS .cdf files start with 'CDF'). "
                              "Re-export it from the instrument software as AIA/ANDI netCDF, or as mzML.")
    from scipy.io import netcdf_file
    _say(progress, f"Reading {path.name}")
    try:
        f = netcdf_file(str(path), "r", mmap=False)
    except Exception as e:  # noqa: BLE001
        raise UserFacingError(f"{path.name}: the netCDF file is damaged or truncated ({e}).") from e
    try:
        v = f.variables
        g = {k: (x.decode("latin-1", "replace").strip("\x00 ") if isinstance(x, bytes) else x) for k, x in f._attributes.items()}
        return _andi_build(path, lambda k: np.asarray(v[k].data) if k in v else None,
                           lambda k: dict(getattr(v[k], "_attributes", {})) if k in v else {}, g,
                           {k: _nc_str(v[k].data) for k in v if k.startswith("instrument_")})
    finally:
        f.close()


def _nc_str(a) -> str:
    try:
        return b"".join(np.asarray(a).ravel().tolist()).decode("latin-1", "replace").strip("\x00 ")
    except Exception:  # noqa: BLE001
        return ""


def _read_andi_hdf5(path: Path) -> msio.Run:
    try:
        import h5py
    except ImportError as e:
        raise UserFacingError(f"{path.name} is a netCDF-4 (HDF5) file, not the classic netCDF-3 that ANDI-MS uses. "
                              "Re-export it as ANDI/AIA netCDF (classic) or as mzML from the instrument software.") from e
    with h5py.File(path, "r") as h:
        g = {k: (x.decode("latin-1", "replace") if isinstance(x, bytes) else x) for k, x in h.attrs.items()}
        get = lambda k: np.asarray(h[k][()]) if k in h else None  # noqa: E731
        attrs = lambda k: {a: b for a, b in h[k].attrs.items()} if k in h else {}  # noqa: E731
        return _andi_build(path, get, attrs, g, {})


def _andi_build(path: Path, get, attrs, g: dict, inst: dict) -> msio.Run:
    for k in ("scan_acquisition_time", "mass_values", "intensity_values"):
        if get(k) is None:
            raise UserFacingError(f"{path.name} is a netCDF file but not ANDI-MS: it has no '{k}' variable. It may be an "
                                  "ANDI chromatography (AIA-Chrom) file with no mass spectra.")
    t = get("scan_acquisition_time").astype(np.float64)
    mz = get("mass_values").astype(np.float64)
    it = get("intensity_values").astype(np.float64)
    for arr, k in ((mz, "mass_values"), (it, "intensity_values")):
        a = attrs(k)
        sf, off = float(a.get("scale_factor", 1.0) or 1.0), float(a.get("add_offset", 0.0) or 0.0)
        if sf != 1.0 or off != 0.0:
            arr *= sf
            arr += off
    n, N = len(t), len(mz)
    si = get("scan_index")
    pc = get("point_count")
    if si is None and pc is None:
        raise UserFacingError(f"{path.name}: ANDI file without scan_index/point_count — scans cannot be separated.")
    notes = []
    if si is not None:
        si = si.astype(np.int64)
        ends = np.append(si[1:], N)
        diffs = ends - si
        if pc is None:
            lens = diffs
        else:
            pc = pc.astype(np.int64)
            lens = np.minimum(pc, np.maximum(diffs, 0))
            if (pc > diffs).any() and np.all(np.diff(si) >= 0):
                notes.append("point_count disagrees with scan_index (the file was cut down after export); scan_index used")
    else:
        lens = pc.astype(np.int64)
        si = np.concatenate([[0], np.cumsum(lens)[:-1]])
    tunit = str(attrs("scan_acquisition_time").get("units", b"") or "")
    if "min" in tunit.lower():
        t = t * 60.0
    pol_s = str(g.get("test_ionization_polarity", "")).lower()
    pol = 1 if "pos" in pol_s else -1 if "neg" in pol_s else 0
    exp = str(g.get("experiment_type", "")).lower()
    cen_flag = True if "centroid" in exp else False if ("continuum" in exp or "profile" in exp) else None
    spectra = []
    for i in range(n):
        a, b = int(si[i]), int(si[i] + max(lens[i], 0))
        m, v = mz[a:b], it[a:b]
        nz = v > 0
        m, v = m[nz], v[nz]
        spectra.append({"rt": float(t[i]), "ms_level": 1, "mz": m, "intensity": v, "polarity": pol,
                        "centroided": cen_flag if cen_flag is not None else msio._looks_centroided(m),
                        "scan_id": f"scan={i + 1}"})
    ins = " ".join(x for x in (inst.get("instrument_mfr", ""), inst.get("instrument_model", "") or inst.get("instrument_name", "")) if x)
    meta = {"vendor": inst.get("instrument_mfr", "") or "ANDI export", "instrument": ins or "ANDI-MS export",
            "date": str(g.get("experiment_date_time_stamp", "")), "operator": str(g.get("operator_name", "")),
            "experiment_type": str(g.get("experiment_type", "")), "ionization": str(g.get("test_ionization_mode", "")),
            "inlet": str(g.get("test_ms_inlet", "")), "reader": "scipy.io.netcdf_file (ANDI-MS)"}
    meta = {k: v for k, v in meta.items() if v}
    if notes:
        meta["note"] = "; ".join(notes)
    return msio.from_lists(path.stem, str(path), "andi_cdf", spectra, meta=meta)


# ====================================================================== registry

READERS = {
    "thermo_raw": read_thermo,
    "bruker_tdf": read_bruker_tdf,
    "agilent_masshunter": read_agilent,
    "agilent_chemstation": read_agilent,
    "waters_raw": read_waters,
    "andi_cdf": read_andi,
}
