"""Reading mass-spec data into one in-memory form, whatever instrument wrote it.

A `Run` holds every spectrum of one acquisition as flat NumPy arrays: per-spectrum metadata (retention
time, MS level, precursor, polarity…) and all peaks concatenated, with `offsets` marking where each
spectrum starts. That is small enough to cache as a single .npz per file, fast to slice for chromatograms,
and independent of the vendor, so every analysis downstream is written once.

Open formats (mzML, mzXML, MGF, mzData) are read here with pyOpenMS / pyteomics. Vendor formats are
dispatched to `vendor.py`, which either reads them natively (Bruker timsTOF, Agilent, Waters, ANDI/netCDF)
or converts them (Thermo .raw via ThermoRawFileParser).
"""
from __future__ import annotations

import gzip
import io
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import HOME, UserFacingError, log_exc

CACHE = HOME / "cache"

# ------------------------------------------------------------------ format detection

OPEN_EXT = {".mzml": "mzML", ".mzxml": "mzXML", ".mgf": "MGF", ".mzdata": "mzData"}

# Formats with no reader that runs on macOS. The message tells the user exactly how to get an mzML.
UNSUPPORTED = {
    "sciex_wiff": "Sciex .wiff/.wiff2",
    "shimadzu_lcd": "Shimadzu .lcd",
    "bruker_baf": "Bruker .d (BAF — maXis/impact/compact/solariX)",
    "bruker_fid": "Bruker MALDI/FT-ICR (fid/ser)",
}
CONVERT_HELP = (
    "Convert it to mzML first with ProteoWizard msconvert (free, Windows: proteowizard.sourceforge.io; "
    "on a Mac it runs through its Docker image), using the vendor peak-picking filter: "
    "msconvert FILE --mzML --filter \"peakPicking vendor msLevel=1-\". Then open the .mzML here."
)


def detect_format(path: Path) -> str:
    """Name of the format of a file or folder (a vendor .d / .raw can be a folder)."""
    p = Path(path)
    name = p.name.lower()
    if p.is_dir():
        if name.endswith(".d"):
            if (p / "analysis.tdf").exists():
                return "bruker_tdf"
            if (p / "analysis.baf").exists():
                return "bruker_baf"
            if (p / "AcqData").is_dir():
                return "agilent_masshunter"
            if any(p.glob("*.ms")) or any(p.glob("*.MS")) or any(p.glob("*.ch")) or any(p.glob("*.CH")):
                return "agilent_chemstation"
            if (p / "fid").exists() or (p / "ser").exists() or any(p.glob("*/fid")):
                return "bruker_fid"
            return "unknown_d"
        if name.endswith(".raw"):
            if any(p.glob("_FUNC*.DAT")) or any(p.glob("_func*.dat")):
                return "waters_raw"
            return "unknown_raw_dir"
        return "folder"
    for ext in (".mzml.gz", ".mzxml.gz", ".mgf.gz"):
        if name.endswith(ext):
            return OPEN_EXT[ext[:-3]] + ".gz"
    suf = p.suffix.lower()
    if suf in OPEN_EXT:
        return OPEN_EXT[suf]
    if suf == ".raw":
        return "thermo_raw"
    if suf in (".wiff", ".wiff2", ".wiff.scan"):
        return "sciex_wiff"
    if suf == ".lcd":
        return "shimadzu_lcd"
    if suf in (".cdf", ".netcdf", ".nc"):
        return "andi_cdf"
    if suf == ".ms":
        return "agilent_chemstation"
    if suf == ".baf":
        return "bruker_baf"
    return "unknown"


FORMAT_LABEL = {
    "mzML": "mzML", "mzXML": "mzXML", "MGF": "MGF", "mzData": "mzData",
    "mzML.gz": "mzML (gzip)", "mzXML.gz": "mzXML (gzip)", "MGF.gz": "MGF (gzip)",
    "thermo_raw": "Thermo .raw", "bruker_tdf": "Bruker timsTOF .d (TDF)", "bruker_baf": "Bruker .d (BAF)",
    "agilent_masshunter": "Agilent MassHunter .d", "agilent_chemstation": "Agilent ChemStation .D/.ms",
    "waters_raw": "Waters .raw", "andi_cdf": "ANDI-MS netCDF (.cdf)", "sciex_wiff": "Sciex .wiff",
    "shimadzu_lcd": "Shimadzu .lcd", "bruker_fid": "Bruker fid/ser",
}


def is_ms_path(p: Path) -> bool:
    """True for anything this app can at least recognise as mass-spec data (used by the file browser)."""
    f = detect_format(p)
    return f not in ("unknown", "folder", "unknown_d", "unknown_raw_dir")


# ------------------------------------------------------------------ the Run

@dataclass
class Run:
    name: str
    source: str
    fmt: str
    rt: np.ndarray                   # seconds, per spectrum
    ms_level: np.ndarray             # int8
    polarity: np.ndarray             # int8: +1 / -1 / 0 unknown
    precursor_mz: np.ndarray         # nan for MS1
    precursor_charge: np.ndarray     # int8, 0 unknown
    precursor_intensity: np.ndarray  # nan when not recorded
    centroided: np.ndarray           # bool per spectrum
    offsets: np.ndarray              # int64, len n+1
    mz: np.ndarray                   # float64, all peaks
    intensity: np.ndarray            # float32, all peaks
    scan_id: list = field(default_factory=list)
    ion_mobility: np.ndarray | None = None   # per spectrum (1/K0) for frame-collapsed timsTOF data, else None
    chromatograms: list = field(default_factory=list)   # [{id, q1, q3, time(np), intensity(np)}] — SRM/MRM traces
    meta: dict = field(default_factory=dict)

    # ---- derived ----
    @property
    def n(self) -> int:
        return len(self.rt)

    def spectrum(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.mz[a:b], self.intensity[a:b]

    def idx(self, level: int = 1, polarity: int | None = None) -> np.ndarray:
        m = self.ms_level == level
        if polarity:
            m &= self.polarity == polarity
        return np.flatnonzero(m)

    @property
    def tic(self) -> np.ndarray:
        return _seg_reduce(self.intensity.astype(np.float64), self.offsets, np.add)

    @property
    def bpi(self) -> np.ndarray:
        return _seg_reduce(self.intensity, self.offsets, np.maximum)

    def bp_mz(self) -> np.ndarray:
        out = np.full(self.n, np.nan)
        for i in range(self.n):
            a, b = self.offsets[i], self.offsets[i + 1]
            if b > a:
                out[i] = self.mz[a + int(np.argmax(self.intensity[a:b]))]
        return out

    @property
    def is_centroided(self) -> bool:
        ms1 = self.idx(1)
        if len(ms1):
            return bool(self.centroided[ms1].mean() > 0.5)
        return bool(self.centroided.mean() > 0.5) if self.n else True

    def polarities(self) -> list[str]:
        out = []
        if (self.polarity > 0).any():
            out.append("+")
        if (self.polarity < 0).any():
            out.append("−")
        return out or ["?"]

    # ---- chromatograms ----
    def xic(self, mz: float, ppm: float = 10.0, level: int = 1, rt_range=None, polarity=None, how="sum",
            precursor=None, precursor_tol=0.01):
        """Extracted-ion chromatogram: intensity within mz ± ppm in every spectrum of an MS level.

        how='sum' adds every peak in the window (right for centroided data), 'max' takes the tallest
        (less sensitive to a neighbouring peak sneaking into a wide window on profile data).
        precursor restricts MS2 spectra to one isolated precursor (PRM / DDA fragment traces).
        """
        sel = self.idx(level, polarity)
        if precursor is not None:
            sel = sel[np.abs(self.precursor_mz[sel] - precursor) <= precursor_tol]
        if rt_range is not None:
            sel = sel[(self.rt[sel] >= rt_range[0]) & (self.rt[sel] <= rt_range[1])]
        tol = mz * ppm * 1e-6
        lo, hi = mz - tol, mz + tol
        out = np.zeros(len(sel))
        for k, i in enumerate(sel):
            a, b = self.offsets[i], self.offsets[i + 1]
            seg = self.mz[a:b]
            l, r = np.searchsorted(seg, lo), np.searchsorted(seg, hi, side="right")
            if r > l:
                v = self.intensity[a + l:a + r]
                out[k] = v.sum() if how == "sum" else v.max()
        return self.rt[sel], out

    def average_spectrum(self, rt_lo: float, rt_hi: float, level: int = 1, polarity=None, bin_ppm: float = 5.0):
        """Sum the spectra between two retention times into one (for a peak or an infusion)."""
        sel = self.idx(level, polarity)
        sel = sel[(self.rt[sel] >= rt_lo) & (self.rt[sel] <= rt_hi)]
        if not len(sel):
            return np.array([]), np.array([])
        mz = np.concatenate([self.spectrum(i)[0] for i in sel])
        it = np.concatenate([self.spectrum(i)[1] for i in sel]).astype(np.float64)
        if not len(mz):
            return mz, it
        if not self.is_centroided:
            # profile data: sum onto a common grid so the peak shapes add up rather than interleave
            o = np.argsort(mz)
            mz, it = mz[o], it[o]
            step = np.median(np.diff(np.unique(mz))) if len(mz) > 2 else 0.001
            grid = np.round(mz / step).astype(np.int64)
            u, inv = np.unique(grid, return_inverse=True)
            s = np.bincount(inv, weights=it)
            w = np.bincount(inv, weights=it * mz)
            return w / np.where(s > 0, s, 1), s / len(sel)
        return merge_peaks(mz, it, bin_ppm, n_spectra=len(sel))

    # ---- storage ----
    def save(self, path: Path):
        chrom = self.chromatograms or []
        np.savez(
            path, rt=self.rt, ms_level=self.ms_level, polarity=self.polarity, precursor_mz=self.precursor_mz,
            precursor_charge=self.precursor_charge, precursor_intensity=self.precursor_intensity,
            centroided=self.centroided, offsets=self.offsets, mz=self.mz, intensity=self.intensity,
            scan_id=np.array(self.scan_id if self.scan_id else [], dtype=str),
            ion_mobility=self.ion_mobility if self.ion_mobility is not None else np.array([]),
            chrom_meta=np.array([__import__("json").dumps({k: v for k, v in c.items() if k not in ("time", "intensity")}) for c in chrom], dtype=str),
            chrom_len=np.array([len(c["time"]) for c in chrom], dtype=np.int64),
            chrom_time=np.concatenate([np.asarray(c["time"], float) for c in chrom]) if chrom else np.array([]),
            chrom_int=np.concatenate([np.asarray(c["intensity"], float) for c in chrom]) if chrom else np.array([]),
            info=np.array([__import__("json").dumps({"name": self.name, "source": self.source, "fmt": self.fmt, "meta": self.meta},
                                                    default=str)]),
        )

    @classmethod
    def load(cls, path: Path) -> "Run":
        import json
        z = np.load(path, allow_pickle=False)
        info = json.loads(str(z["info"][0]))
        chrom = []
        if len(z["chrom_len"]):
            pos = np.concatenate([[0], np.cumsum(z["chrom_len"])])
            for k, m in enumerate(z["chrom_meta"]):
                c = json.loads(str(m))
                c["time"] = z["chrom_time"][pos[k]:pos[k + 1]]
                c["intensity"] = z["chrom_int"][pos[k]:pos[k + 1]]
                chrom.append(c)
        im = z["ion_mobility"]
        return cls(name=info["name"], source=info["source"], fmt=info["fmt"], rt=z["rt"], ms_level=z["ms_level"],
                   polarity=z["polarity"], precursor_mz=z["precursor_mz"], precursor_charge=z["precursor_charge"],
                   precursor_intensity=z["precursor_intensity"], centroided=z["centroided"], offsets=z["offsets"],
                   mz=z["mz"], intensity=z["intensity"], scan_id=[str(s) for s in z["scan_id"]],
                   ion_mobility=im if len(im) else None, chromatograms=chrom, meta=info.get("meta", {}))

    def summary(self) -> dict:
        ms1, ms2 = self.idx(1), self.idx(2)
        return {
            "name": self.name, "format": FORMAT_LABEL.get(self.fmt, self.fmt), "spectra": int(self.n),
            "ms1": int(len(ms1)), "ms2": int(len(ms2)), "msn": int((self.ms_level > 2).sum()),
            "chromatograms": len(self.chromatograms),
            "rt_min": float(self.rt.min() / 60) if self.n else 0, "rt_max": float(self.rt.max() / 60) if self.n else 0,
            "mz_min": float(self.mz.min()) if len(self.mz) else 0, "mz_max": float(self.mz.max()) if len(self.mz) else 0,
            "peaks": int(len(self.mz)), "centroided": self.is_centroided, "polarity": "/".join(self.polarities()),
            "instrument": self.meta.get("instrument", ""), "date": self.meta.get("date", ""),
        }


def _seg_reduce(values: np.ndarray, offsets: np.ndarray, ufunc) -> np.ndarray:
    """Per-spectrum sum/max over concatenated peaks; empty spectra give 0 (reduceat alone would not)."""
    n = len(offsets) - 1
    out = np.zeros(n, dtype=np.float64)
    if not len(values):
        return out
    starts = offsets[:-1]
    nonempty = offsets[1:] > starts
    if nonempty.any():
        red = ufunc.reduceat(values, starts[nonempty].clip(max=len(values) - 1))
        out[nonempty] = red
    return out


def merge_peaks(mz: np.ndarray, it: np.ndarray, ppm: float, n_spectra: int = 1):
    """Merge centroids that fall within `ppm` of each other (intensity-weighted m/z, summed intensity)."""
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    if len(mz) < 2:
        return mz, it / max(n_spectra, 1)
    gap = np.diff(mz) / mz[1:] * 1e6
    grp = np.concatenate([[0], np.cumsum(gap > ppm)])
    s = np.bincount(grp, weights=it)
    w = np.bincount(grp, weights=it * mz)
    return w / np.where(s > 0, s, 1), s / max(n_spectra, 1)


def from_lists(name, source, fmt, spectra: list[dict], chromatograms=None, meta=None) -> Run:
    """Build a Run from a list of {rt, ms_level, mz, intensity, [polarity, precursor_mz, precursor_charge,
    precursor_intensity, centroided, scan_id, ion_mobility]} dicts (rt in seconds). Used by every reader."""
    spectra = sorted(spectra, key=lambda s: s["rt"])
    n = len(spectra)
    lens = np.array([len(s["mz"]) for s in spectra], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    mz = np.empty(int(offsets[-1]), dtype=np.float64)
    it = np.empty(int(offsets[-1]), dtype=np.float32)
    for k, s in enumerate(spectra):
        a, b = offsets[k], offsets[k + 1]
        m = np.asarray(s["mz"], dtype=np.float64)
        v = np.asarray(s["intensity"], dtype=np.float32)
        if len(m) > 1 and (np.diff(m) < 0).any():
            o = np.argsort(m, kind="stable")
            m, v = m[o], v[o]
        mz[a:b] = m
        it[a:b] = v
    g = lambda key, default: np.array([s.get(key, default) if s.get(key) is not None else default for s in spectra])  # noqa: E731
    im = g("ion_mobility", np.nan)
    return Run(
        name=name, source=str(source), fmt=fmt,
        rt=g("rt", 0.0).astype(np.float64), ms_level=g("ms_level", 1).astype(np.int8),
        polarity=g("polarity", 0).astype(np.int8), precursor_mz=g("precursor_mz", np.nan).astype(np.float64),
        precursor_charge=g("precursor_charge", 0).astype(np.int8),
        precursor_intensity=g("precursor_intensity", np.nan).astype(np.float64),
        centroided=g("centroided", True).astype(bool), offsets=offsets, mz=mz, intensity=it,
        scan_id=[str(s.get("scan_id", k)) for k, s in enumerate(spectra)],
        ion_mobility=im.astype(np.float32) if np.isfinite(im).any() else None,
        chromatograms=chromatograms or [], meta=meta or {},
    )


# ------------------------------------------------------------------ open-format readers

def _ungzip(path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="msb_")) / path.name[:-3]
    with gzip.open(path, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    return tmp


def read_openms(path: Path, fmt: str, name: str | None = None, source: str | None = None) -> Run:
    """mzML / mzXML / mzData through pyOpenMS (C++ parser; much faster than pure Python for large files)."""
    import pyopenms as oms
    exp = oms.MSExperiment()
    loader = {"mzML": oms.MzMLFile, "mzXML": oms.MzXMLFile, "mzData": oms.MzDataFile}[fmt]()
    try:
        loader.load(str(path), exp)
    except Exception as e:  # noqa: BLE001
        raise UserFacingError(f"{Path(path).name} could not be read as {fmt}: {e}") from e
    return from_openms(exp, name or Path(path).stem, source or str(path), fmt)


def from_openms(exp, name: str, source: str, fmt: str) -> Run:
    import pyopenms as oms
    pos = oms.IonSource.Polarity.POSITIVE
    neg = oms.IonSource.Polarity.NEGATIVE
    est = None
    spectra = []
    for k, s in enumerate(exp):
        mz, it = s.get_peaks()
        t = s.getType()
        if t == oms.SpectrumSettings.SpectrumType.CENTROID:
            cen = True
        elif t == oms.SpectrumSettings.SpectrumType.PROFILE:
            cen = False
        else:
            cen = _looks_centroided(mz)
        pol = s.getInstrumentSettings().getPolarity()
        d = {"rt": s.getRT(), "ms_level": s.getMSLevel(), "mz": mz, "intensity": it, "centroided": cen,
             "polarity": 1 if pol == pos else -1 if pol == neg else 0, "scan_id": s.getNativeID() or k}
        pcs = s.getPrecursors()
        if pcs:
            p = pcs[0]
            d.update(precursor_mz=p.getMZ(), precursor_charge=p.getCharge(), precursor_intensity=p.getIntensity() or np.nan)
        try:
            d["ion_mobility"] = s.getDriftTime() if s.getDriftTime() > 0 else None
        except Exception:  # noqa: BLE001
            pass
        spectra.append(d)
    chroms = []
    for c in exp.getChromatograms():
        t, v = c.get_peaks()
        cid = c.getNativeID()
        if not len(t) or cid.lower() in ("tic", "bpc") or "total ion" in cid.lower():
            continue          # the run's TIC/BPC is recomputed from spectra; only SRM/SIM traces are kept
        q1 = c.getPrecursor().getMZ()
        q3 = c.getProduct().getMZ()
        chroms.append({"id": cid, "q1": float(q1), "q3": float(q3), "time": np.asarray(t, float), "intensity": np.asarray(v, float)})
    meta = {}
    try:
        ins = exp.getInstrument()
        meta["instrument"] = " ".join(x for x in (ins.getVendor(), ins.getModel(), ins.getName()) if x).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        dt = exp.getDateTime().get()
        if dt and not dt.startswith("0000"):
            meta["date"] = dt
    except Exception:  # noqa: BLE001
        pass
    return from_lists(name, source, fmt, spectra, chroms, meta)


def _looks_centroided(mz: np.ndarray) -> bool:
    """Profile spectra are sampled on a fine, nearly regular grid; centroids are sparse and irregular."""
    if len(mz) < 20:
        return True
    d = np.diff(mz)
    d = d[d > 0]
    if not len(d):
        return True
    # in profile mode most neighbouring points are closer than 0.02 Da; centroids of real peaks are not
    return float(np.mean(d < 0.02)) < 0.5


def read_mgf(path: Path, name=None, source=None) -> Run:
    from pyteomics import mgf
    spectra = []
    with mgf.read(str(path), use_index=False, convert_arrays=1) as r:
        for k, s in enumerate(r):
            p = s["params"]
            pm = p.get("pepmass") or (None,)
            ch = p.get("charge")
            z = 0
            if ch:
                try:
                    z = int(list(ch)[0]) if not isinstance(ch, (int, np.integer)) else int(ch)
                except Exception:  # noqa: BLE001
                    z = 0
            rt = p.get("rtinseconds")
            try:
                rt = float(rt[0] if isinstance(rt, (list, tuple)) else rt) if rt is not None else float(k)
            except Exception:  # noqa: BLE001
                rt = float(k)
            spectra.append({"rt": rt, "ms_level": 2, "mz": s["m/z array"], "intensity": s["intensity array"],
                            "precursor_mz": pm[0], "precursor_charge": abs(z), "centroided": True,
                            "polarity": -1 if z < 0 else (1 if z > 0 else 0),
                            "precursor_intensity": pm[1] if len(pm) > 1 and pm[1] else np.nan,
                            "scan_id": p.get("title", k)})
    return from_lists(name or Path(path).stem, source or str(path), "MGF", spectra)


# ------------------------------------------------------------------ entry point

def read_any(path: Path, progress=None) -> Run:
    """Read any supported file or folder into a Run. Raises UserFacingError with a fix for unsupported ones."""
    path = Path(path)
    fmt = detect_format(path)
    name = re.sub(r"\.(mzml|mzxml|mgf|mzdata)(\.gz)?$", "", path.name, flags=re.I)
    name = re.sub(r"\.(raw|d|cdf|wiff2?|lcd|ms)$", "", name, flags=re.I)
    if fmt in UNSUPPORTED:
        raise UserFacingError(f"{path.name} is {UNSUPPORTED[fmt]} data, which no reader can open on macOS. " + CONVERT_HELP)
    if fmt.endswith(".gz"):
        tmp = _ungzip(path)
        try:
            base = fmt[:-3]
            return read_mgf(tmp, name, str(path)) if base == "MGF" else read_openms(tmp, base, name, str(path))
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
    if fmt == "MGF":
        return read_mgf(path, name)
    if fmt in ("mzML", "mzXML", "mzData"):
        return read_openms(path, fmt, name)
    from . import vendor
    reader = vendor.READERS.get(fmt)
    if reader is None:
        raise UserFacingError(f"{path.name}: not a mass-spec format this app recognises "
                              "(mzML, mzXML, MGF, Thermo .raw, Bruker timsTOF .d, Agilent .d, Waters .raw, ANDI .cdf). " + CONVERT_HELP)
    run = reader(path, progress=progress)
    run.name = name
    return run


def cache_path(path: Path) -> Path:
    """Where the parsed copy of a data file lives: one .npz per source path, shared by every job."""
    import hashlib
    p = Path(path).resolve()
    h = hashlib.sha1(str(p).encode()).hexdigest()[:10]
    return CACHE / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', p.name)}.{h}.npz"


def load_cached(path: Path, progress=None) -> Run:
    """read_any, but keep a parsed .npz so the explorer and re-analyses never re-parse a vendor file.

    The cache is keyed by the file's absolute path and invalidated when the file is newer than it.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    npz = cache_path(path)
    p = Path(path)
    src_m = max([p.stat().st_mtime] + ([q.stat().st_mtime for q in p.iterdir()] if p.is_dir() else []))
    if npz.exists() and npz.stat().st_mtime >= src_m:
        try:
            return Run.load(npz)
        except Exception:  # noqa: BLE001 - a stale or truncated cache is simply rebuilt
            log_exc("cache load")
    run = read_any(path, progress=progress)
    run.save(npz)
    return run


def load_runs(job, files: list[Path], pct0: int = 2, pct1: int = 25, centroid_profile: bool = True) -> list[Run]:
    """Read every data file of a job (cached), centroiding profile data, and record them in result["runs"]
    so the interactive explorer can open the same parsed copies later."""
    runs = []
    for k, f in enumerate(files):
        job.step(pct0 + int((pct1 - pct0) * k / max(len(files), 1)), f"Reading {Path(f).name} ({k + 1}/{len(files)})")
        r = load_cached(Path(f))
        if centroid_profile and not r.centroided.all():
            job.step(pct0 + int((pct1 - pct0) * (k + 0.5) / max(len(files), 1)), f"Centroiding {Path(f).name}")
            r = centroid(r)
            r.save(cache_path(Path(f)))   # the explorer then works on the centroided copy as well
        runs.append(r)
    job.result["runs"] = [{"name": r.name, "file": str(f), "npz": str(cache_path(Path(f))), **r.summary()} for r, f in zip(runs, files)]
    return runs


# ------------------------------------------------------------------ pyOpenMS bridge + centroiding

def to_openms(run: Run, levels=(1,), polarity=None):
    """MSExperiment with the spectra of the given MS levels (for OpenMS algorithms)."""
    import pyopenms as oms
    exp = oms.MSExperiment()
    for i in range(run.n):
        if levels and run.ms_level[i] not in levels:
            continue
        if polarity and run.polarity[i] != polarity:
            continue
        s = oms.MSSpectrum()
        mz, it = run.spectrum(i)
        s.set_peaks((mz, it.astype(np.float32)))
        s.setRT(float(run.rt[i]))
        s.setMSLevel(int(run.ms_level[i]))
        s.setType(oms.SpectrumSettings.SpectrumType.CENTROID if run.centroided[i] else oms.SpectrumSettings.SpectrumType.PROFILE)
        s.setNativeID(str(run.scan_id[i]) if run.scan_id else f"scan={i}")
        if run.ms_level[i] > 1 and np.isfinite(run.precursor_mz[i]):
            p = oms.Precursor()
            p.setMZ(float(run.precursor_mz[i]))
            p.setCharge(int(run.precursor_charge[i]))
            s.setPrecursors([p])
        exp.addSpectrum(s)
    exp.updateRanges()
    return exp


def centroid(run: Run, signal_to_noise: float = 0.0) -> Run:
    """Profile → centroid with OpenMS PeakPickerHiRes (the standard high-resolution peak picker).

    Spectra that are already centroided pass through unchanged, so this is safe to call on anything.
    """
    if run.centroided.all():
        return run
    import pyopenms as oms
    pp = oms.PeakPickerHiRes()
    prm = pp.getParameters()
    prm.setValue("signal_to_noise", float(signal_to_noise))
    pp.setParameters(prm)
    spectra = []
    for i in range(run.n):
        mz, it = run.spectrum(i)
        if not run.centroided[i] and len(mz) > 3:
            s = oms.MSSpectrum()
            s.set_peaks((mz, it.astype(np.float32)))
            out = oms.MSSpectrum()
            pp.pick(s, out)
            mz, it = out.get_peaks()
        spectra.append({"rt": run.rt[i], "ms_level": run.ms_level[i], "mz": mz, "intensity": it,
                        "polarity": run.polarity[i], "precursor_mz": run.precursor_mz[i],
                        "precursor_charge": run.precursor_charge[i], "precursor_intensity": run.precursor_intensity[i],
                        "centroided": True, "scan_id": run.scan_id[i] if run.scan_id else i,
                        "ion_mobility": None if run.ion_mobility is None else run.ion_mobility[i]})
    out = from_lists(run.name, run.source, run.fmt, spectra, run.chromatograms, {**run.meta, "centroided_by": "PeakPickerHiRes"})
    return out


# ------------------------------------------------------------------ sample sheets

SHEET_ALIASES = {
    "file": ["file", "filename", "file_name", "raw_file", "rawfile", "data_file", "path", "run"],
    "sample": ["sample", "sample_name", "samplename", "name", "sample_id", "id"],
    "group": ["group", "condition", "class", "treatment", "genotype", "phenotype", "status"],
    "type": ["type", "sample_type", "sampletype", "role", "kind"],
    "batch": ["batch", "plate"],
    "order": ["order", "injection_order", "run_order", "injection", "sequence", "acquisition_order"],
    "concentration": ["concentration", "conc", "amount", "level", "spike"],
}


def read_table(path: Path):
    import pandas as pd
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    txt = p.read_text(errors="replace")
    sep = "\t" if txt.count("\t") > txt.count(",") else ","
    return pd.read_csv(io.StringIO(txt), sep=sep)


def normalise_sheet(df):
    """Rename recognisable columns to file/sample/group/type/batch/order/concentration; keep the rest."""
    ren = {}
    low = {c: re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in df.columns}
    for canon, al in SHEET_ALIASES.items():
        for c, l in low.items():
            if l in al and canon not in ren.values() and c not in ren:
                ren[c] = canon
                break
    return df.rename(columns=ren)


def infer_type(name: str) -> str:
    n = name.lower()
    if re.search(r"(^|[_\-. ])(qc|pool(ed)?|sst)([_\-. \d]|$)", n):
        return "qc"
    if re.search(r"blank|solvent|process_?blk|(^|[_\-. ])blk([_\-. \d]|$)|extraction_?blank", n):
        return "blank"
    if re.search(r"(^|[_\-. ])(std|cal|calib(rant|ration)?|standard)([_\-. \d]|$)", n):
        return "standard"
    return "sample"


def match_sheet(files: list[Path], sheet) -> list[dict]:
    """One row per data file: sample, group, type, batch, order, concentration.

    Files are matched to sheet rows by file name (with or without extension) or by sample name. Types
    missing from the sheet are inferred from the name (QC / blank / standard); groups are never guessed.
    """
    rows = []
    sh = normalise_sheet(sheet) if sheet is not None else None
    lut = {}
    if sh is not None:
        for _, r in sh.iterrows():
            for key in ("file", "sample"):
                v = r.get(key) if key in sh.columns else None
                if v is not None and v == v and str(v).strip():
                    lut.setdefault(_stem(Path(str(v).strip()).name), r)
    for k, f in enumerate(files):
        st = _stem(Path(f).name)
        r = lut.get(st)
        if r is None:  # a sheet name that is a prefix/suffix of the file (common with vendor suffixes)
            for key, rr in lut.items():
                if key and (st.startswith(key) or key.startswith(st)):
                    r = rr
                    break
        get = (lambda c: (None if r is None or c not in r.index or r[c] != r[c] else r[c]))  # noqa: E731
        typ = str(get("type") or "").strip().lower()
        typ = {"pooled": "qc", "pool": "qc", "quality control": "qc", "std": "standard", "calibrant": "standard",
               "cal": "standard", "calibration": "standard", "blk": "blank", "unknown": "sample"}.get(typ, typ)
        if typ not in ("sample", "qc", "blank", "standard"):
            typ = infer_type(Path(f).name)
        order = get("order")
        conc = get("concentration")
        rows.append({
            "file": str(f), "sample": str(get("sample") or _stem(Path(f).name, keep_case=True)), "group": (str(get("group")).strip() if get("group") is not None else ""),
            "type": typ, "batch": str(get("batch") or "1"), "order": float(order) if order is not None else float(k + 1),
            "concentration": float(conc) if conc is not None and str(conc).strip() not in ("", "nan") else None,
            "matched": r is not None,
        })
    return rows


def _stem(name: str, keep_case: bool = False) -> str:
    n = name.strip()
    n = re.sub(r"\.(mzml|mzxml|mgf|mzdata)(\.gz)?$", "", n, flags=re.I)
    n = re.sub(r"\.(raw|d|cdf|wiff2?|lcd|ms)$", "", n, flags=re.I)
    return n if keep_case else n.lower()
