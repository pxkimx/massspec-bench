"""MassSpec Bench web server.  Run:  uvicorn server.app:app --port 8766"""
from __future__ import annotations

import importlib
import json
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, chem, lifecycle, msio, peaks
from .common import HOME, JOBS, ROOT, Job, load_settings, read_json, run_safely, save_settings, write_json

VERSION = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "dev"
UPLOADS = HOME / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="MassSpec Bench", version=VERSION)
print(f"MassSpec Bench {VERSION} — data in {HOME} — python {__import__('platform').python_version()}", flush=True)
_lock = threading.Semaphore(1)  # one heavy analysis at a time


def _close_orphaned_jobs():
    """Analyses still marked 'running' belong to a server that has since quit (its window was closed)."""
    for st in JOBS.glob("*/status.json"):
        try:
            d = read_json(st)
            if d.get("state") == "running":
                d.update(state="cancelled", message="Stopped",
                         error="MassSpec Bench was closed while this analysis was running. Run it again to finish it.")
                write_json(st, d)
        except Exception:  # noqa: BLE001 - one unreadable job must not stop the app
            continue


_close_orphaned_jobs()
if lifecycle.start():
    print("Quits by itself when its last browser tab is closed (set MB_KEEP_ALIVE=1 to keep it running).", flush=True)

KINDS = {
    "explore": ("explore", "Raw data overview"),
    "untargeted": ("untargeted", "Untargeted LC-MS"),
    "proteomics": ("proteomics", "Proteomics (DDA)"),
    "targeted": ("targeted", "Targeted quantification"),
}


def _pipeline(kind: str):
    if kind not in KINDS:
        raise HTTPException(400, f"Unknown analysis type {kind}")
    try:
        return importlib.import_module(f".{KINDS[kind][0]}", __package__).run
    except ModuleNotFoundError as e:
        raise HTTPException(501, f"The {KINDS[kind][1]} module is not installed in this build ({e}).")


def _start(kind: str, files: list[str], name: str, params: dict, job_dir: Path):
    fn = _pipeline(kind)
    job_dir.mkdir(parents=True, exist_ok=True)
    job = Job(job_dir, kind, name)
    (job_dir / "request.json").write_text(json.dumps({"kind": kind, "name": name, "params": params, "files": files}))

    def work():
        job.set_status("running", 0, "Waiting for the previous analysis to finish" if _lock._value == 0 else "Starting")
        with _lock:
            run_safely(job, fn, [Path(f) for f in files], params)

    threading.Thread(target=work, daemon=True).start()
    return {"job": job_dir.name, "kind": kind}


# ---------------------------------------------------------------- choosing data
@app.get("/api/browse")
def browse(path: str = ""):
    """List a folder on this Mac. Vendor 'folders' (.d, Waters .raw) are shown as data files, not folders.

    Raw files are often several GB, so they are read where they are rather than uploaded into the app.
    """
    p = Path(path).expanduser() if path else Path.home()
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, f"No such folder: {p}")
    items = []
    try:
        entries = sorted(p.iterdir(), key=lambda x: x.name.lower())
    except PermissionError:
        raise HTTPException(403, f"macOS does not allow reading {p}. Grant access in System Settings → Privacy & Security → Files and Folders.")
    for e in entries:
        if e.name.startswith("."):
            continue
        try:
            fmt = msio.detect_format(e)
            ms = msio.is_ms_path(e)
            if e.is_dir() and not ms:
                items.append({"name": e.name, "path": str(e), "dir": True})
            elif ms or e.suffix.lower() in (".csv", ".tsv", ".txt", ".xlsx", ".fasta", ".fa", ".msp"):
                size = e.stat().st_size if e.is_file() else sum(f.stat().st_size for f in e.rglob("*") if f.is_file())
                items.append({"name": e.name, "path": str(e), "dir": False, "ms": ms, "format": msio.FORMAT_LABEL.get(fmt, ""),
                              "supported": ms and fmt not in msio.UNSUPPORTED, "size": size})
        except Exception:  # noqa: BLE001 - one unreadable entry must not hide the folder
            continue
    roots = [{"name": "Home", "path": str(Path.home())}]
    for n in ("Desktop", "Documents", "Downloads"):
        if (Path.home() / n).exists():
            roots.append({"name": n, "path": str(Path.home() / n)})
    if Path("/Volumes").exists():
        roots += [{"name": v.name, "path": str(v)} for v in Path("/Volumes").iterdir() if not v.name.startswith(".")]
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "items": items, "roots": roots}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    """Files dropped onto the page are saved once under MB_HOME/uploads and then used by path like any other."""
    d = UPLOADS / uuid.uuid4().hex[:8]
    d.mkdir(parents=True)
    out = []
    for f in files:
        rel = Path(f.filename)
        parts = [x for x in rel.parts if x not in ("", ".", "..")]
        p = d.joinpath(*parts) if parts else d / "file"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            while chunk := await f.read(1 << 20):
                fh.write(chunk)
        out.append(p)
    # a dropped vendor folder (.d / .raw) arrives as its individual files: report the folder itself
    tops = []
    for p in out:
        rel = p.relative_to(d)
        top = d / rel.parts[0]
        if top not in tops:
            tops.append(top)
    return {"paths": [str(t) for t in tops]}


class Inspect(BaseModel):
    paths: list[str]
    sheet: str | None = None


@app.post("/api/inspect")
def inspect(body: Inspect):
    """What each chosen path is, and one sample-sheet row per data file (matched to the sheet if one is given)."""
    data, other, problems = [], [], []
    for s in body.paths:
        p = Path(s).expanduser()
        if not p.exists():
            problems.append(f"{p.name}: not found")
            continue
        if p.is_dir() and not msio.is_ms_path(p):
            data += [q for q in sorted(p.iterdir()) if msio.is_ms_path(q)]   # a folder of runs → its runs
            continue
        (data if msio.is_ms_path(p) else other).append(p)
    sheet = None
    sheet_path = body.sheet or next((str(o) for o in other if o.suffix.lower() in (".csv", ".tsv", ".txt", ".xlsx")), None)
    if sheet_path:
        try:
            sheet = msio.read_table(Path(sheet_path))
        except Exception as e:  # noqa: BLE001
            problems.append(f"Sample sheet could not be read: {e}")
    rows = msio.match_sheet(data, sheet)
    for r, p in zip(rows, data):
        fmt = msio.detect_format(p)
        r["format"] = msio.FORMAT_LABEL.get(fmt, fmt)
        r["supported"] = fmt not in msio.UNSUPPORTED
        if not r["supported"]:
            problems.append(f"{p.name}: {msio.UNSUPPORTED[fmt]} cannot be read on macOS — convert it to mzML first.")
    return {"rows": rows, "other": [str(o) for o in other], "sheet": sheet_path, "problems": problems,
            "sheet_columns": [str(c) for c in msio.normalise_sheet(sheet).columns] if sheet is not None else []}


class AnalyzeIn(BaseModel):
    kind: str
    paths: list[str]
    name: str | None = None
    params: dict = {}


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    files = [p for p in body.paths if Path(p).exists()]
    if not files:
        raise HTTPException(400, "None of the selected files exist any more.")
    name = body.name or (Path(files[0]).name if len(files) == 1 else f"{Path(files[0]).parent.name or 'data'} · {len(files)} files")
    return _start(body.kind, files, name, body.params, JOBS / uuid.uuid4().hex[:10])


@app.post("/api/rerun/{jid}")
def rerun(jid: str, params: str = Form("{}")):
    return _rerun(jid, json.loads(params))


def _rerun(jid: str, params: dict):
    req = json.loads((JOBS / jid / "request.json").read_text())
    merged = {**req["params"], **params}
    merged = {k: v for k, v in merged.items() if v is not None}      # null returns a parameter to automatic
    return _start(req["kind"], req["files"], req["name"], merged, JOBS / uuid.uuid4().hex[:10])


@app.post("/api/demo/{which}")
def demo(which: str):
    """Simulated datasets with a known answer — the same ones the self-test uses."""
    from . import simulate
    d = HOME / "examples" / which
    if which == "untargeted":
        if not (d / "sample_sheet.csv").exists():
            simulate.metabolomics_study(d, n_per_group=5, n_compounds=150, n_qc=4)
        sheet = msio.match_sheet(sorted(d.glob("*.mzML")), msio.read_table(d / "sample_sheet.csv"))
        return _start("untargeted", [r["file"] for r in sheet], "Example: simulated metabolomics study", {"sheet": sheet, "example": True},
                      JOBS / uuid.uuid4().hex[:10])
    if which == "explore":
        if not (d / "Example_run.mzML").exists():
            d.mkdir(parents=True, exist_ok=True)
            ions = []
            rng = np.random.default_rng(3)
            for m, rt in zip(rng.uniform(120, 800, 80), rng.uniform(60, 840, 80)):
                ions += simulate.compound_ions(m, rt, 10 ** rng.uniform(4.5, 6.5))
            simulate.write_lcms(d / "Example_run.mzML", ions, rt_max=900)
        return _start("explore", [str(d / "Example_run.mzML")], "Example: one simulated LC-MS run", {}, JOBS / uuid.uuid4().hex[:10])
    raise HTTPException(404)


# ---------------------------------------------------------------- jobs
@app.get("/api/jobs")
def list_jobs(limit: int = 60):
    rows = []
    for d in sorted(JOBS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True) if JOBS.exists() else []:
        if not d.is_dir() or not (d / "request.json").exists():
            continue
        try:
            req = json.loads((d / "request.json").read_text())
            st = read_json(d / "status.json") if (d / "status.json").exists() else {}
            row = {"job": d.name, "kind": req.get("kind"), "name": req.get("name") or d.name, "state": st.get("state"),
                   "t": d.stat().st_mtime * 1000, "files": len(req.get("files", []))}
            if (d / "result.json").exists():
                res = read_json(d / "result.json")
                row["summary"] = res.get("summary") or ({"headline": f"{res['tiles'][0]['value']} {res['tiles'][0]['label']}"}
                                                        if res.get("tiles") else None)
            elif st.get("state") in ("error", "cancelled"):
                row["error"] = (st.get("error") or "")[:160]
            rows.append(row)
        except Exception:  # noqa: BLE001
            continue
        if len(rows) >= limit:
            break
    return rows


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    d = JOBS / jid
    if not (d / "status.json").exists():
        raise HTTPException(404, "Unknown job")
    if read_json(d / "status.json").get("state") != "running":
        return {"ok": False}
    (d / "cancel").write_text("1")
    return {"ok": True}


@app.get("/api/jobs/{jid}/status")
def status(jid: str):
    p = JOBS / jid / "status.json"
    if not p.exists():
        raise HTTPException(404, "Unknown job")
    return JSONResponse(read_json(p))


@app.get("/api/jobs/{jid}/result")
def result(jid: str):
    p = JOBS / jid / "result.json"
    if not p.exists():
        raise HTTPException(404, "Result not ready")
    r = read_json(p)
    r["job"] = jid
    return JSONResponse(r)


@app.get("/api/jobs/{jid}/files/{path:path}")
def files(jid: str, path: str):
    p = (JOBS / jid / path).resolve()
    if not str(p).startswith(str((JOBS / jid).resolve())) or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=p.name if p.suffix in (".pdf", ".csv", ".tsv", ".mzML", ".zip") else None)


# ---------------------------------------------------------------- interactive explorer
_RUNS: OrderedDict = OrderedDict()
_RUNS_LOCK = threading.Lock()


def _run(jid: str, k: int) -> msio.Run:
    """A parsed run of a job, kept in a small LRU so clicking around the explorer stays instant."""
    p = JOBS / jid / "result.json"
    if not p.exists():
        raise HTTPException(404, "Result not ready")
    runs = read_json(p).get("runs") or []
    if not 0 <= k < len(runs):
        raise HTTPException(404, "No such run")
    info = runs[k]
    key = info["npz"]
    with _RUNS_LOCK:
        if key in _RUNS:
            _RUNS.move_to_end(key)
            return _RUNS[key]
    if Path(key).exists():
        r = msio.Run.load(Path(key))
    elif Path(info["file"]).exists():
        r = msio.load_cached(Path(info["file"]))
    else:
        raise HTTPException(410, f"The data file {info['file']} is no longer where it was when this analysis ran.")
    with _RUNS_LOCK:
        _RUNS[key] = r
        while len(_RUNS) > 4:
            _RUNS.popitem(last=False)
    return r


def _thin(x, y, n=6000):
    """At most n points, keeping each bucket's maximum so peaks survive the thinning."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) <= n:
        return x, y
    b = int(np.ceil(len(x) / n))
    m = len(x) // b * b
    yy = y[:m].reshape(-1, b)
    k = yy.argmax(1) + np.arange(yy.shape[0]) * b
    return x[k], y[k]


def _r(a, d=5):
    return [None if not np.isfinite(v) else round(float(v), d) for v in a]


@app.get("/api/jobs/{jid}/runs")
def runs(jid: str):
    p = JOBS / jid / "result.json"
    if not p.exists():
        raise HTTPException(404)
    return read_json(p).get("runs") or []


@app.get("/api/jobs/{jid}/run/{k}/chrom")
def chrom(jid: str, k: int, type: str = "tic", level: int = 1, c: int | None = None):
    r = _run(jid, k)
    if type == "srm":
        ch = r.chromatograms[c or 0]
        t, y = _thin(ch["time"], ch["intensity"])
        return {"t": _r(t / 60, 4), "y": _r(y, 1), "label": ch.get("id")}
    idx = r.idx(level)
    y = r.tic[idx] if type == "tic" else r.bpi[idx]
    t, y = _thin(r.rt[idx], y)
    return {"t": _r(t / 60, 4), "y": _r(y, 1), "n": int(len(idx)),
            "srm": [{"i": i, "id": x.get("id"), "q1": x.get("q1"), "q3": x.get("q3")} for i, x in enumerate(r.chromatograms[:500])]}


@app.get("/api/jobs/{jid}/run/{k}/spectrum")
def spectrum(jid: str, k: int, rt: float | None = None, idx: int | None = None, level: int = 1):
    """The spectrum nearest a retention time (minutes) at an MS level, or a given spectrum index."""
    r = _run(jid, k)
    if idx is None:
        cand = r.idx(level)
        if not len(cand):
            raise HTTPException(404, f"No MS{level} spectra in this file")
        idx = int(cand[np.argmin(np.abs(r.rt[cand] - (rt or 0) * 60))])
    mz, it = r.spectrum(idx)
    if len(mz) > 30000:          # very dense profile spectra: keep the tallest points
        keep = np.sort(np.argsort(it)[-30000:])
        mz, it = mz[keep], it[keep]
    # the MS2 spectra that were triggered from this MS1 scan (DDA), for click-through
    kids = []
    if r.ms_level[idx] == 1:
        j = idx + 1
        while j < r.n and r.ms_level[j] > 1 and len(kids) < 60:
            kids.append({"idx": int(j), "precursor": float(r.precursor_mz[j]), "z": int(r.precursor_charge[j])})
            j += 1
    return {"idx": int(idx), "rt": float(r.rt[idx] / 60), "level": int(r.ms_level[idx]), "centroided": bool(r.centroided[idx]),
            "precursor": None if not np.isfinite(r.precursor_mz[idx]) else float(r.precursor_mz[idx]),
            "z": int(r.precursor_charge[idx]), "polarity": int(r.polarity[idx]), "scan": r.scan_id[idx] if r.scan_id else str(idx),
            "mz": _r(mz, 5), "i": _r(it, 1), "children": kids, "n_spectra": int(r.n)}


@app.get("/api/jobs/{jid}/run/{k}/average")
def average(jid: str, k: int, lo: float, hi: float, level: int = 1):
    r = _run(jid, k)
    mz, it = r.average_spectrum(lo * 60, hi * 60, level=level)
    if len(mz) > 30000:
        keep = np.sort(np.argsort(it)[-30000:])
        mz, it = mz[keep], it[keep]
    return {"mz": _r(mz, 5), "i": _r(it, 1), "centroided": r.is_centroided}


@app.get("/api/jobs/{jid}/xic")
def xic(jid: str, mz: float, ppm: float = 10, runs: str = "all", level: int = 1, rt: float | None = None, window: float = 0.5):
    """Extracted-ion chromatogram of one m/z in every run of the job, each integrated around the chosen time."""
    info = read_json(JOBS / jid / "result.json").get("runs") or []
    ks = range(len(info)) if runs == "all" else [int(x) for x in runs.split(",") if x != ""]
    out = []
    for k in list(ks)[:60]:
        r = _run(jid, k)
        t, y = r.xic(mz, ppm, level=level)
        pk = peaks.integrate(t, y, rt=rt * 60 if rt is not None else None, window=window * 60 if rt is not None else None)
        tt, yy = _thin(t, y, 3000)
        out.append({"k": k, "name": r.name, "t": _r(tt / 60, 4), "y": _r(yy, 1),
                    "peak": None if not pk else {**{a: pk[a] for a in ("area", "height", "snr", "fwhm")},
                                                 "rt": pk["rt"] / 60, "start": pk["start"] / 60, "end": pk["end"] / 60}})
    return {"mz": mz, "ppm": ppm, "traces": out}


@app.get("/api/jobs/{jid}/run/{k}/deconvolute")
def deconvolute(jid: str, k: int, lo: float, hi: float, zmin: int = 5, zmax: int = 60, mmin: float = 5000, mmax: float = 80000,
                step: float = 1.0):
    """Charge-state deconvolution of the spectra between two times (minutes) → zero-charge masses."""
    r = _run(jid, k)
    mz, it = r.average_spectrum(lo * 60, hi * 60)
    if not len(mz):
        raise HTTPException(404, "No MS1 spectra in that time range")
    pol = 1 if (r.polarity >= 0).mean() >= 0.5 else -1
    masses, score = chem.deconvolute(mz, it, (zmin, zmax), (mmin, mmax), step=step, polarity=pol)
    m, s = _thin(masses, score, 4000)
    return {"mass": _r(m, 2), "score": _r(s, 1), "peaks": chem.pick_masses(masses, score)}


@app.get("/api/jobs/{jid}/run/{k}/annotate")
def annotate(jid: str, k: int, idx: int, seq: str, tol: float = 20, charges: str = "1,2"):
    r = _run(jid, k)
    mz, it = r.spectrum(idx)
    zs = tuple(int(z) for z in charges.split(",") if z.strip())
    try:
        hits, theo = chem.annotate_spectrum(mz, it, seq, charges=zs, tol_ppm=tol)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read that peptide: {e}")
    matched = sum(it[h["peak"]] for h in {h["peak"]: h for h in hits}.values())
    z = int(r.precursor_charge[idx]) or 2
    return {"hits": hits, "n_theoretical": len(theo), "explained": float(matched / it.sum()) if it.sum() else 0,
            "peptide_mz": chem.peptide_mass(seq, z), "z": z,
            "precursor": None if not np.isfinite(r.precursor_mz[idx]) else float(r.precursor_mz[idx])}


class Reint(BaseModel):
    row: int
    start: float | None = None      # minutes; omitted = the automatic boundaries
    end: float | None = None


@app.post("/api/jobs/{jid}/reintegrate")
def reintegrate(jid: str, body: Reint):
    """Integrate one target in one file between boundaries the user chose (targeted analyses).

    Nothing in the job is changed: the new area and concentration are returned for review, and the trace
    comes back with them so the UI can draw exactly what was integrated.
    """
    from . import targeted
    res = read_json(JOBS / jid / "result.json")
    rows = next((it["rows"] for s in res["sections"] for it in s["items"] if it.get("type") == "target_review"), None)
    if not rows or not 0 <= body.row < len(rows):
        raise HTTPException(404, "No such integration")
    row = rows[body.row]
    start = body.start if body.start is not None else row.get("start")
    end = body.end if body.end is not None else row.get("end")
    if start is None or end is None:
        rt = row.get("rt") or row.get("expected_rt") or 0
        w = row.get("window") or 0.5
        start, end = (start if start is not None else rt - w / 4), (end if end is not None else rt + w / 4)
    if end <= start:
        raise HTTPException(400, "The end of the peak must come after its start.")
    run = _run(jid, int(row.get("file_index") or 0))
    out = targeted.reintegrate(run, row, start * 60, end * 60)
    for k in ("start", "end", "rt"):
        if out.get(k) is not None:
            out[k] = out[k] / 60
    t = np.asarray(out.pop("t"), float) / 60
    return {**out, "t": _r(t, 4), "y": _r(out.pop("y"), 1), "auto": {k: row.get(k) for k in ("start", "end", "area", "concentration", "report")},
            "target": row.get("target"), "sample": row.get("sample")}


# ---------------------------------------------------------------- calculators
class FormulaIn(BaseModel):
    formula: str
    polarity: int = 1


@app.post("/api/tools/formula")
def formula_tool(body: FormulaIn):
    try:
        mono = chem.formula_mass(body.formula)
        avg = chem.formula_mass(body.formula, average=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Not a formula I can read: {e}")
    adducts = [{"adduct": a, "mz": chem.adduct_mz(mono, a)} for a in chem.DEFAULT_ADDUCTS[1 if body.polarity >= 0 else -1]]
    mzs, ab = chem.isotope_pattern(body.formula, 6, adduct=adducts[0]["adduct"])
    return {"formula": body.formula, "monoisotopic": mono, "average": avg, "adducts": adducts,
            "isotopes": [{"mz": float(m), "rel": float(a)} for m, a in zip(mzs, ab) if a > 0.05]}


class PeptideIn(BaseModel):
    sequence: str
    charges: list[int] = [1, 2]


@app.post("/api/tools/peptide")
def peptide_tool(body: PeptideIn):
    seq = body.sequence.strip()
    try:
        mono = chem.peptide_mass(seq)
        frags = chem.fragment_ions(seq, charges=tuple(body.charges))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Not a peptide sequence I can read: {e}")
    return {"sequence": seq, "monoisotopic": mono, "mz": {z: chem.peptide_mass(seq, z) for z in range(1, 6)}, "fragments": frags,
            "note": "Cysteines carry carbamidomethyl (+57.021); add other modifications in brackets, e.g. PEPM[+15.995]TIDE."}


class MzIn(BaseModel):
    mz: float
    polarity: int = 1
    ppm: float = 10


@app.post("/api/tools/mz")
def mz_tool(body: MzIn):
    """Which neutral masses, and which known metabolites, could an observed m/z be?"""
    pol = 1 if body.polarity >= 0 else -1
    rows = [{"adduct": a, "neutral": chem.neutral_from_mz(body.mz, a)} for a in chem.DEFAULT_ADDUCTS[pol]]
    hits = []
    lib = ROOT / "server" / "data" / "metabolites.csv"
    if lib.exists():
        import pandas as pd
        df = pd.read_csv(lib)
        masses = [chem.formula_mass(f) if isinstance(f, str) else np.nan for f in df["formula"]]
        for r in rows:
            for m, (_, row) in zip(masses, df.iterrows()):
                if np.isfinite(m) and abs(m - r["neutral"]) / m * 1e6 <= body.ppm:
                    hits.append({"name": row["name"], "formula": row["formula"], "adduct": r["adduct"],
                                 "ppm": (r["neutral"] - m) / m * 1e6})
    return {"candidates": rows, "library": sorted(hits, key=lambda h: abs(h["ppm"]))[:30]}


# ---------------------------------------------------------------- code, settings, assistant
@app.get("/api/source")
def source():
    out = {}
    for p in sorted((ROOT / "server").glob("*.py")):
        out[f"server/{p.name}"] = p.read_text()
    return out


class Settings(BaseModel):
    api_key: str | None = None
    model: str | None = None


@app.get("/api/settings")
def get_settings():
    cfg = load_settings()
    key = cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY") or ""
    return {"version": VERSION, "has_key": bool(key), "key_hint": (key[:7] + "…" + key[-4:]) if key else "",
            "model": cfg.get("model") or agent.DEFAULT_MODEL, "home": str(HOME)}


@app.post("/api/settings")
def set_settings(body: Settings):
    save_settings({k: v for k, v in body.model_dump().items() if v})
    return get_settings()


@app.get("/api/models")
def list_models():
    return agent.available_models()


@app.post("/api/quit")
def quit_server():
    """Sidebar 'Quit': stop the server and everything it started (a search, a conversion) right away."""
    threading.Timer(0.5, lambda: lifecycle.quit_now("Quit was chosen in the app")).start()
    return {"ok": True}


# The page's heartbeat and its goodbye. The goodbye comes from navigator.sendBeacon, which posts the page id
# as a plain-text body, so both read the raw body rather than JSON.
@app.post("/api/alive")
async def page_alive(request: Request):
    lifecycle.alive((await request.body()).decode(errors="replace")[:64] or "page")
    return {"ok": True}


@app.get("/api/window")
async def page_window(request: Request, page: str = "page"):
    """An event stream the page keeps open for as long as its window exists. When the window goes, the socket
    drops and the server knows at once — even if the browser never fired 'pagehide'."""
    import asyncio
    page = page[:64]

    async def stream():
        lifecycle.window_open(page)
        try:
            yield "retry: 1000\n\n"
            while not await request.is_disconnected():
                yield ": still here\n\n"
                await asyncio.sleep(2)
        finally:
            lifecycle.window_closed(page)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/closing")
async def page_closing(request: Request):
    lifecycle.closing((await request.body()).decode(errors="replace")[:64] or "page")
    return Response(status_code=204)


class ChatIn(BaseModel):
    conv: str
    message: str
    job: str | None = None


@app.post("/api/agent/chat")
def agent_chat(body: ChatIn):
    return StreamingResponse(agent.chat(body.conv, body.message, body.job, _rerun), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/agent/reset")
def agent_reset(body: ChatIn):
    agent.reset(body.conv)
    return {"ok": True}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
