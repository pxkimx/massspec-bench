"""In-app Claude agent. Streams a tool-using conversation over SSE. Tools let it inspect jobs,
run or re-run analyses, edit metadata, and execute Python inside a job folder to fix things."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

from .common import JOBS, load_settings, read_json

DEFAULT_MODEL = "claude-sonnet-5"
FALLBACK_MODELS = ["claude-sonnet-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"]


def available_models() -> dict:
    cfg = load_settings()
    key = cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"ok": False, "error": "no key", "models": FALLBACK_MODELS}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        ids = [m.id for m in client.models.list(limit=100)]
        ids = [i for i in ids if i.startswith("claude")]
        # newest first as the API returns them; put sonnet models at the top
        ids = sorted(ids, key=lambda i: (0 if "sonnet" in i else 1 if "opus" in i else 2))
        return {"ok": True, "models": ids or FALLBACK_MODELS}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "models": FALLBACK_MODELS}


def _resolve_model(client, wanted: str) -> str:
    """Use the configured model if the key can see it; otherwise the newest Sonnet (models get retired)."""
    try:
        ids = [m.id for m in client.models.list(limit=100)]
    except Exception:  # noqa: BLE001
        return wanted
    if wanted in ids:
        return wanted
    for pat in ("sonnet", "opus", "haiku"):
        for i in ids:
            if pat in i:
                return i
    return wanted

SYSTEM = """You are the analysis assistant built into MassSpec Bench, a local app for mass-spectrometry data. It reads
vendor raw files (Thermo, Bruker timsTOF, Agilent, Waters, ANDI) and open formats (mzML, mzXML, MGF) and runs four
analyses: a raw-data overview (explore), untargeted LC-MS metabolomics/lipidomics (pyOpenMS feature finding, alignment,
gap filling, QC-based drift correction, PQN, PCA, Welch/ANOVA statistics, m/z annotation), bottom-up proteomics (Sage
database search at 1% FDR, MaxLFQ-style protein quantification, limma-style moderated t-test) and targeted
quantification (XIC/SRM integration, weighted calibration curves, LOD/LOQ). You talk to a scientist looking at a
results page. Be concise, concrete and honest about limits: annotations from m/z alone are putative (MSI level 3),
cells of a feature table are not independent of the processing choices, and a p-value needs real biological
replicates. Quote numbers from the result JSON, name features/proteins/compounds, and flag design problems (too few
replicates, groups confounded with injection order or batch, QCs that do not cluster). When the user asks to change
something, prefer re-running with different parameters (rerun_analysis) over ad-hoc code — pass only the parameters
that change. Use run_python for custom plots, extra statistics, or anything the pipeline does not offer: the job
folder holds the CSV outputs, and the parsed raw data can be loaded with
`from server.msio import Run; r = Run.load(path)` using the `npz` paths listed in result.json["runs"] (r.xic(mz, ppm),
r.spectrum(i), r.rt in seconds, r.ms_level, r.precursor_mz). Save figures as PNG in the job folder and mention them by
file name so the UI shows them. Each result carries a params_panel (every parameter, the value used, where it came
from and why) — read it before discussing or changing a parameter. Never invent results — read them with the tools.
Retention times: seconds inside the data, minutes in everything shown to the user. Keep answers short unless asked for depth."""

TOOLS = [
    {"name": "list_jobs", "description": "List recent analyses with their id, kind, name and status.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_result", "description": "Full result of a finished job: tiles, flags (findings), sections, tables, methods, params, and params_panel (every parameter with the value used, where it came from and why). Use this before interpreting anything.",
     "input_schema": {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}},
    {"name": "list_files", "description": "List files in a job folder (inputs, CSV results, figures).",
     "input_schema": {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}},
    {"name": "read_file", "description": "Read the first N lines of a text/CSV file in a job folder (default 40).",
     "input_schema": {"type": "object", "properties": {"job": {"type": "string"}, "path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["job", "path"]}},
    {"name": "rerun_analysis", "description": (
         "Re-run a job with changed parameters (same input files). Pass ONLY the parameters you want to change: anything "
         "omitted keeps the previous value, and null returns a parameter to automatic. get_result's params_panel lists every "
         "parameter key with the value used and why. The sample sheet is params['sheet'] (list of rows with file, sample, "
         "group, type sample/qc/blank/standard, batch, order, concentration) — to regroup samples pass the whole edited sheet.\n"
         "Returns the new job id; poll get_status until done."),
     "input_schema": {"type": "object", "properties": {"job": {"type": "string"}, "params": {"type": "object"}}, "required": ["job", "params"]}},
    {"name": "get_status", "description": "Progress of a running job (state, pct, message, error).",
     "input_schema": {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}},
    {"name": "write_file", "description": "Write a text file into a job's input folder (e.g. a corrected metadata CSV), then rerun_analysis to use it.",
     "input_schema": {"type": "object", "properties": {"job": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["job", "path", "content"]}},
    {"name": "run_python", "description": "Run Python code with the job folder as working directory (numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib, pyopenms, pyteomics available; the app's own modules import as `server.*`). CSV results are in the folder; parsed raw runs load with server.msio.Run.load(npz) using result.json['runs'][k]['npz']. Returns stdout/stderr and new files. Save figures as PNG.",
     "input_schema": {"type": "object", "properties": {"job": {"type": "string"}, "code": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["job", "code"]}},
]


def _job_dir(job: str) -> Path:
    p = (JOBS / job).resolve()
    if not str(p).startswith(str(JOBS.resolve())) or not p.exists():
        raise ValueError(f"unknown job {job}")
    return p


def _safe(p: Path, root: Path) -> Path:
    q = (root / p).resolve()
    if not str(q).startswith(str(root.resolve())):
        raise ValueError("path escapes job folder")
    return q


def run_tool(name: str, inp: dict, start_rerun) -> str:
    try:
        if name == "list_jobs":
            rows = []
            for d in sorted(JOBS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
                if not (d / "request.json").exists():
                    continue
                req = json.loads((d / "request.json").read_text())
                st = read_json(d / "status.json") if (d / "status.json").exists() else {}
                rows.append({"job": d.name, "kind": req.get("kind"), "name": req.get("name"), "state": st.get("state"), "message": st.get("message")})
            return json.dumps(rows)
        job = _job_dir(inp["job"])
        if name == "get_result":
            r = read_json(job / "result.json")
            slim = {k: r[k] for k in ("kind", "name", "tiles", "flags", "params", "methods", "summary", "downloads") if k in r}
            slim["runs"] = [{k: x.get(k) for k in ("name", "file", "npz", "format", "spectra", "ms1", "ms2", "polarity")} for x in r.get("runs", [])]
            # the parameters panel without the UI scaffolding (control type, options, ranges)
            slim["params_panel"] = [{"group": g["title"], "fields": [
                {"key": f["key"], "label": f["label"], "value": f["value"], "source": f["source"], "why": f["why"]}
                for f in g["fields"]]} for g in r.get("params_panel", [])]
            slim["sections"] = [{"id": s["id"], "title": s["title"], "items": [
                {"type": i["type"], "title": i.get("title"), "yours": i.get("yours"), "id": i.get("id"),
                 **({"columns": i["columns"], "rows": i["rows"][:15]} if i["type"] == "table" else {}),
                 **({"file": i["src"]} if i["type"] == "figure" else {})} for i in s["items"]]} for s in r["sections"]]
            return json.dumps(slim)[:60000]
        if name == "get_status":
            return json.dumps(read_json(job / "status.json", {}))
        if name == "list_files":
            out = []
            for p in sorted(job.rglob("*")):
                if p.is_file() and "__pycache__" not in str(p):
                    out.append(f"{p.relative_to(job)}  ({p.stat().st_size // 1024} KB)")
            return "\n".join(out[:300])
        if name == "read_file":
            p = _safe(Path(inp["path"]), job)
            n = int(inp.get("lines") or 40)
            opener = __import__("gzip").open if p.suffix == ".gz" else open
            with opener(p, "rt", errors="replace") as fh:
                return "".join([next(fh, "") for _ in range(n)])[:20000]
        if name == "write_file":
            p = _safe(Path("input") / Path(inp["path"]).name, job)
            p.write_text(inp["content"])
            req = json.loads((job / "request.json").read_text())
            if str(p) not in req["files"]:
                req["files"].append(str(p))
                (job / "request.json").write_text(json.dumps(req))
            return f"wrote {p.relative_to(job)} ({len(inp['content'])} chars) and added it to the job's input files"
        if name == "rerun_analysis":
            new = start_rerun(job.name, inp.get("params") or {})
            return json.dumps(new)
        if name == "run_python":
            before = {p.name for p in job.iterdir()}
            code = inp["code"]
            env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
            proc = subprocess.run([sys.executable, "-c", code], cwd=job, capture_output=True, text=True,
                                  timeout=int(inp.get("timeout") or 180), env=env)
            new = sorted({p.name for p in job.iterdir()} - before)
            out = (proc.stdout[-8000:] + ("\n[stderr]\n" + proc.stderr[-4000:] if proc.stderr.strip() else ""))
            return out + (f"\n[new files] {', '.join(new)}" if new else "") + f"\n[exit {proc.returncode}]"
        return f"unknown tool {name}"
    except subprocess.TimeoutExpired:
        return "error: timed out"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------- conversations
CONV: dict[str, list] = {}
LOCK = threading.Lock()


def chat(conv_id: str, user_text: str, job: str | None, start_rerun) -> Iterator[str]:
    """Yields SSE 'data:' lines: {type: text|tool|tool_result|done|error}."""
    try:
        import anthropic
    except ImportError:
        yield _sse({"type": "error", "text": "The anthropic package is missing: pip install anthropic"}); return
    cfg = load_settings()
    key = cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        yield _sse({"type": "error", "text": "No API key. Open Settings (gear icon) and paste an Anthropic API key."}); return
    client = anthropic.Anthropic(api_key=key)
    model = _resolve_model(client, cfg.get("model") or DEFAULT_MODEL)
    with LOCK:
        hist = CONV.setdefault(conv_id, [])
    ctx = ""
    if job:
        try:
            r = json.loads((_job_dir(job) / "result.json").read_text())
            ctx = (f"\n\nThe user is currently viewing job '{job}' ({r['kind']}): {r['name']}. Key findings:\n"
                   + "\n".join(f"- [{f['level']}] {f['text']}" for f in r["flags"])
                   + f"\nParameters: {json.dumps(r.get('params'))}")
        except Exception:  # noqa: BLE001
            ctx = f"\n\nThe user is viewing job '{job}' (still running or failed)."
    hist.append({"role": "user", "content": user_text})
    for _ in range(12):  # tool-use rounds
        text_buf = ""
        try:
            with client.messages.stream(model=model, max_tokens=4000, system=SYSTEM + ctx, tools=TOOLS, messages=hist) as stream:
                for ev in stream:
                    if ev.type == "content_block_delta" and getattr(ev.delta, "type", "") == "text_delta":
                        text_buf += ev.delta.text
                        yield _sse({"type": "text", "text": ev.delta.text})
                msg = stream.get_final_message()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "authentication" in msg.lower() or "invalid x-api-key" in msg.lower():
                msg = "The API key was rejected. Open Settings & API key and paste a valid key from console.anthropic.com."
            elif "credit" in msg.lower() or "billing" in msg.lower():
                msg = "Anthropic says this key has no credit. Add a prepaid balance at console.anthropic.com → Billing."
            yield _sse({"type": "error", "text": f"{type(e).__name__}: {msg}"}); hist.pop(); return
        hist.append({"role": "assistant", "content": [b.model_dump() for b in msg.content]})
        tool_uses = [b for b in msg.content if b.type == "tool_use"]
        if not tool_uses:
            break
        results = []
        for tu in tool_uses:
            yield _sse({"type": "tool", "name": tu.name, "input": tu.input})
            out = run_tool(tu.name, tu.input, start_rerun)
            yield _sse({"type": "tool_result", "name": tu.name, "text": out[:1500]})
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
        hist.append({"role": "user", "content": results})
    yield _sse({"type": "done"})


def _sse(d: dict) -> str:
    return f"data: {json.dumps(d)}\n\n"


def reset(conv_id: str):
    with LOCK:
        CONV.pop(conv_id, None)
