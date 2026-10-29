"""Shared job bookkeeping and figure helpers.

Every pipeline writes into a Job: tiles (headline numbers), flags (findings), sections of figures and
tables, methods, and a params panel. result.json drives the UI, the PDF report and the assistant, so a
pipeline never talks to any of them directly.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("MB_HOME", Path.home() / "Library" / "Application Support" / "MassSpecBench"))
JOBS = HOME / "jobs"
TOOLS = HOME / "tools"          # downloaded helper programs (ThermoRawFileParser, Sage)
for _d in (JOBS, TOOLS):
    _d.mkdir(parents=True, exist_ok=True)

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
    "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
    "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#393b79",
]
UP, DOWN, NS = "#c9304f", "#2a64b0", "#c7ccd1"
QC_COLOR, BLANK_COLOR = "#222222", "#9aa4ab"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


style()


class Job:
    def __init__(self, root: Path, kind: str, name: str):
        self.root = Path(root)
        self.fig_dir = self.root / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.result = {
            "kind": kind, "name": name, "tiles": [], "flags": [], "sections": [],
            "methods": [], "versions": {}, "downloads": [], "params": {}, "params_panel": [],
        }
        self._section = None
        self._pgroup = None
        self.set_status("running", 0, "Starting")

    # ---- progress ----
    def set_status(self, state, pct, msg, error=None, trace=None):
        s = {"state": state, "pct": pct, "message": msg, "time": time.time()}
        if error:
            s["error"] = error
        if trace:
            s["trace"] = trace
        write_json(self.root / "status.json", s)

    def step(self, pct, msg):
        # Every step is a chance to stop. Cancellation is cooperative rather than a killed thread.
        self.check_cancelled()
        print(f"[{self.kind}] {pct:>3}% {msg}", flush=True)
        self.set_status("running", pct, msg)

    def check_cancelled(self):
        if (self.root / "cancel").exists():
            raise Cancelled("Stopped at your request.")

    # ---- content ----
    def tile(self, value, label):
        self.result["tiles"].append({"value": value, "label": label})

    def flag(self, level, text):
        """level: ok | warn | info. text may contain <b>…</b>."""
        self.result["flags"].append({"level": level, "text": text})

    def section(self, sid, kicker, title, lede=""):
        self._section = {"id": sid, "kicker": kicker, "title": title, "lede": lede, "items": []}
        self.result["sections"].append(self._section)

    def figure(self, fid, title, fig=None, how="", yours="", wide=False, sub=""):
        """Save a matplotlib figure (PNG + vector PDF) into the current section.

        how   — how to read this kind of plot, in plain language (the same for every dataset)
        yours — what this plot says about *this* dataset, computed from the numbers
        """
        fig = fig or plt.gcf()
        png = self.fig_dir / f"{fid}.png"
        fig.savefig(png, dpi=200, facecolor="white")
        vector = False
        try:
            fig.savefig(self.fig_dir / f"{fid}.pdf", facecolor="white", bbox_inches="tight")
            vector = True
        except Exception:  # noqa: BLE001 - a figure that will not vectorise must not lose its PNG
            log_exc(f"vector export {fid}")
        plt.close("all")
        self._section["items"].append({
            "type": "figure", "id": fid, "title": title, "sub": sub,
            "src": f"figures/{fid}.png", "how": how, "yours": yours, "wide": wide,
            **({"vector": f"figures/{fid}.pdf"} if vector else {}),
        })

    def table(self, tid, title, columns, rows, note="", csv=None):
        """rows: list of lists (already formatted for display, at most a few hundred). csv: file in the job folder."""
        item = {"type": "table", "id": tid, "title": title, "columns": columns, "rows": rows, "note": note}
        if csv:
            item["csv"] = csv
        self._section["items"].append(item)

    def widget(self, wtype, **kw):
        """Interactive element the UI knows how to draw (e.g. 'explorer')."""
        self._section["items"].append({"type": wtype, **kw})

    def method(self, head, text):
        self.result["methods"].append([head, text])

    def download(self, label, href):
        self.result["downloads"].append({"label": label, "href": href})

    # ---- parameters panel ----
    def param_group(self, title, note=""):
        self._pgroup = {"title": title, "note": note, "fields": []}
        self.result["params_panel"].append(self._pgroup)

    def param(self, key, label, value, why, source="default", type="number", options=None, **kw):
        """One row of the Parameters section.

        key    — must be a key the pipeline's run(params=...) accepts, or Re-analyze drops it; keys starting
                 with '_' and type='fixed' are display-only
        source — auto (derived from the data) | user (set by hand) | default | data | fixed
        why    — what actually happened with this value, written at analysis time
        type   — number | int | text | bool | select | fixed
        """
        if self._pgroup is None:
            self.param_group("Settings")
        f = {"key": key, "label": label, "value": value, "why": why, "source": source, "type": type}
        if options is not None:
            f["options"] = options
        f.update(kw)
        self._pgroup["fields"].append(f)
        if not key.startswith("_") and type != "fixed":
            self.result["params"][key] = value

    def save(self):
        self.result["sections"] = [s for s in self.result["sections"] if s.get("items")]
        write_json(self.root / "result.json", self.result)


def write_json(path: Path, obj) -> None:
    """Write JSON so a reader never sees a half-written file: write a temporary file, then rename it over the old one
    (a rename within one folder is atomic). status.json is polled every second while it is being rewritten."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, default=_json_default))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    """Read a job's JSON file; `default` if it does not exist. Retries briefly in case a writer from an older
    version (truncate-then-write) is mid-write."""
    path = Path(path)
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return default
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(0.05)


def resolve(params: dict, key: str, default, auto=None):
    """Value + source for a parameter: what the user set, else the data-derived value, else the default."""
    v = params.get(key)
    if v is not None and v != "":
        return v, "user"
    if auto is not None:
        return auto, "auto"
    return default, "default"


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    return str(o)


class Cancelled(Exception):
    """Raised at a step boundary when the user has asked for the run to stop."""


class UserFacingError(Exception):
    """An error whose message is written for the user (bad input, unsupported file) — shown without a traceback."""


def run_safely(job: Job, fn, *args, **kw):
    try:
        fn(job, *args, **kw)
        job.save()
        from .report import build_pdf
        job.step(98, "Writing PDF report")
        try:
            build_pdf(job.root)
        except Exception:  # noqa: BLE001 - a report that fails must not lose the analysis
            job.flag("warn", "The PDF report could not be written: " + log_exc("pdf"))
            job.save()
        job.set_status("done", 100, "Analysis complete")
    except Cancelled:
        job.set_status("cancelled", 100, "Stopped", error="You stopped this analysis.")
    except UserFacingError as e:
        job.set_status("error", 100, "Stopped", error=str(e))
    except Exception as e:  # pragma: no cover
        tb = traceback.format_exc()
        print(tb, flush=True)
        job.set_status("error", 100, "Stopped", error=f"{type(e).__name__}: {e}", trace=tb[-6000:] + "\n" + env_summary())


def log_exc(where: str) -> str:
    """Print the current exception's traceback to the server log and return a short 'Type: msg' string."""
    import sys
    et, ev, _ = sys.exc_info()
    print(f"[{where}] non-fatal error:\n" + traceback.format_exc(), flush=True)
    return f"{et.__name__ if et else 'Error'}: {str(ev)[:160]}"


def versions(*pkgs) -> dict:
    from importlib.metadata import version
    out = {}
    for k in pkgs:
        try:
            out[k] = version(k)
        except Exception:  # noqa: BLE001
            pass
    return out


def env_summary() -> str:
    import platform
    vs = versions("numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "statsmodels", "pyopenms",
                  "pyteomics", "rainbow-api", "timsrust_pyo3")
    return f"[env] python {platform.python_version()} {platform.machine()} | " + " ".join(f"{k}={v}" for k, v in vs.items())


def fmt_n(v):
    v = float(v)
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:.1f}G"
    if a >= 1e6:
        return f"{v/1e6:.1f}M"
    if a >= 1e4:
        return f"{v/1e3:.0f}k"
    if a >= 1e3:
        return f"{v/1e3:.1f}k"
    return f"{v:.0f}" if a >= 10 or v == int(v) else f"{v:.2g}"


def commas(v):
    return f"{int(round(float(v))):,}"


def pct(a, b):
    return f"{100*a/b:.1f}%" if b else "0%"


# ---------------------------------------------------------------- settings (API key, model)
CONFIG = Path.home() / ".massspec-bench" / "config.json"


def load_settings() -> dict:
    """Settings saved from the UI. The API key falls back to RNAseq Bench's, so one key serves both apps."""
    cfg = {}
    try:
        cfg = json.loads(CONFIG.read_text())
    except Exception:  # noqa: BLE001
        pass
    if not cfg.get("api_key"):
        try:
            k = json.loads((Path.home() / ".rnaseq-bench" / "config.json").read_text()).get("api_key")
            if k:
                cfg["api_key"] = k
        except Exception:  # noqa: BLE001
            pass
    return cfg


def save_settings(d: dict):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cur = {}
    try:
        cur = json.loads(CONFIG.read_text())
    except Exception:  # noqa: BLE001
        pass
    cur.update(d)
    CONFIG.write_text(json.dumps(cur))
    try:
        CONFIG.chmod(0o600)
    except Exception:  # noqa: BLE001
        pass
