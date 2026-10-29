"""Vendor-format readers checked against real, public example files.

Run from source/:
    "$HOME/Library/Application Support/MassSpecBench/venv/bin/python" tests/test_vendor.py            # fetch + check
    "$HOME/Library/Application Support/MassSpecBench/venv/bin/python" tests/test_vendor.py --no-fetch # only check what is present

Files go to tests/_out/vendor/ (git-ignored). Every file is fetched from the public GitHub repository it ships
with (URLs below), so the checks can be re-run on a fresh machine. Not pytest-dependent: prints a table and
exits non-zero if any present file fails its checks.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "_out" / "vendor"

RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"
TREE = "https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"

# (local name, repo, branch, path in repo). A path that is a folder is fetched recursively.
SOURCES = [
    # Thermo .raw — ThermoRawFileParser's own test data (Orbitrap, DDA)
    ("thermo/small.RAW", "compomics/ThermoRawFileParser", "master", "ThermoRawFileParserTest/Data/small.RAW"),
    ("thermo/small2.RAW", "compomics/ThermoRawFileParser", "master", "ThermoRawFileParserTest/Data/small2.RAW"),
    # Bruker timsTOF — a real 10 s diaPASEF idle-flow acquisition on a timsTOF SCP (timsrust_pyo3 test data)
    ("bruker/idleflow_diaPASEF_10s.d", "jspaezp/timsrust_pyo3", "main",
     "tests/data/230711_idleflow_400-1000mz_25mz_diaPasef_10sec.d"),
    # ... and a synthetic mini DDA-PASEF file (3 precursors) written by the timsrust author
    ("bruker/dda_test.d", "jspaezp/timsrust_pyo3", "main", "tests/data/dda_test.d"),
    # Agilent / Waters — rainbow-api test data (evanyeyeye/rainbow, tests/inputs)
    ("agilent/amber.D", "evanyeyeye/rainbow", "main", "tests/inputs/amber.D"),      # MassHunter MSProfile.bin
    ("agilent/copper.D", "evanyeyeye/rainbow", "main", "tests/inputs/copper.D"),    # MassHunter MSPeak.bin
    ("agilent/silver.D", "evanyeyeye/rainbow", "main", "tests/inputs/silver.D"),    # MassHunter profile + XSpecific
    ("agilent/yellow.D", "evanyeyeye/rainbow", "main", "tests/inputs/yellow.D"),    # MassHunter GC-MS (MSPeak) + FID + data.ms
    ("agilent/bronze.D", "evanyeyeye/rainbow", "main", "tests/inputs/bronze.D"),    # MassHunter, DAD only (no MS)
    ("agilent/orange.D", "evanyeyeye/rainbow", "main", "tests/inputs/orange.D"),    # ChemStation MSD1.MS + ADC
    ("agilent/red.D", "evanyeyeye/rainbow", "main", "tests/inputs/red.D"),          # ChemStation, UV only (no MS)
    ("waters/blue.raw", "evanyeyeye/rainbow", "main", "tests/inputs/blue.raw"),
    ("waters/violet.raw", "evanyeyeye/rainbow", "main", "tests/inputs/violet.raw"),
    ("waters/white.raw", "evanyeyeye/rainbow", "main", "tests/inputs/white.raw"),
    ("waters/indigo.raw", "evanyeyeye/rainbow", "main", "tests/inputs/indigo.raw"),
    ("waters/turquoise.raw", "evanyeyeye/rainbow", "main", "tests/inputs/turquoise.raw"),
    # ANDI-MS netCDF — LC-MS (xcms faahKO) and GC-MS (PyMassSpec)
    ("andi/ko15.CDF", "bioc/faahKO", "devel", "inst/cdf/KO/ko15.CDF"),
    ("andi/gc01_0812_066.cdf", "PyMassSpec/PyMassSpec", "master", "tests/data/gc01_0812_066.cdf"),
]
# a lone ChemStation .ms file (not inside a .D folder), made by copying one out of orange.D
DERIVED = [("agilent/MSD1.ms", "agilent/orange.D/MSD1.MS")]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "MassSpecBench-tests"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch():
    import server  # noqa: F401 - sets the certifi CA bundle for urllib
    trees = {}
    for local, repo, branch, path in SOURCES:
        dest = OUT / local
        if dest.exists():
            continue
        key = (repo, branch)
        if key not in trees:
            trees[key] = json.loads(_get(TREE.format(repo=repo, branch=branch)))["tree"]
        blobs = [t for t in trees[key] if t["type"] == "blob" and (t["path"] == path or t["path"].startswith(path + "/"))]
        if not blobs:
            print(f"  !! {path} not found in {repo}@{branch}")
            continue
        tmp = dest.with_name(dest.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.unlink(missing_ok=True)
        print(f"  fetching {local} ({sum(b.get('size', 0) for b in blobs) / 1e6:.1f} MB, {len(blobs)} file(s))")
        for b in blobs:
            target = tmp if b["path"] == path else tmp / b["path"][len(path) + 1:]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_get(RAW.format(repo=repo, branch=branch, path=urllib.request.quote(b["path"]))))
        tmp.rename(dest)
    for local, src in DERIVED:
        if not (OUT / local).exists() and (OUT / src).exists():
            shutil.copy(OUT / src, OUT / local)


# ------------------------------------------------------------------ checks

def check(path: Path) -> dict:
    """Read through the app's own entry points (detect → read_any → cache → reload) and sanity-check."""
    import numpy as np
    from server import msio

    fmt = msio.detect_format(path)
    t0 = time.time()
    run = msio.read_any(path)
    dt = time.time() - t0
    s = run.summary()
    problems = []
    if run.n == 0 and not run.chromatograms:
        problems.append("no spectra and no traces")
    if run.n:
        if not np.all(np.diff(run.rt) >= 0):
            problems.append("rt not sorted")
        if len(run.mz) and not (0 < run.mz.min() and run.mz.max() < 20000):
            problems.append(f"m/z out of range {run.mz.min():.1f}-{run.mz.max():.1f}")
        if len(run.intensity) and (run.intensity < 0).any():
            problems.append("negative intensity")
        for i in range(run.n):
            a, b = run.offsets[i], run.offsets[i + 1]
            if b - a > 1 and (np.diff(run.mz[a:b]) < 0).any():
                problems.append(f"spectrum {i} m/z not sorted")
                break
        ms2 = run.idx(2)
        if len(ms2) and not np.isfinite(run.precursor_mz[ms2]).any():
            problems.append("MS2 without precursor m/z")
        tic = run.tic[run.idx(1)] if len(run.idx(1)) else run.tic
        if len(tic) > 5 and not (tic > 0).any():
            problems.append("TIC all zero")
    for c in run.chromatograms:
        if len(c["time"]) != len(c["intensity"]) or not len(c["time"]):
            problems.append(f"trace {c['id']} empty/misaligned")
    # the parse cache must round-trip exactly
    npz = OUT / "_cache" / (path.name + ".npz")
    npz.parent.mkdir(parents=True, exist_ok=True)
    run.save(npz)
    back = msio.Run.load(npz)
    if back.n != run.n or len(back.mz) != len(run.mz) or len(back.chromatograms) != len(run.chromatograms):
        problems.append("cache round-trip changed the data")
    ms2 = run.idx(2)
    return {
        "file": str(path.relative_to(OUT)), "fmt": fmt, "sec": dt, "spectra": s["spectra"], "ms1": s["ms1"],
        "ms2": s["ms2"], "rt": f"{s['rt_min']:.2f}-{s['rt_max']:.2f}", "mz": f"{s['mz_min']:.0f}-{s['mz_max']:.0f}" if run.n else "-",
        "peaks": s["peaks"], "cent": "y" if s["centroided"] else "n", "pol": s["polarity"],
        "prec": int(np.isfinite(run.precursor_mz[ms2]).sum()) if len(ms2) else 0,
        "z": int((run.precursor_charge[ms2] > 0).sum()) if len(ms2) else 0,
        "im": "y" if run.ion_mobility is not None else "",
        "traces": len(run.chromatograms), "instrument": (run.meta.get("instrument") or "")[:28],
        "problems": problems, "run": run,
    }


def tic_plot(results, path: Path):
    """One small TIC per file so a human can see that it looks like a chromatogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rs = [r for r in results if "run" in r]
    if not rs:
        return
    fig, axes = plt.subplots(len(rs), 1, figsize=(8, 1.6 * len(rs)), squeeze=False)
    for ax, r in zip(axes[:, 0], rs):
        run = r["run"]
        if run.n:
            lvl = 1 if len(run.idx(1)) else 2
            pol = 1 if (run.polarity[run.idx(lvl)] >= 0).mean() >= 0.5 else -1   # one polarity: switching runs saw-tooth
            sel = run.idx(lvl, pol) if (run.polarity != 0).any() else run.idx(lvl)
            ax.plot(run.rt[sel] / 60, run.tic[sel], lw=0.7)
        for c in run.chromatograms[:3]:
            ax2 = ax.twinx() if run.n else ax
            ax2.plot(np.asarray(c["time"]) / 60, c["intensity"], lw=0.6, color="grey")
        ax.set_title(r["file"], fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("minutes")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"TIC overview: {path}")


def main():
    if "--no-fetch" not in sys.argv:
        print("Fetching example files into", OUT)
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            fetch()
        except Exception as e:  # noqa: BLE001 - still check whatever is already there
            print("  fetch failed:", type(e).__name__, e)
    targets = [OUT / local for local, *_ in SOURCES] + [OUT / local for local, _ in DERIVED]
    targets = [t for t in targets if t.exists()]
    if not targets:
        print("No example files present.")
        return 1
    from server.common import UserFacingError
    results = []
    for t in targets:
        try:
            results.append(check(t))
        except UserFacingError as e:
            results.append({"file": str(t.relative_to(OUT)), "error": f"UserFacingError: {e}"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results.append({"file": str(t.relative_to(OUT)), "error": f"{type(e).__name__}: {e}"})
    cols = ["file", "fmt", "sec", "spectra", "ms1", "ms2", "prec", "z", "rt", "mz", "peaks", "cent", "pol", "im",
            "traces", "instrument"]
    rows = [[f"{r.get(c):.1f}" if c == "sec" else str(r.get(c, "")) for c in cols] for r in results if "error" not in r]
    w = [max(len(c), *(len(x[i]) for x in rows)) if rows else len(c) for i, c in enumerate(cols)]
    print()
    print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    for x in rows:
        print("  ".join(v.ljust(w[i]) for i, v in enumerate(x)))
    bad = 0
    print()
    for r in results:
        if "error" in r:
            # files that hold no MS data are expected to fail with a clear message
            print(f"ERROR  {r['file']}: {r['error'][:300]}")
            bad += 0 if r["file"] in EXPECTED_ERRORS else 1
        elif r["problems"]:
            print(f"FAIL   {r['file']}: {'; '.join(r['problems'])}")
            bad += 1
        else:
            print(f"ok     {r['file']}  (rt in minutes; traces = non-MS detector channels)")
    tic_plot(results, OUT / "tic_overview.png")
    print(f"\n{len(results)} file(s), {bad} unexpected failure(s)")
    return 1 if bad else 0


EXPECTED_ERRORS: set[str] = set()

if __name__ == "__main__":
    sys.exit(main())
