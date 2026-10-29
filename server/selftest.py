"""Quick end-to-end check of this installation:  python -m server.selftest

Simulates small datasets with a known answer and runs every analysis on them (a few minutes), then
reports pass/fail. Uses a temporary MB_HOME so nothing touches your real analyses.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="msb_selftest_"))
    os.environ["MB_HOME"] = str(tmp / "home")
    from . import explore, msio, simulate
    from .common import Job, env_summary, run_safely

    print(env_summary())
    ok = True

    def check(name, fn, files, params):
        nonlocal ok
        d = tmp / name
        job = Job(d, name, f"selftest-{name}")
        t = time.time()
        run_safely(job, fn, files, params)
        st = json.loads((d / "status.json").read_text())
        print(f"{name}: {st['state']} in {time.time() - t:.0f}s {st.get('error', '')}")
        if st["state"] != "done":
            ok = False
            print(st.get("trace", "")[-2500:])
            return None
        r = json.loads((d / "result.json").read_text())
        print("  " + " · ".join(f"{t['value']} {t['label']}" for t in r["tiles"][:5]))
        return r

    study = simulate.metabolomics_study(tmp / "study", n_per_group=3, n_compounds=60, n_qc=3, rt_max=480)
    sheet = msio.match_sheet(study["files"], msio.read_table(study["sheet"]))
    check("explore", explore.run, study["files"][:3], {"sheet": sheet[:3]})
    try:
        from . import untargeted
        check("untargeted", untargeted.run, study["files"], {"sheet": sheet})
    except ImportError:
        print("untargeted: module missing")
        ok = False
    try:
        from . import targeted
        import pandas as pd
        truth = pd.read_csv(study["truth"])
        rows = truth.nlargest(4, "base_intensity")
        tl = tmp / "targets.csv"
        pd.DataFrame({"name": [f"cpd{i}" for i in rows.index], "mz": rows["mz_MH"], "rt": rows["rt"] / 60}).to_csv(tl, index=False)
        check("targeted", targeted.run, study["files"], {"sheet": sheet, "targets": str(tl)})
    except ImportError:
        print("targeted: module missing")
        ok = False
    try:
        from . import proteomics  # noqa: F401
        print("proteomics: module present (search needs the Sage download; covered by tests/test_proteomics.py)")
    except ImportError:
        print("proteomics: module missing")
        ok = False
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
