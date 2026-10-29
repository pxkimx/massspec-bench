"""Helper programs downloaded on first use: ThermoRawFileParser (Thermo .raw → mzML) and Sage (peptide search).

Both are open source and ship self-contained macOS builds on GitHub, so nothing needs .NET, Mono or
Docker. They are fetched once into MB_HOME/tools/<name>/ and reused. Anything that needs the internet
must degrade to a clear message, never a traceback: callers catch ToolUnavailable.
"""
from __future__ import annotations

import os
import platform
import shutil
import stat
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .common import TOOLS

ARM = platform.machine() == "arm64"

REGISTRY = {
    "ThermoRawFileParser": {
        "url": ("https://github.com/CompOmics/ThermoRawFileParser/releases/download/v.2.0.0-dev/"
                + ("ThermoRawFileParser-v.2.0.0-dev-osx-arm64.zip" if ARM else "ThermoRawFileParser-v.2.0.0-dev-osx.zip")),
        "exe": "ThermoRawFileParser",
        "about": "ThermoRawFileParser (CompOmics, Apache-2.0; uses Thermo's RawFileReader library)",
    },
    "sage": {
        "url": ("https://github.com/lazear/sage/releases/download/v0.14.7/"
                + ("sage-v0.14.7-aarch64-apple-darwin.tar.gz" if ARM else "sage-v0.14.7-x86_64-apple-darwin.tar.gz")),
        "exe": "sage",
        "about": "Sage 0.14.7 (Lazear 2023, MIT) — fast open-search proteomics engine with built-in FDR and LFQ",
    },
}


class ToolUnavailable(Exception):
    pass


def _adhoc_sign(exe: Path):
    """Apple Silicon kills an arm64 executable that has no code signature at all (SIGKILL, before it prints
    anything). Unsigned release binaries get an ad-hoc signature — what Homebrew does for what it builds;
    no identity is involved and no system setting changes. Binaries that already verify are left alone."""
    if platform.system() != "Darwin":
        return
    import subprocess
    try:
        if subprocess.run(["codesign", "-v", str(exe)], capture_output=True, timeout=60).returncode != 0:
            subprocess.run(["codesign", "--force", "--sign", "-", str(exe)], capture_output=True, timeout=120)
    except Exception:  # noqa: BLE001 - vendor.py still retries at first use
        pass


def tool_path(name: str) -> Path | None:
    """The executable if it is already installed (env override MB_<NAME> wins, e.g. MB_SAGE=/usr/local/bin/sage)."""
    env = os.environ.get("MB_" + name.upper())
    if env and Path(env).exists():
        return Path(env)
    d = TOOLS / name
    if d.exists():
        for p in d.rglob(REGISTRY[name]["exe"]):
            if p.is_file() and os.access(p, os.X_OK):
                return p
    found = shutil.which(REGISTRY[name]["exe"])
    return Path(found) if found else None


def ensure_tool(name: str, progress=None) -> Path:
    """Path to the tool's executable, downloading and unpacking it the first time."""
    p = tool_path(name)
    if p:
        return p
    spec = REGISTRY[name]
    d = TOOLS / name
    d.mkdir(parents=True, exist_ok=True)
    arc = d / spec["url"].rsplit("/", 1)[-1]
    try:
        if progress:
            progress(f"Downloading {name} (first use only)")
        with urllib.request.urlopen(spec["url"], timeout=60) as r, open(arc, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    except Exception as e:  # noqa: BLE001
        arc.unlink(missing_ok=True)
        raise ToolUnavailable(f"{name} is downloaded on first use and the download failed ({type(e).__name__}: {e}). "
                              "Check the internet connection and try again.") from e
    try:
        if arc.suffix == ".zip":
            with zipfile.ZipFile(arc) as z:
                z.extractall(d)
        else:
            with tarfile.open(arc) as t:
                t.extractall(d)
    finally:
        arc.unlink(missing_ok=True)
    for p in d.rglob(spec["exe"]):
        if p.is_file():
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            # a quarantine flag on a downloaded binary makes macOS refuse to run it
            os.system(f"xattr -dr com.apple.quarantine '{d}' >/dev/null 2>&1")
            _adhoc_sign(p)
            return p
    raise ToolUnavailable(f"{name} was downloaded but its program '{spec['exe']}' was not in the archive.")
