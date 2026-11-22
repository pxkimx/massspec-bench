"""Other web sites must not be able to quit the app or change anything (a cross-site POST needs no preflight).

    pytest tests/test_origin_guard.py      or      python tests/test_origin_guard.py

Starts the real server on a free port with a throw-away home folder; a few seconds; nothing is left running.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _status(port, path, method="GET", headers=None, body=None) -> int:
    req = urllib.request.Request(f"http://localhost:{port}{path}", data=body, method=method, headers=headers or {})
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code


def run_checks() -> list[tuple[str, bool]]:
    port = _free_port()
    home = tempfile.mkdtemp(prefix="origin_guard_")
    env = {**os.environ, "MB_HOME": home, "PP_HOME": home, "MB_KEEP_ALIVE": "1", "PP_KEEP_ALIVE": "1"}
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(port)], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site", "Content-Type": "text/plain"}
        own = {"Origin": f"http://localhost:{port}", "Sec-Fetch-Site": "same-origin"}
        out = [
            ("another site cannot quit the app", _status(port, "/api/quit", "POST", evil, b"") == 403),
            ("another localhost port is refused", _status(port, "/api/closing", "POST",
                                                          {"Origin": "http://localhost:1"}, b"tab") == 403),
            ("an opaque 'null' origin is refused", _status(port, "/api/alive", "POST", {"Origin": "null"}, b"tab") == 403),
            ("the app's own page may post", _status(port, "/api/alive", "POST", own, b"tab") < 400),
            ("a client without Origin (curl) may post", _status(port, "/api/alive", "POST", None, b"tab") < 400),
            ("reading is not blocked", _status(port, "/", "GET", evil) == 200),
        ]
        time.sleep(0.3)
        out.append(("server still running after the refused quit", p.poll() is None))
        return out
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def test_origin_guard():
    bad = [name for name, ok in run_checks() if not ok]
    assert not bad, bad


if __name__ == "__main__":
    res = run_checks()
    for name, ok in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print("ORIGIN GUARD", "PASSED" if all(ok for _, ok in res) else "FAILED")
    sys.exit(0 if all(ok for _, ok in res) else 1)
