"""The app quits completely when its window is closed (server/lifecycle.py).

    "$HOME/Library/Application Support/MassSpecBench/venv/bin/python" tests/test_lifecycle.py

Starts the real server on a free port with a throw-away MB_HOME and checks each way it can end — and the ways
it must not: a reload, or nobody having opened it yet. About a minute. Nothing is left running afterwards.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(port, path, body=b""):
    return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method="POST"), timeout=3)


def listening(port) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/settings", timeout=1)
        return True
    except Exception:  # noqa: BLE001
        return False


def server(env_extra=None):
    port = free_port()
    env = {**os.environ, "MB_HOME": tempfile.mkdtemp(prefix="msb_life_"), **(env_extra or {})}
    env.pop("MB_KEEP_ALIVE", None)
    p = subprocess.Popen([PY, "-m", "uvicorn", "server.app:app", "--port", str(port)], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        if listening(port):
            return p, port
        time.sleep(0.25)
    p.kill()
    raise RuntimeError("server did not start")


def window(port, page):
    """Open the page's event stream the way EventSource does, and keep it open (like a browser window)."""
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", f"/api/window?page={page}", headers={"Accept": "text/event-stream"})
    r = c.getresponse()
    r.read1(64)                                   # the stream has started
    return c


def ended_within(p, seconds) -> bool:
    try:
        p.wait(timeout=seconds)
        return True
    except subprocess.TimeoutExpired:
        return False


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return ok


def main() -> int:
    from server import lifecycle
    ok = True

    # the rule itself
    q = lifecycle.should_quit
    ok &= check("nobody opened it yet: keep running", q(0, False, 5, True)[0] is False)
    ok &= check("a page is open: keep running", q(1, True, 10, True)[0] is False)
    st = (0, True, 0, True)
    steps = 0
    while not q(*st)[0]:
        _, seen, empty, closing = q(*st)
        st = (0, seen, empty, closing)
        steps += 1
    ok &= check("after a close: quit on the third empty check", steps == lifecycle.AFTER_CLOSE - 1, f"({steps + 1} checks)")

    # 1. the tab is closed -> quits in about six seconds, port released
    p, port = server()
    post(port, "/api/alive", b"tab1")
    time.sleep(1)
    t = time.monotonic()
    post(port, "/api/closing", b"tab1")
    gone = ended_within(p, 15)
    ok &= check("tab closed: server quits", gone, f"after {time.monotonic() - t:.1f} s")
    ok &= check("tab closed: port released", not listening(port))
    if not gone:
        p.kill()

    # 2. a reload (close + a new page right away) does not quit
    p, port = server()
    post(port, "/api/alive", b"tab1")
    post(port, "/api/closing", b"tab1")
    time.sleep(0.5)
    post(port, "/api/alive", b"tab2")
    alive_after = not ended_within(p, 12)
    ok &= check("reload: still running 12 s later", alive_after)
    # two tabs, one closed: keep running
    post(port, "/api/alive", b"tab3")
    post(port, "/api/closing", b"tab2")
    ok &= check("one of two tabs closed: still running", not ended_within(p, 10))
    post(port, "/api/closing", b"tab3")
    ok &= check("then the last tab closed: quits", ended_within(p, 15))

    # 2b. the window goes without a goodbye (the Claude browser pane skips 'pagehide') -> the stream drops -> quits fast
    p, port = server()
    post(port, "/api/alive", b"tab1")
    w = window(port, "tab1")
    time.sleep(1)
    t = time.monotonic()
    w.close()
    ok &= check("window gone, no beacon: quits via the dropped stream", ended_within(p, 15), f"after {time.monotonic() - t:.1f} s")

    # 2c. reload through the stream: old stream drops, new page connects -> keeps running; a late heartbeat from the
    #     closed page must not bring it back to life, nor clear the "a window just closed" state
    p, port = server()
    w1 = window(port, "old")
    time.sleep(0.5)
    w1.close()
    w2 = window(port, "new")
    ok &= check("reload via stream: still running 10 s later", not ended_within(p, 10))
    post(port, "/api/alive", b"old")                 # in flight when the old page went
    t = time.monotonic()
    w2.close()
    ok &= check("then the window goes: quits in seconds despite a stale heartbeat", ended_within(p, 15), f"after {time.monotonic() - t:.1f} s")

    # 3. never opened -> keeps running (a slow browser must not lose the app)
    p, port = server({"MB_LIFECYCLE_FAST": "1"})
    ok &= check("never opened: still running", not ended_within(p, 8))
    p.terminate()
    p.wait()

    # 4. browser vanished without a word (crash) -> quits after the silence limit (shortened clock)
    p, port = server({"MB_LIFECYCLE_FAST": "1"})
    post(port, "/api/alive", b"tab1")
    t = time.monotonic()
    ok &= check("silent disconnect: quits after the silence limit", ended_within(p, 20), f"after {time.monotonic() - t:.1f} s (fast clock)")

    # 5. the launcher is killed (Dock force-quit) -> quits within a few seconds
    launcher = subprocess.Popen(["sleep", "600"])
    p, port = server({"MB_LAUNCHER_PID": str(launcher.pid)})
    post(port, "/api/alive", b"tab1")
    t = time.monotonic()
    launcher.kill()
    launcher.wait()
    ok &= check("launcher killed: server quits", ended_within(p, 8), f"after {time.monotonic() - t:.1f} s")

    # 6. everything the server started goes with it (children and grandchildren)
    marker = "7773.5"
    code = ("import subprocess, time; from server import lifecycle; "
            f"subprocess.Popen(['bash', '-c', 'sleep {marker} & sleep {marker}; wait']); time.sleep(1); lifecycle.quit_now('test')")
    subprocess.run([PY, "-c", code], cwd=ROOT, timeout=30)
    time.sleep(0.5)
    left = subprocess.run(["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True).stdout.split()
    ok &= check("quit stops child and grandchild processes", not left, f"left: {left}" if left else "")
    for pid in left:
        os.kill(int(pid), signal.SIGKILL)

    print("LIFECYCLE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
