"""Quit completely when the app's window is closed.

MassSpec Bench is a local server with a browser tab as its window. Left alone, the server would keep running
after the tab is closed. This module ties the two together (same rules as Cell Surveillance's lifecycle.py):

* Every open page holds an event stream open (/api/window), says "still here" every few seconds
  (/api/alive), and says "closing" as it goes away (/api/closing, navigator.sendBeacon on pagehide). When no
  page is left, the server stops about six seconds later. A reload opens a new page within that time, so reloading does not quit the app.
* If no page has been heard from for about three minutes and none said it was closing (the browser crashed
  or was force-quit), the server stops as well. Background tabs are throttled by the browser to roughly one
  heartbeat a minute, so a page counts as open for 90 s after its last heartbeat.
* If the launcher that started the server is gone (the app was quit or force-quit from the Dock), the
  server stops.
* Whatever the reason, everything the server started — a Sage search, a ThermoRawFileParser conversion, the
  assistant's Python — is stopped with it, children first. Nothing is left listening or running.

Set MB_KEEP_ALIVE=1 to switch this off (dev.sh does, so auto-reload development is not interrupted).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

CHECK_S = 2.0            # how often the server looks at who is connected
PAGE_TTL_S = 90.0        # a page counts as open this long after its last heartbeat (background-tab throttling)
AFTER_CLOSE = 3          # checks with no page open after one said it was closing (about 6 s)
AFTER_SILENCE = 45       # further checks with no page and no closing message (90 s TTL + 90 s ≈ 3 min)
LAUNCHER_ENV = "MB_LAUNCHER_PID"
if os.environ.get("MB_LIFECYCLE_FAST") == "1":     # tests only: the same rules on a shorter clock
    CHECK_S, PAGE_TTL_S, AFTER_SILENCE = 0.5, 3.0, 6

_lock = threading.Lock()
_pages: dict[str, float] = {}          # page id -> time of its last heartbeat
_state = {"closing": False, "started": False, "seen": False}


def _log(text: str) -> None:
    print(f"[{time.strftime('%F %T')}] lifecycle: {text}", file=sys.stderr, flush=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------- what the page reports
# Three signals, strongest first:
#   window  — the page holds an event stream open (/api/window). The socket drops the moment the window goes,
#             even when the browser skips 'pagehide' (the Claude browser pane does), so this alone ends the app
#             within a few seconds.
#   closing — the page's goodbye beacon on 'pagehide' (a second opinion, and the fastest signal on a reload).
#   alive   — a heartbeat every few seconds, for a browser that cannot keep the stream open.
# A page that has closed is forgotten completely: a heartbeat left to go stale would keep the app looking
# occupied, and every check that saw it would clear the "a window just closed" state — six seconds would
# silently become three minutes.
_windows: dict[str, int] = {}          # page id -> open event streams
_gone: set[str] = set()                # pages that have closed; late heartbeats from them are ignored


def _mark_seen():
    _state["closing"] = False
    _state["seen"] = True            # recorded now, not at the next check: a tab closed within two seconds still counts


def alive(page: str) -> None:
    with _lock:
        if page in _gone:
            return
        _pages[page] = time.monotonic()
        _mark_seen()


def closing(page: str) -> None:
    with _lock:
        _pages.pop(page, None)
        _windows.pop(page, None)
        _gone.add(page)
        _state["closing"] = True


def window_open(page: str) -> None:
    with _lock:
        _gone.discard(page)
        _windows[page] = _windows.get(page, 0) + 1
        _mark_seen()


def window_closed(page: str) -> None:
    with _lock:
        n = _windows.get(page, 0) - 1
        if n > 0:
            _windows[page] = n
            return
        _windows.pop(page, None)
        _pages.pop(page, None)
        _gone.add(page)
        _state["closing"] = True


def connected() -> int:
    now = time.monotonic()
    with _lock:
        for p in [p for p, t in _pages.items() if now - t > PAGE_TTL_S]:
            del _pages[p]
        return len(set(_pages) | set(_windows))


def should_quit(n_open: int, seen: bool, empty: int, closing_: bool) -> tuple:
    """One step of the rule. Returns (quit now?, seen, empty checks, closing)."""
    if n_open > 0:
        return False, True, 0, False
    if not seen:                                           # nobody has opened the app yet
        return False, False, 0, closing_
    empty += 1
    return (closing_ and empty >= AFTER_CLOSE) or empty >= AFTER_SILENCE, True, empty, closing_


# ---------------------------------------------------------------- stopping everything
def _descendants(pid: int) -> list[int]:
    """Every process below `pid`, deepest first (so children never outlive a parent killed first)."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    kids: dict[int, list[int]] = {}
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) == 2:
            kids.setdefault(int(parts[1]), []).append(int(parts[0]))
    order: list[int] = []

    def walk(p):
        for c in kids.get(p, []):
            walk(c)
            order.append(c)
    walk(pid)
    return order


def stop_children(grace: float = 3.0) -> None:
    procs = _descendants(os.getpid())
    for p in procs:
        try:
            os.kill(p, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(_alive(p) for p in procs):
        time.sleep(0.1)
    for p in procs:
        if _alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def quit_now(reason: str) -> None:
    _log(f"quitting: {reason}")
    stop_children()
    os._exit(0)          # analysis threads are daemons; the port is released with the process


def _watch() -> None:
    launcher = int(os.environ.get(LAUNCHER_ENV, "0") or 0)
    seen, empty = False, 0
    while True:
        time.sleep(CHECK_S)
        if launcher and not _alive(launcher):
            quit_now("the launcher was closed")
        quit_, seen, empty, closing_ = should_quit(connected(), seen or _state["seen"], empty, _state["closing"])
        _state["closing"] = closing_
        if quit_:
            quit_now("its window was closed" if closing_ else "no window has been open for three minutes")


def start() -> bool:
    """Start watching (once per server process). Returns False when switched off with MB_KEEP_ALIVE=1."""
    if os.environ.get("MB_KEEP_ALIVE") == "1":
        return False
    with _lock:
        if not _state["started"]:
            _state["started"] = True
            threading.Thread(target=_watch, daemon=True, name="lifecycle").start()
    return True
