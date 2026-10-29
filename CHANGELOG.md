# Changelog

## 1.0.3 — Light | Dark theme choice (2026-10-07)
- A Light | Dark switch (sun and moon) at the top of the sidebar, on every page, replaces the single moon button.
  With no choice made the app follows the Mac's appearance, as before. The choice is stored as
  `localStorage["theme"]` = `light`/`dark` and applied in `<head>` before first paint, so the other theme never flashes.
- Charts in the explorer, calculators and integration review look up their colours when they draw, so a theme change
  recolours charts already on screen. Other windows of the app (and the Toolbox Launcher frame) follow a change.
- Light theme: secondary text and warning colours darkened to reach 4.5:1 contrast on every light background.
- Report figures, PNG/PDF downloads and the PDF report are unchanged (print-friendly white).

## 1.0.2 — fixes from the 2026-10-01 test sweep
- status.json and result.json are written to a temporary file and renamed into place, and every reader goes through
  `common.read_json`, so polling a running analysis can no longer hit a half-written file (an occasional HTTP 500).
- Back and Forward move between analyses and the home page (each analysis is a history entry; `popstate`/`hashchange`).
- The four analysis tabs on the home page wrap to 2 × 2 when the upload panel is narrow, so their labels are never cut off.
- The simulated metabolomics example says that no library match is expected for its random masses.

## 1.0.1 — quits completely when closed (2026-09-30)
- Closing the browser tab stops the server about 6 s later (a reload does not); a browser that vanishes without
  closing stops it after about 3 min; quitting or force-quitting the app from the Dock stops it within 2 s.
- Quitting stops everything the server started (Sage search, ThermoRawFileParser, assistant Python), children first.
- The launcher now stays alive while the server runs and ends with it; analyses left "running" by a closed app
  are marked stopped at the next start. `server/lifecycle.py`, `tests/test_lifecycle.py`.
- The page also holds an event stream open (`/api/window`), so a window closed without `pagehide` (the Claude
  browser pane) still quits the app in seconds; a closed page's heartbeat is forgotten at once.
- Port checks in the launcher, dev.sh and the Stop command look only for a listening server
  (`lsof -sTCP:LISTEN`): a browser socket left in CLOSE_WAIT no longer blocks a restart or gets killed.

## 1.0.0 — first release (2026-09-23)
- Reads Thermo .raw (ThermoRawFileParser), Bruker timsTOF .d, Agilent .d, Waters .raw, ANDI .cdf, mzML/mzXML/mzData/MGF.
- Analyses: raw-data overview + interactive explorer, untargeted LC-MS, proteomics (Sage), targeted quantification.
- Mass calculators, sample table with sheet import, PDF report, Claude assistant, Mac app bundle.
