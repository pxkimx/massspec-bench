MassSpec Bench — how to start
================================

1. Unzip, then move "MassSpec Bench.app" anywhere (Applications or Desktop).

2. First open only — macOS Gatekeeper will say the app is "damaged" or "cannot be verified"
   because it is not signed with a paid Apple developer certificate. Fix it once:
      open Terminal, type   xattr -cr    (with a space after it), drag the app into the
      Terminal window so its path is filled in, press Return.
   Then double-click the app. (Alternative: right-click the app -> Open -> Open.)

3. The first launch installs the analysis packages into
   ~/Library/Application Support/MassSpecBench   (2–4 minutes, needs internet).
   Later launches take a few seconds. Your browser opens http://localhost:8766 and the
   app quits by itself when you close its browser tab (a reload does not quit it), and
   quitting its Dock icon stops it too. "Quit MassSpec Bench" in the sidebar and
   "Stop MassSpec Bench.command" still work.
   It uses its own port (8766), so it runs side by side with RNAseq Bench (8765).

4. Thermo .raw files need ThermoRawFileParser and proteomics needs the Sage search engine;
   both are downloaded automatically the first time they are used (~35 MB and ~7 MB).

If the app will not open at all: double-click "Start MassSpec Bench.command" instead —
it runs the same thing inside a Terminal window so you can see every message.
The log is at ~/Library/Application Support/MassSpecBench/massspec-bench.log

Needs Python 3.10–3.13 from python.org (Anaconda's Python does not work with pyOpenMS;
it can stay installed, the app just skips it).
