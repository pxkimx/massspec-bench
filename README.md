# MassSpec Bench

A local Mac app for mass-spectrometry data: open raw files from the major instrument vendors, look inside
them interactively, and run the standard untargeted (metabolomics / lipidomics), proteomics and targeted
quantification workflows — with publication-style figures, a plain-language reading of every plot, a
PDF report, and a built-in Claude assistant. Sister app of RNAseq Bench.

## Start

**Mac app** — unzip, run `xattr -cr` on the app once (see README-FIRST.txt), double-click. The first
launch installs packages into `~/Library/Application Support/MassSpecBench` (2–4 min). It serves
http://localhost:8766, so it runs alongside RNAseq Bench (8765).

**From source**
```bash
./dev.sh                     # uses the app's environment; auto-reloads on edits
python -m server.selftest    # simulated data through every analysis, a few minutes
```
Needs Python 3.10–3.13 from python.org — **not Anaconda** (its OpenMP library breaks pyOpenMS).

## Which files open

| Vendor / format | How | Status |
|---|---|---|
| Thermo `.raw` | ThermoRawFileParser (downloaded on first use) → mzML | reads directly |
| Bruker timsTOF `.d` (TDF) | timsrust + the TDF SQLite | reads directly |
| Agilent `.d` (MassHunter, ChemStation) | rainbow | reads directly |
| Waters `.raw` | rainbow | reads directly |
| mzML, mzXML, mzData, MGF (also `.gz`) | pyOpenMS / pyteomics | reads directly |
| ANDI-MS `.cdf` | netCDF | reads directly |
| Sciex `.wiff`, Shimadzu `.lcd`, Bruker BAF `.d` | — | convert once with ProteoWizard `msconvert` |

Large files are never uploaded: **Choose files on this Mac** reads them where they are, and each file is
parsed once into a cache (`MassSpecBench/cache`) that every later analysis and the explorer reuse.

## Analyses

**Explore** — per-file summary, TIC/BPC overlays, RT × m/z ion maps, MS/MS acquisition (rate, charge
states, precursors), the largest chromatographic peaks with their spectrum, common contaminants and a mass
accuracy estimate from them, SRM/MRM traces. Plus the **interactive explorer**: click the chromatogram for
the spectrum at that time, click a peak for its extracted-ion chromatogram in every file (integrated),
average a time range, charge-state deconvolution for intact proteins, b/y annotation of MS/MS spectra.

**Untargeted LC-MS** — pyOpenMS feature detection, RT alignment, feature linking, gap filling by
re-integration, adduct/isotope grouping, blank and QC-CV filters, QC-RLSC drift correction, PQN
normalisation, PCA, Welch/ANOVA with BH-FDR, volcano, heatmaps, and annotation against a built-in
metabolite list, your own compound list (m/z + RT) or an MS/MS library (cosine) — each labelled with its
MSI confidence level.

**Proteomics (DDA)** — Sage database search with target–decoy FDR (downloaded on first use), ID
quality control, LFQ or TMT quantification, protein roll-up, imputation shown transparently, and a
limma-style moderated t-test between groups.

**Targeted quantification** — full-scan XIC, PRM or SRM/MRM transitions; integration review for every
target in every sample; weighted linear/quadratic calibration with back-calculated accuracy, LOD/LOQ,
QC accuracy and blank carry-over; concentrations for the unknowns.

**Calculators** — formula → monoisotopic mass, adduct m/z and isotope pattern; observed m/z → candidate
neutral masses and library matches; peptide → m/z per charge and b/y fragment ladder.

**Sample table** — every data file gets a row (sample, type = sample/QC/blank/standard, group, injection
order, concentration). Load a CSV/XLSX sheet or edit the table in place; QC/blank/standard types are
inferred from file names, groups never are (a *Groups from names* button does it only when asked).

## Outputs
Per job in `MassSpecBench/jobs/<id>/`: `report.pdf`, `figures/*.png` + vector `.pdf`, and CSVs
(feature tables, statistics, annotations, PSMs/peptides/proteins, calibration and concentrations).

## Layout
```
server/msio.py      Run model, readers, cache     server/explore.py     raw-data overview
server/vendor.py    vendor formats                server/untargeted.py  metabolomics / lipidomics
server/tools.py     helper downloads              server/proteomics.py  Sage search + quant
server/chem.py      masses, isotopes, peptides    server/targeted.py    calibration + quant
server/peaks.py     peak integration              server/app.py         HTTP API
server/report.py    PDF                           server/agent.py       Claude assistant
web/index.html      UI                            server/simulate.py    synthetic data (tests)
```

## Caveats
Annotations from m/z alone are putative (MSI level 3). Everything runs on your machine; only the
assistant talks to Anthropic's API, and only with what you type plus the tool results it requests.

## Download and licence
The ready-to-run Mac app and a short guide: https://paul-h-kim.com/software/massspec-bench
Source: https://github.com/pxkimx/massspec-bench · MIT licence (see `LICENSE`), © 2026 Paul H. Kim.
