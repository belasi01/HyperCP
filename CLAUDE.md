# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HyperCP (HyperInSPACE Community Processor) is a PyQt5 desktop application, developed for NASA/EUMETSAT,
that processes above-water hyperspectral/multispectral ocean color radiometry (Sea-Bird HyperSAS w/ or
w/o SolarTracker/pySAS, TriOS RAMSES, IMO DALEC, So-Rad) from raw instrument binary through to
water-leaving reflectance products, with uncertainty propagation and SeaBASS/OCDB/HDF5 output.

## Environment setup

Uses conda, not pip/venv, as the primary dependency manager (see `environment.yml` for pinned versions,
e.g. Python 3.11, PyQt 5.15, numpy 1.x, pandas 1.5):

```
conda env create -f environment.yml
conda activate hypercp
```

If the environment breaks after a pull, update rather than recreate: `conda env update --file environment.yml --prune`.

## Running the application

```
python Main.py
```

This launches the PyQt GUI. On first launch it auto-creates `Config/`, `Logs/`, `Plots/` directories and
downloads large auxiliary databases into `Data/` (Zhang et al. 2017 glint-correction LUT/db, several hundred
MB–GB) — expect this on a clean checkout.

For headless/scripted runs (no GUI), call the `Command` class in `Main.py` directly, as demonstrated in
`run_Sample_Data.py`: copy that script, point it at your own input/output directories and a `.cfg` file, and
it can process multiple levels across multiple files using multiple core threads. This is the recommended
way to batch-process or to drive HyperCP from other tooling. The `-cmd` single-level CLI flag on `Main.py`
is currently obsolete.

## Tests

There is one test module, run with `unittest` (this is exactly what CI runs, see
`.github/workflows/ApplicationTesting.yml`):

```
python -m unittest Tests.test_sample_data
```

CI runs this across ubuntu/macos/windows and requires ECMWF (`~/.ecmwf_ads_credentials.json`) and GMAO/MERRA2
(`~/.netrc`) credentials for ancillary-data retrieval — tests that hit `GetAnc*.py` will fail without these
locally.

## Building a distributable bundle

`python make.py` invokes PyInstaller with platform-specific options (icon, splash screen, hidden imports) to
produce a standalone app bundle under `Bundled/`. This mirrors `.github/workflows/Build.yml`; you generally
only need this when working on packaging/distribution, not day-to-day feature work.

## Architecture: processing levels

The core mental model is a linear pipeline of processing levels, mirroring satellite ocean-color processing
conventions, each level reading the HDF5 output of the previous one and writing new HDF5 (raw binary for L0):

`L0 (raw binary) → L1A → L1AQC → L1B → L1BQC → L2`

- **L1A**: raw binary → HDF5. Instrument-specific readers (`ProcessL1aSeaBird.py`, `ProcessL1aTriOS.py`,
  `ProcessL1aDALEC.py`, `ProcessL1aSoRad.py`) parse raw frames using calibration/telemetry files
  (`CalibrationFileReader.py`, `RawSeaBirdReader.py`).
- **L1AQC**: filters by vessel attitude (pitch/roll/yaw) and viewing/solar geometry (`ProcessL1aqc.py`,
  deglitching in `ProcessL1aqc_deglitch.py`).
- **L1B**: dark current correction, application of calibrations (factory or FRM — see
  `ProcessL1b_FactoryCal.py` / `ProcessL1b_FRMCal.py`), interpolation (`ProcessL1b_Interp.py`), and timestamp/
  waveband matching across all radiometers in the suite (`ProcessL1b.py`, plus per-instrument variants
  `ProcessL1bDALEC.py`, `ProcessL1bTriOS.py`).
- **L1BQC**: further QC filtering before ensemble binning (`ProcessL1bqc.py`).
- **L2**: time-ensemble averaging, Rrs calculation, sky/sunglint correction (`RhoCorrections.py`,
  `ocbrdf/` BRDF models, `ZhangRho.py`), NIR similarity spectrum QC, and derived ocean color products
  (`L2chlor_a.py`, `L2kd490.py`, `L2par.py`, `L2poc.py`, `L2pic.py`, `L2qaa.py`, `L2avw.py`, `L2gocad.py`,
  `L2ipar.py`, `L2qwip.py`, `L2wei_QA.py`), driven from `ProcessL2.py` / `ProcessL2OCproducts.py` /
  `ProcessL2BRDF.py`.

`Source/Controller.py` (`Controller` class) is the orchestrator: `processL1a`, `processL1aqc`, `processL2`,
etc. are the entry points that dispatch to the level/instrument-specific modules above, and
`processSingleLevel`/multi-level batch logic lives here too. Multi-level batch runs will only carry a file
forward if the previous level's output for that file was created within the last minute — a stale existing
file at a level is treated as "discarded," and processing for that file stops.

## Architecture: data model and config

- **HDF5 in-memory data model**: `HDFRoot.py` / `HDFGroup.py` / `HDFDataset.py` wrap h5py structures and are
  passed between processing stages as the primary in-memory representation (referred to as `root`/`node` in
  the Controller code) before being written to disk at each level.
- **Configuration**: `ConfigFile.py` manages the `.cfg`/`.hdr` file pair that defines which calibration files
  are enabled and all per-level processing parameters; `ConfigWindow.py` is its GUI. Calibration/telemetry
  files (`.cal`, `.tdf`, `.sip`) live under `Config/<ConfigName>_Calibration/` and are managed via
  `CalibrationFile.py`/`CalibrationFileReader.py`. There is no universal default configuration — parameters
  and enabled calibration files are inherently cruise/instrument-package specific.
- **Uncertainty propagation**: `Source/PIU/` (per-instrument-uncertainty) implements FRM-style uncertainty
  budgets (`Uncertainty_Analysis.py`, `Breakdown_FRM.py`, `Breakdown_CB.py`, per-instrument classes
  `HyperOCR.py`/`TriOS.py`/`DALEC.py`) using `punpy`/`comet_maths`; `Source/matheo/` provides spectral band
  integration/convolution to satellite wavebands (RSR weighting).
- **Ancillary/model data**: `AncillaryReader.py` and `GetAnc*.py` pull environmental ancillary data (wind,
  AOD, etc.) from SeaBASS-format files or fall back to GMAO/ECMWF reanalysis models when field logs are
  incomplete; required for the Zhang et al. (2017) glint correction.
- **Output**: `SeaBASSWriter.py`/`SeaBASSHeader.py` produce SeaBASS/OCDB-format text files at L2;
  `PDFreport.py` aggregates logs and plots from all levels into a processing report.

## Conventions worth knowing

- Relative azimuth (`relAz`) convention: the angle between sensor viewing direction and solar azimuth, both
  measured sensor-to-target/sensor-to-sun. Some modules (Zhang et al. 2017 glint correction, Morel 2002 BRDF
  correction) internally use `180 - relAz` instead — check which convention a given module expects before
  wiring in new geometry-dependent code.
- Input/output directories are organized by processing-level subdirectory under a parent directory (auto
  created: `L1A`, `L1AQC`, `L1B`, `L1BQC`, `L2`), not by arbitrary user-chosen paths per level.
