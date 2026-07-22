# Daily pySAS/HyperCP pipeline (Amundsen CASCADE 2026)

`download_and_run_hypercp.sh` is the end-to-end automation for Simon's operational
processing of the pySAS HyperSAS system on the Amundsen (CASCADE 2026 cruise): it syncs
raw data from the ship over WebDAV, runs it through HyperCP L1A -> L1AQC -> L1B -> L1BQC
-> L2 for every requested sky-glint correction model, and produces the daily QC/ranking
report. It's designed to run once a day, unattended, via cron.

For the general HyperCP processing-level architecture (what L1A/L1AQC/.../L2 actually
do), see the root `CLAUDE.md`. This README only covers the automation layer in
`MyScripts/` on top of that.

For the interactive tool used to browse the L2 results this pipeline produces, see
[`README_rrs_explorer.md`](README_rrs_explorer.md) (`rrs_explorer_app.py`).

## Setup (once per machine)

1. Create the base conda env from the repo root:
   ```bash
   conda env create -f environment.yml
   conda activate hypercp
   ```
2. Add the MyScripts-specific extra (`cartopy`, used by `extract_l2_qc_tables.py` for the
   track-map coastlines):
   ```bash
   conda env update -n hypercp -f MyScripts/environment.yml
   ```
   **Never add `--prune` to that command.** `MyScripts/environment.yml` intentionally
   lists only the MyScripts-specific addition, not the full environment — `--prune`
   removes anything installed in the target env that isn't listed in the file you point
   it at, so it will silently uninstall unrelated packages (this has happened: it once
   removed `h5py`). Without `--prune`, `conda env update` is purely additive and safe to
   re-run any time.
3. Copy the config template and fill in your values:
   ```bash
   cp MyScripts/pipeline_config.env.template MyScripts/pipeline_config.env
   ```
   `pipeline_config.env` is git-ignored (machine-specific paths + WebDAV credentials —
   never commit it). See the reference table below for every key.

## `pipeline_config.env` reference

| Key | Meaning |
|---|---|
| `RUN_WebDAV` | `true`/`false`. Skip the WebDAV sync step entirely (e.g. for a local reprocessing test where the raw data is already on disk). |
| `RUN_HCP` | `true`/`false`. Skip the HyperCP processing step entirely (e.g. to only test the WebDAV sync). |
| `VERBOSE_TERMINAL` | `true`: print everything live to the terminal (use for manual/interactive runs). `false`: redirect all output to `<MAIN_DATA_PATH>pySAS/Automated_Pipeline_Log/pySAS_processing_<date>.log` (use for cron). |
| `CLOBBER` | `true`: `run_pySAS006_processing.py` overwrites existing outputs at every level — needed for a full reprocessing from RAW. `false`: only recomputes files that are missing or previously failed, skips everything already successfully processed. Leave `true` for routine daily cron use (the WebDAV sync is already incremental, and Controller-level staleness checks prevent reprocessing untouched files — see `CLAUDE.md`); set to `false` only if you deliberately want to protect existing outputs while patching a subset of dates. |
| `WEBDAV_HOST` / `WEBDAV_USER` / `WEBDAV_PASS` | Amundsen Science WebDAV credentials. |
| `CRUISE` / `EXPERIMENT` | Used to build ancillary filenames (`<CRUISE>_<EXPERIMENT>_Ancillary_<date>.sb`) and passed through to SeaBASS metadata. |
| `ROLL_OFFSET` | Static degrees subtracted from raw ROLL in `correct_L1A_files.py::process_roll_offset`, applied unconditionally to every date processed. Currently `-5`. **This needs periodic re-validation** — see "IMU roll bias" below. |
| `PATH_HCP` | Absolute path to the HyperCP repo root (trailing slash). |
| `MAIN_DATA_PATH` | Absolute path to the data root; `TSG/`, `ATS/`, and `pySAS/` subfolders live under it (auto-created if missing). |
| `LFTP_BIN` | Path to the `lftp` binary used for the WebDAV mirror. |
| `CONDA_SH_PATH` | Path to `conda.sh` (e.g. `~/miniconda3/etc/profile.d/conda.sh`). Only used by `download_and_run_hypercp.sh`, sourced before `conda activate hypercp`. **Required for cron** — cron runs with a minimal environment and does not load `.bashrc`/`.zshrc`, so without this the script can't find `conda` or `python` at all (verified: a clean cron-like shell has neither on `PATH`). If this key is missing or wrong, the job fails silently with no processing done and no obvious error in the daily log (the failure happens before the script's own log redirection is set up). |
| `CFG_FILE_NAME` / `HDR_FILE_NAME` | The `.cfg`/`.hdr` pair under `Config/` that defines calibration files and processing parameters for this leg. |
| `SKY_MODEL` | Which sky-glint correction model(s) to run at L2: `ALL` (M99+Z17+3C), a comma-separated subset (`M99,Z17`), or a single model. Each `<model>NN` runs the full L2 pipeline; `apply_nir_corrections.py` then derives the NIR/SimSpec variants cheaply (patches Rrs/nLw in place, no rerun) — except for `3C`, see "7 methods, not 9" in `README_rrs_explorer.md`. |

## Running manually

Full day, all steps (same as cron would do):
```bash
conda activate hypercp
bash MyScripts/download_and_run_hypercp.sh 20260721   # date argument optional; defaults to today
```

A single level for a single date (bypasses the WebDAV sync, useful for debugging one
step):
```bash
python MyScripts/run_pySAS006_processing.py --date 20260721 --level L1AQC
```

## Cron (daily unattended run)

```cron
30 23 * * * /bin/bash /path/to/HyperCP/MyScripts/download_and_run_hypercp.sh >> /path/to/HyperCP/MyScripts/cron_stderr.log 2>&1
```

- No date argument -> the script computes `DATE_TARGET` itself. As currently written
  this is **today's date** (`date +%Y%m%d`), not yesterday's, despite an old comment in
  the script suggesting otherwise — running at 23:30 processes the day that's about to
  end, which is the intended behavior.
- The `>> cron_stderr.log` redirect is a safety net independent of the script's own
  `VERBOSE_TERMINAL=false` logging: it catches failures that happen *before* that
  internal redirection is set up (missing `pipeline_config.env`, bad `CONDA_SH_PATH`,
  etc.), which would otherwise be silently lost.
- Confirm it actually ran: check `<MAIN_DATA_PATH>pySAS/Automated_Pipeline_Log/pySAS_processing_<date>.log`
  for the full processing trace, and `cron_stderr.log` for anything that failed earlier.

## Known quirks / open items

- **IMU roll bias**: `ROLL_OFFSET=-5` is applied to every date unconditionally. Raw ROLL
  actually drifted from about +5° (July 1) to +8.7° (July 17-18) rather than staying
  fixed, then jumped to ~14° along with PITCH on 2026-07-18 afternoon (consistent with a
  physical knock/reorientation of the sensor, not electronic drift), before both dropped
  back to near 0° around 2026-07-19 20:20 UTC and have stayed there since (confirmed
  through 2026-07-20, the last full day checked). A single static `-5°` offset is
  therefore already an approximation pre-2026-07-19, and is likely *wrong* (would
  introduce a new -5° bias) for dates from 2026-07-19 20:20 UTC onward. Worth revisiting
  before reprocessing that range from RAW — either date-conditional offsets or dropping
  the correction entirely for recent dates.
- **Pre-CASCADE-leg dates**: dates before the `_Leg1-3` calibration convention (e.g.
  2026-07-05) may need the older `pySAS_Amundsen_2026.cfg` config and `AMD_LEG_00`
  ancillary naming instead of the current `CFG_FILE_NAME`/`HDR_FILE_NAME` /
  `CRUISE_EXPERIMENT` pair. If backfilling an early date fails or looks wrong, check this
  before assuming a bug.
- **Design decisions baked into `apply_nir_corrections.py` / the active `.cfg`** (not
  arbitrary — don't re-litigate without reason):
  - NIR/SimSpec correction always subtracts the offset regardless of sign, deviating
    from upstream HyperCP's revert-to-0-if-negative guard — based on Simon's field
    experience with low-wind/cloud conditions producing genuinely negative Rrs.
  - `bL2NegativeSpec=0` in the active `.cfg` — negative Rrs spectra are kept rather than
    QC-filtered out.
- **`MyScripts/environment.yml`**: only ever apply with plain `conda env update` (no
  `--prune`) — see Setup above.
