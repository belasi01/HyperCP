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
| `RUN_WebDAV` | `true`/`false`. Skip the Step 1 data sync entirely, regardless of `SYNC_MODE` (e.g. for a local reprocessing test where the raw data is already on disk). Name predates `SYNC_MODE`; kept for backward compatibility. |
| `RUN_HCP` | `true`/`false`. Skip the HyperCP processing step entirely (e.g. to only test the data sync). |
| `VERBOSE_TERMINAL` | `true`: print everything live to the terminal (use for manual/interactive runs). `false`: redirect all output to `<MAIN_DATA_PATH>pySAS/Automated_Pipeline_Log/pySAS_processing_<date>.log` (use for cron). |
| `CLOBBER` | `true`: `run_pySAS006_processing.py` overwrites existing outputs at every level — needed for a full reprocessing from RAW. `false`: only recomputes files that are missing or previously failed, skips everything already successfully processed. Leave `true` for routine daily cron use (the sync step is already incremental, and Controller-level staleness checks prevent reprocessing untouched files — see `CLAUDE.md`); set to `false` only if you deliberately want to protect existing outputs while patching a subset of dates. |
| `SYNC_MODE` | `webdav` (default, sync from shore over the internet via `lftp`) or `smb` (copy from a locally-mounted SMB share — use while physically on board the Amundsen, on the ship's own network, no internet needed). See "On-board mode (SMB)" below. |
| `WEBDAV_HOST` / `WEBDAV_USER` / `WEBDAV_PASS` | Amundsen Science WebDAV credentials. Only used when `SYNC_MODE=webdav`. |
| `SMB_MOUNT_POINT` / `SMB_TSG_SUBPATH` / `SMB_ATS_SUBPATH` / `SMB_PYSAS_SUBPATH` | Local mount point and per-data-type subpaths for `smb://10.0.0.10/data`. Only used when `SYNC_MODE=smb`. See "On-board mode (SMB)" below. |
| `CRUISE` / `EXPERIMENT` | Used to build ancillary filenames (`<CRUISE>_<EXPERIMENT>_Ancillary_<date>.sb`) and passed through to SeaBASS metadata. |
| `ROLL_OFFSET` | Static degrees added to raw ROLL in `correct_L1A_files.py::process_roll_offset`, applied unconditionally to every date processed. Currently `-5`. **This needs periodic re-validation** — see "IMU roll bias" below. |
| `PITCH_OFFSET` | Same mechanism as `ROLL_OFFSET`, but for PITCH (`correct_L1A_files.py::process_pitch_offset`). Added 2026-07-31 after a second physical-disturbance event (2026-07-30, see below) made it clear a single-axis correction isn't enough. Currently `0` (not yet corrected — wiring only, values/reprocessing to be decided per date). |
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

A single level for a single date (bypasses the data sync, useful for debugging one
step):
```bash
python MyScripts/run_pySAS006_processing.py --date 20260721 --level L1AQC
```

## On-board mode (SMB)

While physically on the Amundsen, data is reachable directly on the ship's network at
`smb://10.0.0.10/data` instead of over WebDAV from shore. Step 1 of
`download_and_run_hypercp.sh` supports this as an alternative to the WebDAV sync,
selected by `SYNC_MODE=smb` in `pipeline_config.env` (default is `webdav`).

1. Mount the share (Finder → Go → Connect to Server → `smb://10.0.0.10/data`, or
   `mount_smbfs //user@10.0.0.10/data /Volumes/data`).
2. Set in `pipeline_config.env`:
   ```
   SYNC_MODE=smb
   SMB_MOUNT_POINT=/Volumes/data          # wherever it actually mounted
   SMB_TSG_SUBPATH=...                    # confirm real subpaths on board
   SMB_ATS_SUBPATH=...
   SMB_PYSAS_SUBPATH=...
   ```
   The `SMB_*_SUBPATH` defaults mirror the WebDAV server's own `TSG_CDOM/`/`ATS/`/`PySAS/`
   layout, but the SMB share's actual structure hasn't been confirmed yet (as of
   2026-08-05) — check it once mounted rather than assuming.
3. Run as usual (manually or via cron) — everything downstream (Step 2 onward,
   including the "no RAW data" guard) is identical regardless of `SYNC_MODE`.

`sync_via_smb()` in the script copies with `cp -n` (never overwrites a file already
present locally), matching the incremental behavior of the WebDAV `lftp mirror -c -n`.
If `SMB_MOUNT_POINT` isn't actually mounted, it logs an error and returns without
copying anything — which then hits the "no RAW data" guard below and exits cleanly
rather than processing garbage.

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
- If no pySAS RAW file matching `*<date>*.raw` exists under `<MAIN_DATA_PATH>pySAS/RAW_NoHeaders/`
  after the WebDAV sync (instrument down, network issue, or just not synced yet), the
  script logs a one-line warning to that date's log and exits cleanly instead of running
  the full L1A->L2->NIR->`extract_l2_qc_tables.py` chain against nothing — that used to
  run to completion on empty data and only fail at the very end, in
  `extract_l2_qc_tables.py`'s pivot step (`KeyError: 'Filename'`), wasting a full cycle
  and leaving a misleading traceback instead of a clear "no data" message.

## Known quirks / open items

- **IMU roll/pitch bias**: `ROLL_OFFSET` (and, as of 2026-07-31, `PITCH_OFFSET`) are
  applied to every date unconditionally, but the real bias is not a single constant —
  it has moved at least four times so far, always in a way consistent with the sensor
  physically getting knocked or re-seated rather than electronic drift (raw PITCH and
  ROLL jump together, by comparable amounts, and are otherwise very stable within a
  period):
  - **2026-07-01 to 07-17**: raw ROLL drifted from about +5° to +8.7°, raw PITCH stayed
    near 0°.
  - **2026-07-18 afternoon to 07-19 ~20:20 UTC**: both ROLL and PITCH jumped to ~14°
    together, then dropped back to near 0° around 2026-07-19 20:20 UTC.
  - **2026-07-19 20:20 UTC to 07-29**: both near 0° (confirmed through 2026-07-20; the
    07-29 GPS failure — see below — doesn't appear to have disturbed the IMU).
  - **2026-07-30**: both jumped again, to a very stable ROLL median ~18.1° and PITCH
    median ~19.8° for the entire day (two mid-day L1A files are even missing the
    SATTHS1500A group entirely), almost certainly from physically handling the mast
    while fixing the GPS that day.
  A single global static offset per axis is therefore only ever an approximation for
  whichever period it was tuned to, and actively wrong for every other period —
  `ROLL_OFFSET=-5` in particular has been *wrong* (introducing a new bias rather than
  removing one) for most of the cruise since 2026-07-19. `PITCH_OFFSET` was added
  2026-07-31 (currently `0`, not yet corrected) so the mechanism exists, but the
  question of per-date-range values — and reprocessing the affected date ranges from
  RAW once decided — is still open.
- **GPS outage (2026-07-29)**: raw GPRMC STATUS degraded starting ~16:31 UTC and went
  fully void (`V`, frozen last-known position) from ~18:31 UTC through the end of that
  day's data. HyperCP's own L1AQC GPS-status filter (`ProcessL1aqc.py`) already flags
  this correctly (logged "Percentage of data failed on GPS Status: 100%"), so this
  shouldn't silently contaminate output, but no `L1AQC` output was produced for
  2026-07-29 at all as of this writing. Resolved "momentarily" per field report on
  2026-07-30 — the pitch/roll disturbance above happened the same day, likely from the
  physical fix.
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
