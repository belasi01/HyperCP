# Rrs Explorer

`rrs_explorer_app.py` is a local, interactive web tool (Dash/Plotly) for exploring the
L2 Rrs results produced by the pySAS/HyperCP daily pipeline (`download_and_run_hypercp.sh`
+ `extract_l2_qc_tables.py`). It complements the static daily PDF report with an
interactive map, per-cast spectral comparison across sky-correction methods, and
inter-method statistics on any subset of casts you select.

It is designed to run fully offline: no map tiles, no external APIs. Each user (e.g. on
the ship, with direct access to the data, and a collaborator on land working from a
synced copy) runs their own local instance -- there is no shared server.

## Requirements

- `hypercp` conda environment (`dash` and `plotly` are in `environment.yml`; run
  `conda env update --file environment.yml --prune` if they're missing).
- `MyScripts/pipeline_config.env` filled in (same file used by `download_and_run_hypercp.sh`
  and `extract_l2_qc_tables.py` -- `MAIN_DATA_PATH` and `PATH_HCP` in particular).
- At least one date already processed through `extract_l2_qc_tables.py`, i.e.
  `AnalysisComparison/L2_Methods_Quotes_<date>.csv` must exist under
  `<MAIN_DATA_PATH>/pySAS/`. This is produced automatically by the daily pipeline;
  no separate step is needed if you're running from `download_and_run_hypercp.sh`.

## Launching

```
conda activate hypercp
python MyScripts/rrs_explorer_app.py
```

Then open **http://127.0.0.1:8050** in a browser. The terminal running the command must
stay open (or backgrounded) for as long as you want the tool available; closing the
browser tab does not stop the server. To stop it: `Ctrl-C` in the terminal, or
`pkill -f rrs_explorer_app.py`.

The first time a given date is selected, all 7 methods' spectra for that day are read
from the L2 HDF5 files and cached in memory (a few seconds for a full day). Every
interaction after that (selection, wavelength slider, color variable) is instant --
nothing is re-read from disk.

## Layout

### Date and color variable

- **Date(s)**: multi-select, lists every date with an `AnalysisComparison/L2_Methods_Quotes_<date>.csv`.
  Defaults to only the most recent date; add more to combine several days into one view
  (map, spectra, SPLOM). Each day is read/cached independently the first time it's picked,
  then simply concatenated -- cheap even for many days. If selected dates don't share the
  exact same waveband grid (the L1B binning can differ by a band or two day to day), every
  day beyond the first (chronologically) is linearly resampled onto the first day's grid.
  With more than one date selected, the map does not draw a connecting line between casts
  (it would otherwise draw a spurious line from the last cast of one day to the first of
  the next), and cast labels/legends include the date alongside the time.
- **Couleur des points (carte)**: what the map markers (and, by extension, the "Spectres
  comparés" panel below) are colored by:
  - Heure UTC, Vent (m/s), Li/Es (nuage, cloud-flag ratio), Angle zénithal solaire,
    SST/Salinité/Fluorescence TSG (matched to each cast by nearest timestamp, 15 min
    tolerance -- same logic as the daily PDF report's TSG section).
  - **QWIP**: a second dropdown ("Méthode (QWIP)") appears to pick which of the 7
    processing methods' QWIP score to map, since QWIP is per-method rather than a
    single ancillary value. The color scale is fixed to 0-0.15 (not autoscaled) so a
    rare extreme outlier doesn't wash out the color for every other point.
  - **Date**: useful specifically when several dates are selected, to tell casts from
    different days apart at a glance.

### Map (top-left) and per-point spectra (top-right)

- **Click** a point on the map: the panel on the right shows all 7 methods' Rrs spectra
  for that single cast (M99/Z17/3C x NN, plus NIR/SimSpec for M99 and Z17 only -- see
  "7 methods, not 9" below).
  - Line **width/style** encodes QWIP (Dierssen et al. thresholds): thick solid =
    QWIP < 0.05 (valid), thin solid = 0.05-0.1 (caution), dotted = >= 0.1 (likely invalid
    for optically-deep water).
  - Line **opacity** encodes WEI_QA (Wei, Lee & Shang 2016 totScore, 0-1, independent of
    QWIP): more opaque = better agreement with the 23 reference water-type spectral
    shapes, more transparent = worse.
  - Exact QWIP/WEI values are in the legend for each method.
- **Lasso or box select** (icons in the map's toolbar) a group of points instead: this
  drives both the "Spectres comparés" and "Intercomparaison" sections below. The zoom
  icon in the same toolbar switches back out of selection mode. Mouse-wheel zoom also
  works directly. With nothing selected, both sections below default to using all casts
  of the day.

### Spectres comparés entre points sélectionnés (une méthode)

Pick one method (default M99NN) and see the Rrs spectra of every selected cast
overlaid, colored by the same variable currently chosen for the map (with a matching
colorbar). Useful for looking at spatial/temporal variability for a single method,
as opposed to the top-right panel's inter-method comparison at a single point.

### Intercomparaison des méthodes

For the current point selection and a chosen wavelength (slider, snapped to the nearest
actual waveband), shows:
- A scatterplot matrix (SPLOM) of all 7 methods' Rrs against each other at that
  wavelength.
- A sortable table of pairwise statistics (N, bias, RMSD, R²) for every method pair,
  sorted by RMSD descending (worst agreement first) by default -- click a column header
  to re-sort.

## 7 methods, not 9

As of 2026-07-17, NIR/SimSpec offset corrections are not produced for the 3C sky model
(found not to be meaningful for it during exploration) -- only `3CNN` is kept, alongside
the full `M99NN/NIR/SimSpec` and `Z17NN/NIR/SimSpec` set. This applies to
`apply_nir_corrections.py`, `download_and_run_hypercp.sh`, `extract_l2_qc_tables.py`,
and this tool consistently -- dates processed before that change may still have
`3CNIR`/`3CSimSpec` HDF5 files on disk, but they are no longer produced going forward
and are not read by this tool (`rrs_explorer_app.py`'s `METHODS` list only includes the
7 current combinations).
