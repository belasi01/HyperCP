"""
apply_nir_corrections.py

Post-processes L2 NN HDF5 files (M99NN, Z17NN, 3CNN) to produce NIR and SimSpec
corrected versions, with matching SeaBASS outputs and Rrs plots.

Workflow:
  1. Run M99NN, Z17NN, 3CNN with run_pySAS006_processing.py (full L2 pipeline)
  2. Run this script to derive M99NIR, M99SimSpec, Z17NIR, Z17SimSpec, 3CNIR, 3CSimSpec
     without re-running 95% of the L2 computation.

Usage:
    python MyScripts/apply_nir_corrections.py --date 20260701
    python MyScripts/apply_nir_corrections.py --date 20260701 --time 180621
    python MyScripts/apply_nir_corrections.py --date 20260701 --model M99
"""

import os
import sys
import glob
import shutil
import argparse
import json
import re
import collections
import numpy as np
import h5py
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt

if not hasattr(plt.cm, 'get_cmap'):
    plt.cm.get_cmap = mpl.colormaps.get_cmap

# ==============================================================================
# PIPELINE CONFIG
# ==============================================================================

MY_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pipeline_config(path):
    config = {}
    if not os.path.exists(path):
        print(f"❌ Config file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            config[key.strip()] = val.strip()
    return config


env = load_pipeline_config(os.path.join(MY_DIR, 'pipeline_config.env'))
PATH_HCP = env['PATH_HCP']
PATH_DATA = os.path.join(env['MAIN_DATA_PATH'], 'pySAS')

if PATH_HCP not in sys.path:
    sys.path.insert(0, PATH_HCP)

os.environ["HYPERINSPACE_CMD"] = "true"

from Source.HDFRoot import HDFRoot
from Source.ConfigFile import ConfigFile
from Source.MainConfig import MainConfig
from Source.SeaBASSWriter import SeaBASSWriter
from Source.SeaBASSHeader import SeaBASSHeader
from Source.ProcessL2OCproducts import ProcessL2OCproducts
from Source.utils import plotting
import Source.utils.dating as dating


# ==============================================================================
# NIR CORRECTION HELPERS
# ==============================================================================

def _wl_fields(dtype):
    """Return wavelength field names (numeric-only) sorted by float value."""
    return sorted(
        [f for f in dtype.names if re.match(r'^[\d.]+$', f)],
        key=float
    )


def compute_f0_from_nn_file(h5_path):
    """
    Reconstruct F0 {wl_str: float} from existing nLw/Rrs in a NN file.

    In NN files, Rrs and nLw are uncorrected, so F0[λ] = nLw[λ] / Rrs[λ].
    Uses median over valid ensembles to be robust to outliers.
    """
    with h5py.File(h5_path, 'r') as f:
        rrs = f['REFLECTANCE/Rrs_HYPER'][...]
        nlw = f['REFLECTANCE/nLw_HYPER'][...]

    f0 = {}
    for wl in _wl_fields(rrs.dtype):
        r = rrs[wl].astype(float)
        n = nlw[wl].astype(float)
        valid = (~np.isnan(r)) & (~np.isnan(n)) & (np.abs(r) > 1e-10)
        if np.any(valid):
            f0[wl] = float(np.median(n[valid] / r[valid]))
    return f0


# ==============================================================================
# NIR CORRECTION ALGORITHMS (applied in-place on HDF5 file)
# ==============================================================================

def apply_simple_nir(h5_path, f0=None):
    """
    Simple NIR correction — Mueller & Austin (1995).
    Deviates from Source/ProcessL2.py::nirCorrection by design: HyperCP reverts
    to a 0 offset whenever the NIR minimum is negative ("never ADD reflectance").
    In practice, a negative NIR baseline is common under overcast/white-sky
    conditions and is a legitimate "white" residual to remove, not a reason to
    skip the correction -- so here the offset is always subtracted, whatever
    its sign. Rrs and nLw each get their OWN independent, spectrally-flat offset
    (the minimum in 700-800 nm of that same dataset), subtracted uniformly
    across all wavelengths.
    """
    with h5py.File(h5_path, 'r+') as f:
        rrs = f['REFLECTANCE/Rrs_HYPER'][...]
        nlw = f['REFLECTANCE/nLw_HYPER'][...]

        wls = _wl_fields(rrs.dtype)
        nir_wls = [wl for wl in wls if 700 <= float(wl) <= 800]

        for i in range(len(rrs)):
            rrs_nir_vals = [float(rrs[wl][i]) for wl in nir_wls
                            if not np.isnan(rrs[wl][i])]
            if rrs_nir_vals:
                rrs_corr = min(rrs_nir_vals)
                for wl in wls:
                    rrs[wl][i] -= rrs_corr

            nlw_nir_vals = [float(nlw[wl][i]) for wl in nir_wls
                            if not np.isnan(nlw[wl][i])]
            if nlw_nir_vals:
                nlw_corr = min(nlw_nir_vals)
                for wl in wls:
                    nlw[wl][i] -= nlw_corr

        f['REFLECTANCE/Rrs_HYPER'][...] = rrs
        f['REFLECTANCE/nLw_HYPER'][...] = nlw


def apply_simspec_nir(h5_path, f0):
    """
    Similarity Spectrum NIR correction — Ruddick et al. (2005/2006).
    Matches Source/ProcessL2.py::nirCorrection: rho = pi*Rrs is interpolated to
    720/780/870 nm (each from its own ~50 nm window, like the real pipeline),
    F0 is interpolated at the same three points to derive an nLw-equivalent
    offset, and a single flat scalar (rrs_corr / nlw_corr) is subtracted from
    every waveband -- not a per-wavelength F0-scaled offset. Unlike the real
    pipeline, the offset is always subtracted even when negative (see
    apply_simple_nir docstring for the rationale).
    """
    ALPHA1 = 2.35   # expected ρ(720)/ρ(780) ratio
    ALPHA2 = 1.91   # expected ρ(780)/ρ(870) ratio
    THRESH = 0.03   # ρ(720) threshold separating clear from turbid water

    f0_wls = sorted(f0.keys(), key=float)
    f0_x = np.array([float(w) for w in f0_wls])
    f0_y = np.array([f0[w] for w in f0_wls])

    def f0_at(target):
        return float(interp1d(f0_x, f0_y)(target))

    with h5py.File(h5_path, 'r+') as f:
        rrs = f['REFLECTANCE/Rrs_HYPER'][...]
        nlw = f['REFLECTANCE/nLw_HYPER'][...]

        wls = _wl_fields(rrs.dtype)
        win720 = [wl for wl in wls if 700 <= float(wl) <= 750]
        win780 = [wl for wl in wls if 760 <= float(wl) <= 800]
        win870 = [wl for wl in wls if 850 <= float(wl) <= 890]

        def interp_rho(win, i, target):
            x = [float(wl) for wl in win]
            y = [float(rrs[wl][i]) * np.pi for wl in win]
            if len(x) == 1:
                return y[0]
            return float(interp1d(x, y)(target))

        for i in range(len(rrs)):
            rho720 = interp_rho(win720, i, 720)
            rho780 = interp_rho(win780, i, 780)
            F01, F02 = f0_at(720), f0_at(780)

            if win870:
                rho870 = interp_rho(win870, i, 870)
                F03 = f0_at(870)
            else:
                rho870 = None

            # Reverts to primary (720/780) mode when no 870 nm data is available,
            # same fallback as the real pipeline.
            if rho720 < THRESH or rho870 is None:
                eps = (ALPHA1 * rho780 - rho720) / (ALPHA1 - 1.0)
                eps_nlw = (ALPHA1 * rho780 * F02 - rho720 * F01) / (ALPHA1 - 1.0)
            else:
                eps = (ALPHA2 * rho870 - rho780) / (ALPHA2 - 1.0)
                eps_nlw = (ALPHA2 * rho870 * F03 - rho780 * F02) / (ALPHA2 - 1.0)

            rrs_corr = eps / np.pi
            nlw_corr = eps_nlw / np.pi
            # Deviates from Source/ProcessL2.py by design: no revert-to-0 guard for
            # a negative offset (see apply_simple_nir docstring) -- always subtract.

            for wl in wls:
                rrs[wl][i] -= rrs_corr
                nlw[wl][i] -= nlw_corr

        f['REFLECTANCE/Rrs_HYPER'][...] = rrs
        f['REFLECTANCE/nLw_HYPER'][...] = nlw


CORRECTION_FUNCS = {
    'NIR': apply_simple_nir,
    'SimSpec': apply_simspec_nir,
}


# ==============================================================================
# OUTPUT GENERATION (SeaBASS + plots via HyperCP APIs)
# ==============================================================================

def _read_main_version():
    """Read Main.py's VERSION constant without importing Main (avoids pulling in PyQt5)."""
    with open(os.path.join(PATH_HCP, 'Main.py')) as f:
        for line in f:
            if line.startswith('VERSION'):
                return line.split('=')[1].strip(" \n'\"")
    return 'unknown'


def init_hypercp_output(cfg_path, out_dir):
    """Load ConfigFile/MainConfig/SeaBASSHeader (products/settings) and point MainConfig
    output to the version directory -- mirrors what Main.Command.__init__ does for a normal run."""
    MainConfig.createDefaultConfig("cmdline_main.config", _read_main_version())
    ConfigFile.loadConfig(cfg_path)
    SeaBASSHeader.loadSeaBASSHeader(ConfigFile.settings["seaBASSHeaderFileName"])
    MainConfig.settings['outDir'] = out_dir


def generate_seabass(hdf_path):
    try:
        SeaBASSWriter.outputTXT_Type2(hdf_path)
        print(f"    ✅ SeaBASS written")
    except Exception as e:
        print(f"    ⚠️  SeaBASS failed: {e}")


def generate_plots(hdf_path):
    filename = os.path.basename(hdf_path)
    try:
        root = HDFRoot.readHDF5(hdf_path)
        # plotRadiometry annotates Rrs plots with the QWIP/WEI_QA scores, read via
        # `.columns` -- populate them (readHDF5 only fills raw `.data`).
        derived = root.getGroup("DERIVED_PRODUCTS")
        if derived:
            for ds in derived.datasets.values():
                ds.datasetToColumns()
        plotting.plotRadiometry(root, filename, rType='Rrs', plotDelta=False)
        print(f"    ✅ Rrs plot written")
        # Generate nLw plot as well (same cost, useful for QC)
        try:
            plotting.plotRadiometry(root, filename, rType='nLw', plotDelta=False)
            print(f"    ✅ nLw plot written")
        except Exception:
            pass
    except Exception as e:
        print(f"    ⚠️  Plot generation failed: {e}")


# ==============================================================================
# CORE FILE PROCESSING
# ==============================================================================

def process_one_file(nn_path, out_dir, correction, cfg_path, clobber):
    """Copy a single NN HDF5 file, apply correction, recompute derived products, generate outputs."""
    filename = os.path.basename(nn_path)
    # HyperCP's own convention (and extract_l2_qc_tables.py) expects L2 HDF5
    # files under <version>/L2/, with Plots/SeaBASS resolved relative to that.
    out_path = os.path.join(out_dir, "L2", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and not clobber:
        print(f"    ⏩ Already exists, skipping: {filename}")
        return

    # 1. Copy NN → output dir (preserve all upstream data intact)
    shutil.copy2(nn_path, out_path)

    # 2. Load HyperCP config (products/settings) -- needed both to recompute
    #    derived products below and to generate SeaBASS/plots afterward.
    init_hypercp_output(cfg_path, out_dir)

    # 3. Reconstruct F0 from the original NN file before any modification
    #    (only strictly required for SimSpec's nLw offset).
    f0 = compute_f0_from_nn_file(nn_path)
    if not f0 and correction == 'SimSpec':
        print(f"    ❌ Could not reconstruct F0 from {filename}, skipping.")
        os.remove(out_path)
        return

    # 4. Apply NIR correction in-place on the copied file
    CORRECTION_FUNCS[correction](out_path, f0)
    print(f"    ✅ {correction} correction applied")

    # 5. Recompute derived products (QWIP, WEI_QA, chlor_a, etc.) from the
    #    corrected Rrs. These were copied stale from the NN file in step 1 and
    #    are NOT automatically consistent with the NIR/SimSpec-corrected Rrs
    #    (Source/ProcessL2.py runs procProds() right after nirCorrection()).
    #    NOTE: WEI_QA also depends on convolved satellite-band Rrs (Rrs_MODISA),
    #    which this script does not correct, so WEI_QA will remain approximate;
    #    QWIP depends only on Rrs_HYPER and will be fully correct.
    root = HDFRoot.readHDF5(out_path)
    reflectance = root.getGroup("REFLECTANCE")
    # HDFRoot.readHDF5 only populates raw `.data`; procProds() reads via `.columns`
    # (normally populated by datasetToColumns() during the live L2 run) and expects
    # a synthesized 'Datetime' column that HyperCP builds in-memory and never
    # persists to HDF5 -- neither survives a plain read-from-disk round trip.
    for ds in reflectance.datasets.values():
        ds.datasetToColumns()
    rrs_hyper = reflectance.getDataset("Rrs_HYPER")
    date_time = [
        dating.timeTag2ToDateTime(dating.dateTagToDateTime(dt), tt)
        for dt, tt in zip(rrs_hyper.columns["Datetag"], rrs_hyper.columns["Timetag2"])
    ]
    # procProds() assumes column order [Datetime, Datetag, Timetag2, <wavelengths>]
    # (it slices columns.keys()[3:] to get the wavebands) -- Datetime must be first.
    reordered = collections.OrderedDict()
    reordered["Datetime"] = date_time
    reordered.update(rrs_hyper.columns)
    rrs_hyper.columns = reordered

    existing_derived = root.getGroup("DERIVED_PRODUCTS")
    if existing_derived:
        root.removeGroup(existing_derived)  # HDFRoot.removeGroup expects the group object, not a name
    if sum(ConfigFile.products.values()) > 0:
        ProcessL2OCproducts.procProds(root)

    # Mirror Source/ProcessL2.py's final step: Datetime is a scratch column (Python
    # datetime objects) used to populate DERIVED_PRODUCTS -- h5py can't serialize it,
    # so the real pipeline pops it from every dataset and rebuilds `.data` before
    # writing. procProds() leaves it in place, so we must strip it too.
    derived = root.getGroup("DERIVED_PRODUCTS")
    for gp in [reflectance, derived]:
        if gp is None:
            continue
        for ds in gp.datasets.values():
            if "Datetime" in ds.columns:
                ds.columns.pop("Datetime")
            ds.columnsToDataset()

    root.writeHDF5(out_path)

    # 6. SeaBASS + plots
    generate_seabass(out_path)
    generate_plots(out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Apply NIR/SimSpec corrections to L2 NN files (post-processing)."
    )
    parser.add_argument('--date', required=True,
                        help="Date YYYYMMDD (e.g. 20260701)")
    parser.add_argument('--time', default=None,
                        help="Time HHMMSS — process a single file (e.g. 180621)")
    parser.add_argument('--model', default='ALL',
                        choices=['M99', 'Z17', '3C', 'ALL'],
                        help="Rho-sky model to process (default: ALL → M99 + Z17 + 3C)")
    parser.add_argument('--no-clobber', dest='clobber', action='store_false',
                        help="Skip files that already exist in output dir")
    parser.set_defaults(clobber=True)
    args = parser.parse_args()

    models = ['M99', 'Z17', '3C'] if args.model == 'ALL' else [args.model]
    date_pat = f"*{args.date}_{args.time}*" if args.time else f"*{args.date}*"
    cfg_path = os.path.join(PATH_HCP, 'Config', env['CFG_FILE_NAME'])

    print(f"\n⚡ NIR Post-Processing")
    print(f"   Date   : {args.date}" + (f" | Time: {args.time}" if args.time else ""))
    print(f"   Models : {', '.join(models)}")
    print(f"   Config : {cfg_path}")

    for model in models:
        nn_dir = os.path.join(PATH_DATA, f"{model}NN", "L2")
        if not os.path.isdir(nn_dir):
            print(f"\n⚠️  {model}NN/ not found — run the NN processing first.")
            continue

        nn_files = sorted(glob.glob(os.path.join(nn_dir, f"{date_pat}*_L2.hdf")))
        if not nn_files:
            print(f"\n⚠️  No L2 HDF5 found in {nn_dir} for pattern {date_pat}")
            continue

        print(f"\n▶️  {model} — {len(nn_files)} file(s) in {model}NN/")

        for correction in ['NIR', 'SimSpec']:
            version = f"{model}{correction}"
            out_dir = os.path.join(PATH_DATA, version)
            os.makedirs(out_dir, exist_ok=True)

            # Save version config snapshot (metadata only, not used by Command)
            with open(cfg_path) as fc:
                cfg_snap = json.load(fc)
            cfg_snap['bL2PerformNIRCorrection'] = 1
            cfg_snap['bL2SimpleNIRCorrection'] = 1 if correction == 'NIR' else 0
            cfg_snap['bL2SimSpecNIRCorrection'] = 1 if correction == 'SimSpec' else 0
            with open(os.path.join(out_dir, f"config_run_{version}.cfg"), 'w') as fc:
                json.dump(cfg_snap, fc, indent=4)

            print(f"\n  [{version}] Processing {len(nn_files)} file(s)...")
            for nn_path in nn_files:
                print(f"  → {os.path.basename(nn_path)}")
                process_one_file(nn_path, out_dir, correction, cfg_path, args.clobber)

            print(f"  ✅ {version} complete.")

    print(f"\n✨ Done!")


if __name__ == '__main__':
    os.chdir(PATH_HCP)
    main()
