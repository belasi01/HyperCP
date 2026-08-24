"""
apply_nir_corrections_greenedge.py

GreenEdge 2016 HyperSAS equivalent of apply_nir_corrections.py (pySAS/Amundsen):
derives NIR and SimSpec corrected L2 from the M99NN/Z17NN/3CNN files produced by
run_greenedge_processing.py, without re-running the L2 computation.

Reuses every correction/derived-products/SeaBASS/plot function from
apply_nir_corrections.py as-is (compute_f0_from_nn_file, apply_simple_nir,
apply_simspec_nir, process_one_file -- all already take explicit paths, none of
that logic is pySAS-specific) -- only the per-station file discovery differs from
pySAS's flat date/time layout, so that's the only part reimplemented here.

Workflow:
  1. run_greenedge_processing.py --station <name> --level L2 --version ALL
  2. python apply_nir_corrections_greenedge.py --station <name>

Usage:
    conda activate hypercp
    python apply_nir_corrections_greenedge.py --station 20160609_StationG100
    python apply_nir_corrections_greenedge.py --all
    python apply_nir_corrections_greenedge.py --station 20160609_StationG100 --model M99
"""
import os
import sys
import glob
import argparse

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

# Reuses apply_nir_corrections.py's correction math and HyperCP-output plumbing
# unchanged -- see module docstring. Its own load_pipeline_config()/PATH_HCP/
# PATH_DATA execute at import time (pointed at pySAS's pipeline_config.env) but
# are never referenced below; every function used here takes explicit paths.
import apply_nir_corrections as nir_core  # noqa: E402


def load_pipeline_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split("=", 1)
            config[key.strip()] = val.strip()
    return config


env = load_pipeline_config(os.path.join(MY_DIR, "pipeline_config_greenedge.env"))
PATH_HCP = env["PATH_HCP"]
GREENEDGE_ROOT = env["GREENEDGE_ROOT"]
CFG_FILE_NAME = env["CFG_FILE_NAME"]


def discover_stations():
    stations = []
    for raw_fp in sorted(glob.glob(os.path.join(GREENEDGE_ROOT, "*", "HyperSAS", "*.raw"))):
        station_dir = os.path.dirname(os.path.dirname(raw_fp))
        stations.append(os.path.basename(station_dir))
    return stations


def process_station(station, models, clobber):
    station_dir = os.path.join(GREENEDGE_ROOT, station, "HyperSAS")
    cfg_path = os.path.join(PATH_HCP, "Config", CFG_FILE_NAME)

    for model in models:
        if model in nir_core.SKIP_NIR_CORRECTIONS_FOR:
            print(f"  ⏩ {model}: NIR/SimSpec skipped (not relevant for this model).")
            continue

        nn_dir = os.path.join(station_dir, f"{model}NN", "L2")
        nn_files = sorted(glob.glob(os.path.join(nn_dir, "*_L2.hdf")))
        if not nn_files:
            print(f"  ⚠️  [{station}] No L2 HDF5 in {nn_dir} -- run run_greenedge_processing.py "
                 f"--level L2 --version {model}NN first.")
            continue

        for correction in ["NIR", "SimSpec"]:
            version = f"{model}{correction}"
            out_dir = os.path.join(station_dir, version)
            os.makedirs(out_dir, exist_ok=True)
            print(f"  [{version}] {len(nn_files)} file(s)...")
            for nn_path in nn_files:
                try:
                    nir_core.process_one_file(nn_path, out_dir, correction, cfg_path, clobber)
                except Exception as e:
                    print(f"    ❌ Failed on {os.path.basename(nn_path)}: {e}")
            print(f"  ✅ {version} complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--station", type=str, default=None, help="Station folder name (e.g. 20160609_StationG100)")
    parser.add_argument("--all", action="store_true", help="Process every station discovered under GREENEDGE_ROOT")
    parser.add_argument("--model", default="ALL", choices=["M99", "Z17", "3C", "ALL"])
    parser.add_argument("--no-clobber", dest="clobber", action="store_false",
                        help="Skip files that already exist in the output dir")
    parser.set_defaults(clobber=True)
    args = parser.parse_args()

    if not args.station and not args.all:
        parser.error("Specify --station <name> or --all")

    models = ["M99", "Z17", "3C"] if args.model == "ALL" else [args.model]
    stations = discover_stations() if args.all else [args.station]

    os.chdir(PATH_HCP)
    for station in stations:
        print(f"\n=== {station} ===")
        process_station(station, models, args.clobber)

    print("\n✨ Done!")
