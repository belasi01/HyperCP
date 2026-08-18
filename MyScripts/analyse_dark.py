"""
analyse_dark.py - Analyse un fichier RAW pySAS/HyperSAS acquis capuchons sur un
sous-ensemble de capteurs (typiquement Lt et Li, pas Ed) pour évaluer le niveau de
"noir" (dark current) tel qu'il apparaît dans le canal normal d'acquisition -- les
trames ShutterLight, les mêmes trames utilisées pour produire les Rrs en cast normal --
et le comparer à la référence dark interne de l'instrument (trames ShutterDark,
obturateur électronique périodique du capteur, ~1 trame sur 5) qu'HyperCP utilise pour
la correction dark opérationnelle (voir Source/ProcessL1b.py::processDarkCorrection,
qui interpole les trames ShutterDark dans le temps pour corriger les trames
ShutterLight).

Contexte : anomalies spectrales observées dans les Rrs à faible Ed -- hypothèse que la
correction dark "opérationnelle" (basée sur les trames ShutterDark périodiques) ne
capture pas fidèlement le niveau/la forme spectrale du dark réellement présent dans le
canal ShutterLight. Ce script traite le RAW jusqu'au niveau L1A (en mémoire, sans
écrire de HDF5) en contournant le filtre SZA (qui rejetterait normalement ce fichier --
acquis de nuit/capuchons, donc "sans lumière valide"), puis compare pour chaque capteur
capuchonné :
  (a) ShutterLight  -- le canal normal, tel qu'il serait vu "comme si c'était de la
      lumière" -- c'est la mesure demandée par Simon.
  (b) ShutterDark    -- la référence dark interne périodique de l'instrument.
  (c) DARK_AVE/DARK_SAMP -- l'estimé dark par pixels noirs embarqué dans chaque trame
      ShutterLight (scalaire, pas spectral).

Usage CLI :
    conda activate hypercp
    python MyScripts/analyse_dark.py /path/to/pySAS006_20260815_232312.raw --sensors LT,LI
    python MyScripts/analyse_dark.py <raw_path> --sensors LT,LI --out-dir /tmp/dark_test

Usage programmatique :
    from analyse_dark import analyse_dark
    analyse_dark("/path/to/pySAS006_20260815_232312.raw", capped_sensors=["LT", "LI"])
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)


def load_pipeline_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()
    return config


ENV = load_pipeline_config(os.path.join(MY_DIR, "pipeline_config.env"))
PATH_HCP = ENV["PATH_HCP"]
MAIN_DATA_PATH = ENV["MAIN_DATA_PATH"]
CFG_FILE_NAME = ENV["CFG_FILE_NAME"]

if PATH_HCP not in sys.path:
    sys.path.insert(0, PATH_HCP)
os.chdir(PATH_HCP)
os.environ["HYPERINSPACE_CMD"] = "true"

from Source.ConfigFile import ConfigFile  # noqa: E402
from Source.Controller import Controller  # noqa: E402
from Source.ProcessL1aSeaBird import ProcessL1aSeaBird  # noqa: E402

VALID_SENSORS = ("ES", "LI", "LT")


def _load_l1a_root_bypassing_sza(raw_path):
    """Traite le RAW jusqu'au niveau L1A en mémoire (aucune écriture disque), en
    désactivant le filtre bL1aCleanSZA qui rejetterait un fichier acquis sans
    lumière valide (nuit / capuchons)."""
    ConfigFile.loadConfig(CFG_FILE_NAME)
    ConfigFile.settings["bL1aCleanSZA"] = 0

    calFiles = ConfigFile.settings["CalibrationFiles"]
    calibrationMap = Controller.processCalibrationConfig(ConfigFile.filename, calFiles)

    root = ProcessL1aSeaBird.processL1a(raw_path, calibrationMap)
    if root is None:
        raise RuntimeError(
            f"Le traitement L1A de {raw_path} a échoué même en contournant le filtre "
            f"SZA -- voir Logs/{os.path.splitext(os.path.basename(raw_path))[0]}_L1A.log"
        )
    return root


def _find_group(root, sensor, frame_type):
    """Retourne le premier groupe dont le FrameType correspond (ShutterLight ou
    ShutterDark) et qui porte un dataset pour ce capteur (ES/LI/LT)."""
    for gp in root.groups:
        if gp.attributes.get("FrameType") == frame_type and gp.getDataset(sensor) is not None:
            return gp
    return None


def _spectrum_stats(ds):
    """Extrait (wavelengths, mean, std, n) d'un HDFDataset spectral (colonnes = nm)."""
    names = [n for n in ds.data.dtype.names if n not in ("Datetag", "Timetag2")]
    wl = np.array([float(n) for n in names])
    order = np.argsort(wl)
    wl = wl[order]
    arr = np.vstack([ds.data[names[i]] for i in order]).T.astype(np.float64)  # (N, nWL)
    return wl, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), arr.shape[0]


def _scalar_stats(gp, sensor, name):
    ds = gp.getDataset(name)
    if ds is None or sensor not in ds.data.dtype.names:
        return None
    vals = np.asarray(ds.data[sensor], dtype=np.float64)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def _flag_outlier_channels(wl, std, factor=5.0):
    """Repère les canaux (longueurs d'onde) dont le bruit (écart-type temporel) est
    anormalement élevé par rapport au reste du spectre -- typiquement un pixel/canal
    défectueux isolé (ex: jonction de filtre d'ordre), pas une dérive spectrale large
    bande. Retourne (masque_outlier, médiane_std_globale)."""
    med = float(np.nanmedian(std))
    mask = std > factor * med
    return mask, med


def analyse_dark(raw_path, capped_sensors, out_dir=None):
    """Analyse le dark d'un fichier RAW acquis capuchons sur `capped_sensors`
    (sous-ensemble de {"ES", "LI", "LT"}).

    Produit dans out_dir (par défaut <MAIN_DATA_PATH>/pySAS/DarkAnalysis/) :
      - <basename>_dark_analysis.png : spectres moyens +/- ecart-type, ShutterLight
        ("comme si c'etait de la lumiere") vs ShutterDark (reference interne).
      - <basename>_dark_analysis.md  : rapport texte (stats, mise en garde, comparaison).

    Retourne le dict de résultats {sensor: {...}}.
    """
    capped_sensors = [s.upper() for s in capped_sensors]
    for s in capped_sensors:
        if s not in VALID_SENSORS:
            raise ValueError(f"Capteur inconnu '{s}' -- doit être parmi {VALID_SENSORS}")

    if out_dir is None:
        out_dir = os.path.join(MAIN_DATA_PATH, "pySAS", "DarkAnalysis")
    os.makedirs(out_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(raw_path))[0]

    print(f"⏳ Lecture L1A (filtre SZA contourné) : {os.path.basename(raw_path)} ...")
    root = _load_l1a_root_bypassing_sza(raw_path)

    results = {}
    for sensor in capped_sensors:
        light_gp = _find_group(root, sensor, "ShutterLight")
        dark_gp = _find_group(root, sensor, "ShutterDark")

        if light_gp is None:
            print(f"⚠️  Aucune trame ShutterLight trouvée pour {sensor} -- capteur ignoré.")
            continue

        wl, mean_light, std_light, n_light = _spectrum_stats(light_gp.getDataset(sensor))
        dark_ave = _scalar_stats(light_gp, sensor, "DARK_AVE")
        dark_samp = _scalar_stats(light_gp, sensor, "DARK_SAMP")
        inttime = _scalar_stats(light_gp, sensor, "INTTIME")
        outlier_mask, median_std = _flag_outlier_channels(wl, std_light)

        entry = {
            "wl": wl,
            "light_mean": mean_light,
            "light_std": std_light,
            "n_light": n_light,
            "light_group": light_gp.id,
            "outlier_mask": outlier_mask,
            "median_std": median_std,
            "dark_ave": dark_ave,
            "dark_samp": dark_samp,
            "inttime": inttime,
        }

        if dark_gp is not None:
            wl_d, mean_dark, std_dark, n_dark = _spectrum_stats(dark_gp.getDataset(sensor))
            entry.update({
                "dark_wl": wl_d,
                "dark_mean": mean_dark,
                "dark_std": std_dark,
                "n_dark": n_dark,
                "dark_group": dark_gp.id,
            })
        else:
            print(f"    ℹ️  Aucune trame ShutterDark interne trouvée pour {sensor} (comparaison non disponible).")

        results[sensor] = entry
        print(f"    ✅ {sensor} : {n_light} trame(s) ShutterLight ({light_gp.id}), "
              f"niveau moyen {np.nanmean(mean_light):.2f} compte(s), "
              f"écart-type moyen {np.nanmean(std_light):.2f}")

    if not results:
        raise RuntimeError("Aucun capteur capuchonné n'a produit de données exploitables.")

    _plot_dark_analysis(results, raw_path, basename, out_dir)
    _write_report(results, raw_path, basename, out_dir, capped_sensors)

    return results


def _plot_dark_analysis(results, raw_path, basename, out_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, (sensor, r) in zip(axes, results.items()):
        ax.plot(r["wl"], r["light_mean"], color="crimson", linewidth=1.5,
                label=f'ShutterLight "capuchon" (n={r["n_light"]}, {r["light_group"]})')
        ax.fill_between(r["wl"], r["light_mean"] - r["light_std"], r["light_mean"] + r["light_std"],
                         color="crimson", alpha=0.2)

        if "dark_mean" in r:
            ax.plot(r["dark_wl"], r["dark_mean"], color="steelblue", linewidth=1.5, linestyle="--",
                    label=f'ShutterDark interne (n={r["n_dark"]}, {r["dark_group"]})')
            ax.fill_between(r["dark_wl"], r["dark_mean"] - r["dark_std"], r["dark_mean"] + r["dark_std"],
                             color="steelblue", alpha=0.15)

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Longueur d'onde (nm)")
        ax.set_ylabel(f"{sensor} (comptes numériques bruts)")
        ax.set_title(f"Capteur {sensor} (capuchonné)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(f"Analyse du dark -- {os.path.basename(raw_path)}", fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(out_dir, f"{basename}_dark_analysis.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"📊 Figure sauvegardée : {out_png}")


def _write_report(results, raw_path, basename, out_dir, capped_sensors):
    lines = [
        f"# Analyse du dark -- {os.path.basename(raw_path)}",
        "",
        f"Capteur(s) capuchonné(s) déclaré(s) : {', '.join(capped_sensors)}",
        "",
        "Comparaison entre :",
        "- **ShutterLight** : trames du canal normal d'acquisition (mêmes trames utilisées",
        "  pour produire les Rrs) -- ici lues \"comme si c'était de la lumière\" alors que le",
        "  capteur est physiquement capuchonné.",
        "- **ShutterDark** : trames de la référence dark interne périodique de l'instrument,",
        "  utilisées par HyperCP (`ProcessL1b.py::processDarkCorrection`) pour corriger les",
        "  trames ShutterLight en mode opérationnel normal.",
        "- **DARK_AVE/DARK_SAMP** : estimé dark par pixels noirs, embarqué (scalaire) dans",
        "  chaque trame ShutterLight -- non utilisé par HyperCP pour la correction, fourni",
        "  ici à titre indicatif.",
        "",
    ]

    for sensor, r in results.items():
        mask = r["outlier_mask"]
        wl_ok = r["wl"][~mask]
        mean_ok = r["light_mean"][~mask]

        lines.append(f"## Capteur {sensor}")
        lines.append("")
        lines.append(f"- ShutterLight ({r['light_group']}) : {r['n_light']} trame(s)")
        lines.append(f"  - Niveau moyen (toutes longueurs d'onde) : {np.nanmean(r['light_mean']):.2f} comptes")
        lines.append(f"  - Écart-type moyen (bruit), hors canaux défectueux : {np.nanmean(r['light_std'][~mask]):.2f} comptes "
                     f"(médiane spectrale {r['median_std']:.2f})")
        lines.append(f"  - Étendue spectrale du signal moyen (hors canaux défectueux) : {np.nanmin(mean_ok):.2f} à {np.nanmax(mean_ok):.2f} comptes "
                     f"(delta {np.nanmax(mean_ok) - np.nanmin(mean_ok):.2f})")
        if r["inttime"]:
            lines.append(f"  - Temps d'intégration moyen : {r['inttime'][0]:.2f} ms (écart-type {r['inttime'][1]:.2f})")
        if r["dark_ave"]:
            lines.append(f"  - DARK_AVE embarqué (moyenne des trames) : {r['dark_ave'][0]:.2f} (écart-type {r['dark_ave'][1]:.2f})")
        if r["dark_samp"]:
            lines.append(f"  - DARK_SAMP embarqué (moyenne des trames) : {r['dark_samp'][0]:.2f} (écart-type {r['dark_samp'][1]:.2f})")

        if mask.any():
            lines.append("")
            lines.append(f"  ⚠️ **{int(mask.sum())} canal(aux) anormalement bruyant(s)** (écart-type > 5x la médiane "
                         f"spectrale) -- probablement un pixel/canal défectueux isolé plutôt qu'un biais spectral "
                         f"large bande :")
            for wl_bad, m_bad, s_bad in zip(r["wl"][mask], r["light_mean"][mask], r["light_std"][mask]):
                lines.append(f"    - {wl_bad:.2f} nm : moyenne={m_bad:.1f}, écart-type={s_bad:.1f}")

        if "dark_mean" in r:
            lines.append("")
            lines.append(f"- ShutterDark interne ({r['dark_group']}) : {r['n_dark']} trame(s)")
            lines.append(f"  - Niveau moyen (toutes longueurs d'onde) : {np.nanmean(r['dark_mean']):.2f} comptes")
            lines.append(f"  - Écart-type moyen (bruit) : {np.nanmean(r['dark_std']):.2f} comptes")

            # Comparaison directe sur la grille de longueurs d'onde commune (interpolation simple),
            # canaux défectueux du ShutterLight exclus pour ne pas fausser le diagnostic.
            dark_mask = np.interp(r["dark_wl"], r["wl"], mask.astype(float)) > 0.5
            common_light = np.interp(r["dark_wl"], r["wl"], r["light_mean"])
            delta = common_light - r["dark_mean"]
            delta_ok = delta[~dark_mask]
            lines.append("")
            lines.append(f"- **Écart ShutterLight (capuchon) − ShutterDark interne**, hors canaux défectueux : "
                         f"moyen {np.nanmean(delta_ok):.2f}, min {np.nanmin(delta_ok):.2f}, max {np.nanmax(delta_ok):.2f} comptes.")
            if np.nanmax(np.abs(delta_ok)) > 3 * np.nanmean(r["light_std"][~mask]):
                lines.append("  ⚠️ Écart significatif (supérieur à ~3x le bruit du canal ShutterLight) sur une "
                             "partie du spectre hors canaux défectueux isolés : la référence dark interne ne "
                             "semble pas représenter fidèlement le niveau/la forme spectrale du dark réel du "
                             "canal d'acquisition. Ceci pourrait expliquer les anomalies observées dans les Rrs "
                             "à faible Ed.")
            else:
                lines.append("  ℹ️ Écart dans l'ordre de grandeur du bruit du canal (hors canaux défectueux isolés) "
                             "-- pas d'indication forte d'un biais systématique large bande de la correction dark "
                             "opérationnelle pour ce fichier. Les canaux défectueux isolés listés ci-dessus "
                             "restent, eux, à surveiller spécifiquement.")
        lines.append("")

    lines.append("---")
    lines.append("Note : ce test a été acquis délibérément hors des conditions d'éclairement valides "
                 "(capuchons / faible lumière) ; le filtre SZA (`bL1aCleanSZA`) a donc été contourné "
                 "spécifiquement pour ce traitement diagnostique et n'affecte pas le traitement opérationnel.")

    out_md = os.path.join(out_dir, f"{basename}_dark_analysis.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📝 Rapport sauvegardé : {out_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_path", help="Chemin du fichier RAW pySAS/HyperSAS acquis capuchons")
    parser.add_argument("--sensors", required=True,
                        help="Capteur(s) capuchonné(s), séparés par des virgules, ex: LT,LI (parmi ES,LI,LT)")
    parser.add_argument("--out-dir", default=None,
                        help="Dossier de sortie (par défaut <MAIN_DATA_PATH>/pySAS/DarkAnalysis/)")
    args = parser.parse_args()

    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    analyse_dark(args.raw_path, sensors, out_dir=args.out_dir)
