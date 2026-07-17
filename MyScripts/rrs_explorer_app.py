"""
Rrs Explorer - visionneuse interactive locale (Dash/Plotly) pour les résultats L2
du pipeline pySAS/HyperCP.

Usage :
    conda activate hypercp
    python rrs_explorer_app.py
    -> ouvrir http://127.0.0.1:8050 dans un navigateur

Fonctionnement :
- Lit pipeline_config.env (même fichier que extract_l2_qc_tables.py) pour trouver
  MAIN_DATA_PATH.
- Le sélecteur de date liste les journées déjà traitées (fichiers
  AnalysisComparison/L2_Methods_Quotes_<date>.csv produits par extract_l2_qc_tables.py).
- La carte affiche la trajectoire du bateau pour la journée choisie (un point par cast),
  avec traits de côte (données géo intégrées à Plotly, aucun réseau requis).
- Cliquer sur un point charge les 9 spectres Rrs (M99/Z17/3C x NN/NIR/SimSpec) de ce cast.
- Sélectionner un groupe de points (lasso/rectangle) + choisir une longueur d'onde au
  slider affiche une matrice de corrélation (scatterplot matrix) entre les 9 méthodes
  pour ce sous-ensemble, avec un tableau de stats d'intercomparaison (biais, RMSD, R²)
  par paire de méthodes.

Conçu pour tourner en local, sans connexion réseau requise, pour un usage à bord
(accès réseau limité) aussi bien qu'à terre sur les données synchronisées. Les
spectres de la journée sont lus une fois depuis les HDF5 puis mis en cache en mémoire
(par date) pour que la sélection/slider restent réactifs sans relire les fichiers.
"""
import os
import sys
import glob
import itertools
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from dash import Dash, dcc, html, dash_table, Input, Output, State
from generate_ancillary_file import read_tsg

# NIR/SimSpec offsets are not relevant on top of 3C (decided 2026-07-17) -- only 3CNN
# is produced, matching what apply_nir_corrections.py now generates.
SKIP_NIR_CORRECTIONS_FOR = {"3C"}
METHODS = [
    f"{model}{correction}"
    for model in ("M99", "Z17", "3C")
    for correction in (("NN",) if model in SKIP_NIR_CORRECTIONS_FOR else ("NN", "NIR", "SimSpec"))
]
METHOD_COLORS = dict(zip(METHODS, plotly.colors.qualitative.D3[:len(METHODS)]))

# Variables disponibles pour la coloration des points sur la carte -> colonne de df_casts
COLOR_VARIABLES = {
    "Heure UTC": ("Hour", "Heure (UTC)"),
    "Vent (m/s)": ("WindSpeed", "Vent (m/s)"),
    "Li/Es (nuage)": ("CloudRatio", "Li(750)/Es(750)"),
    "Angle zénithal solaire (°)": ("SZA", "Angle zénithal solaire (°)"),
    "SST TSG (°C)": ("TSG_T2", "Température TSG (°C)"),
    "Salinité TSG (psu)": ("TSG_S", "Salinité (psu)"),
    "Fluorescence TSG": ("TSG_Fluo", "Fluorescence TSG"),
}


def load_pipeline_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#")[0].strip()
            if "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()
    return config


MY_DIR = os.path.dirname(os.path.abspath(__file__))
ENV = load_pipeline_config(os.path.join(MY_DIR, "pipeline_config.env"))
MAIN_DATA_PATH = ENV["MAIN_DATA_PATH"]
BASE_PATH = os.path.join(MAIN_DATA_PATH, "pySAS")
ANALYSIS_DIR = os.path.join(BASE_PATH, "AnalysisComparison")
TSG_DIR = os.path.join(MAIN_DATA_PATH, "TSG")

# dating (Source.utils.dating) sert à convertir Datetag/Timetag2 en datetime pour
# recaler chaque cast sur l'enregistrement TSG le plus proche (cf. extract_l2_qc_tables.py).
PATH_HCP = ENV["PATH_HCP"]
if PATH_HCP not in sys.path:
    sys.path.insert(0, PATH_HCP)
import Source.utils.dating as dating  # noqa: E402

# Cache mémoire par date : {date_str: {"df_casts": ..., "wavelengths": ..., "cube": ...}}
# cube shape = (n_casts, n_wavelengths, n_methods), NaN si un cast est absent d'une méthode.
_DAY_CACHE = {}


def available_dates():
    pattern = os.path.join(ANALYSIS_DIR, "L2_Methods_Quotes_*.csv")
    dates = sorted(
        os.path.basename(f).replace("L2_Methods_Quotes_", "").replace(".csv", "")
        for f in glob.glob(pattern)
    )
    return dates


def load_casts(date_str):
    """Un point par cast (Filename, Timetag2, Lat, Lon, WindSpeed, CloudRatio), indépendant
    de la méthode (ces variables ancillaires sont communes aux 9 méthodes)."""
    csv_path = os.path.join(ANALYSIS_DIR, f"L2_Methods_Quotes_{date_str}.csv")
    df = pd.read_csv(csv_path)
    df_casts = df.drop_duplicates(subset=["Filename", "Timetag2"])[
        ["Filename", "Datetag", "Timetag2", "Latitude", "Longitude", "WindSpeed", "CloudRatio"]
    ].sort_values("Timetag2").reset_index(drop=True)
    return df_casts


def merge_tsg_for_casts(df_casts, date_str):
    """Associe à chaque cast (Datetag/Timetag2) l'enregistrement TSG le plus proche dans le
    temps (temperature t2, salinite s, fluorescence fluo), tolérance 15 min. Même approche que
    extract_tsg_for_casts() dans extract_l2_qc_tables.py."""
    df_out = df_casts.copy()
    df_out["TSG_T2"] = np.nan
    df_out["TSG_S"] = np.nan
    df_out["TSG_Fluo"] = np.nan

    tsg_path = os.path.join(TSG_DIR, f"tsg_convdata_{date_str}.cnv")
    if not os.path.exists(tsg_path):
        return df_out

    df_tsg = read_tsg(tsg_path)[["datetime", "t2", "s", "fluo"]].dropna(subset=["datetime"])
    for col in ("t2", "s", "fluo"):
        df_tsg[col] = pd.to_numeric(df_tsg[col], errors="coerce")
    df_tsg = df_tsg.sort_values("datetime")

    cast_datetimes = [
        pd.Timestamp(dating.timeTag2ToDateTime(dating.dateTagToDateTime(int(row["Datetag"])), int(row["Timetag2"])))
        for _, row in df_out.iterrows()
    ]
    df_out = df_out.assign(_cast_datetime=cast_datetimes).sort_values("_cast_datetime")

    merged = pd.merge_asof(df_out, df_tsg, left_on="_cast_datetime", right_on="datetime",
                            direction="nearest", tolerance=pd.Timedelta("15min"))
    df_out["TSG_T2"] = merged["t2"].values
    df_out["TSG_S"] = merged["s"].values
    df_out["TSG_Fluo"] = merged["fluo"].values
    return df_out.drop(columns=["_cast_datetime"]).sort_values("Timetag2").reset_index(drop=True)


def timetag2_to_hhmmss(ttag):
    ttag = int(ttag)
    h = ttag // 10000000
    m = (ttag % 10000000) // 100000
    s = (ttag % 100000) // 1000
    return f"{h:02d}:{m:02d}:{s:02d}"


def timetag2_to_hour(ttag):
    ttag = int(ttag)
    return (ttag // 10000000) + ((ttag % 10000000) // 100000) / 60.0


def get_rrs_spectrum(method, filename, timetag2):
    fpath = os.path.join(BASE_PATH, method, "L2", filename)
    if not os.path.exists(fpath):
        return None, None
    with h5py.File(fpath, "r") as h5f:
        rrs_path = "/REFLECTANCE/Rrs_HYPER"
        if rrs_path not in h5f:
            return None, None
        rrs_raw = h5f[rrs_path][...]
        colnames = rrs_raw.dtype.names
        wavelengths = np.array([float(w) for w in colnames[2:]])
        timetag2_vector = rrs_raw[colnames[1]]
        line_indices = np.where(timetag2_vector == timetag2)[0]
        if len(line_indices) == 0:
            return None, None
        line_idx = line_indices[0]
        spectrum = np.array([rrs_raw[w][line_idx] for w in colnames[2:]])
        return wavelengths, spectrum


def load_sza_for_casts(df_casts, reference_method=METHODS[0]):
    """Angle zénithal solaire (/ANCILLARY/SZA), lu directement des HDF5 L2 (donnée de
    géométrie commune aux 9 méthodes, donc lue une seule fois via une méthode de référence)."""
    sza_values = np.full(len(df_casts), np.nan)
    for fname, group in df_casts.groupby("Filename"):
        fpath = os.path.join(BASE_PATH, reference_method, "L2", fname)
        if not os.path.exists(fpath):
            continue
        with h5py.File(fpath, "r") as h5f:
            if "/ANCILLARY/SZA" not in h5f:
                continue
            sza_raw = h5f["/ANCILLARY/SZA"][...]
            colnames = sza_raw.dtype.names
            ttag_vec = sza_raw[colnames[1]]
            sza_vec = sza_raw[colnames[2]]
            for idx in group.index:
                match = np.where(ttag_vec == df_casts.loc[idx, "Timetag2"])[0]
                if len(match):
                    sza_values[idx] = float(sza_vec[match[0]])
    return sza_values


def load_qc_matrices(date_str, df_casts):
    """Matrices (n_casts x n_methods) de QWIP et WEI_QA, lues directement du CSV
    L2_Methods_Quotes_<date>.csv (pas besoin de rouvrir les HDF5)."""
    csv_path = os.path.join(ANALYSIS_DIR, f"L2_Methods_Quotes_{date_str}.csv")
    raw_df = pd.read_csv(csv_path)
    key_index = pd.MultiIndex.from_arrays([df_casts["Filename"], df_casts["Timetag2"]])
    qwip_pivot = raw_df.pivot_table(index=["Filename", "Timetag2"], columns="Method", values="QWIP")
    wei_pivot = raw_df.pivot_table(index=["Filename", "Timetag2"], columns="Method", values="WEI_QA")
    qwip_matrix = qwip_pivot.reindex(index=key_index, columns=METHODS).to_numpy()
    wei_matrix = wei_pivot.reindex(index=key_index, columns=METHODS).to_numpy()
    return qwip_matrix, wei_matrix


def get_day_data(date_str):
    """Charge (une fois par date) tous les spectres des 9 méthodes en mémoire."""
    if date_str in _DAY_CACHE:
        return _DAY_CACHE[date_str]

    df_casts = load_casts(date_str)
    df_casts["Hour"] = df_casts["Timetag2"].apply(timetag2_to_hour)
    df_casts = merge_tsg_for_casts(df_casts, date_str)
    df_casts["SZA"] = load_sza_for_casts(df_casts)
    qwip_matrix, wei_matrix = load_qc_matrices(date_str, df_casts)
    wavelengths = None
    cube = None

    for i, row in df_casts.iterrows():
        for j, method in enumerate(METHODS):
            wl, spec = get_rrs_spectrum(method, row["Filename"], row["Timetag2"])
            if wl is None:
                continue
            if wavelengths is None:
                wavelengths = wl
                cube = np.full((len(df_casts), len(wavelengths), len(METHODS)), np.nan)
            cube[i, :, j] = spec

    data = {"df_casts": df_casts, "wavelengths": wavelengths, "cube": cube,
            "qwip": qwip_matrix, "wei": wei_matrix}
    _DAY_CACHE[date_str] = data
    return data


def build_map_figure(df_casts, color_values, color_label, cmin=None, cmax=None):
    color_values = pd.Series(np.asarray(color_values, dtype=float), index=df_casts.index)
    hover_text = [
        f"{row.Filename}<br>{timetag2_to_hhmmss(row.Timetag2)} UTC<br>{color_label}: {val:.3g}"
        if pd.notna(val) else f"{row.Filename}<br>{timetag2_to_hhmmss(row.Timetag2)} UTC<br>{color_label}: n/d"
        for row, val in zip(df_casts.itertuples(), color_values)
    ]
    customdata = list(zip(df_casts["Filename"], df_casts["Timetag2"]))

    marker = dict(
        size=10, color=color_values, colorscale="Viridis",
        colorbar=dict(title=color_label),
        line=dict(color="black", width=0.5),
    )
    if cmin is not None:
        marker["cmin"] = cmin
    if cmax is not None:
        marker["cmax"] = cmax

    fig = go.Figure(go.Scattergeo(
        lon=df_casts["Longitude"],
        lat=df_casts["Latitude"],
        mode="lines+markers",
        line=dict(color="#888", width=1),
        marker=marker,
        text=hover_text,
        hoverinfo="text",
        customdata=customdata,
    ))
    fig.update_geos(
        fitbounds="locations",
        resolution=50,
        showland=True, landcolor="#f4f3ef",
        showocean=True, oceancolor="#e0f3ff",
        showlakes=True, lakecolor="#e0f3ff",
        showrivers=True, rivercolor="#e0f3ff",
        showcoastlines=True, coastlinecolor="#7f8c8d",
        showcountries=True, countrycolor="#bdc3c7",
        projection_type="mercator",
    )
    fig.update_layout(
        title="Trajectoire du navire (cliquer un point ; icône lasso/rectangle dans la barre "
              "d'outils pour sélectionner un groupe -- l'icône zoom permet d'y revenir)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=600,
    )
    return fig


def build_empty_spectra_figure():
    fig = go.Figure()
    fig.update_layout(
        title="Cliquer un point sur la carte pour afficher les spectres Rrs",
        xaxis_title="Longueur d'onde (nm)", yaxis_title="Rrs (sr⁻¹)",
        height=600,
    )
    return fig


def find_cast_index(df_casts, filename, timetag2):
    matches = df_casts.index[(df_casts["Filename"] == filename) & (df_casts["Timetag2"] == int(timetag2))]
    return int(matches[0]) if len(matches) else None


# Seuils Dierssen et al. (FRM4SOC) : QWIP < 0.05 valide, 0.05-0.1 douteux, >= 0.1
# probablement invalide pour un milieu optiquement profond.
def qwip_line_style(qwip):
    if qwip is None or not np.isfinite(qwip):
        return dict(width=1.5, dash="solid")
    if qwip >= 0.1:
        return dict(width=1.5, dash="dot")
    if qwip >= 0.05:
        return dict(width=1.0, dash="solid")
    return dict(width=3.0, dash="solid")


# WEI_QA (Wei, Lee & Shang 2016 totScore) : 1 = bon accord de forme spectrale avec les
# 23 types d'eau de référence (5 bandes), 0 = mauvais -- indépendant du QWIP, donc encodé
# séparément via l'opacité du trait plutôt que mélangé aux mêmes seuils.
def wei_opacity(wei):
    if wei is None or not np.isfinite(wei):
        return 0.6
    return float(np.clip(0.25 + 0.75 * wei, 0.25, 1.0))


def build_spectra_figure(day, idx):
    wavelengths = day["wavelengths"]
    cube = day["cube"]
    qwip_matrix = day["qwip"]
    wei_matrix = day["wei"]
    row = day["df_casts"].iloc[idx]

    fig = go.Figure()
    for j, method in enumerate(METHODS):
        spectrum = cube[idx, :, j]
        if np.all(np.isnan(spectrum)):
            continue
        qwip = qwip_matrix[idx, j]
        wei = wei_matrix[idx, j]
        style = qwip_line_style(qwip)
        qwip_txt = f"{qwip:.3f}" if np.isfinite(qwip) else "n/d"
        wei_txt = f"{wei:.2f}" if np.isfinite(wei) else "n/d"
        fig.add_trace(go.Scatter(
            x=wavelengths, y=spectrum, mode="lines",
            name=f"{method} (QWIP={qwip_txt}, WEI={wei_txt})",
            line=dict(color=METHOD_COLORS[method], width=style["width"], dash=style["dash"]),
            opacity=wei_opacity(wei),
        ))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(
        title=f"Spectres Rrs -- {row['Filename']} @ {timetag2_to_hhmmss(row['Timetag2'])} UTC<br>"
              "<sup>Épais = QWIP&lt;0.05 (valide) | fin = 0.05-0.1 (douteux) | pointillé = "
              "QWIP&ge;0.1 (invalide, Dierssen et al.) -- opacité ∝ WEI (Wei et al. 2016)</sup>",
        xaxis_title="Longueur d'onde (nm)", yaxis_title="Rrs (sr⁻¹)",
        xaxis_range=[380, 800],
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, font=dict(size=10)),
        margin=dict(t=100),
    )
    return fig


def build_splom_figure(df_wl, wavelength):
    fig = go.Figure(go.Splom(
        dimensions=[dict(label=m, values=df_wl[m]) for m in METHODS],
        diagonal_visible=False,
        showupperhalf=True,
        marker=dict(size=5, color="#1f77b4", opacity=0.6, line=dict(width=0.5, color="white")),
    ))
    fig.update_layout(
        title=f"Matrice de corrélation inter-méthodes à {wavelength:.1f} nm (N={len(df_wl)} points sélectionnés)",
        height=750,
    )
    return fig


def compute_pairwise_stats(df_wl):
    records = []
    for m1, m2 in itertools.combinations(METHODS, 2):
        x = df_wl[m1].to_numpy()
        y = df_wl[m2].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        n = int(mask.sum())
        if n < 2:
            continue
        x, y = x[mask], y[mask]
        bias = float(np.mean(y - x))
        rmsd = float(np.sqrt(np.mean((y - x) ** 2)))
        r = np.corrcoef(x, y)[0, 1]
        records.append({
            "Méthode A": m1, "Méthode B": m2, "N": n,
            "Biais (B-A)": round(bias, 5), "RMSD": round(rmsd, 5),
            "R²": round(float(r ** 2), 4) if np.isfinite(r) else None,
        })
    df_stats = pd.DataFrame(records).sort_values("RMSD", ascending=False)
    return df_stats


def build_empty_splom_figure():
    fig = go.Figure()
    fig.update_layout(
        title="Sélectionner des points sur la carte (lasso/rectangle) pour comparer les méthodes",
        height=750,
    )
    return fig


def selected_indices(selected_data, n_casts):
    """Indices des casts sélectionnés sur la carte, ou tous les casts si rien n'est sélectionné."""
    if selected_data is None or not selected_data.get("points"):
        return list(range(n_casts))
    return sorted({p["pointIndex"] for p in selected_data["points"]})


def build_selected_spectra_figure(day, idx, method, color_values, color_label="Heure (UTC)"):
    wavelengths = day["wavelengths"]
    df_casts = day["df_casts"]
    cube = day["cube"]
    method_idx = METHODS.index(method)

    values = pd.Series(np.asarray(color_values, dtype=float), index=df_casts.index)
    finite = values[np.isfinite(values)]
    vmin, vmax = (finite.min(), finite.max()) if len(finite) else (0.0, 1.0)
    span = (vmax - vmin) or 1.0

    fig = go.Figure()
    for i in idx:
        row = df_casts.iloc[i]
        spectrum = cube[i, :, method_idx]
        if np.all(np.isnan(spectrum)):
            continue
        val = values.iloc[i]
        if pd.notna(val):
            color = plotly.colors.sample_colorscale("Viridis", [(val - vmin) / span])[0]
        else:
            color = "#999999"
        val_txt = f"{val:.3g}" if pd.notna(val) else "n/d"
        fig.add_trace(go.Scatter(
            x=wavelengths, y=spectrum, mode="lines",
            line=dict(color=color), opacity=0.8,
            name=timetag2_to_hhmmss(row["Timetag2"]),
            hovertext=f"{row['Filename']}<br>{color_label}: {val_txt}",
        ))

    # Trace fantôme pour afficher une colorbar cohérente avec la carte
    # (chaque spectre est une trace de couleur fixe, donc pas de colorbar native).
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Viridis", cmin=vmin, cmax=vmax, color=[vmin],
                    showscale=True, colorbar=dict(title=color_label)),
        showlegend=False, hoverinfo="none",
    ))

    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(
        title=f"Spectres Rrs -- méthode {method} ({len(idx)} points sélectionnés), "
              f"couleur = {color_label}",
        xaxis_title="Longueur d'onde (nm)", yaxis_title="Rrs (sr⁻¹)",
        xaxis_range=[380, 800],
        height=550,
        showlegend=len(idx) <= 20,
    )
    return fig


app = Dash(__name__)
app.title = "Rrs Explorer - pySAS"

dates = available_dates()
default_date = dates[-1] if dates else None
_default_day = get_day_data(default_date) if default_date else None
_default_wl = _default_day["wavelengths"] if _default_day is not None else np.array([550.0])

app.layout = html.Div([
    html.H2("Rrs Explorer -- pySAS / HyperCP"),
    html.Div([
        html.Div([
            html.Label("Date :"),
            dcc.Dropdown(
                id="date-dropdown",
                options=[{"label": d, "value": d} for d in dates],
                value=default_date,
                clearable=False,
                style={"width": "200px"},
            ),
        ], style={"display": "inline-block", "marginRight": "40px"}),
        html.Div([
            html.Label("Couleur des points (carte) :"),
            dcc.Dropdown(
                id="color-dropdown",
                options=[{"label": k, "value": k} for k in COLOR_VARIABLES] + [{"label": "QWIP", "value": "QWIP"}],
                value="Heure UTC",
                clearable=False,
                style={"width": "220px"},
            ),
        ], style={"display": "inline-block", "marginRight": "40px"}),
        html.Div([
            html.Label("Méthode (QWIP) :"),
            dcc.Dropdown(
                id="qwip-method-dropdown",
                options=[{"label": m, "value": m} for m in METHODS],
                value="M99NN",
                clearable=False,
                style={"width": "200px"},
            ),
        ], id="qwip-method-container", style={"display": "none"}),
    ], style={"marginBottom": "20px"}),
    html.Div([
        html.Div(dcc.Graph(id="map-graph", config={"scrollZoom": True}),
                 style={"width": "48%", "display": "inline-block"}),
        html.Div(dcc.Graph(id="spectra-graph", figure=build_empty_spectra_figure()),
                 style={"width": "48%", "display": "inline-block", "float": "right"}),
    ]),
    html.Hr(),
    html.H3("Spectres comparés entre points sélectionnés (une méthode)"),
    html.Div([
        html.Label("Méthode :"),
        dcc.Dropdown(
            id="method-dropdown",
            options=[{"label": m, "value": m} for m in METHODS],
            value="M99NN",
            clearable=False,
            style={"width": "200px"},
        ),
    ], style={"marginBottom": "10px"}),
    dcc.Graph(id="selected-spectra-graph"),
    html.Hr(),
    html.H3("Intercomparaison des méthodes (sur la sélection de points)"),
    html.Div([
        html.Label("Longueur d'onde (nm) :"),
        dcc.Slider(
            id="wl-slider",
            min=float(_default_wl.min()), max=float(_default_wl.max()),
            step=None,
            marks={float(w): f"{w:.0f}" for w in _default_wl[::10]},
            value=float(_default_wl[len(_default_wl) // 2]),
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ], style={"width": "80%", "margin": "20px auto"}),
    html.Div([
        html.Div(dcc.Graph(id="splom-graph", figure=build_empty_splom_figure()),
                 style={"width": "58%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div(
            dash_table.DataTable(
                id="stats-table",
                columns=[{"name": c, "id": c} for c in
                         ["Méthode A", "Méthode B", "N", "Biais (B-A)", "RMSD", "R²"]],
                data=[],
                sort_action="native",
                style_table={"height": "700px", "overflowY": "auto"},
                style_cell={"fontSize": 12, "padding": "4px"},
                style_header={"fontWeight": "bold"},
            ),
            style={"width": "40%", "display": "inline-block", "float": "right"},
        ),
    ]),
])


@app.callback(
    Output("qwip-method-container", "style"),
    Input("color-dropdown", "value"),
)
def toggle_qwip_method_dropdown(color_choice):
    return {"display": "inline-block"} if color_choice == "QWIP" else {"display": "none"}


@app.callback(
    Output("map-graph", "figure"),
    Output("wl-slider", "min"),
    Output("wl-slider", "max"),
    Output("wl-slider", "marks"),
    Output("wl-slider", "value"),
    Input("date-dropdown", "value"),
    Input("color-dropdown", "value"),
    Input("qwip-method-dropdown", "value"),
)
def update_map(date_str, color_choice, qwip_method):
    if not date_str:
        return go.Figure(), 0, 1, {}, 0
    day = get_day_data(date_str)
    wavelengths = day["wavelengths"]
    marks = {float(w): f"{w:.0f}" for w in wavelengths[::10]}
    default_wl = float(wavelengths[len(wavelengths) // 2])

    if color_choice == "QWIP":
        method_idx = METHODS.index(qwip_method) if qwip_method in METHODS else 0
        color_values = day["qwip"][:, method_idx]
        color_label = f"QWIP ({METHODS[method_idx]})"
        # Fixed range keyed to the Dierssen et al. thresholds (0.05 caution, 0.1 invalid)
        # so a rare extreme outlier doesn't wash out the color scale for everyone else.
        map_fig = build_map_figure(day["df_casts"], color_values, color_label, cmin=0, cmax=0.15)
    else:
        color_col, color_label = COLOR_VARIABLES.get(color_choice, ("Hour", "Heure (UTC)"))
        color_values = day["df_casts"][color_col]
        map_fig = build_map_figure(day["df_casts"], color_values, color_label)

    return (
        map_fig,
        float(wavelengths.min()), float(wavelengths.max()), marks, default_wl,
    )


@app.callback(
    Output("spectra-graph", "figure"),
    Input("map-graph", "clickData"),
    State("date-dropdown", "value"),
)
def update_spectra(click_data, date_str):
    if click_data is None or not date_str:
        return build_empty_spectra_figure()
    point = click_data["points"][0]
    filename, timetag2 = point["customdata"]
    day = get_day_data(date_str)
    idx = find_cast_index(day["df_casts"], filename, int(timetag2))
    if idx is None:
        return build_empty_spectra_figure()
    return build_spectra_figure(day, idx)


@app.callback(
    Output("splom-graph", "figure"),
    Output("stats-table", "data"),
    Input("map-graph", "selectedData"),
    Input("wl-slider", "value"),
    State("date-dropdown", "value"),
)
def update_intercomparison(selected_data, wavelength, date_str):
    if not date_str or wavelength is None:
        return build_empty_splom_figure(), []

    day = get_day_data(date_str)
    wavelengths = day["wavelengths"]
    cube = day["cube"]
    idx = selected_indices(selected_data, cube.shape[0])

    if len(idx) < 2:
        return build_empty_splom_figure(), []

    wl_idx = int(np.argmin(np.abs(wavelengths - wavelength)))
    actual_wl = wavelengths[wl_idx]

    df_wl = pd.DataFrame(cube[idx, wl_idx, :], columns=METHODS)

    fig = build_splom_figure(df_wl, actual_wl)
    df_stats = compute_pairwise_stats(df_wl)
    return fig, df_stats.to_dict("records")


@app.callback(
    Output("selected-spectra-graph", "figure"),
    Input("map-graph", "selectedData"),
    Input("method-dropdown", "value"),
    Input("color-dropdown", "value"),
    State("date-dropdown", "value"),
)
def update_selected_spectra(selected_data, method, color_choice, date_str):
    if not date_str or not method:
        return go.Figure()
    day = get_day_data(date_str)
    idx = selected_indices(selected_data, day["cube"].shape[0])

    if color_choice == "QWIP":
        method_idx = METHODS.index(method)
        color_values = day["qwip"][:, method_idx]
        color_label = f"QWIP ({method})"
    else:
        color_col, color_label = COLOR_VARIABLES.get(color_choice, ("Hour", "Heure (UTC)"))
        color_values = day["df_casts"][color_col]

    return build_selected_spectra_figure(day, idx, method, color_values, color_label)


if __name__ == "__main__":
    if not dates:
        print(f"⚠️ Aucune journée traitée trouvée dans {ANALYSIS_DIR}")
        print("   (lancer extract_l2_qc_tables.py sur au moins une date d'abord)")
    app.run(debug=True, use_reloader=False)
