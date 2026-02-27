import io
import re

import numpy as np
import pandas as pd
import streamlit as st
from alphacast import Alphacast

# Optional plotting stack (kept because your original notebook uses it)
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from scipy.optimize import curve_fit


# -----------------------------
# Defaults (from your notebook)
# -----------------------------

DEFAULT_DATASET_ID = 41886

DEFAULT_ONS = {
    "YMCHO": {"calificacion": "AAA(arg)", "min_nominal": 1},
    "MGC9O": {"calificacion": "AA+(arg)", "min_nominal": 1},
    "MTCGO": {"calificacion": "N/D", "min_nominal": 1},
    "TLC1O": {"calificacion": "AA+(arg)", "min_nominal": 1000},
    "PNDCO": {"calificacion": "AAA(arg)", "min_nominal": 1000},
    "GNCXO": {"calificacion": "A+(arg)", "min_nominal": 1000},
    "BACAO": {"calificacion": "AAA(arg)", "min_nominal": 150000},
    "CAC5O": {"calificacion": "AA(arg)", "min_nominal": 1},
    "YCAMO": {"calificacion": "AAA(arg)", "min_nominal": 10000},
    "IRCFO": {"calificacion": "AAA(arg)", "min_nominal": 1},
    "YMCIO": {"calificacion": "AAA(arg)", "min_nominal": 1},
    "BYCHO": {"calificacion": "A1+(arg)", "min_nominal": 2000},
    "ARC1O": {"calificacion": "AA(arg)", "min_nominal": 1000},
    "BACGO": {"calificacion": "CCC+ (Fitch)", "min_nominal": 1000},
    "PNXCO": {"calificacion": "AAA(arg)", "min_nominal": 1000},
    "DNC7O": {"calificacion": "CCC+ (Fitch)", "min_nominal": 100},
    "RUCDO": {"calificacion": "CCC (Fitch)", "min_nominal": 1},
    "TLCMO": {"calificacion": "AA+(arg)", "min_nominal": 1000},
    "YMCXO": {"calificacion": "AAA(arg)", "min_nominal": 1},
    "TSC3O": {"calificacion": "B (Fitch)", "min_nominal": 10000},
    "MGCMO": {"calificacion": "AA+(arg)", "min_nominal": 1000},
    "YFCJO": {"calificacion": "AAA(arg)", "min_nominal": 1000},
    "PLC4O": {"calificacion": "BB (Fitch)", "min_nominal": 1000},
    "YMCJO": {"calificacion": "AAA(arg)", "min_nominal": 1},
    "TTCAO": {"calificacion": "AAA(arg)", "min_nominal": 1000},
    "VSCVO": {"calificacion": "BB- (Fitch)", "min_nominal": 1000},
    "TLCPO": {"calificacion": "B (Fitch)", "min_nominal": 100},
    "RC1CO": {"calificacion": "B+ (Fitch)", "min_nominal": 1000},
    "YM34O": {"calificacion": "AAA(arg)", "min_nominal": 1000},
    "IRCPO": {"calificacion": "AAA(arg)", "min_nominal": 1},
    "MGCOO": {"calificacion": "AAA(arg)", "min_nominal": 10000},
    "VSCTO": {"calificacion": "AAA(arg)", "min_nominal": 10000},
    "DNC5O": {"calificacion": "A(arg)", "min_nominal": 1},
    "LOC5O": {"calificacion": "AAA(arg)", "min_nominal": 50},
    "PN38O": {"calificacion": "AAA(arg)", "min_nominal": 1},
    "RCCJO": {"calificacion": "AAA(arg)", "min_nominal": "N/D"},
    "VSCRO": {"calificacion": "AAA(arg)", "min_nominal": 1},
}

DEFAULT_LAMINA_MAX = 10000
DEFAULT_MD_MAX = 5.0

BONOS_GD = ["GD29", "GD30", "GD35", "GD38", "GD41", "GD46"]
BONOS_AL = ["AL29", "AL30", "AL35", "AE38", "AL41", "AL46"]
BOPREAL = ["BPOC7", "BPOC8", "BPOC9"]


# -----------------------------
# Helpers
# -----------------------------
def default_ons_df() -> pd.DataFrame:
    rows = []
    for t, meta in DEFAULT_ONS.items():
        rows.append(
            {
                "Ticker": t,
                "Calificacion": meta.get("calificacion", "N/D"),
                "Lamina_Minima": meta.get("min_nominal", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    df["Lamina_Minima"] = pd.to_numeric(df["Lamina_Minima"], errors="coerce")
    return df


@st.cache_data(show_spinner=False, ttl=15 * 60)
def download_dataset(api_key: str, dataset_id: int) -> pd.DataFrame:
    """
    Downloads the Alphacast dataset as CSV and returns a raw DataFrame.
    Cached for 15 minutes to avoid hammering the API.
    """
    alphacast = Alphacast(api_key)
    csv_data_bytes = alphacast.datasets.dataset(int(dataset_id)).download_data(format="csv")
    csv_data_string = csv_data_bytes.decode("utf-8")
    return pd.read_csv(io.StringIO(csv_data_string))


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors your notebook transformations:
    - drops Market Segment if present
    - renames columns
    - type conversions
    - keeps useful columns
    """
    df = df.copy()

    if "Market Segment" in df.columns:
        df.drop(columns=["Market Segment"], inplace=True)

    new_column_names = {
        "symbol": "Ticker",
        "irr": "TIR",
        "modified duration": "MD",
        "convexity": "Convexidad",
        "parity": "Paridad",
        "residual value": "Valor Residual",
    }
    df.rename(columns={k: v for k, v in new_column_names.items() if k in df.columns}, inplace=True)

    # Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # numeric conversions (match notebook intent)
    if "TIR" in df.columns:
        df["TIR"] = pd.to_numeric(df["TIR"], errors="coerce") * 100
    if "MD" in df.columns:
        df["MD"] = pd.to_numeric(df["MD"], errors="coerce")

    # columns to keep (only those that exist)
    wanted = ["Date", "Ticker", "Industry", "law", "TIR", "MD", "Convexidad", "Paridad", "Valor Residual"]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].copy()

    # round
    for c in ["TIR", "MD", "Convexidad", "Paridad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    return df


def latest_snapshot(df_norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if "Date" not in df_norm.columns:
        return df_norm.copy(), None
    most_recent_date = df_norm["Date"].max()
    out = df_norm[df_norm["Date"] == most_recent_date].copy()
    return out, most_recent_date


def build_ons_table(latest_df: pd.DataFrame, ons_df: pd.DataFrame) -> pd.DataFrame:
    """
    latest_df: dataframe for the most recent date (already normalized)
    ons_df: columns: Ticker, Calificacion, Lamina_Minima
    """
    if latest_df.empty:
        return latest_df.copy()

    df = latest_df.copy()
    df["Ticker"] = df["Ticker"].astype(str)

    ons_df2 = ons_df.copy()
    ons_df2["Ticker"] = ons_df2["Ticker"].astype(str)
    ons_df2["Lamina_Minima"] = pd.to_numeric(ons_df2["Lamina_Minima"], errors="coerce")

    df = df[df["Ticker"].isin(ons_df2["Ticker"].dropna().unique())].copy()
    df = df.merge(ons_df2[["Ticker", "Calificacion", "Lamina_Minima"]], on="Ticker", how="left")
    return df


def nelson_siegel(MD, b0, b1, b2, t):
    return b0 + b1 * ((1 - np.exp(-MD / t)) / (MD / t)) + b2 * (((1 - np.exp(-MD / t)) / (MD / t)) - np.exp(-MD / t))


def plot_tir_vs_md_fit(df_plot: pd.DataFrame, title: str):
    """
    Plot a Nelson-Siegel fit + scatter. Returns a matplotlib fig.
    """
    dfp = df_plot.dropna(subset=["MD", "TIR"]).copy()
    if dfp.empty:
        return None

    try:
        params, cov = curve_fit(nelson_siegel, dfp["MD"], dfp["TIR"], maxfev=5000)
        fit_dom = np.linspace(dfp["MD"].min(), dfp["MD"].max(), 100)
        fit_vals = nelson_siegel(fit_dom, *params)
    except Exception:
        params, cov, fit_dom, fit_vals = None, None, None, None

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=dfp, x="MD", y="TIR", s=90, ax=ax)

    if fit_dom is not None:
        ax.plot(fit_dom, fit_vals, linestyle="--", alpha=0.8)

    texts = []
    if "Ticker" in dfp.columns:
        for _, row in dfp.iterrows():
            texts.append(ax.text(row["MD"] + 0.05, row["TIR"] + 0.05, str(row["Ticker"]), fontsize=9))
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.5))

    ax.set_xlabel("Modified Duration (MD)")
    ax.set_ylabel("TIR [%]")
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.4)

    if params is not None and cov is not None:
        err = np.sqrt(np.diag(cov))
        st.caption(
            f"Parámetros ajustados (Nelson–Siegel): "
            f"b0={params[0]:.3f}±{err[0]:.3f}, "
            f"b1={params[1]:.3f}±{err[1]:.3f}, "
            f"b2={params[2]:.3f}±{err[2]:.3f}, "
            f"tau={params[3]:.3f}±{err[3]:.3f}"
        )

    return fig


def compute_spreads_252(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Re-implements your 'Spread Promedio 252' logic using the normalized DF (not only latest date).
    Returns a df with Numero, TIR_AL_AE_Promedio_252, TIR_GD_Promedio_252, Spread_Promedio_252
    """
    todos = BONOS_GD + BONOS_AL + BOPREAL
    df_tir = df_norm[df_norm["Ticker"].isin(todos)].copy()
    if df_tir.empty:
        return pd.DataFrame(columns=["Numero", "TIR_AL_AE_Promedio_252", "TIR_GD_Promedio_252", "Spread_Promedio_252"])

    df_tir["TIR"] = pd.to_numeric(df_tir["TIR"], errors="coerce")

    if "Date" in df_tir.columns:
        df_tir = df_tir.sort_values(["Ticker", "Date"])

    df_last_252 = df_tir.groupby("Ticker", group_keys=False).tail(252)

    df_avg = (
        df_last_252.groupby("Ticker", as_index=False)["TIR"]
        .mean()
        .rename(columns={"TIR": "TIR_Promedio_252"})
    )

    df_avg["Numero"] = df_avg["Ticker"].str.extract(r"(\d+)")
    df_avg["Grupo"] = df_avg["Ticker"].apply(
        lambda x: "GD" if str(x).startswith("GD") else ("AL" if str(x).startswith(("AL", "AE")) else "Otro")
    )

    df_pairs = (
        df_avg.pivot_table(index="Numero", columns="Grupo", values="TIR_Promedio_252", aggfunc="first")
        .reset_index()
    )
    if not {"AL", "GD"}.issubset(set(df_pairs.columns)):
        return pd.DataFrame(columns=["Numero", "TIR_AL_AE_Promedio_252", "TIR_GD_Promedio_252", "Spread_Promedio_252"])

    df_pairs = df_pairs[df_pairs[["AL", "GD"]].notna().all(axis=1)].copy()
    df_pairs["Spread_Promedio_252"] = df_pairs["AL"] - df_pairs["GD"]

    out = (
        df_pairs.rename(columns={"AL": "TIR_AL_AE_Promedio_252", "GD": "TIR_GD_Promedio_252"})
        .assign(Numero=lambda d: pd.to_numeric(d["Numero"], errors="coerce"))
        .dropna(subset=["Numero"])
        .assign(Numero=lambda d: d["Numero"].astype(int))
        .sort_values("Numero")
        [["Numero", "TIR_AL_AE_Promedio_252", "TIR_GD_Promedio_252", "Spread_Promedio_252"]]
    )
    out["Spread_Promedio_252"] = out["Spread_Promedio_252"].round(2)
    return out


def classify_bono(ticker: str) -> str:
    if ticker in BONOS_GD:
        return "GD"
    if ticker in BONOS_AL:
        return "AL"
    if ticker in BOPREAL:
        return "BOPREAL"
    return "Otro"


def add_spread_to_latest(latest_bonos_df: pd.DataFrame, spreads_252: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'Spread%' (AL - GD for same Numero, using latest snapshot)
    and 'Spread_Promedio_252' (from historical averages).
    """
    df = latest_bonos_df.copy()
    if df.empty:
        return df

    df["Grupo"] = df["Ticker"].apply(classify_bono)
    df["Numero"] = df["Ticker"].apply(lambda x: re.findall(r"\d+", str(x))[0] if re.findall(r"\d+", str(x)) else None)
    df["Spread%"] = np.nan

    for num in df["Numero"].dropna().unique():
        dtemp = df[df["Numero"] == num]
        if set(dtemp["Grupo"]) >= {"GD", "AL"}:
            tir_gd = dtemp.loc[dtemp["Grupo"] == "GD", "TIR"].values[0]
            tir_al = dtemp.loc[dtemp["Grupo"] == "AL", "TIR"].values[0]
            df.loc[df["Numero"] == num, "Spread%"] = float(tir_al) - float(tir_gd)

    spreads = spreads_252.copy()
    df["Numero_str"] = df["Numero"].astype(str)
    spreads["Numero_str"] = spreads["Numero"].astype(str)

    df = df.merge(spreads[["Numero_str", "Spread_Promedio_252"]], on="Numero_str", how="left")
    df["Spread_Promedio_252"] = pd.to_numeric(df["Spread_Promedio_252"], errors="coerce").round(2)

    df.drop(columns=["Numero_str"], inplace=True)
    return df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Curvas ONs y Bonos (Alphacast)", layout="wide")

st.title("Curvas de ONs y Bonos (Alphacast)")

with st.sidebar:
    st.header("Configuración")

    # ✅ CAMBIO: NO USAMOS st.secrets (evita StreamlitSecretNotFoundError)
    api_key = st.text_input(
        "Alphacast API Key",
        value="",
        type="password",
        help="Ingresá tu API Key acá (no se lee desde secrets.toml).",
    )

    dataset_id = st.number_input("Dataset ID", value=int(DEFAULT_DATASET_ID), step=1)

    st.divider()
    st.subheader("Filtros de ONs (salida)")
    lamina_max = st.number_input("Lámina mínima máxima (<=)", value=float(DEFAULT_LAMINA_MAX), step=1.0)
    md_max = st.number_input("MD máxima (<=)", value=float(DEFAULT_MD_MAX), step=0.1)

    rating_default = "AAA(arg)"
    rating = st.text_input("Calificación (exacta)", value=rating_default, help="Ej: AAA(arg) | AA+(arg) | N/D")

    show_plots = st.checkbox("Mostrar gráficos (Nelson–Siegel)", value=True)

if not api_key.strip():
    st.warning("Ingresá tu Alphacast API Key para descargar el dataset.")
    st.stop()

with st.spinner("Descargando dataset desde Alphacast..."):
    raw = download_dataset(api_key=api_key.strip(), dataset_id=int(dataset_id))

df_norm = normalize_dataset(raw)
latest_df, most_recent_date = latest_snapshot(df_norm)

st.header("ONs")

st.markdown(
    "Editá la tabla (agregar / modificar tickers, calificación o lámina mínima). "
    "La salida se filtra por **Lámina mínima** y **MD**."
)

if "ons_table" not in st.session_state:
    st.session_state["ons_table"] = default_ons_df()

left, right = st.columns([2, 1])
with right:
    st.subheader("Carga rápida (opcional)")
    uploaded = st.file_uploader("Importar ONs desde CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            needed = {"Ticker", "Calificacion", "Lamina_Minima"}
            if not needed.issubset(set(df_up.columns)):
                st.error("El CSV debe tener columnas: Ticker, Calificacion, Lamina_Minima")
            else:
                df_up["Lamina_Minima"] = pd.to_numeric(df_up["Lamina_Minima"], errors="coerce")
                st.session_state["ons_table"] = df_up[list(needed)].copy()
                st.success("ONs importadas y cargadas en la tabla.")
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")

    if st.button("Restaurar ONs precargadas"):
        st.session_state["ons_table"] = default_ons_df()
        st.success("Restaurado.")

with left:
    st.subheader("Tabla de ONs")
    edited = st.data_editor(
        st.session_state["ons_table"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "Calificacion": st.column_config.TextColumn("Calificación", required=False),
            "Lamina_Minima": st.column_config.NumberColumn("Lámina mínima", required=False, min_value=0.0, step=1.0),
        },
        key="ons_editor",
    )
    st.session_state["ons_table"] = edited.copy()

df_ons = build_ons_table(latest_df, st.session_state["ons_table"])

df_ons["Lamina_Minima"] = pd.to_numeric(df_ons.get("Lamina_Minima", np.nan), errors="coerce")
df_ons["MD"] = pd.to_numeric(df_ons.get("MD", np.nan), errors="coerce")

df_ons_filtered = df_ons.copy()
df_ons_filtered = df_ons_filtered[pd.to_numeric(df_ons_filtered["Lamina_Minima"], errors="coerce") <= float(lamina_max)]
df_ons_filtered = df_ons_filtered[pd.to_numeric(df_ons_filtered["MD"], errors="coerce") <= float(md_max)]

if rating.strip():
    df_ons_filtered = df_ons_filtered[df_ons_filtered["Calificacion"].astype(str) == rating.strip()]

st.subheader("Resultado ONs (filtrado)")
st.dataframe(df_ons_filtered.sort_values(["MD", "TIR"], ascending=[True, False]), use_container_width=True, height=360)

if show_plots and not df_ons_filtered.empty:
    st.subheader("Gráfico ONs: TIR vs MD (fit Nelson–Siegel)")
    fig = plot_tir_vs_md_fit(df_ons_filtered, title="Curva de ONs (TIR vs MD)")
    if fig is not None:
        st.pyplot(fig, clear_figure=True)

st.divider()
st.header("Bonos Hard Dollar")

all_bonds = BONOS_GD + BONOS_AL + BOPREAL
selected_bonds = st.multiselect("Seleccionar bonos", options=all_bonds, default=all_bonds)

latest_bonos = latest_df[latest_df["Ticker"].isin(selected_bonds)].copy()
spreads_252 = compute_spreads_252(df_norm)
latest_bonos = add_spread_to_latest(latest_bonos, spreads_252)

st.subheader("Snapshot bonos (fecha más reciente) + spreads")
st.dataframe(latest_bonos.sort_values(["Ticker"]), use_container_width=True, height=320)

if not spreads_252.empty:
    st.subheader("Spreads promedio (últimos 252) por número (AL/AE - GD)")
    st.dataframe(spreads_252, use_container_width=True, height=220)

if show_plots and not latest_bonos.empty:
    st.subheader("Gráfico Bonos: TIR vs MD (fit Nelson–Siegel por grupo)")
    tmp = latest_bonos.copy()
    tmp["Grupo"] = tmp["Ticker"].apply(lambda t: classify_bono(str(t)))

    for gname in ["GD", "AL", "BOPREAL"]:
        df_g = tmp[tmp["Grupo"] == gname].copy()
        if df_g.empty:
            continue
        fig = plot_tir_vs_md_fit(df_g, title=f"{gname}: TIR vs MD")
        if fig is not None:
            st.pyplot(fig, clear_figure=True)