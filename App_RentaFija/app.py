# -*- coding: utf-8 -*-
"""
Curvas de ONs y Bonos — versión PRO
-----------------------------------
Mejoras sobre la versión original:
  * Gráficos interactivos Plotly con estética institucional (reemplazan matplotlib/seaborn)
  * Fit Nelson-Siegel con análisis cheap/rich (residuos vs curva, en bps)
  * Z-score del spread por legislación (AL/AE vs GD) con señal de arbitraje
  * Métricas clave para el asesor: carry, TIR/MD, convexidad, pendiente de curva
  * Breakeven de inflación (CER vs tasa fija) y de devaluación (Dollar Linked vs tasa fija)
  * Comparación breakeven vs REM (BCRA, dataset 5621) y vs futuros MATBA-ROFEX (dataset 5331)
  * Insights automáticos generados a partir de los datos

Requisitos:
  pip install streamlit pandas numpy plotly scipy alphacast
"""

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from alphacast import Alphacast
from scipy.optimize import curve_fit

# =====================================================================
# Configuración general y tema visual
# =====================================================================

DEFAULT_DATASET_ID = 41886        # ONs / Bonos (curvas)
DATASET_REM = 5621                # REM - BCRA (expectativas)
DATASET_FUTUROS = 5331            # Futuros de dólar MATBA-ROFEX
DATASET_RF_SOBERANA = 42760       # Renta fija soberana (CER / fija / DL) para breakevens

COLORS = {
    "primary": "#1f2a44",     # azul institucional
    "accent": "#c8963e",      # dorado
    "gd": "#2563eb",          # ley NY
    "al": "#dc2626",          # ley local
    "bopreal": "#059669",     # BCRA
    "cer": "#7c3aed",
    "fija": "#0891b2",
    "dl": "#ea580c",
    "cheap": "#16a34a",
    "rich": "#dc2626",
    "grid": "rgba(120,130,150,0.18)",
    "fit": "#64748b",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Helvetica, Arial", size=13, color="#1e293b"),
    title_font=dict(size=17, color=COLORS["primary"]),
    margin=dict(l=60, r=30, t=70, b=55),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#cbd5e1"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                bgcolor="rgba(255,255,255,0.6)"),
    plot_bgcolor="white",
)


def style_axes(fig: go.Figure, xtitle: str, ytitle: str) -> go.Figure:
    fig.update_xaxes(title=xtitle, showgrid=True, gridcolor=COLORS["grid"],
                     zeroline=False, showline=True, linecolor="#94a3b8", ticks="outside")
    fig.update_yaxes(title=ytitle, showgrid=True, gridcolor=COLORS["grid"],
                     zeroline=False, showline=True, linecolor="#94a3b8", ticks="outside")
    return fig


def add_source_watermark(fig: go.Figure, fecha=None) -> go.Figure:
    txt = "Fuente: Alphacast"
    if fecha is not None:
        txt += f" · Datos al {pd.Timestamp(fecha).strftime('%d/%m/%Y')}"
    fig.add_annotation(text=txt, xref="paper", yref="paper", x=0, y=-0.16,
                       showarrow=False, font=dict(size=10, color="#94a3b8"), align="left")
    return fig


# =====================================================================
# Universos por defecto
# =====================================================================

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

# Regex heurísticos para clasificar soberanos en pesos (editable desde la UI)
REGEX_CER = r"^(TX|TZX|T2X|TC|PARP|DICP|CUAP|TZXD?)"
REGEX_FIJA = r"^(S\d{2}|T\d{2}[A-Z]\d|TT[DJMS]\d|BNA|TO\d)"
REGEX_DL = r"^(TV\d|TZV|D\d{2}[A-Z]\d)"


# =====================================================================
# Descarga y normalización
# =====================================================================

def default_ons_df() -> pd.DataFrame:
    rows = [{"Ticker": t, "Calificacion": m.get("calificacion", "N/D"),
             "Lamina_Minima": m.get("min_nominal", np.nan)} for t, m in DEFAULT_ONS.items()]
    df = pd.DataFrame(rows)
    df["Lamina_Minima"] = pd.to_numeric(df["Lamina_Minima"], errors="coerce")
    return df


@st.cache_data(show_spinner=False, ttl=15 * 60)
def download_dataset(api_key: str, dataset_id: int) -> pd.DataFrame:
    alphacast = Alphacast(api_key)
    csv_bytes = alphacast.datasets.dataset(int(dataset_id)).download_data(format="csv")
    return pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Market Segment" in df.columns:
        df.drop(columns=["Market Segment"], inplace=True)

    ren = {"symbol": "Ticker", "irr": "TIR", "modified duration": "MD",
           "convexity": "Convexidad", "parity": "Paridad", "residual value": "Valor Residual"}
    df.rename(columns={k: v for k, v in ren.items() if k in df.columns}, inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "TIR" in df.columns:
        df["TIR"] = pd.to_numeric(df["TIR"], errors="coerce") * 100
    if "MD" in df.columns:
        df["MD"] = pd.to_numeric(df["MD"], errors="coerce")

    wanted = ["Date", "Ticker", "Industry", "law", "TIR", "MD", "Convexidad", "Paridad", "Valor Residual"]
    df = df[[c for c in wanted if c in df.columns]].copy()
    for c in ["TIR", "MD", "Convexidad", "Paridad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    return df


def latest_snapshot(df_norm: pd.DataFrame):
    if "Date" not in df_norm.columns:
        return df_norm.copy(), None
    d = df_norm["Date"].max()
    return df_norm[df_norm["Date"] == d].copy(), d


def build_ons_table(latest_df: pd.DataFrame, ons_df: pd.DataFrame) -> pd.DataFrame:
    if latest_df.empty:
        return latest_df.copy()
    df = latest_df.copy()
    df["Ticker"] = df["Ticker"].astype(str)
    o = ons_df.copy()
    o["Ticker"] = o["Ticker"].astype(str)
    o["Lamina_Minima"] = pd.to_numeric(o["Lamina_Minima"], errors="coerce")
    df = df[df["Ticker"].isin(o["Ticker"].dropna().unique())].copy()
    return df.merge(o[["Ticker", "Calificacion", "Lamina_Minima"]], on="Ticker", how="left")


# =====================================================================
# Nelson-Siegel + cheap/rich
# =====================================================================

def nelson_siegel(md, b0, b1, b2, tau):
    md = np.asarray(md, dtype=float)
    x = md / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(x == 0, 1.0, (1 - np.exp(-x)) / x)
    return b0 + b1 * f + b2 * (f - np.exp(-x))


def fit_ns(md: pd.Series, tir: pd.Series):
    """Ajusta NS con bounds razonables. Devuelve (params, cov) o (None, None)."""
    md, tir = np.asarray(md, float), np.asarray(tir, float)
    if len(md) < 5:
        return None, None
    try:
        p0 = [float(np.nanmean(tir)), -1.0, 1.0, 1.5]
        bounds = ([-50, -80, -80, 0.05], [120, 80, 80, 30])
        params, cov = curve_fit(nelson_siegel, md, tir, p0=p0, bounds=bounds, maxfev=20000)
        return params, cov
    except Exception:
        return None, None


def cheap_rich(df: pd.DataFrame, params) -> pd.DataFrame:
    """Residuo vs curva NS en bps. Positivo = rinde por encima de la curva (barato)."""
    out = df.copy()
    if params is None:
        out["TIR_Curva"] = np.nan
        out["Residuo_bps"] = np.nan
        return out
    out["TIR_Curva"] = nelson_siegel(out["MD"], *params).round(2)
    out["Residuo_bps"] = ((out["TIR"] - out["TIR_Curva"]) * 100).round(0)
    out["Señal"] = np.select(
        [out["Residuo_bps"] >= 50, out["Residuo_bps"] <= -50],
        ["🟢 Barato vs curva", "🔴 Caro vs curva"], default="⚪ En línea",
    )
    return out


def plot_curve_pro(df: pd.DataFrame, title: str, fecha=None, group_col: str | None = None,
                   group_colors: dict | None = None, fit_per_group: bool = False):
    """Scatter TIR vs MD con etiquetas, fit NS y coloreo cheap/rich."""
    dfp = df.dropna(subset=["MD", "TIR"]).copy()
    if dfp.empty:
        return None, None

    fig = go.Figure()
    all_params = {}

    def _hover(d):
        cols = [c for c in ["Paridad", "Convexidad", "Calificacion", "law"] if c in d.columns]
        base = "<b>%{text}</b><br>TIR: %{y:.2f}%<br>MD: %{x:.2f}"
        cd = None
        if cols:
            cd = d[cols].values
            for i, c in enumerate(cols):
                base += f"<br>{c}: %{{customdata[{i}]}}"
        return base + "<extra></extra>", cd

    groups = [(None, dfp)] if not group_col else list(dfp.groupby(group_col))
    for gname, dg in groups:
        color = (group_colors or {}).get(gname, COLORS["primary"])
        hv, cd = _hover(dg)
        fig.add_trace(go.Scatter(
            x=dg["MD"], y=dg["TIR"], mode="markers+text",
            text=dg["Ticker"], textposition="top center",
            textfont=dict(size=10, color="#334155"),
            marker=dict(size=11, color=color, line=dict(width=1.2, color="white"), opacity=0.9),
            name=str(gname) if gname else "Instrumentos",
            customdata=cd, hovertemplate=hv,
        ))
        if fit_per_group:
            p, _ = fit_ns(dg["MD"], dg["TIR"])
            all_params[gname] = p
            if p is not None and len(dg) >= 5:
                dom = np.linspace(dg["MD"].min(), dg["MD"].max(), 120)
                fig.add_trace(go.Scatter(x=dom, y=nelson_siegel(dom, *p), mode="lines",
                                         line=dict(dash="dash", width=1.6, color=color),
                                         name=f"NS {gname}", hoverinfo="skip", opacity=0.6))

    if not fit_per_group:
        p, cov = fit_ns(dfp["MD"], dfp["TIR"])
        all_params["global"] = p
        if p is not None:
            dom = np.linspace(dfp["MD"].min(), dfp["MD"].max(), 150)
            fitted = nelson_siegel(dom, *p)
            fig.add_trace(go.Scatter(x=dom, y=fitted, mode="lines",
                                     line=dict(dash="dash", width=2, color=COLORS["fit"]),
                                     name="Curva Nelson-Siegel"))
            # Banda ±50 bps como referencia de valor relativo
            fig.add_trace(go.Scatter(x=np.concatenate([dom, dom[::-1]]),
                                     y=np.concatenate([fitted + 0.5, (fitted - 0.5)[::-1]]),
                                     fill="toself", fillcolor="rgba(100,116,139,0.10)",
                                     line=dict(width=0), name="±50 bps", hoverinfo="skip"))

    fig.update_layout(title=title, height=520, **PLOTLY_LAYOUT)
    style_axes(fig, "Modified Duration (años)", "TIR (%)")
    add_source_watermark(fig, fecha)
    return fig, all_params


# =====================================================================
# Spreads por legislación (AL/AE vs GD) + z-score
# =====================================================================

def classify_bono(ticker: str) -> str:
    if ticker in BONOS_GD:
        return "GD"
    if ticker in BONOS_AL:
        return "AL"
    if ticker in BOPREAL:
        return "BOPREAL"
    return "Otro"


def spread_series(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Serie diaria de spread AL/AE - GD por 'Numero' (año de vencimiento)."""
    todos = BONOS_GD + BONOS_AL
    d = df_norm[df_norm["Ticker"].isin(todos)].dropna(subset=["TIR"]).copy()
    if d.empty or "Date" not in d.columns:
        return pd.DataFrame()
    d["Numero"] = d["Ticker"].str.extract(r"(\d+)")
    d["Grupo"] = np.where(d["Ticker"].str.startswith("GD"), "GD", "AL")
    piv = d.pivot_table(index=["Date", "Numero"], columns="Grupo", values="TIR", aggfunc="first").reset_index()
    if not {"AL", "GD"}.issubset(piv.columns):
        return pd.DataFrame()
    piv = piv.dropna(subset=["AL", "GD"])
    piv["Spread"] = piv["AL"] - piv["GD"]
    return piv


def spread_stats(spreads: pd.DataFrame) -> pd.DataFrame:
    """Último spread, promedio/desvío 252 ruedas y z-score por Numero."""
    if spreads.empty:
        return pd.DataFrame()
    rows = []
    for num, g in spreads.groupby("Numero"):
        g = g.sort_values("Date").tail(252)
        if g.empty:
            continue
        cur, mu, sd = g["Spread"].iloc[-1], g["Spread"].mean(), g["Spread"].std()
        z = (cur - mu) / sd if sd and sd > 0 else np.nan
        rows.append({"Numero": int(num), "Spread_Actual": round(cur, 2),
                     "Spread_Prom_252": round(mu, 2), "Desvio_252": round(sd, 2) if pd.notna(sd) else np.nan,
                     "Z_Score": round(z, 2) if pd.notna(z) else np.nan})
    out = pd.DataFrame(rows).sort_values("Numero")
    if not out.empty:
        out["Señal"] = np.select(
            [out["Z_Score"] >= 1.5, out["Z_Score"] <= -1.5],
            ["🟢 Spread caro → favorece ley local (AL)", "🔵 Spread comprimido → favorece ley NY (GD)"],
            default="⚪ Neutral",
        )
    return out


def plot_spread_history(spreads: pd.DataFrame):
    if spreads.empty:
        return None
    fig = go.Figure()
    palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    for i, (num, g) in enumerate(spreads.groupby("Numero")):
        g = g.sort_values("Date").tail(252)
        fig.add_trace(go.Scatter(x=g["Date"], y=g["Spread"], mode="lines",
                                 name=f"20{num}" if len(str(num)) == 2 else str(num),
                                 line=dict(width=1.8, color=palette[i % len(palette)])))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(title="Spread por legislación (AL/AE − GD) · últimas 252 ruedas",
                      height=440, **PLOTLY_LAYOUT)
    style_axes(fig, "Fecha", "Spread (puntos de TIR)")
    add_source_watermark(fig)
    return fig


# =====================================================================
# Breakevens (CER / DL vs tasa fija) + REM + Futuros
# =====================================================================

def classify_soberano(ticker: str, rx_cer: str, rx_fija: str, rx_dl: str) -> str:
    t = str(ticker).upper().strip()
    if re.match(rx_cer, t):
        return "CER"
    if re.match(rx_dl, t):
        return "DL"
    if re.match(rx_fija, t):
        return "Fija"
    return "Otro"


def compute_breakevens(df_latest: pd.DataFrame, kind: str = "CER") -> pd.DataFrame:
    """
    Breakeven = (1 + TIR_fija) / (1 + TIR_real_o_DL) − 1, interpolando la curva de
    tasa fija (fit NS o interpolación lineal) en la MD de cada bono CER / DL.
      * CER → inflación breakeven anual implícita
      * DL  → devaluación breakeven anual implícita
    """
    fija = df_latest[df_latest["Clase"] == "Fija"].dropna(subset=["MD", "TIR"])
    target = df_latest[df_latest["Clase"] == kind].dropna(subset=["MD", "TIR"])
    if len(fija) < 2 or target.empty:
        return pd.DataFrame()

    params, _ = fit_ns(fija["MD"], fija["TIR"])

    def tasa_fija_en(md):
        lo, hi = fija["MD"].min(), fija["MD"].max()
        if params is not None and lo <= md <= hi:
            return float(nelson_siegel(np.array([md]), *params)[0])
        f = fija.sort_values("MD")
        return float(np.interp(md, f["MD"], f["TIR"]))

    rows = []
    lo, hi = fija["MD"].min(), fija["MD"].max()
    for _, r in target.iterrows():
        md = float(r["MD"])
        tn = tasa_fija_en(md)
        be = ((1 + tn / 100) / (1 + float(r["TIR"]) / 100) - 1) * 100
        rows.append({"Ticker": r["Ticker"], "MD": round(md, 2),
                     "TIR_" + ("Real" if kind == "CER" else "DL"): round(float(r["TIR"]), 2),
                     "TIR_Fija_Interp": round(tn, 2),
                     "Breakeven_%anual": round(be, 2),
                     "Extrapolado": md < lo or md > hi})
    return pd.DataFrame(rows).sort_values("MD")


def plot_breakeven(df_be: pd.DataFrame, title: str, color: str, ref_value: float | None = None,
                   ref_label: str | None = None):
    if df_be.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_be["MD"], y=df_be["Breakeven_%anual"], mode="markers+lines+text",
        text=df_be["Ticker"], textposition="top center", textfont=dict(size=10),
        line=dict(width=2, color=color), marker=dict(size=10, color=color, line=dict(width=1, color="white")),
        name="Breakeven implícito",
        hovertemplate="<b>%{text}</b><br>MD: %{x:.2f}<br>Breakeven: %{y:.2f}% anual<extra></extra>",
    ))
    if ref_value is not None:
        fig.add_hline(y=ref_value, line_dash="dash", line_color=COLORS["accent"],
                      annotation_text=ref_label or f"Referencia: {ref_value:.1f}%",
                      annotation_font_color=COLORS["accent"])
    fig.update_layout(title=title, height=460, **PLOTLY_LAYOUT)
    style_axes(fig, "Modified Duration (años)", "Breakeven (% anual)")
    add_source_watermark(fig)
    return fig


def extract_rem_inflacion(df_rem: pd.DataFrame):
    """
    Intenta detectar en el REM la mediana de inflación esperada (próximos 12 meses
    o interanual). Devuelve (valor, nombre_columna) o (None, None).
    """
    if df_rem is None or df_rem.empty:
        return None, None
    d = df_rem.copy()
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.sort_values("Date")
    cands = [c for c in d.columns if isinstance(c, str)
             and re.search(r"inflaci", c, re.I)
             and re.search(r"median", c, re.I)]
    pref = [c for c in cands if re.search(r"12|interanual|i\.?a\.?|next 12", c, re.I)]
    col = (pref or cands or [None])[0]
    if col is None:
        return None, None
    serie = pd.to_numeric(d[col], errors="coerce").dropna()
    if serie.empty:
        return None, None
    return float(serie.iloc[-1]), col


# =====================================================================
# UI
# =====================================================================

st.set_page_config(page_title="Renta Fija PRO — Curvas, Spreads y Breakevens", layout="wide")

st.markdown(
    f"""
    <div style="padding:14px 20px;border-radius:12px;
         background:linear-gradient(90deg,{COLORS['primary']},#31456e);color:white;">
      <span style="font-size:1.45rem;font-weight:700;">📊 Renta Fija Argentina — Panel PRO</span><br>
      <span style="opacity:.85;">Curvas ONs y soberanos · valor relativo · spreads por legislación · breakevens vs REM y ROFEX</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Herramienta de análisis. No constituye recomendación de inversión.")

with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Alphacast API Key", value="", type="password")
    dataset_id = st.number_input("Dataset curvas (ONs/Bonos)", value=int(DEFAULT_DATASET_ID), step=1)

    st.divider()
    st.subheader("Filtros de ONs")
    lamina_max = st.number_input("Lámina mínima (≤)", value=float(DEFAULT_LAMINA_MAX), step=1.0)
    md_max = st.number_input("MD máxima (≤)", value=float(DEFAULT_MD_MAX), step=0.1)
    rating = st.text_input("Calificación exacta (vacío = todas)", value="AAA(arg)")

    st.divider()
    st.subheader("Breakevens")
    ds_soberanos = st.number_input("Dataset RF soberana (pesos)", value=int(DATASET_RF_SOBERANA), step=1)
    ds_rem = st.number_input("Dataset REM (BCRA)", value=int(DATASET_REM), step=1)
    ds_fut = st.number_input("Dataset Futuros dólar", value=int(DATASET_FUTUROS), step=1)

if not api_key.strip():
    st.warning("Ingresá tu Alphacast API Key para descargar los datasets.")
    st.stop()

with st.spinner("Descargando dataset de curvas..."):
    raw = download_dataset(api_key.strip(), int(dataset_id))

df_norm = normalize_dataset(raw)
latest_df, most_recent_date = latest_snapshot(df_norm)

tab_ons, tab_bonos, tab_be, tab_insights = st.tabs(
    ["🏢 ONs", "💵 Bonos Hard Dollar", "⚖️ Breakevens (CER · DL · REM · ROFEX)", "🎯 Insights"]
)

# ---------------------------------------------------------------------
# TAB 1 — ONs
# ---------------------------------------------------------------------
with tab_ons:
    st.markdown("Editá el universo de ONs (tickers, calificación, lámina). La salida se filtra por lámina, MD y calificación.")

    if "ons_table" not in st.session_state:
        st.session_state["ons_table"] = default_ons_df()

    left, right = st.columns([2, 1])
    with right:
        st.subheader("Carga rápida")
        up = st.file_uploader("Importar ONs desde CSV", type=["csv"])
        if up is not None:
            try:
                du = pd.read_csv(up)
                needed = {"Ticker", "Calificacion", "Lamina_Minima"}
                if not needed.issubset(du.columns):
                    st.error("El CSV debe tener columnas: Ticker, Calificacion, Lamina_Minima")
                else:
                    du["Lamina_Minima"] = pd.to_numeric(du["Lamina_Minima"], errors="coerce")
                    st.session_state["ons_table"] = du[list(needed)].copy()
                    st.success("ONs importadas.")
            except Exception as e:
                st.error(f"No se pudo leer el CSV: {e}")
        if st.button("Restaurar ONs precargadas"):
            st.session_state["ons_table"] = default_ons_df()

    with left:
        st.subheader("Universo de ONs")
        edited = st.data_editor(
            st.session_state["ons_table"], num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", required=True),
                "Calificacion": st.column_config.TextColumn("Calificación"),
                "Lamina_Minima": st.column_config.NumberColumn("Lámina mínima", min_value=0.0, step=1.0),
            },
            key="ons_editor",
        )
        st.session_state["ons_table"] = edited.copy()

    df_ons = build_ons_table(latest_df, st.session_state["ons_table"])
    df_ons["Lamina_Minima"] = pd.to_numeric(df_ons.get("Lamina_Minima"), errors="coerce")
    df_ons_f = df_ons[
        (pd.to_numeric(df_ons["Lamina_Minima"], errors="coerce") <= float(lamina_max))
        & (pd.to_numeric(df_ons["MD"], errors="coerce") <= float(md_max))
    ].copy()
    if rating.strip():
        df_ons_f = df_ons_f[df_ons_f["Calificacion"].astype(str) == rating.strip()]

    fig_ons, params_ons = (None, {})
    if not df_ons_f.empty:
        fig_ons, params_ons = plot_curve_pro(df_ons_f, "Curva de ONs — TIR vs MD (fit Nelson-Siegel, banda ±50 bps)",
                                             fecha=most_recent_date)
        df_ons_f = cheap_rich(df_ons_f, params_ons.get("global"))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Instrumentos", len(df_ons_f))
        c2.metric("TIR promedio", f"{df_ons_f['TIR'].mean():.2f}%")
        c3.metric("MD promedio", f"{df_ons_f['MD'].mean():.2f}")
        best = df_ons_f.loc[df_ons_f["Residuo_bps"].idxmax()] if df_ons_f["Residuo_bps"].notna().any() else None
        c4.metric("Más barato vs curva", best["Ticker"] if best is not None else "—",
                  f"+{best['Residuo_bps']:.0f} bps" if best is not None else "")

        if fig_ons is not None:
            st.plotly_chart(fig_ons, use_container_width=True)

        st.subheader("Valor relativo (cheap / rich vs curva NS)")
        cols_show = [c for c in ["Ticker", "TIR", "MD", "Convexidad", "Paridad", "Calificacion",
                                 "Lamina_Minima", "TIR_Curva", "Residuo_bps", "Señal"] if c in df_ons_f.columns]
        st.dataframe(df_ons_f[cols_show].sort_values("Residuo_bps", ascending=False),
                     use_container_width=True, height=380)
        st.caption("Residuo = TIR observada − TIR teórica de la curva. Positivo (>50 bps): rinde por encima de "
                   "sus comparables de igual duration → candidato *barato*. Negativo: *caro*. "
                   "Validar siempre contra liquidez, calificación y riesgo emisor puntual.")
    else:
        st.info("No hay ONs que pasen los filtros actuales.")

# ---------------------------------------------------------------------
# TAB 2 — Bonos HD
# ---------------------------------------------------------------------
with tab_bonos:
    all_bonds = BONOS_GD + BONOS_AL + BOPREAL
    sel = st.multiselect("Seleccionar bonos", options=all_bonds, default=all_bonds)

    lb = latest_df[latest_df["Ticker"].isin(sel)].copy()
    lb["Grupo"] = lb["Ticker"].apply(classify_bono)

    if not lb.empty:
        fig_b, params_b = plot_curve_pro(
            lb, "Soberanos Hard Dollar — TIR vs MD por legislación", fecha=most_recent_date,
            group_col="Grupo", group_colors={"GD": COLORS["gd"], "AL": COLORS["al"], "BOPREAL": COLORS["bopreal"]},
            fit_per_group=True,
        )
        if fig_b is not None:
            st.plotly_chart(fig_b, use_container_width=True)

        # Ratio TIR/MD como proxy de carry por unidad de riesgo tasa
        lb["TIR/MD"] = (lb["TIR"] / lb["MD"]).round(2)
        st.subheader("Snapshot (fecha más reciente)")
        st.dataframe(lb.sort_values(["Grupo", "Ticker"]), use_container_width=True, height=320)

    st.subheader("Spread por legislación: AL/AE − GD")
    spreads = spread_series(df_norm)
    stats = spread_stats(spreads)
    if not stats.empty:
        st.dataframe(stats, use_container_width=True, height=240)
        st.caption("Z-score sobre 252 ruedas. Un spread muy por encima de su media histórica (z ≥ 1.5) indica que la ley "
                   "local está inusualmente castigada: si se espera compresión, el bono AL captura ese exceso de rendimiento. "
                   "Un z ≤ −1.5 sugiere que el premio por ley NY está barato en términos relativos.")
        fig_sp = plot_spread_history(spreads)
        if fig_sp is not None:
            st.plotly_chart(fig_sp, use_container_width=True)
    else:
        st.info("No hay pares AL/GD suficientes para calcular spreads históricos.")

# ---------------------------------------------------------------------
# TAB 3 — Breakevens
# ---------------------------------------------------------------------
with tab_be:
    st.markdown(
        "El **breakeven** es la inflación (o devaluación) anual que iguala el rendimiento de un bono CER "
        "(o Dollar Linked) con el de un bono a **tasa fija** de igual duration: "
        "`BE = (1 + TIR_fija) / (1 + TIR_real) − 1`. "
        "Si esperás inflación **mayor** al breakeven, conviene CER; si esperás **menor**, conviene tasa fija. "
        "Análogo para DL con la devaluación."
    )

    try:
        with st.spinner("Descargando renta fija soberana en pesos..."):
            raw_sob = download_dataset(api_key.strip(), int(ds_soberanos))
        df_sob = normalize_dataset(raw_sob)
        latest_sob, fecha_sob = latest_snapshot(df_sob)
    except Exception as e:
        latest_sob, fecha_sob = pd.DataFrame(), None
        st.error(f"No se pudo descargar el dataset {ds_soberanos}: {e}")

    if not latest_sob.empty and {"Ticker", "TIR", "MD"}.issubset(latest_sob.columns):
        with st.expander("Ajustar clasificación CER / Fija / DL (regex y selección manual)"):
            rx_cer = st.text_input("Regex CER", value=REGEX_CER)
            rx_fija = st.text_input("Regex Tasa fija", value=REGEX_FIJA)
            rx_dl = st.text_input("Regex Dollar Linked", value=REGEX_DL)

        latest_sob["Clase"] = latest_sob["Ticker"].apply(lambda t: classify_soberano(t, rx_cer, rx_fija, rx_dl))

        with st.expander("Revisar / corregir universo detectado"):
            for k in ["CER", "Fija", "DL"]:
                auto = sorted(latest_sob.loc[latest_sob["Clase"] == k, "Ticker"].astype(str).unique())
                pick = st.multiselect(f"Bonos {k}", options=sorted(latest_sob["Ticker"].astype(str).unique()),
                                      default=auto, key=f"pick_{k}")
                latest_sob.loc[latest_sob["Ticker"].astype(str).isin(pick), "Clase"] = k

        counts = latest_sob["Clase"].value_counts()
        st.caption(f"Detectados → CER: {counts.get('CER', 0)} · Fija: {counts.get('Fija', 0)} · DL: {counts.get('DL', 0)}")

        # --- REM ---
        rem_val, rem_col = None, None
        try:
            with st.spinner("Descargando REM (BCRA)..."):
                raw_rem = download_dataset(api_key.strip(), int(ds_rem))
            rem_val, rem_col = extract_rem_inflacion(raw_rem)
            with st.expander("Ver REM crudo / elegir otra columna"):
                st.dataframe(raw_rem.tail(15), use_container_width=True)
                manual_col = st.selectbox("Columna de expectativa a usar",
                                          options=["(auto)"] + [c for c in raw_rem.columns if c != "Date"])
                if manual_col != "(auto)":
                    s = pd.to_numeric(raw_rem[manual_col], errors="coerce").dropna()
                    if not s.empty:
                        rem_val, rem_col = float(s.iloc[-1]), manual_col
        except Exception as e:
            st.warning(f"No se pudo procesar el REM (dataset {ds_rem}): {e}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔥 Breakeven de inflación (CER vs fija)")
            be_cer = compute_breakevens(latest_sob, kind="CER")
            if not be_cer.empty:
                fig = plot_breakeven(be_cer, "Inflación breakeven implícita por plazo", COLORS["cer"],
                                     ref_value=rem_val,
                                     ref_label=f"REM: {rem_val:.1f}%" if rem_val is not None else None)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(be_cer, use_container_width=True, height=260)
                if rem_val is not None:
                    corto = be_cer[be_cer["MD"] <= 1.5]
                    if not corto.empty:
                        be_prom = corto["Breakeven_%anual"].mean()
                        gap = be_prom - rem_val
                        if gap > 3:
                            st.success(f"**Insight:** el mercado descuenta {be_prom:.1f}% de inflación en el tramo corto, "
                                       f"{gap:+.1f} pp por **encima** del REM ({rem_val:.1f}%). Si el REM acierta, "
                                       f"la **tasa fija** ofrece mejor retorno esperado que CER en ese tramo.")
                        elif gap < -3:
                            st.success(f"**Insight:** breakeven corto ({be_prom:.1f}%) {abs(gap):.1f} pp por **debajo** del REM "
                                       f"({rem_val:.1f}%). El CER luce **barato**: si la inflación converge al REM, "
                                       f"CER supera a tasa fija.")
                        else:
                            st.info(f"Breakeven corto ({be_prom:.1f}%) alineado con el REM ({rem_val:.1f}%): "
                                    f"el mercado y los analistas están pricing similar. Decisión por perfil de riesgo/liquidez.")
            else:
                st.info("No hay suficientes bonos CER y tasa fija clasificados para calcular breakevens.")

        with col_b:
            st.subheader("💱 Breakeven de devaluación (DL vs fija)")
            be_dl = compute_breakevens(latest_sob, kind="DL")

            # Futuros ROFEX como referencia cruzada
            fut_ref = None
            try:
                with st.spinner("Descargando futuros de dólar..."):
                    raw_fut = download_dataset(api_key.strip(), int(ds_fut))
                with st.expander("Futuros MATBA-ROFEX (crudo) + devaluación implícita"):
                    st.dataframe(raw_fut.tail(20), use_container_width=True)
                    spot = st.number_input("Spot de referencia (A3500 / mayorista)", value=0.0, step=1.0,
                                           help="Ingresá el spot para calcular devaluación implícita de cada posición.")
                    num_cols = [c for c in raw_fut.columns
                                if c != "Date" and pd.to_numeric(raw_fut[c], errors="coerce").notna().any()]
                    fut_cols = st.multiselect("Columnas de posiciones de futuros", options=num_cols)
                    if spot > 0 and fut_cols:
                        last = raw_fut.sort_values("Date").iloc[-1] if "Date" in raw_fut.columns else raw_fut.iloc[-1]
                        rows = []
                        for c in fut_cols:
                            px = pd.to_numeric(pd.Series([last[c]]), errors="coerce").iloc[0]
                            if pd.notna(px):
                                rows.append({"Posición": c, "Precio": round(float(px), 2),
                                             "Devaluación_directa_%": round((float(px) / spot - 1) * 100, 2)})
                        if rows:
                            df_fut = pd.DataFrame(rows)
                            st.dataframe(df_fut, use_container_width=True)
                            fut_ref = df_fut["Devaluación_directa_%"].iloc[-1]
            except Exception as e:
                st.warning(f"No se pudo procesar futuros (dataset {ds_fut}): {e}")

            if not be_dl.empty:
                fig = plot_breakeven(be_dl, "Devaluación breakeven implícita por plazo", COLORS["dl"],
                                     ref_value=fut_ref,
                                     ref_label=f"ROFEX (última posición): {fut_ref:.1f}%" if fut_ref is not None else None)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(be_dl, use_container_width=True, height=260)
                st.caption("Si el breakeven de devaluación de la curva DL difiere mucho de lo que descuenta ROFEX "
                           "para plazos comparables, hay una inconsistencia explotable entre ambos mercados "
                           "(sujeta a costos de rolleo, garantías y liquidez).")
            else:
                st.info("No hay suficientes bonos DL y tasa fija clasificados para calcular breakevens.")

        # Curvas en pesos superpuestas
        st.subheader("Curvas en pesos: Fija vs CER vs DL")
        d3 = latest_sob[latest_sob["Clase"].isin(["CER", "Fija", "DL"])].copy()
        if not d3.empty:
            fig3, _ = plot_curve_pro(d3, "TIR vs MD por clase (fit NS por grupo)", fecha=fecha_sob,
                                     group_col="Clase",
                                     group_colors={"CER": COLORS["cer"], "Fija": COLORS["fija"], "DL": COLORS["dl"]},
                                     fit_per_group=True)
            if fig3 is not None:
                st.plotly_chart(fig3, use_container_width=True)
            st.caption("Ojo: la curva CER está en tasa **real** y la fija en tasa **nominal** — la brecha vertical "
                       "entre ambas es, justamente, la inflación breakeven.")
    else:
        st.info("El dataset de soberanos no tiene la estructura esperada (Ticker/TIR/MD). "
                "Revisá el ID o los nombres de columnas.")

# ---------------------------------------------------------------------
# TAB 4 — Insights automáticos
# ---------------------------------------------------------------------
with tab_insights:
    st.subheader("🎯 Resumen ejecutivo para el asesor")
    bullets = []

    # Forma de la curva HD (usa fit global de GD)
    lb_all = latest_df[latest_df["Ticker"].isin(BONOS_GD)].dropna(subset=["MD", "TIR"])
    if len(lb_all) >= 3:
        corto = lb_all.nsmallest(2, "MD")["TIR"].mean()
        largo = lb_all.nlargest(2, "MD")["TIR"].mean()
        pend = largo - corto
        forma = "invertida (el mercado exige más tasa en el tramo corto → estrés de corto plazo)" if pend < -0.5 \
            else ("empinada (premio por extender duration)" if pend > 0.5 else "plana")
        bullets.append(f"**Curva GD {forma}.** Tramo corto ≈ {corto:.1f}%, tramo largo ≈ {largo:.1f}% "
                       f"(pendiente {pend:+.1f} pp).")

    # Spread ley
    try:
        stats_i = spread_stats(spread_series(df_norm))
        if not stats_i.empty:
            ext = stats_i.loc[stats_i["Z_Score"].abs().idxmax()]
            if abs(ext["Z_Score"]) >= 1.5:
                bullets.append(f"**Spread por legislación 20{int(ext['Numero'])} en extremo histórico** "
                               f"(z = {ext['Z_Score']:+.2f}; actual {ext['Spread_Actual']:.2f} pp vs promedio "
                               f"{ext['Spread_Prom_252']:.2f} pp). {ext['Señal']}.")
            else:
                bullets.append("Spreads AL/GD dentro de rangos normales (|z| < 1.5 en todos los pares): "
                               "sin señal fuerte de arbitraje por legislación.")
    except Exception:
        pass

    # ONs cheap/rich
    try:
        if "df_ons_f" in dir() and isinstance(df_ons_f, pd.DataFrame) and "Residuo_bps" in df_ons_f.columns \
                and df_ons_f["Residuo_bps"].notna().any():
            top = df_ons_f.nlargest(3, "Residuo_bps")
            names = ", ".join(f"{r.Ticker} (+{r.Residuo_bps:.0f} bps)" for r in top.itertuples())
            bullets.append(f"**ONs con mayor exceso de TIR vs curva:** {names}. "
                           f"Chequear liquidez y riesgo emisor antes de armar posición.")
    except Exception:
        pass

    if bullets:
        for b in bullets:
            st.markdown(f"- {b}")
    else:
        st.info("Cargá datos en las otras pestañas para generar insights.")

    st.divider()
    st.markdown(
        """
        **Cómo leer este panel (guía rápida):**
        - *Cheap/rich*: un bono ±50 bps fuera de la curva NS no es automáticamente compra/venta;
          suele reflejar liquidez, lámina mínima, o riesgo idiosincrático. Es un **filtro**, no una orden.
        - *Spread AL−GD*: mide el premio que paga la ley local. Su z-score histórico indica si ese premio
          está caro o barato **en términos relativos**, no si el país mejora o empeora.
        - *Breakeven CER*: es la inflación que "empata" fija vs CER. Compararlo contra el REM te dice
          qué está pricing el mercado vs qué esperan los analistas — el gap es la oportunidad (o el riesgo).
        - *Breakeven DL vs ROFEX*: dos mercados que descuentan devaluación; cuando divergen para el mismo
          plazo, uno de los dos está "equivocado" (o hay fricciones de acceso/garantías que lo explican).
        """
    )
