import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import urllib.parse
from datetime import date

# =========================
# V30 + FORECAST (JAN/FEV/MAR) + RANKING
# =========================

ARQUIVO_EXCEL = "PROD-PRODT.xlsx"
LOGO = "logo.png"
SENHA_APP = "neworder2026"  # <- troque aqui se quiser

# ====== TEMA ======
BG = "#000000"
BG2 = "#070707"
WHITE = "#ffffff"
GRID = "#2a2a2a"
BORDER = "#1a2a1a"

ORANGE_BAR_DARK = "#e56a00"
ORANGE_BAR_LIGHT = "#ffb36b"
ORANGE_LINE = "#ffd0a3"

GREEN = "#00c853"
RED = "#ff1744"
DIFF_LINE = "#ffffff"

st.set_page_config(page_title="Dashboard", layout="wide")

# ====== CSS 100% preto ======
st.markdown(
    f"""
    <style>
      .stApp {{ background:{BG}; color:{WHITE}; }}
      section[data-testid="stSidebar"] {{ background:{BG}; }}
      section[data-testid="stSidebar"] * {{ color:{WHITE} !important; }}
      .stTextInput input, .stDateInput input, .stSelectbox div, .stMultiSelect div {{
        background:#111111 !important; color:{WHITE} !important;
      }}
      .stPlotlyChart > div {{ background:{BG} !important; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOGIN SIMPLES (sem secrets.toml)
# =========================
def login():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        logo_path = Path(__file__).parent / LOGO
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)

        st.caption("Acesso restrito")
        senha = st.text_input("Senha", type="password")

        if senha == SENHA_APP:
            st.session_state["auth"] = True
            st.rerun()
        elif senha:
            st.error("Senha incorreta")

if not st.session_state.get("auth", False):
    login()
    st.stop()

# =========================
# UTILIDADES
# =========================
def to_number(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"nan": "", "None": ""})
    s = s.str.replace("\u00a0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)

    # "1.234,56" -> "1234.56"
    mask = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False)
    s.loc[mask] = s.loc[mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    # "123,45" -> "123.45"
    mask2 = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    s.loc[mask2] = s.loc[mask2].str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")

def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def month_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%m/%Y")

def business_days_in_month(ts: pd.Timestamp) -> int:
    """Dias úteis seg-sex. (Não considera feriados)."""
    start = pd.Timestamp(year=ts.year, month=ts.month, day=1)
    end = (start + pd.offsets.MonthEnd(1))
    return len(pd.bdate_range(start, end))

def style_dark(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        margin=dict(l=10, r=60, t=55, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0),
    )
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig

def add_termometro(fig: go.Figure, height_px: int):
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="{height_px}" viewBox="0 0 80 {height_px}">
      <defs>
        <linearGradient id="g" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0%" stop-color="#001a33"/>
          <stop offset="100%" stop-color="#00bfff"/>
        </linearGradient>
      </defs>
      <rect x="30" y="60" width="18" height="{max(200, height_px-140)}" rx="9"
            fill="url(#g)" stroke="#0aa" stroke-width="1"/>
      <polygon points="39,10 62,60 16,60" fill="#00bfff" stroke="#0aa" stroke-width="1"/>
      <text x="72" y="{int(height_px/2)}" fill="white" font-size="14" font-family="Arial" font-weight="700"
            text-anchor="middle" transform="rotate(-90 72 {int(height_px/2)})">MELHOR</text>
    </svg>
    """.strip()

    src = "data:image/svg+xml;utf8," + urllib.parse.quote(svg)
    fig.add_layout_image(
        dict(
            source=src,
            xref="paper", yref="paper",
            x=1.02, y=0.5,
            sizex=0.10, sizey=1.05,
            xanchor="left", yanchor="middle",
            layer="above"
        )
    )

def add_gradient_bars(fig: go.Figure, x, y, name="Produção"):
    fig.add_trace(go.Bar(
        x=x, y=y, name=name,
        marker=dict(color=ORANGE_BAR_DARK, line=dict(color="#000000", width=0.3)),
        opacity=0.95
    ))
    y2 = [float(v) * 0.75 if pd.notna(v) else 0 for v in y]
    fig.add_trace(go.Bar(
        x=x, y=y2, name="",
        marker=dict(color=ORANGE_BAR_LIGHT, line=dict(color="#000000", width=0.0)),
        opacity=0.55,
        hoverinfo="skip",
        showlegend=False
    ))

def render_minitabela_html(dias, meta_list, real_list, dif_list, decimals=0):
    meta_s = pd.Series(meta_list, dtype="float").fillna(0)
    real_s = pd.Series(real_list, dtype="float").fillna(0)
    dif_s  = pd.Series(dif_list,  dtype="float").fillna(0)

    meta_media = float(meta_s.mean()) if len(meta_s) else 0.0
    real_media = float(real_s.mean()) if len(real_s) else 0.0
    dif_media  = float(dif_s.mean())  if len(dif_s)  else 0.0

    def fmt(v):
        if decimals == 0:
            return f"{int(round(float(v), 0))}"
        return f"{round(float(v), decimals):.{decimals}f}"

    cols = dias + ["MÉDIA"]

    rows = [
        ("Meta",      [fmt(v) for v in meta_s.tolist()] + [fmt(meta_media)]),
        ("Realizado", [fmt(v) for v in real_s.tolist()] + [fmt(real_media)]),
        ("Diferença", [fmt(v) for v in dif_s.tolist()]  + [fmt(dif_media)]),
    ]

    css = f"""
    <style>
      .mini-wrap {{ background:{BG}; padding:6px 0; }}
      table.mini {{
        width:100%;
        border-collapse:collapse;
        background:{BG};
        color:{WHITE};
        font-size:12px;
      }}
      table.mini th, table.mini td {{
        border:0.5px solid {BORDER};
        padding:6px 6px;
        text-align:center;
        background:{BG};
        color:{WHITE};
        white-space:nowrap;
      }}
      table.mini thead th {{
        background:{BG};
        color:{WHITE};
        font-weight:700;
      }}
      table.mini tbody th {{
        background:{BG2};
        color:{WHITE};
        font-weight:700;
        text-align:left;
        padding-left:10px;
      }}
    </style>
    """

    thead = "<thead><tr><th></th>" + "".join([f"<th>{c}</th>" for c in cols]) + "</tr></thead>"

    body_rows = []
    for label, values in rows:
        tds = []
        for v in values:
            if label == "Diferença":
                try:
                    vv = float(str(v).replace(",", "."))
                except:
                    vv = 0.0
                bgc = GREEN if vv >= 0 else RED
                tds.append(f"<td style='background:{bgc}; color:{WHITE}; font-weight:700;'>{v}</td>")
            else:
                tds.append(f"<td>{v}</td>")
        body_rows.append(f"<tr><th>{label}</th>{''.join(tds)}</tr>")

    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    html = f"{css}<div class='mini-wrap'><table class='mini'>{thead}{tbody}</table></div>"
    st.markdown(html, unsafe_allow_html=True)

# =========================
# CARREGAMENTO EXCEL (auto-atualiza quando o arquivo muda)
# =========================
@st.cache_data(show_spinner=False)
def load_data(mtime_key: float) -> pd.DataFrame:
    path = Path(__file__).parent / ARQUIVO_EXCEL
    return pd.read_excel(path)

excel_path = Path(__file__).parent / ARQUIVO_EXCEL
if not excel_path.exists():
    st.error(f"Não encontrei '{ARQUIVO_EXCEL}' na pasta do app.")
    st.stop()

mtime = excel_path.stat().st_mtime
df = load_data(mtime)
df.columns = [str(c).strip() for c in df.columns]

# =========================
# VALIDAR COLUNAS
# =========================
required = ["DATA", "MÊS", "LINHA", "META", "PRODUÇÃO", "EFETIVO", "PRODT."]
faltando = [c for c in required if c not in df.columns]
if faltando:
    st.error(f"Faltam colunas no Excel: {faltando}")
    st.write("Colunas encontradas:", list(df.columns))
    st.stop()

df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
df = df.dropna(subset=["DATA"])
df["MÊS"] = df["MÊS"].astype(str).str.strip().str.upper()
df["LINHA"] = df["LINHA"].astype(str).str.strip()

df["META"] = to_number(df["META"])
df["PRODUÇÃO"] = to_number(df["PRODUÇÃO"])
df["EFETIVO"] = to_number(df["EFETIVO"])
df["PRODT."] = to_number(df["PRODT."])
if "M. PRODT" in df.columns:
    df["M. PRODT"] = to_number(df["M. PRODT"])

# =========================
# TOPO (só logo)
# =========================
logo_path = Path(__file__).parent / LOGO
if logo_path.exists():
    st.image(str(logo_path), width=170)

# =========================
# FILTROS (produçao/produtividade do mês)
# =========================
st.sidebar.header("Filtros — Operação")

meses = sorted(df["MÊS"].dropna().unique().tolist())
mes_sel = st.sidebar.selectbox("Mês", meses, index=meses.index("JANEIRO") if "JANEIRO" in meses else 0)

df_m = df[df["MÊS"] == mes_sel].copy()
if df_m.empty:
    st.warning("Sem dados para o mês selecionado.")
    st.stop()

linhas = sorted(df_m["LINHA"].dropna().unique().tolist())
linha_sel = st.sidebar.multiselect("Linha", linhas, default=linhas)

df_m = df_m[df_m["LINHA"].isin(linha_sel)].copy()
if df_m.empty:
    st.warning("Sem dados para as linhas selecionadas.")
    st.stop()

data_min = df_m["DATA"].min().date()
data_max = df_m["DATA"].max().date()

periodo = st.sidebar.date_input("Período", value=(data_min, data_max))
if isinstance(periodo, tuple) and len(periodo) == 2:
    data_ini, data_fim = periodo
else:
    data_ini = data_min
    data_fim = data_max

df_f = df_m[
    (df_m["DATA"] >= pd.to_datetime(data_ini)) &
    (df_m["DATA"] <= pd.to_datetime(data_fim))
].copy()

df_f = df_f.sort_values("DATA")
if df_f.empty:
    st.warning("Sem dados para o período selecionado.")
    st.stop()

st.sidebar.divider()
mostrar_minitabela = st.sidebar.checkbox("Mostrar mini tabela", value=True)
mostrar_termometro = st.sidebar.checkbox("Mostrar termômetro (MELHOR)", value=True)

# =========================
# AGREGAÇÃO POR DIA (OPERAÇÃO)
# =========================
df_f["DIA_ORD"] = df_f["DATA"].dt.normalize()

# Produção: soma
agg_prod = (
    df_f.groupby("DIA_ORD", as_index=False)
        .agg({"META": "sum", "PRODUÇÃO": "sum"})
        .sort_values("DIA_ORD")
)

# Produtividade: ponderada por efetivo (Produção_total / Efetivo_total)
def _agg_prodt_day(g: pd.DataFrame) -> pd.Series:
    prod_total = pd.to_numeric(g["PRODUÇÃO"], errors="coerce").fillna(0).sum()
    efet_total = pd.to_numeric(g["EFETIVO"], errors="coerce").fillna(0).sum()
    prodt_real = (prod_total / efet_total) if efet_total > 0 else None

    prodt_meta = None
    if "M. PRODT" in g.columns:
        m = pd.to_numeric(g["M. PRODT"], errors="coerce")
        w = pd.to_numeric(g["EFETIVO"], errors="coerce").fillna(0)
        den = w[m.notna()].sum()
        num = (m * w).sum(skipna=True)
        prodt_meta = (num / den) if den and den > 0 else None

    return pd.Series({"PRODT_REAL": prodt_real, "PRODT_META": prodt_meta})

agg_prodt = (
    df_f.groupby("DIA_ORD")
        .apply(_agg_prodt_day)
        .reset_index()
        .sort_values("DIA_ORD")
)

agg = agg_prod.merge(agg_prodt, on="DIA_ORD", how="left")

agg["DIA_TXT"] = pd.to_datetime(agg["DIA_ORD"]).dt.strftime("%d/%m")
x_order = agg["DIA_TXT"].tolist()

agg["DIF_PROD"] = (
    pd.to_numeric(agg["PRODUÇÃO"], errors="coerce").fillna(0)
    - pd.to_numeric(agg["META"], errors="coerce").fillna(0)
)

if agg["PRODT_META"].notna().any():
    agg["META_PRODT"] = agg["PRODT_META"]
else:
    agg["META_PRODT"] = pd.to_numeric(df_f["PRODT."], errors="coerce").mean()

agg["DIF_PRODT"] = (
    pd.to_numeric(agg["PRODT_REAL"], errors="coerce")
    - pd.to_numeric(agg["META_PRODT"], errors="coerce")
).fillna(0)

# =========================
# 1) PRODUÇÃO (MÊS)
# =========================
st.markdown(f"## PRODUÇÃO — {mes_sel}")

fig_prod = go.Figure()
add_gradient_bars(fig_prod, agg["DIA_TXT"], agg["PRODUÇÃO"], name="Produção")

fig_prod.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["META"],
    name="Meta",
    mode="lines+markers",
    line=dict(color=ORANGE_LINE, width=1.2),
    marker=dict(color=ORANGE_LINE, size=5)
))

diff_colors = [GREEN if v >= 0 else RED for v in agg["DIF_PROD"].tolist()]
fig_prod.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["DIF_PROD"],
    name="Real − Meta",
    mode="lines+markers",
    line=dict(color=DIFF_LINE, width=1.2, dash="dash"),
    marker=dict(color=diff_colors, size=7),
    yaxis="y2"
))

dmin = float(pd.to_numeric(agg["DIF_PROD"], errors="coerce").min())
dmax = float(pd.to_numeric(agg["DIF_PROD"], errors="coerce").max())
pad = max(10.0, (dmax - dmin) * 0.15) if (dmax - dmin) != 0 else 50.0

fig_prod.update_layout(
    barmode="overlay",
    height=520,
    xaxis=dict(
        type="category",
        categoryorder="array",
        categoryarray=x_order,
        tickmode="array",
        tickvals=x_order,
        ticktext=x_order,
        showgrid=False
    ),
    yaxis2=dict(
        title="Diferença",
        overlaying="y",
        side="right",
        showgrid=False,
        range=[dmin - pad, dmax + pad]
    )
)

if mostrar_termometro:
    add_termometro(fig_prod, 520)

st.plotly_chart(style_dark(fig_prod), use_container_width=True)

if mostrar_minitabela:
    dias = agg["DIA_TXT"].tolist()
    render_minitabela_html(
        dias,
        pd.Series(agg["META"]).fillna(0).tolist(),
        pd.Series(agg["PRODUÇÃO"]).fillna(0).tolist(),
        pd.Series(agg["DIF_PROD"]).fillna(0).tolist(),
        decimals=0
    )

st.divider()

# =========================
# 2) PRODUTIVIDADE (MÊS)
# =========================
st.markdown(f"## PRODUTIVIDADE — {mes_sel}")

fig_prodt = go.Figure()
fig_prodt.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["PRODT_REAL"],
    name="Produtividade (Real)",
    mode="lines+markers",
    line=dict(color=ORANGE_BAR_LIGHT, width=1.2),
    marker=dict(color=ORANGE_BAR_LIGHT, size=5),
))
fig_prodt.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["META_PRODT"],
    name="Meta Produtividade",
    mode="lines",
    line=dict(color=ORANGE_LINE, width=1.2, dash="dot"),
))

diff_colors2 = [GREEN if v >= 0 else RED for v in agg["DIF_PRODT"].tolist()]
fig_prodt.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["DIF_PRODT"],
    name="Real − Meta (Prodt.)",
    mode="lines+markers",
    line=dict(color=DIFF_LINE, width=1.2, dash="dash"),
    marker=dict(color=diff_colors2, size=7),
    yaxis="y2"
))

dmin2 = float(pd.to_numeric(agg["DIF_PRODT"], errors="coerce").min())
dmax2 = float(pd.to_numeric(agg["DIF_PRODT"], errors="coerce").max())
pad2 = max(0.1, (dmax2 - dmin2) * 0.15) if (dmax2 - dmin2) != 0 else 1.0

fig_prodt.update_layout(
    height=460,
    xaxis=dict(
        type="category",
        categoryorder="array",
        categoryarray=x_order,
        tickmode="array",
        tickvals=x_order,
        ticktext=x_order,
        showgrid=False
    ),
    yaxis2=dict(
        title="Diferença",
        overlaying="y",
        side="right",
        showgrid=False,
        range=[dmin2 - pad2, dmax2 + pad2]
    )
)

if mostrar_termometro:
    add_termometro(fig_prodt, 460)

st.plotly_chart(style_dark(fig_prodt), use_container_width=True)

if mostrar_minitabela:
    dias = agg["DIA_TXT"].tolist()
    render_minitabela_html(
        dias,
        pd.Series(agg["META_PRODT"]).fillna(0).tolist(),
        pd.Series(agg["PRODT_REAL"]).fillna(0).tolist(),
        pd.Series(agg["DIF_PRODT"]).fillna(0).tolist(),
        decimals=2
    )

st.divider()

# =========================
# 3) ANÁLISE & TENDÊNCIA (MESES SEGUINTES / DIAS ÚTEIS)
# =========================
st.sidebar.header("Análise — Tendência (meses seguintes)")
meses_base = st.sidebar.slider("Meses históricos para calcular tendência", 1, 12, 3)
meses_a_frente = st.sidebar.slider("Projetar quantos meses à frente", 1, 12, 3)

st.markdown("## ANÁLISE & TENDÊNCIA — Próximos meses (Dias úteis)")
st.caption("Projeção por Linha usando dias úteis (seg–sex). ⚠️ Não considera feriados; se quiser feriados BR, eu adapto.")

df_t = df[df["LINHA"].isin(linha_sel)].copy()
df_t["MES_START"] = df_t["DATA"].apply(lambda x: month_start(pd.Timestamp(x)))

monthly = (
    df_t.groupby(["LINHA", "MES_START"], as_index=False)
        .agg({"PRODUÇÃO": "sum", "EFETIVO": "sum"})
)

monthly["PRODT_REAL_MES"] = np.where(monthly["EFETIVO"] > 0, monthly["PRODUÇÃO"] / monthly["EFETIVO"], np.nan)
monthly["DIAS_UTEIS"] = monthly["MES_START"].apply(business_days_in_month)
monthly["PROD_POR_DU"] = np.where(monthly["DIAS_UTEIS"] > 0, monthly["PRODUÇÃO"] / monthly["DIAS_UTEIS"], np.nan)
monthly["EFET_POR_DU"] = np.where(monthly["DIAS_UTEIS"] > 0, monthly["EFETIVO"] / monthly["DIAS_UTEIS"], np.nan)

all_months_sorted = sorted(monthly["MES_START"].dropna().unique().tolist())
if len(all_months_sorted) == 0:
    st.warning("Sem meses suficientes para tendência.")
    st.stop()

base_months = all_months_sorted[-meses_base:] if len(all_months_sorted) >= meses_base else all_months_sorted
base = monthly[monthly["MES_START"].isin(base_months)].copy()

trend = (
    base.groupby("LINHA", as_index=False)
        .agg({"PROD_POR_DU": "mean", "EFET_POR_DU": "mean", "PRODT_REAL_MES": "mean"})
).rename(columns={
    "PROD_POR_DU": "TEND_PROD_POR_DU",
    "EFET_POR_DU": "TEND_EFET_POR_DU",
    "PRODT_REAL_MES": "TEND_PRODT"
})

last_month = all_months_sorted[-1]
future_months = [(last_month + pd.offsets.MonthBegin(i)) for i in range(1, meses_a_frente + 1)]
future = pd.DataFrame({"MES_START": future_months})
future["DIAS_UTEIS"] = future["MES_START"].apply(business_days_in_month)
future["MES_TXT"] = future["MES_START"].apply(month_label)

proj = future.merge(trend, how="cross")
proj["PROD_PROJ"] = proj["TEND_PROD_POR_DU"] * proj["DIAS_UTEIS"]
proj["EFET_PROJ"] = proj["TEND_EFET_POR_DU"] * proj["DIAS_UTEIS"]
proj["PRODT_PROJ"] = np.where(proj["EFET_PROJ"] > 0, proj["PROD_PROJ"] / proj["EFET_PROJ"], np.nan)

hist = monthly.copy()
hist["MES_TXT"] = hist["MES_START"].apply(month_label)
hist_last12 = hist[hist["MES_START"].isin(all_months_sorted[-12:])].copy()

st.markdown("### Tendência de Produção por Linha — Histórico mensal + Projeção")
linha_tend = st.selectbox("Escolha uma linha para análise", sorted(trend["LINHA"].unique().tolist()))
hist_l = hist_last12[hist_last12["LINHA"] == linha_tend].sort_values("MES_START")
proj_l = proj[proj["LINHA"] == linha_tend].sort_values("MES_START")

fig_t_prod = go.Figure()
add_gradient_bars(fig_t_prod, hist_l["MES_TXT"], hist_l["PRODUÇÃO"], name="Produção (Hist)")
fig_t_prod.add_trace(go.Bar(
    x=proj_l["MES_TXT"], y=proj_l["PROD_PROJ"],
    name="Produção (Proj)",
    marker=dict(color="#444444", line=dict(color="#000000", width=0.2)),
    opacity=0.85
))
fig_t_prod.update_layout(height=430, barmode="group", xaxis=dict(type="category", showgrid=False))
st.plotly_chart(style_dark(fig_t_prod), use_container_width=True)

st.markdown("### Tendência de Produtividade por Linha — Histórico mensal + Projeção")
fig_t_prodt = go.Figure()
fig_t_prodt.add_trace(go.Scatter(
    x=hist_l["MES_TXT"], y=hist_l["PRODT_REAL_MES"],
    name="Produtividade (Hist)",
    mode="lines+markers",
    line=dict(color=ORANGE_BAR_LIGHT, width=1.2),
    marker=dict(color=ORANGE_BAR_LIGHT, size=6),
))
fig_t_prodt.add_trace(go.Scatter(
    x=proj_l["MES_TXT"], y=proj_l["PRODT_PROJ"],
    name="Produtividade (Proj)",
    mode="lines+markers",
    line=dict(color="#aaaaaa", width=1.2, dash="dot"),
    marker=dict(color="#aaaaaa", size=6),
))
fig_t_prodt.update_layout(height=380, xaxis=dict(type="category", showgrid=False))
st.plotly_chart(style_dark(fig_t_prodt), use_container_width=True)

st.markdown("### Resumo da Projeção (por dias úteis)")
res = proj_l[["MES_TXT", "DIAS_UTEIS", "PROD_PROJ", "PRODT_PROJ"]].copy()
res["PROD_PROJ"] = res["PROD_PROJ"].round(0).astype("Int64")
res["PRODT_PROJ"] = res["PRODT_PROJ"].round(2)
st.dataframe(res, use_container_width=True)

# =====================================================================
# =====================================================================
# 4) BLOCO — FORECAST x PRODUÇÃO PROJETADA — JAN/FEV/MAR + RANKING (FIX)
# =====================================================================
st.divider()
st.markdown("## FORECAST x PRODUÇÃO PROJETADA — JAN / FEV / MAR")
st.caption("Comparativo por Linha: Forecast x Produção Projetada (tendência por dias úteis). Inclui ranking.")

def normalize_cols(cols):
    return [str(c).strip().upper() for c in cols]

def find_forecast_sheet(path: Path):
    """Procura uma aba provável de forecast no mesmo Excel (se existir)."""
    try:
        xls = pd.ExcelFile(path)
        for sh in xls.sheet_names:
            up = sh.strip().upper()
            if any(k in up for k in ["FORECAST", "FOR", "PREV", "PLANO", "VENDA"]):
                try:
                    tmp = pd.read_excel(path, sheet_name=sh)
                    cols = normalize_cols(tmp.columns)
                    if "LINHA" in cols and any("FOR" in c for c in cols):
                        return sh
                except:
                    pass
    except:
        pass
    return None

def load_forecast_from_df(df_fc: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """
    Padroniza o forecast para formato longo: LINHA, MES_START, FORECAST.
    Aceita colunas como: FOR. JAN / FOR. FEV / FOR. MAR etc.
    """
    df_fc = df_fc.copy()
    df_fc.columns = normalize_cols(df_fc.columns)

    if "LINHA" not in df_fc.columns:
        return pd.DataFrame()

    # mapa mês -> número
    month_map = {
        "JAN": 1, "JANEIRO": 1,
        "FEV": 2, "FEVEREIRO": 2,
        "MAR": 3, "MARCO": 3, "MARÇO": 3,
    }

    fc_cols = []
    for c in df_fc.columns:
        cu = str(c).upper().replace("Ç", "C")
        if "FOR" in cu:
            fc_cols.append(c)

    if not fc_cols:
        return pd.DataFrame()

    rows = []
    for _, r in df_fc.iterrows():
        linha = str(r.get("LINHA", "")).strip()
        if not linha:
            continue

        for c in fc_cols:
            cu = str(c).upper().replace("Ç", "C")
            tok = None
            for k in month_map.keys():
                if k in cu:
                    tok = k
                    break
            if not tok:
                continue

            mes_num = month_map[tok]
            mes_start = pd.Timestamp(year=base_year, month=mes_num, day=1)
            val = r.get(c, np.nan)
            val = to_number(pd.Series([val])).iloc[0]
            if pd.isna(val):
                continue

            rows.append({"LINHA": linha, "MES_START": mes_start, "FORECAST": float(val)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.groupby(["LINHA", "MES_START"], as_index=False)["FORECAST"].sum()
    return out

# --------------------------
# Carregar forecast (auto ou upload)
# --------------------------
st.sidebar.header("Forecast — Comparativo")
modo_fc = st.sidebar.radio("Fonte do Forecast", ["Procurar no mesmo Excel", "Enviar arquivo Forecast"], index=0)

df_forecast_long = pd.DataFrame()

# ano base = ano do último mês do seu histórico (dataset)
# (mantém coerente com seu banco)
base_year = int(pd.Timestamp(last_month).year)

if modo_fc == "Procurar no mesmo Excel":
    sh = find_forecast_sheet(excel_path)
    if sh:
        try:
            df_fc = pd.read_excel(excel_path, sheet_name=sh)
            df_forecast_long = load_forecast_from_df(df_fc, base_year=base_year)
            st.sidebar.success(f"Forecast encontrado na aba: {sh}")
        except:
            st.sidebar.error("Não consegui ler a aba de Forecast.")
    else:
        st.sidebar.warning("Não achei aba de Forecast no Excel. Selecione 'Enviar arquivo Forecast'.")
else:
    up = st.sidebar.file_uploader("Envie o Forecast (xlsx ou csv)", type=["xlsx", "csv"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df_fc = pd.read_csv(up)
            else:
                df_fc = pd.read_excel(up)
            df_forecast_long = load_forecast_from_df(df_fc, base_year=base_year)
            st.sidebar.success("Forecast carregado do arquivo enviado.")
        except:
            st.sidebar.error("Não consegui ler esse arquivo de Forecast.")

# --------------------------
# Montar comparação (FIX: projeção calculada do ZERO para meses do forecast)
# --------------------------
if df_forecast_long.empty:
    st.info("⚠️ Forecast não carregado. Esse bloco só aparece quando o forecast é encontrado/enviado.")
else:
    # filtra linhas selecionadas
    df_forecast_long["LINHA"] = df_forecast_long["LINHA"].astype(str).str.strip()
    df_forecast_long = df_forecast_long[df_forecast_long["LINHA"].isin(linha_sel)].copy()

    # meses disponíveis no forecast (normalmente JAN/FEV/MAR do ano base)
    meses_lbl = sorted(df_forecast_long["MES_START"].dropna().unique().tolist())
    meses_lbl_txt = [pd.Timestamp(m).strftime("%b/%Y").upper() for m in meses_lbl]

    mes_view_txt = st.selectbox("Visualizar mês", meses_lbl_txt, index=0 if meses_lbl_txt else 0)
    mes_view = meses_lbl[meses_lbl_txt.index(mes_view_txt)] if meses_lbl_txt else None

    # tendência por linha (já calculada no bloco 3) = TEND_PROD_POR_DU
    trend_use = trend.copy()
    trend_use["LINHA"] = trend_use["LINHA"].astype(str).str.strip()
    trend_use = trend_use[trend_use["LINHA"].isin(linha_sel)].copy()

    # calcula projeção para cada mês que existe no forecast:
    # PROJETADO = TEND_PROD_POR_DU * DIAS_UTEIS(mês)
    months_fc = pd.DataFrame({"MES_START": meses_lbl})
    months_fc["DIAS_UTEIS"] = months_fc["MES_START"].apply(business_days_in_month)

    proj_fc = months_fc.merge(trend_use, how="cross")
    proj_fc["PROJETADO"] = proj_fc["TEND_PROD_POR_DU"] * proj_fc["DIAS_UTEIS"]

    # juntar com forecast
    fc_month = df_forecast_long.groupby(["LINHA", "MES_START"], as_index=False)["FORECAST"].sum()
    comp = proj_fc.merge(fc_month, on=["LINHA", "MES_START"], how="left")

    comp["FORECAST"] = pd.to_numeric(comp["FORECAST"], errors="coerce").fillna(0)
    comp["PROJETADO"] = pd.to_numeric(comp["PROJETADO"], errors="coerce").fillna(0)

    comp["DIF"] = comp["PROJETADO"] - comp["FORECAST"]
    comp["DIF_%"] = np.where(comp["FORECAST"] != 0, (comp["DIF"] / comp["FORECAST"]) * 100, np.nan)
    comp["MES_TXT"] = pd.to_datetime(comp["MES_START"]).dt.strftime("%b/%Y").str.upper()

    # filtra mês selecionado
    comp_m = comp[comp["MES_START"] == pd.Timestamp(mes_view)].copy() if mes_view is not None else comp.copy()

    # ordena por forecast (ou projetado)
    comp_m = comp_m.sort_values("FORECAST", ascending=False)

    # =========================
    # GRÁFICO — Forecast x Projetado (por linha)
    # =========================
    st.markdown("### Comparativo por Linha — Forecast x Projetado")
    fig_fc = go.Figure()

    fig_fc.add_trace(go.Bar(
        x=comp_m["LINHA"],
        y=comp_m["FORECAST"],
        name="Forecast",
        marker=dict(color="#5a5a5a", line=dict(color="#000000", width=0.2)),
        opacity=0.9
    ))
    fig_fc.add_trace(go.Bar(
        x=comp_m["LINHA"],
        y=comp_m["PROJETADO"],
        name="Projetado",
        marker=dict(color=ORANGE_BAR_DARK, line=dict(color="#000000", width=0.2)),
        opacity=0.9
    ))

    fig_fc.update_layout(
        height=460,
        barmode="group",
        xaxis=dict(type="category", tickangle=-25, showgrid=False),
        yaxis=dict(title="Quantidade"),
    )
    st.plotly_chart(style_dark(fig_fc), use_container_width=True)

    # =========================
    # RANKING — Projetado − Forecast
    # =========================
    st.markdown("### Ranking (Projetado − Forecast)")

    rank = comp_m.copy()
    rank["DIF"] = rank["DIF"].round(0)
    rank["PROJETADO"] = rank["PROJETADO"].round(0)
    rank["FORECAST"] = rank["FORECAST"].round(0)
    rank["DIF_%"] = rank["DIF_%"].round(1)

    # maior diferença primeiro
    rank = rank.sort_values("DIF", ascending=False)

    def color_diff(v):
        try:
            v = float(v)
        except:
            return ""
        return f"color: {WHITE}; background-color: {GREEN if v >= 0 else RED}; font-weight:700;"

    show = rank[["LINHA", "FORECAST", "PROJETADO", "DIF", "DIF_%"]].rename(columns={
        "LINHA": "Linha",
        "FORECAST": "Forecast",
        "PROJETADO": "Projetado",
        "DIF": "Projetado − Forecast",
        "DIF_%": "Dif %"
    })

    st.dataframe(
        show.style.applymap(color_diff, subset=["Projetado − Forecast"]).format({
            "Forecast": "{:,.0f}".format,
            "Projetado": "{:,.0f}".format,
            "Projetado − Forecast": "{:,.0f}".format,
            "Dif %": lambda x: "" if pd.isna(x) else f"{x:.1f}%"
        }),
        use_container_width=True
    )

    st.markdown("#### Top 5 (melhor) e Top 5 (pior)")
    top5 = show.head(5)
    bot5 = show.tail(5).sort_values("Projetado − Forecast")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Top 5 — acima do forecast")
        st.dataframe(top5, use_container_width=True, height=220)
    with c2:
        st.caption("Top 5 — abaixo do forecast")
        st.dataframe(bot5, use_container_width=True, height=220)
