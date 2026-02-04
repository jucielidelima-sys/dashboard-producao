import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import urllib.parse

# =========================
# V34 — Completo + Ranking + DIF(PROD-FORE) + MiniTabela Compacta
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

# Produzido
ORANGE_BAR_DARK = "#d85d00"
ORANGE_BAR_LIGHT = "#ffb981"
ORANGE_LINE = "#ffd2a8"

# Forecast/Faturado
FORE_BAR = "#2f2f2f"
FAT_BAR = "#ff8f00"  # tom diferente do Produzido

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

      /* ajuda a caber mini tabela abaixo do gráfico */
      .block-container {{ padding-top: 0.8rem; padding-bottom: 1rem; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOGIN SIMPLES
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
def norm_col(c: str) -> str:
    return str(c).strip().replace("\n", " ")

def to_number(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"nan": "", "None": ""})
    s = s.str.replace("\u00a0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)

    mask = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False)
    s.loc[mask] = s.loc[mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    mask2 = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    s.loc[mask2] = s.loc[mask2].str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")

def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def month_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%m/%Y")

def business_days_in_month(ts: pd.Timestamp) -> int:
    start = pd.Timestamp(year=ts.year, month=ts.month, day=1)
    end = (start + pd.offsets.MonthEnd(1))
    return len(pd.bdate_range(start, end))

def style_dark(fig: go.Figure) -> go.Figure:
    # margem menor para sobrar espaço da mini tabela
    fig.update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        margin=dict(l=10, r=70, t=50, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
    )
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
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
        marker=dict(color=ORANGE_BAR_DARK, line=dict(color="#000000", width=0.25)),
        opacity=0.95
    ))
    y2 = [float(v) * 0.78 if pd.notna(v) else 0 for v in y]
    fig.add_trace(go.Bar(
        x=x, y=y2, name="",
        marker=dict(color=ORANGE_BAR_LIGHT, line=dict(color="#000000", width=0.0)),
        opacity=0.45,
        hoverinfo="skip",
        showlegend=False
    ))

def render_minitabela_html(dias, meta_list, real_list, dif_list, decimals=0):
    """
    Mini tabela compacta:
    - Fonte menor
    - Padding menor
    - Scroll horizontal se precisar
    """
    meta_s = pd.Series(meta_list, dtype="float").fillna(0)
    real_s = pd.Series(real_list, dtype="float").fillna(0)
    dif_s  = pd.Series(dif_list,  dtype="float").fillna(0)

    eff = np.where(meta_s.values != 0, (real_s.values / meta_s.values) * 100.0, np.nan)
    eff_s = pd.Series(eff).fillna(0)

    meta_media = float(meta_s.mean()) if len(meta_s) else 0.0
    real_media = float(real_s.mean()) if len(real_s) else 0.0
    dif_media  = float(dif_s.mean())  if len(dif_s)  else 0.0
    eff_media  = float(pd.Series(eff).mean()) if len(meta_s) else 0.0

    def fmt(v):
        if decimals == 0:
            return f"{int(round(float(v), 0))}"
        return f"{round(float(v), decimals):.{decimals}f}"

    def fmt_pct(v):
        try:
            return f"{round(float(v), 1):.1f}%"
        except:
            return "0.0%"

    cols = dias + ["MÉDIA"]

    rows = [
        ("Meta",        [fmt(v) for v in meta_s.tolist()] + [fmt(meta_media)]),
        ("Realizado",   [fmt(v) for v in real_s.tolist()] + [fmt(real_media)]),
        ("Diferença",   [fmt(v) for v in dif_s.tolist()]  + [fmt(dif_media)]),
        ("Eficiência %",[fmt_pct(v) for v in eff_s.tolist()] + [fmt_pct(eff_media)]),
    ]

    css = f"""
    <style>
      .mini-wrap {{
        background:{BG};
        padding:6px 0;
        overflow-x:auto;
      }}
      table.mini {{
        border-collapse:collapse;
        background:{BG};
        color:{WHITE};
        font-size:10px; /* menor */
        width:max-content; /* não estoura */
        min-width:100%;
      }}
      table.mini th, table.mini td {{
        border:0.5px solid {BORDER};
        padding:3px 4px; /* menor */
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
        padding-left:8px;
        min-width:90px;
      }}
      .thermo {{
        display:inline-block;
        width:10px; height:10px;
        border-radius:3px;
        margin-right:5px;
        border:1px solid #000;
        vertical-align:middle;
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
                arrow = "↑" if vv >= 0 else "↓"
                thermo = f"<span class='thermo' style='background:{bgc}'></span>{arrow} "
                tds.append(f"<td style='background:{bgc}; color:{WHITE}; font-weight:700;'>{thermo}{v}</td>")
            else:
                tds.append(f"<td>{v}</td>")
        body_rows.append(f"<tr><th>{label}</th>{''.join(tds)}</tr>")

    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    html = f"{css}<div class='mini-wrap'><table class='mini'>{thead}{tbody}</table></div>"
    st.markdown(html, unsafe_allow_html=True)

def find_sheet(sheetnames, wanted_keywords):
    up = [s.upper().strip() for s in sheetnames]
    for kw in wanted_keywords:
        kwu = kw.upper()
        for i, s in enumerate(up):
            if kwu in s:
                return sheetnames[i]
    return None

@st.cache_data(show_spinner=False)
def load_sheets(mtime_key: float):
    path = Path(__file__).parent / ARQUIVO_EXCEL
    xls = pd.ExcelFile(path)
    sheetnames = xls.sheet_names

    base_sheet = find_sheet(sheetnames, ["LANC", "LANÇ", "DADOS", "BASE"]) or sheetnames[0]
    forecast_sheet = find_sheet(sheetnames, ["FORECAST", "FORECAT", "FOREC"])
    faturado_sheet = find_sheet(sheetnames, ["FATURADO", "FAT"])

    df_base = pd.read_excel(path, sheet_name=base_sheet)
    df_forecast = pd.read_excel(path, sheet_name=forecast_sheet) if forecast_sheet else pd.DataFrame()
    df_faturado = pd.read_excel(path, sheet_name=faturado_sheet) if faturado_sheet else pd.DataFrame()

    return df_base, df_forecast, df_faturado, base_sheet, forecast_sheet, faturado_sheet

def month_cols(df_in, prefix):
    cols = []
    for c in df_in.columns:
        cu = str(c).upper().replace(" ", "")
        if cu.startswith(prefix):
            cols.append(c)
    return cols

def extract_month_tag(colname):
    cu = str(colname).upper().replace(" ", "")
    parts = cu.split(".")
    if len(parts) >= 2:
        return parts[1]
    return cu

def sum_months(df_in, prefix, months_selected, out_col):
    if df_in.empty or "LINHA" not in df_in.columns:
        return pd.DataFrame(columns=["LINHA", out_col])

    work = df_in.copy()
    work["LINHA"] = work["LINHA"].astype(str).str.strip()

    cols_to_sum = []
    for c in work.columns:
        cu = str(c).upper().replace(" ", "")
        if cu.startswith(prefix):
            tag = extract_month_tag(c)
            if tag in months_selected:
                cols_to_sum.append(c)

    if not cols_to_sum:
        return pd.DataFrame(columns=["LINHA", out_col])

    for c in cols_to_sum:
        work[c] = to_number(work[c])

    out = work.groupby("LINHA", as_index=False)[cols_to_sum].sum(numeric_only=True)
    out[out_col] = out[cols_to_sum].sum(axis=1)
    return out[["LINHA", out_col]]

# =========================
# CARREGAR EXCEL
# =========================
excel_path = Path(__file__).parent / ARQUIVO_EXCEL
if not excel_path.exists():
    st.error(f"Não encontrei '{ARQUIVO_EXCEL}' na pasta do app.")
    st.stop()

mtime = excel_path.stat().st_mtime
df, df_forecast, df_faturado, base_sheet, forecast_sheet, faturado_sheet = load_sheets(mtime)

# TOPO: só logo
logo_path = Path(__file__).parent / LOGO
if logo_path.exists():
    st.image(str(logo_path), width=170)

# =========================
# NORMALIZAR BASE
# =========================
df.columns = [norm_col(c) for c in df.columns]

required = ["DATA", "MÊS", "LINHA", "META", "PRODUÇÃO", "EFETIVO", "PRODT."]
faltando = [c for c in required if c not in df.columns]
if faltando:
    st.error(f"Faltam colunas no Excel (aba base: {base_sheet}): {faltando}")
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
# FILTROS (SEM TURNO)
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
data_ini, data_fim = st.sidebar.date_input("Período", value=(data_min, data_max))

df_f = df_m[
    (df_m["DATA"] >= pd.to_datetime(data_ini)) &
    (df_m["DATA"] <= pd.to_datetime(data_fim))
].copy().sort_values("DATA")

if df_f.empty:
    st.warning("Sem dados para o período selecionado.")
    st.stop()

st.sidebar.divider()
mostrar_minitabela = st.sidebar.checkbox("Mostrar mini tabela", value=True)
mostrar_termometro = st.sidebar.checkbox("Mostrar termômetro (MELHOR)", value=True)

# =========================
# AGREGAÇÃO POR DIA (SÓ DIAS COM DADOS)
# =========================
df_f["DIA_ORD"] = df_f["DATA"].dt.normalize()

agg_prod = (
    df_f.groupby("DIA_ORD", as_index=False)
        .agg({"META": "sum", "PRODUÇÃO": "sum"})
        .sort_values("DIA_ORD")
)

def _agg_prodt_day(g: pd.DataFrame) -> pd.Series:
    prod_total = pd.to_numeric(g["PRODUÇÃO"], errors="coerce").fillna(0).sum()
    efet_total = pd.to_numeric(g["EFETIVO"], errors="coerce").fillna(0).sum()
    prodt_real = (prod_total / efet_total) if efet_total > 0 else np.nan

    prodt_meta = np.nan
    if "M. PRODT" in g.columns:
        m = pd.to_numeric(g["M. PRODT"], errors="coerce")
        w = pd.to_numeric(g["EFETIVO"], errors="coerce").fillna(0)
        den = w[m.notna()].sum()
        num = (m * w).sum(skipna=True)
        prodt_meta = (num / den) if den and den > 0 else np.nan

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

agg["DIF_PROD"] = pd.to_numeric(agg["PRODUÇÃO"], errors="coerce").fillna(0) - pd.to_numeric(agg["META"], errors="coerce").fillna(0)

if agg["PRODT_META"].notna().any():
    agg["META_PRODT"] = agg["PRODT_META"]
else:
    agg["META_PRODT"] = float(pd.to_numeric(df_f["PRODT."], errors="coerce").mean())

agg["DIF_PRODT"] = (pd.to_numeric(agg["PRODT_REAL"], errors="coerce") - pd.to_numeric(agg["META_PRODT"], errors="coerce")).fillna(0)

# =========================
# 1) PRODUÇÃO
# =========================
st.markdown(f"## PRODUÇÃO — {mes_sel}")

fig_prod = go.Figure()
add_gradient_bars(fig_prod, agg["DIA_TXT"], agg["PRODUÇÃO"], name="Produção")

fig_prod.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["META"],
    name="Meta",
    mode="lines+markers",
    line=dict(color=ORANGE_LINE, width=1.0),
    marker=dict(color=ORANGE_LINE, size=4),
))

diff_colors = [GREEN if v >= 0 else RED for v in agg["DIF_PROD"].tolist()]
fig_prod.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["DIF_PROD"],
    name="Real − Meta",
    mode="lines+markers",
    line=dict(color=DIFF_LINE, width=1.0, dash="dash"),
    marker=dict(color=diff_colors, size=6),
    yaxis="y2"
))

dmin = float(pd.to_numeric(agg["DIF_PROD"], errors="coerce").min())
dmax = float(pd.to_numeric(agg["DIF_PROD"], errors="coerce").max())
pad = max(10.0, (dmax - dmin) * 0.15) if (dmax - dmin) != 0 else 50.0

fig_prod.update_layout(
    barmode="overlay",
    height=480,
    xaxis=dict(type="category", categoryorder="array", categoryarray=x_order, tickmode="array", tickvals=x_order, ticktext=x_order),
    yaxis2=dict(title="Diferença", overlaying="y", side="right", showgrid=False, range=[dmin - pad, dmax + pad])
)

if mostrar_termometro:
    add_termometro(fig_prod, 480)

st.plotly_chart(style_dark(fig_prod), use_container_width=True)

if mostrar_minitabela:
    render_minitabela_html(
        dias=agg["DIA_TXT"].tolist(),
        meta_list=pd.Series(agg["META"]).fillna(0).tolist(),
        real_list=pd.Series(agg["PRODUÇÃO"]).fillna(0).tolist(),
        dif_list=pd.Series(agg["DIF_PROD"]).fillna(0).tolist(),
        decimals=0
    )

st.divider()

# =========================
# 2) PRODUTIVIDADE
# =========================
st.markdown(f"## PRODUTIVIDADE — {mes_sel}")

fig_prodt = go.Figure()
fig_prodt.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["PRODT_REAL"],
    name="Produtividade (Real)",
    mode="lines+markers",
    line=dict(color=ORANGE_BAR_LIGHT, width=1.0),
    marker=dict(color=ORANGE_BAR_LIGHT, size=4),
))
fig_prodt.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["META_PRODT"],
    name="Meta Produtividade",
    mode="lines",
    line=dict(color=ORANGE_LINE, width=1.0, dash="dot"),
))

diff_colors2 = [GREEN if v >= 0 else RED for v in agg["DIF_PRODT"].tolist()]
fig_prodt.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["DIF_PRODT"],
    name="Real − Meta",
    mode="lines+markers",
    line=dict(color=DIFF_LINE, width=1.0, dash="dash"),
    marker=dict(color=diff_colors2, size=6),
    yaxis="y2"
))

dmin2 = float(pd.to_numeric(agg["DIF_PRODT"], errors="coerce").min())
dmax2 = float(pd.to_numeric(agg["DIF_PRODT"], errors="coerce").max())
pad2 = max(0.1, (dmax2 - dmin2) * 0.15) if (dmax2 - dmin2) != 0 else 1.0

fig_prodt.update_layout(
    height=440,
    xaxis=dict(type="category", categoryorder="array", categoryarray=x_order, tickmode="array", tickvals=x_order, ticktext=x_order),
    yaxis2=dict(title="Diferença", overlaying="y", side="right", showgrid=False, range=[dmin2 - pad2, dmax2 + pad2])
)

if mostrar_termometro:
    add_termometro(fig_prodt, 440)

st.plotly_chart(style_dark(fig_prodt), use_container_width=True)

if mostrar_minitabela:
    render_minitabela_html(
        dias=agg["DIA_TXT"].tolist(),
        meta_list=pd.Series(agg["META_PRODT"]).fillna(0).tolist(),
        real_list=pd.Series(agg["PRODT_REAL"]).fillna(0).tolist(),
        dif_list=pd.Series(agg["DIF_PRODT"]).fillna(0).tolist(),
        decimals=2
    )

st.divider()

# =========================
# 3) ANÁLISE & TENDÊNCIA (12 meses / DIAS ÚTEIS)
# =========================
st.sidebar.header("Análise — Tendência (meses seguintes)")
meses_base = st.sidebar.slider("Meses históricos para calcular tendência", 1, 12, 12)
meses_a_frente = st.sidebar.slider("Projetar quantos meses à frente", 1, 12, 3)

st.markdown("## ANÁLISE & TENDÊNCIA — Próximos meses (Dias úteis)")
st.caption("Projeção por Linha usando dias úteis (seg–sex). ⚠️ ")

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
        .rename(columns={
            "PROD_POR_DU": "TEND_PROD_POR_DU",
            "EFET_POR_DU": "TEND_EFET_POR_DU",
            "PRODT_REAL_MES": "TEND_PRODT"
        })
)

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
    x=proj_l["MES_TXT"],
    y=proj_l["PROD_PROJ"],
    name="Produção (Proj)",
    marker=dict(color="#444444", line=dict(color="#000000", width=0.2)),
    opacity=0.85
))
fig_t_prod.update_layout(height=420, barmode="group", xaxis=dict(type="category"))
st.plotly_chart(style_dark(fig_t_prod), use_container_width=True)

st.markdown("### Tendência de Produtividade por Linha — Histórico mensal + Projeção")
fig_t_prodt = go.Figure()
fig_t_prodt.add_trace(go.Scatter(
    x=hist_l["MES_TXT"],
    y=hist_l["PRODT_REAL_MES"],
    name="Produtividade (Hist)",
    mode="lines+markers",
    line=dict(color=ORANGE_BAR_LIGHT, width=1.0),
    marker=dict(color=ORANGE_BAR_LIGHT, size=5),
))
fig_t_prodt.add_trace(go.Scatter(
    x=proj_l["MES_TXT"],
    y=proj_l["PRODT_PROJ"],
    name="Produtividade (Proj)",
    mode="lines+markers",
    line=dict(color="#aaaaaa", width=1.0, dash="dot"),
    marker=dict(color="#aaaaaa", size=5),
))
fig_t_prodt.update_layout(height=360, xaxis=dict(type="category"))
st.plotly_chart(style_dark(fig_t_prodt), use_container_width=True)

st.markdown("### Resumo da Projeção (por dias úteis)")
res = proj_l[["MES_TXT", "DIAS_UTEIS", "PROD_PROJ", "PRODT_PROJ"]].copy()
res["PROD_PROJ"] = res["PROD_PROJ"].round(0).astype("Int64")
res["PRODT_PROJ"] = res["PRODT_PROJ"].round(2)
st.dataframe(res, use_container_width=True)

st.divider()

# =========================
# 4) FORECAST × PRODUZIDO × FATURADO + Tabelas DIF + RANKINGS
# =========================
st.markdown("## FORECAST × PRODUZIDO × FATURADO (Ranking por Linha)")

# PRODUZIDO = soma de PRODUÇÃO no período filtrado
produzido = (
    df_f.groupby("LINHA", as_index=False)["PRODUÇÃO"]
        .sum()
        .rename(columns={"PRODUÇÃO": "PRODUZIDO"})
)

if not df_forecast.empty:
    df_forecast.columns = [norm_col(c) for c in df_forecast.columns]
if not df_faturado.empty:
    df_faturado.columns = [norm_col(c) for c in df_faturado.columns]

forecast_cols = month_cols(df_forecast, "FOR.") if (not df_forecast.empty and "LINHA" in df_forecast.columns) else []
faturado_cols = month_cols(df_faturado, "FAT.") if (not df_faturado.empty and "LINHA" in df_faturado.columns) else []
month_tags = sorted(list({extract_month_tag(c) for c in forecast_cols + faturado_cols}))

if not month_tags:
    st.info("Não encontrei colunas FOR.JAN / FAT.JAN nas abas FORECAST/FATURADO.")
else:
    st.sidebar.header("Comparativo — Forecast/Faturado")
    meses_comp = st.sidebar.multiselect(
        "Meses para comparar (FORECAST/FATURADO)",
        options=month_tags,
        default=[m for m in ["JAN", "FEV", "MAR"] if m in month_tags] or month_tags[:1]
    )

    forecast_sum = sum_months(df_forecast, "FOR.", meses_comp, "FORECAST")
    faturado_sum = sum_months(df_faturado, "FAT.", meses_comp, "FATURADO")

    comp = produzido.merge(forecast_sum, on="LINHA", how="outer").merge(faturado_sum, on="LINHA", how="outer")
    comp["PRODUZIDO"] = pd.to_numeric(comp.get("PRODUZIDO", 0), errors="coerce").fillna(0)
    comp["FORECAST"] = pd.to_numeric(comp.get("FORECAST", 0), errors="coerce").fillna(0)
    comp["FATURADO"] = pd.to_numeric(comp.get("FATURADO", 0), errors="coerce").fillna(0)

    comp = comp[comp["LINHA"].isin(linha_sel)].copy()
    comp["DIF (PROD − FORE)"] = comp["PRODUZIDO"] - comp["FORECAST"]
    comp["DIF (FAT − FORE)"] = comp["FATURADO"] - comp["FORECAST"]

    if comp.empty:
        st.info("Sem dados suficientes para montar o comparativo.")
    else:
        # Gráfico comparativo
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=comp["LINHA"], y=comp["FORECAST"],
            name="Forecast",
            marker=dict(color=FORE_BAR, line=dict(color="#000000", width=0.2)),
            opacity=0.95
        ))
        fig_comp.add_trace(go.Bar(
            x=comp["LINHA"], y=comp["PRODUZIDO"],
            name="Produzido",
            marker=dict(color=ORANGE_BAR_DARK, line=dict(color="#000000", width=0.2)),
            opacity=0.95
        ))
        fig_comp.add_trace(go.Bar(
            x=comp["LINHA"], y=comp["FATURADO"],
            name="Faturado",
            marker=dict(color=FAT_BAR, line=dict(color="#000000", width=0.2)),
            opacity=0.95
        ))
        fig_comp.update_layout(height=520, barmode="group", xaxis=dict(type="category", tickangle=-15))
        st.plotly_chart(style_dark(fig_comp), use_container_width=True)

        # ===== TABELAS DIF =====
        st.markdown("### Tabelas de Diferença")

        def thermo_cell(v):
            try:
                vv = float(v)
            except:
                vv = 0.0
            return "🟩 ↑" if vv >= 0 else "🟥 ↓"

        def style_diff(s):
            out = []
            for v in s:
                try:
                    vv = float(v)
                except:
                    vv = 0.0
                out.append(
                    f"background-color:{GREEN}; color:{WHITE}; font-weight:700;"
                    if vv >= 0 else
                    f"background-color:{RED}; color:{WHITE}; font-weight:700;"
                )
            return out

        cA, cB = st.columns(2)

        with cA:
            st.markdown("**DIF (PROD − FORE)**")
            t_prod = comp[["LINHA", "FORECAST", "PRODUZIDO", "DIF (PROD − FORE)"]].copy()
            for col in ["FORECAST", "PRODUZIDO", "DIF (PROD − FORE)"]:
                t_prod[col] = pd.to_numeric(t_prod[col], errors="coerce").fillna(0).round(0).astype("Int64")
            t_prod["TERMÔMETRO"] = t_prod["DIF (PROD − FORE)"].apply(thermo_cell)
            st.dataframe(
                t_prod.style.apply(style_diff, subset=["DIF (PROD − FORE)"]).set_table_styles([
                    {"selector":"th","props":[("background-color",BG),("color",WHITE),("font-weight","700")]},
                    {"selector":"td","props":[("background-color",BG),("color",WHITE)]},
                    {"selector":"table","props":[("background-color",BG),("color",WHITE)]},
                ]),
                use_container_width=True
            )

        with cB:
            st.markdown("**DIF (FAT − FORE)**")
            t_fat = comp[["LINHA", "FORECAST", "FATURADO", "DIF (FAT − FORE)"]].copy()
            for col in ["FORECAST", "FATURADO", "DIF (FAT − FORE)"]:
                t_fat[col] = pd.to_numeric(t_fat[col], errors="coerce").fillna(0).round(0).astype("Int64")
            t_fat["TERMÔMETRO"] = t_fat["DIF (FAT − FORE)"].apply(thermo_cell)
            st.dataframe(
                t_fat.style.apply(style_diff, subset=["DIF (FAT − FORE)"]).set_table_styles([
                    {"selector":"th","props":[("background-color",BG),("color",WHITE),("font-weight","700")]},
                    {"selector":"td","props":[("background-color",BG),("color",WHITE)]},
                    {"selector":"table","props":[("background-color",BG),("color",WHITE)]},
                ]),
                use_container_width=True
            )

        # ===== RANKINGS TOP 5 =====
        st.markdown("### Top 5 (melhor) e Top 5 (pior)")

        def top5_block(df_in, col_diff, title):
            base = df_in[["LINHA", "FORECAST", "PRODUZIDO", "FATURADO", col_diff]].copy()
            base = base.sort_values(col_diff, ascending=False)

            top = base.head(5).copy()
            bottom = base.tail(5).sort_values(col_diff, ascending=True).copy()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Top 5 — {title} (melhor)**")
                st.dataframe(top, use_container_width=True)
            with c2:
                st.markdown(f"**Top 5 — {title} (pior)**")
                st.dataframe(bottom, use_container_width=True)

        top5_block(comp, "DIF (PROD − FORE)", "Produzido − Forecast")
        top5_block(comp, "DIF (FAT − FORE)", "Faturado − Forecast")
