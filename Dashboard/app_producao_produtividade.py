import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Indicadores — Montagem", layout="wide")

# =========================
# TEMA ESCURO NA TELA TODA
# =========================
st.markdown(
    """
    <style>
      .stApp { background:#000000; color:#ffffff; }
      section[data-testid="stSidebar"] { background:#000000; }
      section[data-testid="stSidebar"] * { color:#ffffff; }
      h1,h2,h3,h4,h5,h6,p,label,div { color:#ffffff; }

      .stTextInput input, .stDateInput input, .stSelectbox div, .stMultiSelect div {
        background:#111111 !important; color:#ffffff !important;
      }

      /* Dataframe escuro */
      div[data-testid="stDataFrame"] { background:#000000; }

      /* métricas */
      div[data-testid="stMetric"] { background:#000000; border: 1px solid #222; border-radius: 10px; padding: 8px; }

      /* tabela mini (html) */
      table.mini { border-collapse: collapse; width: 100%; font-size: 12px; }
      table.mini th, table.mini td { border: 1px solid #333; padding: 4px 6px; text-align: center; }
      table.mini th { background: #0b0b0b; font-weight: 800; }
      table.mini td.rowhead { text-align:left; font-weight:800; background:#0b0b0b; }
      table.mini td.meta { background:#0f2a0f; color:#ffffff; }
      table.mini td.real { background:#0f1c2a; color:#ffffff; }
      table.mini td.diff_pos { background:#0a5a0a; color:#ffffff; font-weight:800; }
      table.mini td.diff_neg { background:#7a0a0a; color:#ffffff; font-weight:800; }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("✅ VERSÃO — Diferença verde/vermelha + linha Diferença (v10)")

# =====================
# CARREGAR DADOS
# =====================
@st.cache_data
def load_data():
    return pd.read_excel("PROD-PRODT.xlsx")

df = load_data()
df.columns = [str(c).strip() for c in df.columns]

def find_col_contains(tokens, exclude_tokens=()):
    tokens = [t.upper() for t in tokens]
    exclude_tokens = [t.upper() for t in exclude_tokens]
    for c in df.columns:
        cu = c.upper()
        if any(t in cu for t in tokens) and not any(et in cu for et in exclude_tokens):
            return c
    return None

COL_DATA   = find_col_contains(["DATA"])
COL_LINHA  = find_col_contains(["LINHA", "SETOR", "LINE"])

COL_META_P = find_col_contains(["META"])                 # meta produção
COL_PROD   = find_col_contains(["PRODUÇÃO", "PRODUCAO"])  # produção realizada

# Produtividade
COL_PRODT_META = find_col_contains(["M. PRODT", "META PRODT", "M PRODT"])
COL_PRODT_REAL = find_col_contains(["PRODT."], exclude_tokens=["M. PRODT", "META"]) or \
                 find_col_contains(["PRODT"], exclude_tokens=["M. PRODT", "META PRODT", "M PRODT"])

missing = [("DATA", COL_DATA), ("LINHA", COL_LINHA), ("META", COL_META_P),
           ("PRODUÇÃO", COL_PROD), ("M. PRODT", COL_PRODT_META), ("PRODT.", COL_PRODT_REAL)]
missing = [name for name, col in missing if col is None]
if missing:
    st.error(f"Faltando colunas no Excel: {missing}\n\nColunas encontradas: {list(df.columns)}")
    st.stop()

df = df.rename(columns={
    COL_DATA: "DATA",
    COL_LINHA: "LINHA",
    COL_META_P: "META_PROD",
    COL_PROD: "PRODUCAO",
    COL_PRODT_META: "META_PRODT",
    COL_PRODT_REAL: "PRODT_REAL",
})

df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
df = df.dropna(subset=["DATA"])

for c in ["META_PROD", "PRODUCAO", "META_PRODT", "PRODT_REAL"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# =====================
# FILTROS
# =====================
st.sidebar.header("Filtros")
linhas = sorted(df["LINHA"].dropna().astype(str).unique().tolist())
linha_sel = st.sidebar.multiselect("Linha", linhas, default=linhas, key="f_linha")

data_min = df["DATA"].min().date()
data_max = df["DATA"].max().date()
data_ini, data_fim = st.sidebar.date_input("Período", value=(data_min, data_max), key="f_data")

df_f = df[
    (df["LINHA"].astype(str).isin(linha_sel)) &
    (df["DATA"] >= pd.to_datetime(data_ini)) &
    (df["DATA"] <= pd.to_datetime(data_fim))
].copy()

if df_f.empty:
    st.warning("Nenhum dado para os filtros selecionados.")
    st.stop()

# =====================
# AGREGAÇÃO (SÓ DIAS COM DADOS)
# =====================
df_f["DIA_ORD"] = df_f["DATA"].dt.normalize()
df_f["DIA_TXT"] = df_f["DATA"].dt.strftime("%d/%m")

# Produção: soma
agg_prod = (
    df_f.groupby(["DIA_ORD", "DIA_TXT"], as_index=False)[["PRODUCAO", "META_PROD"]]
      .sum()
      .sort_values("DIA_ORD")
)

# Produtividade: média
agg_prodt = (
    df_f.groupby(["DIA_ORD", "DIA_TXT"], as_index=False)[["PRODT_REAL", "META_PRODT"]]
      .mean()
      .sort_values("DIA_ORD")
)

x_order = agg_prod["DIA_TXT"].tolist()

# Diferença (Real - Meta) para produção e produtividade
agg_prod["DIF"] = agg_prod["PRODUCAO"] - agg_prod["META_PROD"]
agg_prodt["DIF"] = agg_prodt["PRODT_REAL"] - agg_prodt["META_PRODT"]

# =====================
# CORES
# =====================
ORANGE = "#ff7a00"
ORANGE_LIGHT = "#ffa64d"
GRID = "#333333"

GREEN = "#00c853"   # verde
RED   = "#ff1744"   # vermelho

def style_dark(fig):
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="left", x=0),
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=x_order,
        tickmode="array",
        tickvals=x_order,
        ticktext=x_order,
        showgrid=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig

# =====================
# SETA "MELHOR"
# =====================
def melhor_arrow():
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:center; height:100%;">
          <div style="
              width:46px; height:280px;
              background: linear-gradient(#0b4cff, #001a4d);
              border: 1px solid #1a1a1a;
              position: relative;
              border-radius: 4px;
            ">
            <div style="
                position:absolute;
                top:-36px; left:50%;
                transform:translateX(-50%);
                width:0; height:0;
                border-left:23px solid transparent;
                border-right:23px solid transparent;
                border-bottom:36px solid #0b4cff;
              "></div>
            <div style="
                position:absolute;
                top:55px; left:50%;
                transform:translateX(-50%);
                writing-mode: vertical-rl;
                text-orientation: upright;
                color:white; font-weight:800;
                letter-spacing: 2px;
                font-size: 14px;
              ">MELHOR</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================
# MINI TABELA HTML (verde/vermelho na diferença)
# =====================
def mini_tabela_html(dias, meta, real, kind="int"):
    meta = pd.Series(meta)
    real = pd.Series(real)
    meta = pd.to_numeric(meta, errors="coerce").fillna(0)
    real = pd.to_numeric(real, errors="coerce").fillna(0)
    dif = real - meta  # Real - Meta

    if kind == "int":
        fmt = lambda x: f"{int(round(float(x)))}"
    else:
        fmt = lambda x: f"{float(x):.1f}".replace(".", ",")

    # cabeçalho
    html = "<table class='mini'><tr><th></th>"
    for d in dias:
        html += f"<th>{d}</th>"
    html += "</tr>"

    # linha meta
    html += "<tr><td class='rowhead'>Meta</td>"
    for v in meta:
        html += f"<td class='meta'>{fmt(v)}</td>"
    html += "</tr>"

    # linha realizado
    html += "<tr><td class='rowhead'>Realizado</td>"
    for v in real:
        html += f"<td class='real'>{fmt(v)}</td>"
    html += "</tr>"

    # linha diferença
    html += "<tr><td class='rowhead'>Diferença</td>"
    for v in dif:
        cls = "diff_pos" if float(v) >= 0 else "diff_neg"
        html += f"<td class='{cls}'>{fmt(v)}</td>"
    html += "</tr>"

    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

# =====================
# TÍTULO PRINCIPAL
# =====================
st.markdown("<h2 style='text-align:center; font-weight:900; margin:0;'>GRUPO NEW ORDER</h2>", unsafe_allow_html=True)

# =====================
# PRODUÇÃO
# =====================
st.markdown(
    "<h3 style='text-align:center; font-weight:800; margin-top:8px;'>"
    "INDICADOR DE MONITORAMENTO DE PRODUÇÃO — JANEIRO"
    "</h3>",
    unsafe_allow_html=True
)

col_g, col_a = st.columns([0.92, 0.08])

with col_g:
    fig1 = go.Figure()

    # barra: realizado
    fig1.add_trace(go.Bar(
        x=agg_prod["DIA_TXT"],
        y=agg_prod["PRODUCAO"],
        name="Realizado",
        marker_color=ORANGE
    ))

    # linha: meta
    fig1.add_trace(go.Scatter(
        x=agg_prod["DIA_TXT"],
        y=agg_prod["META_PROD"],
        name="Meta",
        mode="lines+markers",
        line=dict(color=ORANGE_LIGHT, width=3),
        marker=dict(color=ORANGE_LIGHT, size=7),
    ))

    # linha: diferença (Real - Meta) com verde/vermelho por ponto
    diff_colors = [GREEN if v >= 0 else RED for v in agg_prod["DIF"].fillna(0).tolist()]
    fig1.add_trace(go.Scatter(
        x=agg_prod["DIA_TXT"],
        y=agg_prod["DIF"],
        name="Diferença (Real - Meta)",
        mode="lines+markers",
        line=dict(color="#ffffff", width=2, dash="dash"),
        marker=dict(color=diff_colors, size=8),
        yaxis="y2"
    ))

    fig1.update_layout(
        height=420,
        yaxis2=dict(
            title="Diferença",
            overlaying="y",
            side="right",
            showgrid=False
        )
    )

    st.plotly_chart(style_dark(fig1), use_container_width=True)

with col_a:
    melhor_arrow()

# mini tabela produção (verde/vermelho na diferença)
mini_tabela_html(
    dias=x_order,
    meta=agg_prod["META_PROD"].tolist(),
    real=agg_prod["PRODUCAO"].tolist(),
    kind="int"
)

st.divider()

# =====================
# PRODUTIVIDADE
# =====================
st.markdown(
    "<h3 style='text-align:center; font-weight:800; margin-top:0;'>"
    "INDICADOR DE MONITORAMENTO DE PRODUTIVIDADE — JANEIRO"
    "</h3>",
    unsafe_allow_html=True
)

col_g2, col_a2 = st.columns([0.92, 0.08])

with col_g2:
    agg_prodt2 = agg_prodt.set_index("DIA_TXT").reindex(x_order).reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=agg_prodt2["DIA_TXT"],
        y=agg_prodt2["PRODT_REAL"],
        name="Realizado",
        mode="lines+markers",
        line=dict(color=ORANGE, width=3),
        marker=dict(color=ORANGE, size=7),
    ))
    fig2.add_trace(go.Scatter(
        x=agg_prodt2["DIA_TXT"],
        y=agg_prodt2["META_PRODT"],
        name="Meta",
        mode="lines+markers",
        line=dict(color=ORANGE_LIGHT, width=3, dash="dot"),
        marker=dict(color=ORANGE_LIGHT, size=7),
    ))
    fig2.update_layout(height=420)
    st.plotly_chart(style_dark(fig2), use_container_width=True)

with col_a2:
    melhor_arrow()

mini_tabela_html(
    dias=x_order,
    meta=agg_prodt2["META_PRODT"].tolist(),
    real=agg_prodt2["PRODT_REAL"].tolist(),
    kind="float"
)

with st.expander("Ver dados filtrados"):
    st.dataframe(df_f.sort_values("DATA"), use_container_width=True, height=520)
