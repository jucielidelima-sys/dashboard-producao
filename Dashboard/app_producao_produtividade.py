import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import urllib.parse

# =========================
# CONFIG (V22)
# =========================
st.set_page_config(page_title="Dashboard", layout="wide")

SENHA_APP = "neworder2026"
ARQUIVO_EXCEL = "PROD-PRODT.xlsx"
LOGO = "logo.png"

# Paleta (preto/laranja suave)
ORANGE_BAR_LIGHT = "#ffb36b"   # bem mais suave
ORANGE_BAR_DARK  = "#e56a00"   # base mais escura p/ efeito gradiente
ORANGE_LINE = "#ffd0a3"        # meta
WHITE = "#ffffff"
GRID = "#2a2a2a"
GREEN = "#00c853"
RED = "#ff1744"
BG = "#000000"
BG2 = "#070707"
BORDER = "#1a1a1a"

# =========================
# ESTILO GLOBAL (PRETO)
# =========================
st.markdown(
    """
    <style>
      .stApp { background:#000000; color:#ffffff; }
      section[data-testid="stSidebar"] { background:#000000; }
      section[data-testid="stSidebar"] * { color:#ffffff !important; }
      .stTextInput input, .stDateInput input, .stSelectbox div, .stMultiSelect div {
        background:#111111 !important; color:#ffffff !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOGIN (SÓ LOGO)
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
# TOPO (SÓ LOGO)
# =========================
logo_path = Path(__file__).parent / LOGO
if logo_path.exists():
    st.image(str(logo_path), width=170)

# =========================
# CONVERSÃO NUMÉRICA FORTE (pt-BR / en-US)
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

# =========================
# CARREGAR DADOS
# =========================
@st.cache_data
def load_data():
    path = Path(__file__).parent / ARQUIVO_EXCEL
    return pd.read_excel(path)

df = load_data()
df.columns = [str(c).strip() for c in df.columns]

required = ["DATA", "MÊS", "LINHA", "META", "PRODUÇÃO", "PRODT."]
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
df["PRODT."] = to_number(df["PRODT."])

# =========================
# FILTROS (SEM TURNO)
# =========================
st.sidebar.header("Filtros")

meses = sorted(df["MÊS"].dropna().unique().tolist())
mes_sel = st.sidebar.selectbox(
    "Mês",
    meses,
    index=meses.index("JANEIRO") if "JANEIRO" in meses else 0
)

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
# CONSOLIDADO POR DIA (SÓ DIAS COM DADOS)
# =========================
df_f["DIA_ORD"] = df_f["DATA"].dt.normalize()

agg = (
    df_f.groupby("DIA_ORD", as_index=False)
        .agg({"META": "sum", "PRODUÇÃO": "sum", "PRODT.": "mean"})
        .sort_values("DIA_ORD")
)

agg["DIA_TXT"] = agg["DIA_ORD"].dt.strftime("%d/%m")
x_order = agg["DIA_TXT"].tolist()

agg["DIF_PROD"] = (agg["PRODUÇÃO"] - agg["META"]).fillna(0)

meta_prodt = float(pd.to_numeric(df_f["PRODT."], errors="coerce").mean())
agg["META_PRODT"] = meta_prodt
agg["DIF_PRODT"] = (agg["PRODT."] - meta_prodt).fillna(0)

# =========================
# ESTILO PLOTLY
# =========================
def style_dark(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(color="white"),
        margin=dict(l=10, r=60, t=55, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0),
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=x_order,
        tickmode="array",
        tickvals=x_order,
        ticktext=x_order,
        showgrid=False
    )
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig

# =========================
# TERMÔMETRO (SVG dentro do gráfico)
# =========================
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

# =========================
# BARRAS “GRADIENTE” (mais visível)
# =========================
def add_gradient_bars(fig: go.Figure, x, y, name="Produção"):
    # Base escura
    fig.add_trace(go.Bar(
        x=x, y=y, name=name,
        marker=dict(color=ORANGE_BAR_DARK, line=dict(color="#000000", width=0.3)),
        opacity=0.95
    ))
    # Faixa clara por cima (efeito gradiente)
    y2 = [v * 0.75 for v in y]  # aumenta a área clara pra ficar MAIS perceptível
    fig.add_trace(go.Bar(
        x=x, y=y2, name="",
        marker=dict(color=ORANGE_BAR_LIGHT, line=dict(color="#000000", width=0.0)),
        opacity=0.55,
        hoverinfo="skip",
        showlegend=False
    ))

# =========================
# MINI TABELA 100% PRETA (HTML) + DIF e MÉDIA verde/vermelha
# =========================
def render_minitabela_html(dias, meta_list, real_list, dif_list, decimals=0):
    meta_s = pd.Series(meta_list, dtype="float").fillna(0)
    real_s = pd.Series(real_list, dtype="float").fillna(0)
    dif_s = pd.Series(dif_list, dtype="float").fillna(0)

    meta_media = float(meta_s.mean()) if len(meta_s) else 0.0
    real_media = float(real_s.mean()) if len(real_s) else 0.0
    dif_media = float(dif_s.mean()) if len(dif_s) else 0.0

    def fmt(v):
        if decimals == 0:
            return f"{int(round(float(v), 0))}"
        return f"{round(float(v), decimals):.{decimals}f}"

    cols = dias + ["MÉDIA"]

    rows = [
        ("Meta",       [fmt(v) for v in meta_s.tolist()] + [fmt(meta_media)]),
        ("Realizado",  [fmt(v) for v in real_s.tolist()] + [fmt(real_media)]),
        ("Diferença",  [fmt(v) for v in dif_s.tolist()]  + [fmt(dif_media)]),
    ]

    # CSS travado
    css = f"""
    <style>
      .mini-wrap {{
        background:{BG};
        padding:6px 0;
      }}
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

    # HTML
    thead = "<thead><tr><th></th>" + "".join([f"<th>{c}</th>" for c in cols]) + "</tr></thead>"

    body_rows = []
    for label, values in rows:
        tds = []
        for i, v in enumerate(values):
            if label == "Diferença":
                # cor verde/vermelha também na MÉDIA
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
# 1) PRODUÇÃO
# =========================
st.markdown(f"## PRODUÇÃO — {mes_sel}")

fig_prod = go.Figure()

add_gradient_bars(fig_prod, agg["DIA_TXT"], agg["PRODUÇÃO"], name="Produção")

# Meta (linhas finas)
fig_prod.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["META"],
    name="Meta",
    mode="lines+markers",
    line=dict(color=ORANGE_LINE, width=1.3),
    marker=dict(color=ORANGE_LINE, size=5)
))

# Diferença (dentro do gráfico, fininha)
diff_colors = [GREEN if v >= 0 else RED for v in agg["DIF_PROD"].tolist()]
fig_prod.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["DIF_PROD"],
    name="Real − Meta",
    mode="lines+markers",
    line=dict(color=WHITE, width=1.2, dash="dash"),
    marker=dict(color=diff_colors, size=7),
    yaxis="y2"
))

dmin = float(agg["DIF_PROD"].min())
dmax = float(agg["DIF_PROD"].max())
pad = max(10.0, (dmax - dmin) * 0.15) if (dmax - dmin) != 0 else 50.0

fig_prod.update_layout(
    barmode="overlay",
    height=520,
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
    meta_list = pd.Series(agg["META"]).fillna(0).tolist()
    real_list = pd.Series(agg["PRODUÇÃO"]).fillna(0).tolist()
    dif_list = pd.Series(agg["DIF_PROD"]).fillna(0).tolist()
    render_minitabela_html(dias, meta_list, real_list, dif_list, decimals=0)

st.divider()

# =========================
# 2) PRODUTIVIDADE
# =========================
st.markdown(f"## PRODUTIVIDADE — {mes_sel}")

fig_prodt = go.Figure()

fig_prodt.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["PRODT."],
    name="Produtividade (Real)",
    mode="lines+markers",
    line=dict(color=ORANGE_BAR_LIGHT, width=1.4),
    marker=dict(color=ORANGE_BAR_LIGHT, size=5),
))

fig_prodt.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=[meta_prodt] * len(agg),
    name="Meta Produtividade",
    mode="lines",
    line=dict(color=ORANGE_LINE, width=1.3, dash="dot"),
))

diff_colors2 = [GREEN if v >= 0 else RED for v in agg["DIF_PRODT"].tolist()]
fig_prodt.add_trace(go.Scatter(
    x=agg["DIA_TXT"], y=agg["DIF_PRODT"],
    name="Real − Meta (Prodt.)",
    mode="lines+markers",
    line=dict(color=WHITE, width=1.2, dash="dash"),
    marker=dict(color=diff_colors2, size=7),
    yaxis="y2"
))

dmin2 = float(agg["DIF_PRODT"].min())
dmax2 = float(agg["DIF_PRODT"].max())
pad2 = max(0.1, (dmax2 - dmin2) * 0.15) if (dmax2 - dmin2) != 0 else 1.0

fig_prodt.update_layout(
    height=460,
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
    meta_list = [meta_prodt] * len(dias)
    real_list = pd.Series(agg["PRODT."]).fillna(0).tolist()
    dif_list = pd.Series(agg["DIF_PRODT"]).fillna(0).tolist()
    render_minitabela_html(dias, meta_list, real_list, dif_list, decimals=2)
