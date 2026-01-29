# app_producao_produtividade.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =========================
# CONFIG / ESTILO
# =========================
APP_VERSION = "V20 — Produção/Produtividade + Forecast + Ranking (Fix total)"
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_FILE = "PROD-PRODT.xlsx"  # seu arquivo principal
LOGO_FILE = "logo.png"                # seu logo (na mesma pasta do .py)

# Paleta (preto/laranja + verde/vermelho)
BG = "#000000"
FG = "#FFFFFF"
GRID = "#2b2b2b"
ORANGE = "#ff7a00"
ORANGE_SOFT = "#ff9a3c"   # “capa” mais clara pro gradiente
META_LINE = "#ffb14a"
DIFF_LINE = "#ffffff"
GREEN = "#00c853"
RED = "#ff1744"
CARD = "#0a0a0a"
BORDER = "#1d1d1d"

st.set_page_config(
    page_title="Dashboard Produção/Produtividade",
    layout="wide",
)

# CSS global (fundo 100% preto + sidebar preta + textos brancos)
st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{
        background-color: {BG} !important;
        color: {FG} !important;
      }}
      .stApp {{
        background: {BG} !important;
      }}
      section[data-testid="stSidebar"] {{
        background-color: {BG} !important;
        border-right: 1px solid {BORDER} !important;
      }}
      section[data-testid="stSidebar"] * {{
        color: {FG} !important;
      }}
      div[data-testid="stToolbar"] {{
        background: {BG} !important;
      }}
      .stPlotlyChart, .stDataFrame {{
        background: {BG} !important;
      }}
      .block-container {{
        padding-top: 1.2rem;
      }}
      /* Remover “menu” visual excessivo em telas pequenas */
      header {{
        background: {BG} !important;
      }}
      /* Caixa tipo card */
      .gno-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 16px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# UTILITÁRIOS
# =========================
def to_number(s: pd.Series | np.ndarray | list) -> pd.Series:
    """Converte números aceitando formatos BR/planilha (ex: 8.010, 1.660)."""
    ser = pd.Series(s).copy()
    ser = ser.astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
    # se tiver ponto como separador de milhar e vírgula decimal (BR), trata:
    # exemplos comuns do Excel: "8.010" (milhar) e "4,89" (decimal)
    # Estratégia:
    # - se tiver "," => considera decimal e remove pontos
    # - se não tiver "," => remove pontos (milhar)
    has_comma = ser.str.contains(",", na=False)
    ser.loc[has_comma] = ser.loc[has_comma].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    ser.loc[~has_comma] = ser.loc[~has_comma].str.replace(".", "", regex=False)
    return pd.to_numeric(ser, errors="coerce")


def safe_read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def find_data_file() -> Optional[Path]:
    """Procura PROD-PRODT.xlsx na pasta do app."""
    p = BASE_DIR / DEFAULT_DATA_FILE
    if p.exists():
        return p
    # tenta achar qualquer xlsx com PROD-PRODT no nome
    for f in BASE_DIR.glob("*.xlsx"):
        if "PROD" in f.name.upper() and "PRODT" in f.name.upper():
            return f
    return None


@st.cache_data(show_spinner=False)
def load_data(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    df = safe_read_excel(path)

    # Normaliza nomes de colunas esperadas
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Colunas essenciais
    # DATA / MÊS / LINHA / META / PRODUÇÃO / M. PRODT / PRODT.
    # Aceita variações
    col_map = {}
    for c in df.columns:
        cu = c.upper().strip()
        if cu in ["DATA", "DATE"]:
            col_map[c] = "DATA"
        elif cu in ["MÊS", "MES", "MÊS ", "MES "]:
            col_map[c] = "MES"
        elif cu in ["LINHA", "SETOR"]:
            col_map[c] = "LINHA"
        elif cu in ["META", "META PROD", "META_PROD", "META_PRODUCAO"]:
            col_map[c] = "META_PROD"
        elif cu in ["PRODUÇÃO", "PRODUCAO", "REALIZADO", "PROD", "PRODUCAO_REAL"]:
            col_map[c] = "PRODUCAO"
        elif cu in ["M. PRODT", "M PRODT", "META PRODT", "META_PRODT", "META PRODUTIVIDADE"]:
            col_map[c] = "META_PRODT"
        elif cu in ["PRODT.", "PRODT", "PRODUTIVIDADE", "PRODUTIVIDADE_REAL", "REAL PRODT"]:
            col_map[c] = "PRODT"
        elif cu in ["TURNO"]:
            col_map[c] = "TURNO"

    df = df.rename(columns=col_map)

    # Garante colunas mínimas
    needed = ["DATA", "MES", "LINHA", "META_PROD", "PRODUCAO", "META_PRODT", "PRODT"]
    for n in needed:
        if n not in df.columns:
            df[n] = np.nan

    # Tipos
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
    df["MES"] = df["MES"].astype(str).str.strip().str.upper()
    df["LINHA"] = df["LINHA"].astype(str).str.strip()

    df["META_PROD"] = to_number(df["META_PROD"])
    df["PRODUCAO"] = to_number(df["PRODUCAO"])
    df["META_PRODT"] = to_number(df["META_PRODT"])
    df["PRODT"] = to_number(df["PRODT"])

    # Mantém só linhas válidas com data
    df = df[df["DATA"].notna()].copy()
    return df


def get_password_from_secrets() -> Optional[str]:
    # 1) Secrets do Streamlit
    try:
        if "APP_PASSWORD" in st.secrets:
            v = str(st.secrets["APP_PASSWORD"]).strip()
            return v or None
    except Exception:
        pass
    # 2) Variável de ambiente
    v = os.environ.get("APP_PASSWORD", "").strip()
    return v or None


def require_login():
    pwd = get_password_from_secrets()
    if not pwd:
        st.markdown(
            f"""
            <div class="gno-card" style="border-color:#3b0000;">
              <div style="color:#ffb3b3; font-weight:700;">Senha não configurada.</div>
              <div style="margin-top:8px; color:#fff;">
                <div>✅ <b>Local</b>: crie <code>%USERPROFILE%\\.streamlit\\secrets.toml</code> com:</div>
                <pre style="background:#0b0b0b;border:1px solid #222;padding:10px;border-radius:8px;color:#fff;">APP_PASSWORD = "sua_senha"</pre>
                <div>✅ <b>Streamlit Cloud</b>: Manage app → Secrets →</div>
                <pre style="background:#0b0b0b;border:1px solid #222;padding:10px;border-radius:8px;color:#fff;">APP_PASSWORD = "sua_senha"</pre>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    if st.session_state.get("auth") is True:
        return

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div class='gno-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:20px;font-weight:800;text-align:center;'>Acesso restrito</div>", unsafe_allow_html=True)
        senha = st.text_input("Senha", type="password")
        ok = st.button("Entrar", use_container_width=True)
        if ok:
            if senha == pwd:
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Senha incorreta.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def style_dark(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FG),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=FG),
        ),
        margin=dict(l=30, r=30, t=60, b=40),
    )
    fig.update_xaxes(
        showgrid=False,
        color=FG,
        tickfont=dict(color=FG),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        color=FG,
        tickfont=dict(color=FG),
        zeroline=False,
    )
    return fig


def add_termometro(fig: go.Figure, height: int = 520):
    # Termômetro simples na direita (seta vertical)
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=1.02,
        x1=1.04,
        y0=0.1,
        y1=0.9,
        fillcolor="#0b0b0b",
        line=dict(color="#111", width=1),
        layer="above",
    )
    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path="M 1.03 0.92 L 1.055 0.86 L 1.045 0.86 L 1.045 0.12 L 1.015 0.12 L 1.015 0.86 L 1.005 0.86 Z",
        fillcolor="#1475ff",
        line=dict(color="#1475ff", width=1),
        layer="above",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.03,
        y=0.5,
        text="<b>M<br>E<br>L<br>H<br>O<br>R</b>",
        showarrow=False,
        font=dict(color="white", size=10),
        align="center",
    )


def render_minitabela_html(dates_lbl: list[str], meta: list[float], real: list[float], dif: list[float], decimals=2):
    # Média
    def _mean(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce")
        return float(x.mean()) if x.notna().any() else np.nan

    meta_m = _mean(meta)
    real_m = _mean(real)
    dif_m = _mean(dif)

    cols = dates_lbl + ["MÉDIA"]

    def fmt(v):
        if pd.isna(v):
            return ""
        return f"{v:.{decimals}f}"

    def cell_color(val):
        if pd.isna(val):
            return "#111"
        return GREEN if val >= 0 else RED

    # monta HTML
    html = []
    html.append("<div class='gno-card' style='padding:0; overflow:auto;'>")
    html.append("<table style='border-collapse:collapse; width:100%; min-width:920px; font-size:14px;'>")
    # header
    html.append("<tr>")
    html.append("<th style='background:#000; color:#fff; border:1px solid #111; padding:10px; text-align:left;'> </th>")
    for c in cols:
        html.append(f"<th style='background:#000; color:#fff; border:1px solid #111; padding:10px; text-align:center; font-weight:800;'>{c}</th>")
    html.append("</tr>")

    # META
    html.append("<tr>")
    html.append("<td style='background:#000; color:#fff; border:1px solid #111; padding:10px; font-weight:800;'>Meta</td>")
    for v in meta + [meta_m]:
        html.append(f"<td style='background:#000; color:#fff; border:1px solid #111; padding:10px; text-align:center; font-weight:700;'>{fmt(v)}</td>")
    html.append("</tr>")

    # REAL
    html.append("<tr>")
    html.append("<td style='background:#000; color:#fff; border:1px solid #111; padding:10px; font-weight:800;'>Realizado</td>")
    for v in real + [real_m]:
        html.append(f"<td style='background:#000; color:#fff; border:1px solid #111; padding:10px; text-align:center; font-weight:700;'>{fmt(v)}</td>")
    html.append("</tr>")

    # DIF (Real - Meta) com verde/vermelho
    html.append("<tr>")
    html.append("<td style='background:#000; color:#fff; border:1px solid #111; padding:10px; font-weight:800;'>Diferença</td>")
    for v in dif + [dif_m]:
        bg = cell_color(v)
        txt = "#000" if bg == GREEN else "#fff"
        html.append(
            f"<td style='background:{bg}; color:{txt}; border:1px solid #111; padding:10px; text-align:center; font-weight:900;'>{fmt(v)}</td>"
        )
    html.append("</tr>")

    html.append("</table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def month_order():
    return ["JANEIRO", "FEVEREIRO", "MARCO", "MARÇO", "ABRIL", "MAIO", "JUNHO", "JULHO", "AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO"]


# =========================
# FORECAST: DETECTOR + READER
# =========================
def norm_col(c: str) -> str:
    c = str(c).strip().upper()
    c = (
        c.replace("Á", "A").replace("Ã", "A").replace("Â", "A")
        .replace("É", "E").replace("Ê", "E")
        .replace("Í", "I")
        .replace("Ó", "O").replace("Ô", "O")
        .replace("Ç", "C")
    )
    c = c.replace(".", "").replace("  ", " ")
    return c


def find_forecast_table_any(excel_path: Path) -> Optional[pd.DataFrame]:
    """Procura, em qualquer aba do Excel, uma tabela com LINHA + FOR JAN/FEV/MAR."""
    try:
        xls = pd.ExcelFile(excel_path)
    except Exception:
        return None

    best = None
    best_score = -1

    for sh in xls.sheet_names:
        try:
            tmp = pd.read_excel(excel_path, sheet_name=sh)
            if tmp is None or tmp.empty:
                continue
            tmp = tmp.copy()
            tmp.columns = [norm_col(c) for c in tmp.columns]

            if "LINHA" not in tmp.columns:
                continue

            # normaliza variações de forecast
            # aceita: FOR JAN, FOR FEV, FOR MAR, FOR M, FOR_M, FOR. M...
            # depois renomeia para FOR JAN/FEV/MAR
            cols = set(tmp.columns)

            rename = {}
            if "FOR JAN" not in cols:
                for c in tmp.columns:
                    if c in ["FORJAN", "FORECAST JAN"]:
                        rename[c] = "FOR JAN"
            if "FOR FEV" not in cols:
                for c in tmp.columns:
                    if c in ["FORFEV", "FORECAST FEV"]:
                        rename[c] = "FOR FEV"
            # MAR
            if "FOR MAR" not in cols:
                for c in tmp.columns:
                    if c in ["FOR M", "FORM", "FORMAR", "FORECAST MAR"]:
                        rename[c] = "FOR MAR"

            if rename:
                tmp = tmp.rename(columns=rename)

            score = 0
            for c in ["FOR JAN", "FOR FEV", "FOR MAR"]:
                if c in tmp.columns:
                    score += 1

            if score > best_score and score >= 1:
                tmp["_SHEET"] = sh
                best = tmp
                best_score = score

        except Exception:
            continue

    return best


# =========================
# APP
# =========================
require_login()

# Header (somente logo, sem título)
c_logo, c_spacer = st.columns([1, 6])
with c_logo:
    logo_path = BASE_DIR / LOGO_FILE
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)

st.caption(APP_VERSION)

# Carregar arquivo principal
data_path = find_data_file()
uploaded = st.sidebar.file_uploader("📄 (Opcional) Enviar Excel principal (PROD-PRODT.xlsx)", type=["xlsx"])

if uploaded is not None:
    # salva local temporário no cache da sessão
    tmp_path = BASE_DIR / "_uploaded_main.xlsx"
    tmp_path.write_bytes(uploaded.getbuffer())
    data_path = tmp_path

if not data_path:
    st.error(f"Não encontrei o arquivo {DEFAULT_DATA_FILE} na pasta do app.")
    st.stop()

df = load_data(str(data_path))

# Filtros (SEM TURNO)
st.sidebar.markdown("## Filtros")

meses = [m for m in month_order() if m in df["MES"].unique()]
if not meses:
    meses = sorted(df["MES"].dropna().unique().tolist())

mes_sel = st.sidebar.selectbox("Mês", meses, index=0 if "JANEIRO" not in meses else meses.index("JANEIRO"))

linhas = sorted(df.loc[df["MES"] == mes_sel, "LINHA"].dropna().unique().tolist())
linha_sel = st.sidebar.multiselect("Linha", linhas, default=linhas[:3] if len(linhas) >= 3 else linhas)

df_f = df[(df["MES"] == mes_sel) & (df["LINHA"].isin(linha_sel))].copy()

if df_f.empty:
    st.warning("Sem dados para os filtros selecionados.")
    st.stop()

# período
min_d = df_f["DATA"].min().date()
max_d = df_f["DATA"].max().date()
periodo = st.sidebar.date_input("Período", value=(min_d, max_d))
if isinstance(periodo, tuple) and len(periodo) == 2:
    d0, d1 = periodo
else:
    d0, d1 = min_d, max_d

df_f = df_f[(df_f["DATA"].dt.date >= d0) & (df_f["DATA"].dt.date <= d1)].copy()
df_f = df_f.sort_values("DATA")

# Agregação diária (soma por dia)
# (mantém só dias que existem dados lançados)
agg = (
    df_f.groupby("DATA", as_index=False)
        .agg(
            META_PROD=("META_PROD", "sum"),
            PRODUCAO=("PRODUCAO", "sum"),
            META_PRODT=("META_PRODT", "mean"),
            PRODT=("PRODT", "mean"),
        )
)

# Labels das datas (dd/mm) somente dias existentes
agg["DIA_TXT"] = agg["DATA"].dt.strftime("%d/%m")
dates_lbl = agg["DIA_TXT"].tolist()

# Diferenças (Realizado - Meta) — conforme sua referência do Excel
agg["DIF_PROD"] = agg["PRODUCAO"] - agg["META_PROD"]
agg["DIF_PRODT"] = agg["PRODT"] - agg["META_PRODT"]

# =========================
# CONTROLES VISUAIS
# =========================
st.sidebar.markdown("---")
mostrar_minitabela = st.sidebar.checkbox("Mostrar mini-tabela", value=True)
mostrar_termometro = st.sidebar.checkbox("Mostrar termômetro", value=True)

# =========================
# 1) PRODUÇÃO
# =========================
st.markdown(f"## PRODUÇÃO — {mes_sel}")

fig1 = go.Figure()

# Barra Produção (laranja suave com “gradiente” por sobreposição)
fig1.add_trace(go.Bar(
    x=dates_lbl,
    y=agg["PRODUCAO"],
    name="Produção",
    marker=dict(color=ORANGE, line=dict(color="#000", width=0.2)),
    opacity=0.95,
))
fig1.add_trace(go.Bar(
    x=dates_lbl,
    y=(agg["PRODUCAO"] * 0.75).fillna(0),
    name="",
    showlegend=False,
    hoverinfo="skip",
    marker=dict(color=ORANGE_SOFT, line=dict(color="#000", width=0.0)),
    opacity=0.45,
))

# Linha Meta
fig1.add_trace(go.Scatter(
    x=dates_lbl,
    y=agg["META_PROD"],
    name="Meta",
    mode="lines+markers",
    line=dict(color=META_LINE, width=1.1),
    marker=dict(size=5, color=META_LINE),
))

# Linha Diferença (Real - Meta) dentro do gráfico (eixo secundário)
dif = agg["DIF_PROD"].tolist()
dif_colors = [GREEN if (pd.notna(v) and v >= 0) else RED for v in dif]

fig1.add_trace(go.Scatter(
    x=dates_lbl,
    y=dif,
    name="Diferença (Real - Meta)",
    mode="lines+markers",
    line=dict(color=DIFF_LINE, width=1.0, dash="dash"),
    marker=dict(size=7, color=dif_colors),
    yaxis="y2",
))

# eixo secundário
dmin = float(np.nanmin(agg["DIF_PROD"])) if agg["DIF_PROD"].notna().any() else -1
dmax = float(np.nanmax(agg["DIF_PROD"])) if agg["DIF_PROD"].notna().any() else 1
pad = max(10.0, (dmax - dmin) * 0.25) if (dmax - dmin) != 0 else 50.0

fig1.update_layout(
    height=520,
    barmode="overlay",
    xaxis=dict(type="category"),
    yaxis=dict(title="Produção"),
    yaxis2=dict(title="Diferença", overlaying="y", side="right", showgrid=False, range=[dmin - pad, dmax + pad]),
)

if mostrar_termometro:
    add_termometro(fig1, 520)

st.plotly_chart(style_dark(fig1), use_container_width=True)

if mostrar_minitabela:
    render_minitabela_html(
        dates_lbl=dates_lbl,
        meta=agg["META_PROD"].tolist(),
        real=agg["PRODUCAO"].tolist(),
        dif=agg["DIF_PROD"].tolist(),
        decimals=0
    )

st.markdown("---")

# =========================
# 2) PRODUTIVIDADE
# =========================
st.markdown(f"## PRODUTIVIDADE — {mes_sel}")

fig2 = go.Figure()

# Linha Realizado
fig2.add_trace(go.Scatter(
    x=dates_lbl,
    y=agg["PRODT"],
    name="Realizado",
    mode="lines+markers",
    line=dict(color=ORANGE, width=1.1),
    marker=dict(size=6, color=ORANGE),
))
# Linha Meta
fig2.add_trace(go.Scatter(
    x=dates_lbl,
    y=agg["META_PRODT"],
    name="Meta",
    mode="lines+markers",
    line=dict(color=META_LINE, width=1.1),
    marker=dict(size=5, color=META_LINE),
))

# Linha Diferença (eixo secundário)
difp = agg["DIF_PRODT"].tolist()
difp_colors = [GREEN if (pd.notna(v) and v >= 0) else RED for v in difp]

fig2.add_trace(go.Scatter(
    x=dates_lbl,
    y=difp,
    name="Diferença (Real - Meta)",
    mode="lines+markers",
    line=dict(color=DIFF_LINE, width=1.0, dash="dash"),
    marker=dict(size=7, color=difp_colors),
    yaxis="y2",
))

dmin2 = float(np.nanmin(agg["DIF_PRODT"])) if agg["DIF_PRODT"].notna().any() else -1
dmax2 = float(np.nanmax(agg["DIF_PRODT"])) if agg["DIF_PRODT"].notna().any() else 1
pad2 = max(0.2, (dmax2 - dmin2) * 0.25) if (dmax2 - dmin2) != 0 else 1.0

fig2.update_layout(
    height=520,
    xaxis=dict(type="category"),
    yaxis=dict(title="Produtividade"),
    yaxis2=dict(title="Diferença", overlaying="y", side="right", showgrid=False, range=[dmin2 - pad2, dmax2 + pad2]),
)

if mostrar_termometro:
    add_termometro(fig2, 520)

st.plotly_chart(style_dark(fig2), use_container_width=True)

if mostrar_minitabela:
    render_minitabela_html(
        dates_lbl=dates_lbl,
        meta=agg["META_PRODT"].tolist(),
        real=agg["PRODT"].tolist(),
        dif=agg["DIF_PRODT"].tolist(),
        decimals=2
    )

st.markdown("---")

# =========================
# 3) TENDÊNCIA / PROJEÇÃO (Produção e Produtividade)
#    (Simples e estável: usa média diária do período filtrado * dias úteis)
# =========================
st.markdown("## ANÁLISE — TENDÊNCIA (Próximos meses)")

# meses futuros (3 meses à frente)
last_date = df_f["DATA"].max()
base_year = last_date.year
base_month = last_date.month

def business_days_in_month(year: int, month: int) -> int:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthBegin(1))
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    # seg-sex
    return int(sum(d.weekday() < 5 for d in days))

# taxa diária observada (somente dias com dados)
prod_daily = agg["PRODUCAO"].mean()
prodt_daily = agg["PRODT"].mean()

future = []
for i in range(1, 4):
    t = (pd.Timestamp(year=base_year, month=base_month, day=1) + pd.DateOffset(months=i))
    bd = business_days_in_month(t.year, t.month)
    future.append({
        "MES_START": pd.Timestamp(year=t.year, month=t.month, day=1),
        "BUSINESS_DAYS": bd,
        "PROD_PROJ": float(prod_daily * bd) if pd.notna(prod_daily) else np.nan,
        "PRODT_PROJ": float(prodt_daily) if pd.notna(prodt_daily) else np.nan,  # produtividade projetada como nível
    })

proj = pd.DataFrame(future)

# mostra gráfico simples de tendência (por linha selecionada já agregada)
fig_tr = go.Figure()
fig_tr.add_trace(go.Bar(
    x=proj["MES_START"].dt.strftime("%b/%Y"),
    y=proj["PROD_PROJ"],
    name="Produção projetada",
    marker=dict(color=ORANGE, line=dict(color="#000", width=0.2)),
    opacity=0.95
))
fig_tr.add_trace(go.Bar(
    x=proj["MES_START"].dt.strftime("%b/%Y"),
    y=(proj["PROD_PROJ"] * 0.75).fillna(0),
    name="",
    showlegend=False,
    hoverinfo="skip",
    marker=dict(color=ORANGE_SOFT, line=dict(color="#000", width=0.0)),
    opacity=0.45
))
fig_tr.update_layout(height=420, yaxis=dict(title="Produção (projeção)"))
st.plotly_chart(style_dark(fig_tr), use_container_width=True)

# =========================
# 4) FORECAST x PRODUÇÃO PROJETADA — JAN/FEV/MAR + RANKING
# =========================
st.markdown("## FORECAST x PRODUÇÃO PROJETADA — JAN/FEV/MAR")

# leitura do forecast: tenta achar dentro do MESMO arquivo principal
forecast_df = find_forecast_table_any(data_path)

# permite upload de outro arquivo só com forecast (opcional)
up_fc = st.sidebar.file_uploader("📈 (Opcional) Enviar Excel de Forecast (JAN/FEV/MAR)", type=["xlsx"], key="fc_up")
if up_fc is not None:
    tmp_fc = BASE_DIR / "_uploaded_forecast.xlsx"
    tmp_fc.write_bytes(up_fc.getbuffer())
    found = find_forecast_table_any(tmp_fc)
    if found is not None:
        forecast_df = found

if forecast_df is None:
    st.warning(
        "Não encontrei a tabela de Forecast automaticamente.\n\n"
        "✅ Solução: envie um Excel (na lateral) com colunas: LINHA + FOR. JAN + FOR. FEV + FOR. MAR (ou FOR. M)."
    )
else:
    # normaliza
    forecast_df = forecast_df.copy()
    forecast_df.columns = [norm_col(c) for c in forecast_df.columns]

    # renomeia MAR se vier como FOR M
    if "FOR MAR" not in forecast_df.columns and "FOR M" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"FOR M": "FOR MAR"})

    for c in ["FOR JAN", "FOR FEV", "FOR MAR"]:
        if c not in forecast_df.columns:
            forecast_df[c] = np.nan

    forecast_df["LINHA"] = forecast_df["LINHA"].astype(str).str.strip()
    for c in ["FOR JAN", "FOR FEV", "FOR MAR"]:
        forecast_df[c] = to_number(forecast_df[c])

    # Seleciona somente linhas que estão no filtro atual (se houver)
    forecast_use = forecast_df[forecast_df["LINHA"].isin(linha_sel)].copy()
    if forecast_use.empty:
        st.warning("Forecast encontrado, porém nenhuma LINHA bate com as linhas filtradas.")
    else:
        # --- Define ano alvo: próximo ano do último mês observado (seguro)
        last_month_ts = df["DATA"].max().to_period("M").to_timestamp()
        target_year = (last_month_ts + pd.offsets.MonthBegin(1)).year

        months_target = [
            pd.Timestamp(year=target_year, month=1, day=1),
            pd.Timestamp(year=target_year, month=2, day=1),
            pd.Timestamp(year=target_year, month=3, day=1),
        ]

        # Projeção por LINHA (usa média diária por LINHA * dias úteis)
        df_line = df[(df["LINHA"].isin(linha_sel)) & (df["MES"] == mes_sel)].copy()
        df_line = df_line[(df_line["DATA"].dt.date >= d0) & (df_line["DATA"].dt.date <= d1)].copy()

        # média diária por linha no período filtrado
        daily_line = (
            df_line.groupby(["LINHA", "DATA"], as_index=False)
                   .agg(PRODUCAO=("PRODUCAO", "sum"))
        )
        mean_daily_line = daily_line.groupby("LINHA", as_index=False)["PRODUCAO"].mean()
        mean_daily_line = mean_daily_line.rename(columns={"PRODUCAO": "MEAN_DAILY"})

        # proj por linha para JAN/FEV/MAR
        rows = []
        for m in months_target:
            bd = business_days_in_month(m.year, m.month)
            tmp = mean_daily_line.copy()
            tmp["MES_START"] = m
            tmp["BUSINESS_DAYS"] = bd
            tmp["PROD_PROJ"] = tmp["MEAN_DAILY"] * bd
            rows.append(tmp)

        proj_line = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["LINHA", "MES_START", "PROD_PROJ"])
        proj_pivot = (
            proj_line.assign(M=proj_line["MES_START"].dt.month)
                     .pivot_table(index="LINHA", columns="M", values="PROD_PROJ", aggfunc="sum")
                     .reset_index()
        )
        proj_pivot = proj_pivot.rename(columns={1: "PROJ JAN", 2: "PROJ FEV", 3: "PROJ MAR"})
        for c in ["PROJ JAN", "PROJ FEV", "PROJ MAR"]:
            if c not in proj_pivot.columns:
                proj_pivot[c] = np.nan

        comp = forecast_use.merge(proj_pivot, on="LINHA", how="left")

        # ===== GRÁFICO POR LINHA
        linhas_disp = sorted(comp["LINHA"].dropna().unique().tolist())
        linha_fc = st.selectbox("Linha (Forecast x Projeção)", linhas_disp, index=0)
        row = comp[comp["LINHA"] == linha_fc].iloc[0]

        meses = ["JAN", "FEV", "MAR"]
        y_fore = [row["FOR JAN"], row["FOR FEV"], row["FOR MAR"]]
        y_proj = [row["PROJ JAN"], row["PROJ FEV"], row["PROJ MAR"]]

        dif = [((p if pd.notna(p) else 0) - (f if pd.notna(f) else 0)) for p, f in zip(y_proj, y_fore)]
        dif_colors = [GREEN if d >= 0 else RED for d in dif]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Bar(
            x=meses, y=y_fore, name="Forecast",
            marker=dict(color="#2f2f2f", line=dict(color="#000", width=0.2)),
            opacity=0.95
        ))
        fig_fc.add_trace(go.Bar(
            x=meses, y=y_proj, name="Produção Projetada",
            marker=dict(color=ORANGE, line=dict(color="#000", width=0.2)),
            opacity=0.95
        ))
        fig_fc.add_trace(go.Bar(
            x=meses, y=[(v * 0.75 if pd.notna(v) else 0) for v in y_proj],
            name="", showlegend=False, hoverinfo="skip",
            marker=dict(color=ORANGE_SOFT, line=dict(color="#000", width=0.0)),
            opacity=0.45
        ))
        fig_fc.add_trace(go.Scatter(
            x=meses, y=dif, name="Diferença (Proj - For)",
            mode="lines+markers",
            line=dict(color=DIFF_LINE, width=1.0, dash="dash"),
            marker=dict(color=dif_colors, size=8),
            yaxis="y2"
        ))

        dmin = float(np.nanmin(dif)) if len(dif) else -1
        dmax = float(np.nanmax(dif)) if len(dif) else 1
        pad = max(10.0, (dmax - dmin) * 0.25) if (dmax - dmin) != 0 else 50.0

        fig_fc.update_layout(
            height=460,
            barmode="group",
            yaxis=dict(title="Quantidade"),
            yaxis2=dict(title="Diferença", overlaying="y", side="right", showgrid=False, range=[dmin - pad, dmax + pad]),
        )
        if mostrar_termometro:
            add_termometro(fig_fc, 460)

        st.plotly_chart(style_dark(fig_fc), use_container_width=True)

        if mostrar_minitabela:
            # mini tabela do forecast vs proj (meta=forecast, real=proj)
            render_minitabela_html(meses, y_fore, y_proj, dif, decimals=0)

        st.markdown("---")
        st.markdown("## RANKING — (PROJ - FORECAST) | JAN+FEV+MAR")

        # ===== RANKING
        comp_rank = comp.copy()
        comp_rank["SUM_FORE"] = comp_rank[["FOR JAN", "FOR FEV", "FOR MAR"]].sum(axis=1, skipna=True)
        comp_rank["SUM_PROJ"] = comp_rank[["PROJ JAN", "PROJ FEV", "PROJ MAR"]].sum(axis=1, skipna=True)
        comp_rank["DELTA"] = comp_rank["SUM_PROJ"] - comp_rank["SUM_FORE"]

        comp_rank = comp_rank.sort_values("DELTA", ascending=False)

        top_n = st.slider("Top N", 5, 30, 10)
        view = comp_rank.head(top_n)

        fig_rank = go.Figure()
        colors = [GREEN if v >= 0 else RED for v in view["DELTA"].fillna(0)]
        fig_rank.add_trace(go.Bar(
            x=view["LINHA"],
            y=view["DELTA"],
            name="Δ (Proj - For)",
            marker=dict(color=colors, line=dict(color="#000", width=0.2)),
            opacity=0.95
        ))
        fig_rank.update_layout(
            height=420,
            xaxis=dict(title="", tickangle=-25),
            yaxis=dict(title="Diferença total (JAN+FEV+MAR)")
        )
        st.plotly_chart(style_dark(fig_rank), use_container_width=True)

        # tabela pequena do ranking
        st.dataframe(
            view[["LINHA", "SUM_PROJ", "SUM_FORE", "DELTA"]].rename(columns={
                "SUM_PROJ": "PROJ (JAN+FEV+MAR)",
                "SUM_FORE": "FORECAST (JAN+FEV+MAR)",
                "DELTA": "DIFERENÇA"
            }),
            use_container_width=True,
            height=280
        )
