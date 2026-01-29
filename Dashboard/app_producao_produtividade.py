# =========================
# 4) FORECAST x PRODUÇÃO PROJETADA (JAN/FEV/MAR)
# =========================
st.markdown("## FORECAST x PRODUÇÃO PROJETADA — JAN/FEV/MAR")

def _norm_col(c: str) -> str:
    c = str(c).strip().upper()
    c = c.replace("Á", "A").replace("Ã", "A").replace("Â", "A")
    c = c.replace("É", "E").replace("Ê", "E")
    c = c.replace("Í", "I")
    c = c.replace("Ó", "O").replace("Ô", "O")
    c = c.replace("Ç", "C")
    c = c.replace(".", "").replace("  ", " ")
    return c

def _find_forecast_table_in_excel(excel_path: Path) -> pd.DataFrame | None:
    """Tenta localizar automaticamente uma tabela com LINHA + FOR JAN/FEV/MAR em qualquer aba."""
    try:
        xls = pd.ExcelFile(excel_path)
    except Exception:
        return None

    candidates = []
    for sh in xls.sheet_names:
        try:
            tmp = pd.read_excel(excel_path, sheet_name=sh)
            if tmp is None or tmp.empty:
                continue
            cols = {c: _norm_col(c) for c in tmp.columns}
            norm = [cols[c] for c in tmp.columns]

            has_linha = any(n == "LINHA" for n in norm)
            has_jan = any(n in ["FOR JAN", "FORJAN", "FORECAST JAN"] for n in norm)
            has_fev = any(n in ["FOR FEV", "FORFEV", "FORECAST FEV"] for n in norm)
            has_mar = any(n in ["FOR M", "FOR MAR", "FORMAR", "FORECAST MAR"] for n in norm)

            if has_linha and (has_jan or has_fev or has_mar):
                tmp2 = tmp.copy()
                tmp2.columns = [_norm_col(c) for c in tmp2.columns]
                tmp2["_SHEET"] = sh
                candidates.append(tmp2)
        except Exception:
            continue

    if not candidates:
        return None

    # pega a candidata com mais colunas FOR encontradas
    def score(d):
        s = 0
        for c in d.columns:
            if c in ["FOR JAN", "FOR FEV", "FOR M", "FOR MAR"]:
                s += 1
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0]

forecast_df = _find_forecast_table_in_excel(excel_path)

if forecast_df is None:
    st.warning(
        "Não encontrei automaticamente a tabela de Forecast no Excel.\n\n"
        "Garanta que exista uma aba com colunas: LINHA, FOR. JAN, FOR. FEV, FOR. MAR (ou FOR. M)."
    )
else:
    # Normaliza/seleciona colunas necessárias
    if "FOR MAR" not in forecast_df.columns and "FOR M" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"FOR M": "FOR MAR"})

    # Algumas planilhas podem ter "FOR M" e "FOR MAR" juntos: prioriza FOR MAR
    if "FOR MAR" not in forecast_df.columns:
        # tenta achar qualquer coisa que pareça MAR
        for c in forecast_df.columns:
            if c.startswith("FOR ") and (" MAR" in c or c.endswith("MAR")):
                forecast_df = forecast_df.rename(columns={c: "FOR MAR"})
                break

    needed_any = ["FOR JAN", "FOR FEV", "FOR MAR"]
    # cria colunas faltantes como NaN pra não quebrar
    for c in needed_any:
        if c not in forecast_df.columns:
            forecast_df[c] = np.nan

    # converte números (aceita 8.010 / 9510 etc)
    for c in ["FOR JAN", "FOR FEV", "FOR MAR"]:
        forecast_df[c] = to_number(forecast_df[c])

    forecast_df["LINHA"] = forecast_df["LINHA"].astype(str).str.strip()

    # Filtra só linhas escolhidas no app
    forecast_df = forecast_df[forecast_df["LINHA"].isin(linha_sel)].copy()

    # --- Projeção (proj) -> precisamos do mês alvo JAN/FEV/MAR do ano que você quer
    # Aqui vamos comparar com os 3 primeiros meses projetados (Jan/Fev/Mar) do ANO seguinte ao último mês do dataset.
    # Se você quiser comparar com Jan/Fev/Mar do mesmo ano, me diga que eu ajusto.
    last_month = sorted(df["DATA"].dropna().dt.to_period("M").unique().to_timestamp().tolist())[-1]
    target_year = (last_month + pd.offsets.MonthBegin(1)).year

    months_target = [
        pd.Timestamp(year=target_year, month=1, day=1),
        pd.Timestamp(year=target_year, month=2, day=1),
        pd.Timestamp(year=target_year, month=3, day=1),
    ]

    proj_cmp = proj.copy()
    proj_cmp = proj_cmp[proj_cmp["MES_START"].isin(months_target)].copy()

    # vira colunas "PROJ JAN/FEV/MAR"
    proj_pivot = (
        proj_cmp.assign(MES=proj_cmp["MES_START"].dt.month)
               .pivot_table(index="LINHA", columns="MES", values="PROD_PROJ", aggfunc="sum")
               .reset_index()
    )
    proj_pivot = proj_pivot.rename(columns={
        1: "PROJ JAN",
        2: "PROJ FEV",
        3: "PROJ MAR",
    })
    for c in ["PROJ JAN", "PROJ FEV", "PROJ MAR"]:
        if c not in proj_pivot.columns:
            proj_pivot[c] = np.nan

    # merge Forecast x Projeção
    comp = forecast_df.merge(proj_pivot, on="LINHA", how="left")

    # seletor de linha (pra ficar legível)
    linhas_disp = sorted(comp["LINHA"].dropna().unique().tolist())
    if not linhas_disp:
        st.warning("Nenhuma LINHA em comum entre Forecast e filtros atuais.")
    else:
        linha_fc = st.selectbox("Linha (Forecast x Projeção)", linhas_disp, index=0)
        row = comp[comp["LINHA"] == linha_fc].iloc[0]

        meses = ["JAN", "FEV", "MAR"]
        y_fore = [row["FOR JAN"], row["FOR FEV"], row["FOR MAR"]]
        y_proj = [row["PROJ JAN"], row["PROJ FEV"], row["PROJ MAR"]]

        # gráfico
        fig_fc = go.Figure()

        # Forecast (cinza)
        fig_fc.add_trace(go.Bar(
            x=meses, y=y_fore, name="Forecast",
            marker=dict(color="#3a3a3a", line=dict(color="#000000", width=0.2)),
            opacity=0.95
        ))

        # Projeção (laranja gradiente)
        # (usa a mesma ideia: barra “base” + barra “clara” por cima)
        fig_fc.add_trace(go.Bar(
            x=meses, y=y_proj, name="Produção Projetada",
            marker=dict(color=ORANGE_BAR_DARK, line=dict(color="#000000", width=0.2)),
            opacity=0.95
        ))
        fig_fc.add_trace(go.Bar(
            x=meses, y=[(v * 0.75 if pd.notna(v) else 0) for v in y_proj],
            name="", showlegend=False, hoverinfo="skip",
            marker=dict(color=ORANGE_BAR_LIGHT, line=dict(color="#000000", width=0.0)),
            opacity=0.55
        ))

        # Diferença (Projetada - Forecast) dentro do gráfico
        dif = [( (p if pd.notna(p) else 0) - (f if pd.notna(f) else 0) ) for p, f in zip(y_proj, y_fore)]
        dif_colors = [GREEN if d >= 0 else RED for d in dif]

        fig_fc.add_trace(go.Scatter(
            x=meses, y=dif, name="Diferença (Proj - For)",
            mode="lines+markers",
            line=dict(color=DIFF_LINE, width=1.1, dash="dash"),
            marker=dict(color=dif_colors, size=8),
            yaxis="y2"
        ))

        dmin = float(np.nanmin(dif)) if len(dif) else -1
        dmax = float(np.nanmax(dif)) if len(dif) else 1
        pad = max(10.0, (dmax - dmin) * 0.25) if (dmax - dmin) != 0 else 50.0

        fig_fc.update_layout(
            title=f"{linha_fc} — {target_year} (Forecast x Projeção)",
            height=460,
            barmode="group",
            xaxis=dict(type="category", showgrid=False),
            yaxis=dict(title="Quantidade", showgrid=True, gridcolor=GRID),
            yaxis2=dict(
                title="Diferença",
                overlaying="y",
                side="right",
                showgrid=False,
                range=[dmin - pad, dmax + pad]
            )
        )

        if mostrar_termometro:
            add_termometro(fig_fc, 460)

        st.plotly_chart(style_dark(fig_fc), use_container_width=True)

        # mini-tabela do Forecast x Projeção
        if mostrar_minitabela:
            dias = ["JAN", "FEV", "MAR"]
            render_minitabela_html(
                dias,
                meta_list=y_fore,        # aqui "Meta" = Forecast
                real_list=y_proj,        # aqui "Realizado" = Projeção
                dif_list=dif,
                decimals=0
            )

        st.caption(f"Tabela Forecast encontrada na aba: **{forecast_df.get('_SHEET', 'desconhecida')}**")
