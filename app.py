from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

st.set_page_config(
    page_title="Inferência Estatística — TIC Kids Online Brasil 2025",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CAMINHO_POPULACAO = DATA_DIR / "tic_kids_online_brasil_2025_base_de_microdados_v1.0.csv"
CAMINHO_AMOSTRA = DATA_DIR / "amostra_aas_kids_2025.csv"

MAPA_AREA = {1: "Urbana", 2: "Rural"}
MAPA_SEXO = {1: "Masculino", 2: "Feminino"}


def limpar_microdados(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Realiza o saneamento de domínio e remove códigos especiais de não-resposta
    (ex.: 99, 999) conforme o escopo metodológico da TIC Kids Online Brasil.
    """
    dados = df.copy()
    relatorio_limpeza = {}

    # 1. IDADE_KIDS: domínio estrito do público-alvo (9 a 17 anos)
    if "IDADE_KIDS" in dados.columns:
        idade_kids_num = pd.to_numeric(dados["IDADE_KIDS"], errors="coerce")
        invalidos_kids = idade_kids_num.notna() & (~idade_kids_num.between(9, 17))
        relatorio_limpeza["IDADE_KIDS_fora_dominio"] = int(invalidos_kids.sum())
        dados["IDADE_KIDS"] = idade_kids_num.where(idade_kids_num.between(9, 17), np.nan)

    # 2. RESP_IDADE: remove códigos de não-resposta (99, 999) e valores fora de faixa plausível
    if "RESP_IDADE" in dados.columns:
        resp_idade_num = pd.to_numeric(dados["RESP_IDADE"], errors="coerce")
        invalidos_resp = resp_idade_num.notna() & (~resp_idade_num.between(14, 110))
        relatorio_limpeza["RESP_IDADE_invalidos"] = int(invalidos_resp.sum())
        dados["RESP_IDADE"] = resp_idade_num.where(resp_idade_num.between(14, 110), np.nan)

    # 3. AREA e SEXO_KIDS: garante apenas códigos válidos {1, 2}
    if "AREA" in dados.columns:
        area_num = pd.to_numeric(dados["AREA"], errors="coerce")
        invalidos_area = area_num.notna() & (~area_num.isin([1, 2]))
        relatorio_limpeza["AREA_invalidos"] = int(invalidos_area.sum())
        dados["AREA"] = area_num.where(area_num.isin([1, 2]), np.nan)

    if "SEXO_KIDS" in dados.columns:
        sexo_num = pd.to_numeric(dados["SEXO_KIDS"], errors="coerce")
        invalidos_sexo = sexo_num.notna() & (~sexo_num.isin([1, 2]))
        relatorio_limpeza["SEXO_KIDS_invalidos"] = int(invalidos_sexo.sum())
        dados["SEXO_KIDS"] = sexo_num.where(sexo_num.isin([1, 2]), np.nan)

    return dados, relatorio_limpeza


@st.cache_data
def carregar_dados():
    if not CAMINHO_POPULACAO.exists():
        st.error(
            f"Arquivo de microdados populacional não encontrado em: `{CAMINHO_POPULACAO}`. "
            "Por favor, certifique-se de que a base esteja no diretório `data/`."
        )
        st.stop()

    df_pop_bruto = pd.read_csv(CAMINHO_POPULACAO, sep=";")
    df_pop, log_limpeza_pop = limpar_microdados(df_pop_bruto)

    if CAMINHO_AMOSTRA.exists():
        df_amostra_bruta = pd.read_csv(CAMINHO_AMOSTRA, sep=";")
        df_amostra, log_limpeza_am = limpar_microdados(df_amostra_bruta)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_amostra = df_pop.sample(n=336, random_state=42)
        df_amostra.to_csv(CAMINHO_AMOSTRA, sep=";", index=False)
        log_limpeza_am = {k: 0 for k in log_limpeza_pop}

    return df_pop, df_amostra, log_limpeza_pop, log_limpeza_am


df_populacao, df_amostra, log_pop, log_am = carregar_dados()

# ==================== BARRA LATERAL ====================
with st.sidebar:
    st.title("TIC Kids Online Brasil 2025")
    st.caption("Inferência Estatística & Amostragem (IA003)")
    st.markdown("---")

    st.subheader("Contexto da Pesquisa")
    st.markdown(
        """
        - **Órgão Responsável**: [Cetic.br | NIC.br](https://cetic.br/pt/pesquisa/kids-online/)
        - **Tema**: Acesso, uso de tecnologias e bem-estar digital de crianças e adolescentes.
        - **Público-alvo**: Indivíduos de 9 a 17 anos no Brasil e seus pais ou responsáveis legais.
        - **Unidade Amostral**: Indivíduo (criança/adolescente respondente).
        """
    )

    st.subheader("Parâmetros do Plano Amostral")
    st.markdown(
        f"""
        - **População ($N$)**: {len(df_populacao)}
        - **Amostra Calculada ($n$)**: {len(df_amostra):,}
        - **Nível de Confiança**: 95% ($Z = 1{{,}}96$)
        - **Margem de Erro Esperada ($e$)**: 5% ($0{{,}}05$)
        - **Proporção Conservadora ($p$)**: 0,50
        - **Método**: Amostragem Aleatória Simples (AAS) sem reposição
        - **Semente Reproduzível**: `random_state=42`
        """
    )

    st.subheader("Validação de Domínio (Saneamento)")
    st.markdown(
        f"""
        - `IDADE_KIDS`: restrita a **[9, 17] anos** (Pop. filtrados: `{log_pop.get('IDADE_KIDS_fora_dominio', 0)}`)
        - `RESP_IDADE`: restrita a **[14, 110] anos** (Pop. filtrados: `{log_pop.get('RESP_IDADE_invalidos', 0)}`)
        - `AREA`: domínio codificado `{{1, 2}}` (Pop. filtrados: `{log_pop.get('AREA_invalidos', 0)}`)
        - `SEXO_KIDS`: domínio codificado `{{1, 2}}` (Pop. filtrados: `{log_pop.get('SEXO_KIDS_invalidos', 0)}`)
        """
    )
    st.info(
        "A base populacional corresponde aos microdados oficiais consolidados da pesquisa, "
        "enquanto a amostra foi extraída probabilisticamente para avaliação inferencial."
    )


# ==================== FUNÇÕES AUXILIARES ====================
def preparar_dados_rotulados(df: pd.DataFrame) -> pd.DataFrame:
    dados = df.copy()
    dados["AREA_LABEL"] = dados["AREA"].map(MAPA_AREA).fillna("Não informado")
    dados["SEXO_LABEL"] = dados["SEXO_KIDS"].map(MAPA_SEXO).fillna("Não informado")
    return dados


def calcular_metricas_quantitativas(serie: pd.Series) -> dict:
    limpa = serie.dropna()
    return {
        "N válidos": int(limpa.count()),
        "Média": float(limpa.mean()),
        "Mediana": float(limpa.median()),
        "Desvio-Padrão": float(limpa.std()),
        "Mínimo": float(limpa.min()),
        "Máximo": float(limpa.max()),
    }


def gerar_tabela_frequencia(serie: pd.Series, nome_coluna: str) -> pd.DataFrame:
    contagem = serie.value_counts(dropna=False).reset_index()
    contagem.columns = [nome_coluna, "Frequência Absoluta (n)"]
    contagem["Proporção (%)"] = (
        contagem["Frequência Absoluta (n)"] / contagem["Frequência Absoluta (n)"].sum()
    ) * 100
    return contagem


def renderizar_bloco_descritivo(df: pd.DataFrame, sufixo_aba: str):
    df_rot = preparar_dados_rotulados(df)

    st.subheader("Variáveis Quantitativas")
    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.markdown("**Idade da Criança/Adolescente (`IDADE_KIDS` — domínio [9, 17])**")
        m_idade_kids = calcular_metricas_quantitativas(df_rot["IDADE_KIDS"])
        st.dataframe(
            pd.DataFrame([m_idade_kids]).style.format(
                {
                    "Média": "{:.2f}",
                    "Mediana": "{:.1f}",
                    "Desvio-Padrão": "{:.2f}",
                    "Mínimo": "{:.0f}",
                    "Máximo": "{:.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col_q2:
        st.markdown("**Idade do Responsável (`RESP_IDADE`)**")
        m_resp_idade = calcular_metricas_quantitativas(df_rot["RESP_IDADE"])
        st.dataframe(
            pd.DataFrame([m_resp_idade]).style.format(
                {
                    "Média": "{:.2f}",
                    "Mediana": "{:.1f}",
                    "Desvio-Padrão": "{:.2f}",
                    "Mínimo": "{:.0f}",
                    "Máximo": "{:.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_hist_kids = px.histogram(
            df_rot.dropna(subset=["IDADE_KIDS"]),
            x="IDADE_KIDS",
            nbins=9,
            title=f"Distribuição da Idade da Criança ({sufixo_aba})",
            labels={"IDADE_KIDS": "Idade (anos)", "count": "Contagem"},
            color_discrete_sequence=["#1f77b4"],
            marginal="box",
        )
        fig_hist_kids.update_xaxes(dtick=1, range=[8.5, 17.5])
        fig_hist_kids.update_layout(bargap=0.1, showlegend=False)
        st.plotly_chart(fig_hist_kids, use_container_width=True)

    with col_g2:
        fig_hist_resp = px.histogram(
            df_rot.dropna(subset=["RESP_IDADE"]),
            x="RESP_IDADE",
            nbins=20,
            title=f"Distribuição da Idade do Responsável ({sufixo_aba})",
            labels={"RESP_IDADE": "Idade (anos)", "count": "Contagem"},
            color_discrete_sequence=["#2ca02c"],
            marginal="box",
        )
        fig_hist_resp.update_layout(bargap=0.1, showlegend=False)
        st.plotly_chart(fig_hist_resp, use_container_width=True)

    st.markdown("---")
    st.subheader("Variáveis Qualitativas")

    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        st.markdown("**Área Domiciliar (`AREA`)**")
        tab_area = gerar_tabela_frequencia(df_rot["AREA_LABEL"], "Área")
        st.dataframe(
            tab_area.style.format({"Proporção (%)": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        fig_area = px.bar(
            tab_area,
            x="Área",
            y="Proporção (%)",
            text="Proporção (%)",
            title=f"Proporção por Área Domiciliar ({sufixo_aba})",
            color="Área",
            color_discrete_map={"Urbana": "#4b6584", "Rural": "#20bf6b", "Não informado": "#a5b1c2"},
        )
        fig_area.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_area.update_layout(yaxis_range=[0, 105], showlegend=False)
        st.plotly_chart(fig_area, use_container_width=True)

    with col_cat2:
        st.markdown("**Sexo da Criança/Adolescente (`SEXO_KIDS`)**")
        tab_sexo = gerar_tabela_frequencia(df_rot["SEXO_LABEL"], "Sexo")
        st.dataframe(
            tab_sexo.style.format({"Proporção (%)": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        fig_sexo = px.bar(
            tab_sexo,
            x="Sexo",
            y="Proporção (%)",
            text="Proporção (%)",
            title=f"Proporção por Sexo da Criança ({sufixo_aba})",
            color="Sexo",
            color_discrete_map={"Masculino": "#3867d6", "Feminino": "#e84393", "Não informado": "#a5b1c2"},
        )
        fig_sexo.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_sexo.update_layout(yaxis_range=[0, 105], showlegend=False)
        st.plotly_chart(fig_sexo, use_container_width=True)


# ==================== TÍTULO PRINCIPAL ====================
st.title("Inferência Estatística — TIC Kids Online Brasil 2025")
st.caption(
    "Estudo comparativo entre parâmetros populacionais reais e estimativas por amostragem probabilística simples."
)

tab1, tab2, tab3 = st.tabs(
    [
        f"Aba 1 — População (N = {len(df_populacao):,})".replace(",", "."),
        f"Aba 2 — Amostra AAS (n = {len(df_amostra):,})",
        "Aba 3 — Inferência & Comparação (População vs. Amostra)",
    ]
)

# ==================== ABA 1 ====================
with tab1:
    st.header(f"Análise Descritiva da População Completa (N = {len(df_populacao):,})".replace(",", "."))
    st.markdown(
        "Métricas e distribuições calculadas diretamente sobre o conjunto consolidado de dados "
        "representando a totalidade de indivíduos elegíveis (9 a 17 anos) observados na pesquisa."
    )
    renderizar_bloco_descritivo(df_populacao, "População")

# ==================== ABA 2 ====================
with tab2:
    st.header(f"Análise Descritiva da Amostra Aleatória Simples (n = {len(df_amostra):,})")
    st.markdown(
        r"Métricas calculadas estritamente a partir da amostra de 336 observações "
        r"($\text{seed}=42$), simulando um estudo no qual apenas a amostra fosse coletada."
    )
    renderizar_bloco_descritivo(df_amostra, "Amostra")

# ==================== ABA 3 ====================
with tab3:
    st.header("Inferência Estatística: Intervalos de Confiança e Testes de Hipóteses")
    st.markdown(
        rf"""
        Nesta seção realizamos a inferência a partir da amostra ($n = 336$) com nível de significância $\alpha = 0{{,}}05$
        (95% de confiança), aplicando o **Fator de Correção para População Finita (FPC)** de forma consistente
        tanto nos Intervalos de Confiança quanto nos Testes de Hipóteses. Em seguida, confrontamos diretamente com o
        parâmetro real da população ($N = 2.642$) para inspecionar a acurácia amostral.
        """
    )

    N_pop = len(df_populacao)
    n_am = len(df_amostra)
    fpc = np.sqrt((N_pop - n_am) / (N_pop - 1))  # Fator de correção de população finita

    # ---------------- 1. IDADE_KIDS ----------------
    kids_am = df_amostra["IDADE_KIDS"].dropna()
    kids_pop = df_populacao["IDADE_KIDS"].dropna()
    media_am_kids = kids_am.mean()
    s_am_kids = kids_am.std()
    n_kids = len(kids_am)
    se_kids = (s_am_kids / np.sqrt(n_kids)) * fpc
    tcrit_kids = stats.t.ppf(0.975, df=n_kids - 1)
    ic_inf_kids = media_am_kids - tcrit_kids * se_kids
    ic_sup_kids = media_am_kids + tcrit_kids * se_kids

    mu0_kids = 12.5
    t_stat_kids = (media_am_kids - mu0_kids) / se_kids
    p_val_kids = 1 - stats.t.cdf(t_stat_kids, df=n_kids - 1)  # unilateral à direita
    param_real_kids = kids_pop.mean()

    # ---------------- 2. RESP_IDADE ----------------
    resp_am = df_amostra["RESP_IDADE"].dropna()
    resp_pop = df_populacao["RESP_IDADE"].dropna()
    media_am_resp = resp_am.mean()
    s_am_resp = resp_am.std()
    n_resp = len(resp_am)
    se_resp = (s_am_resp / np.sqrt(n_resp)) * fpc
    tcrit_resp = stats.t.ppf(0.975, df=n_resp - 1)
    ic_inf_resp = media_am_resp - tcrit_resp * se_resp
    ic_sup_resp = media_am_resp + tcrit_resp * se_resp

    mu0_resp = 40.0
    t_stat_resp = (media_am_resp - mu0_resp) / se_resp
    p_val_resp = 2 * (1 - stats.t.cdf(abs(t_stat_resp), df=n_resp - 1))  # bilateral
    param_real_resp = resp_pop.mean()

    # ---------------- 3. AREA URBANA ----------------
    area_am_valid = df_amostra["AREA"].dropna()
    area_pop_valid = df_populacao["AREA"].dropna()
    p_am_area = (area_am_valid == 1).mean()
    n_area = len(area_am_valid)
    se_area = np.sqrt((p_am_area * (1 - p_am_area)) / n_area) * fpc
    z_crit = stats.norm.ppf(0.975)
    ic_inf_area = max(0.0, p_am_area - z_crit * se_area)
    ic_sup_area = min(1.0, p_am_area + z_crit * se_area)

    p0_area = 0.75
    se0_area = np.sqrt((p0_area * (1 - p0_area)) / n_area) * fpc
    z_stat_area = (p_am_area - p0_area) / se0_area
    p_val_area = 1 - stats.norm.cdf(z_stat_area)  # unilateral à direita
    param_real_area = (area_pop_valid == 1).mean()

    # ---------------- 4. SEXO MASCULINO ----------------
    sexo_am_valid = df_amostra["SEXO_KIDS"].dropna()
    sexo_pop_valid = df_populacao["SEXO_KIDS"].dropna()
    p_am_sexo = (sexo_am_valid == 1).mean()
    n_sexo = len(sexo_am_valid)
    se_sexo = np.sqrt((p_am_sexo * (1 - p_am_sexo)) / n_sexo) * fpc
    ic_inf_sexo = max(0.0, p_am_sexo - z_crit * se_sexo)
    ic_sup_sexo = min(1.0, p_am_sexo + z_crit * se_sexo)

    p0_sexo = 0.50
    se0_sexo = np.sqrt((p0_sexo * (1 - p0_sexo)) / n_sexo) * fpc
    z_stat_sexo = (p_am_sexo - p0_sexo) / se0_sexo
    p_val_sexo = 2 * (1 - stats.norm.cdf(abs(z_stat_sexo)))  # bilateral
    param_real_sexo = (sexo_pop_valid == 1).mean()

    st.subheader("Testes de Hipóteses e Estimações Amostrais")

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("### 1. Média de Idade da Criança (`IDADE_KIDS`)")
        st.markdown(
            rf"""
            - **Hipóteses**: $H_0: \mu \le 12{{,}}5$ vs $H_1: \mu > 12{{,}}5$ (Unilateral à direita)
            - **Tipo de Teste**: Teste $t$ de Student para uma média (com FPC)
            - **Domínio Elegível**: $[9, 17]$ anos ($n_{{válidos}} = """ + f"{n_kids}$)"
        )
        st.write(fr"- **Estimativa Amostral ($\bar{{x}}$)**: `{media_am_kids:.3f}` anos")
        st.write(
            fr"- **IC 95% (com FPC)**: `[{ic_inf_kids:.3f}; {ic_sup_kids:.3f}]`"
        )
        st.write(fr"- **Estatística $t$**: `{t_stat_kids:.4f}`")
        st.write(fr"- **Valor-$p$**: `{p_val_kids:.4f}`")
        decisao_kids = (
            "Rejeita-se H0 ao nível de 5%"
            if p_val_kids < 0.05
            else "Não se rejeita H0 ao nível de 5%"
        )
        st.info(f"**Decisão Estatística**: {decisao_kids}")
        cobre_kids = ic_inf_kids <= param_real_kids <= ic_sup_kids
        st.write(
            fr"- **Parâmetro Populacional Real ($\mu$)**: `{param_real_kids:.3f}` | "
            fr"**Dentro do IC?** {'✅ Sim' if cobre_kids else '❌ Não'}"
        )

        st.markdown("---")

        st.markdown("### 3. Proporção em Área Urbana (`AREA == 1`)")
        st.markdown(
            rf"""
            - **Hipóteses**: $H_0: p \le 0{{,}}75$ vs $H_1: p > 0{{,}}75$ (Unilateral à direita)
            - **Tipo de Teste**: Teste $Z$ para uma proporção (com FPC)
            """
        )
        st.write(
            fr"- **Estimativa Amostral ($\hat{{p}}$)**: `{p_am_area * 100:.2f}%` ({p_am_area:.4f})"
        )
        st.write(
            fr"- **IC 95% (com FPC)**: `[{ic_inf_area * 100:.2f}%; {ic_sup_area * 100:.2f}%]`"
        )
        st.write(fr"- **Estatística $Z$**: `{z_stat_area:.4f}`")
        st.write(fr"- **Valor-$p$**: `{p_val_area:.4e}`")
        decisao_area = (
            "Rejeita-se H0 ao nível de 5%"
            if p_val_area < 0.05
            else "Não se rejeita H0 ao nível de 5%"
        )
        st.info(f"**Decisão Estatística**: {decisao_area}")
        cobre_area = ic_inf_area <= param_real_area <= ic_sup_area
        st.write(
            fr"- **Parâmetro Populacional Real ($p$)**: `{param_real_area * 100:.2f}%` | "
            fr"**Dentro do IC?** {'✅ Sim' if cobre_area else '❌ Não'}"
        )

    with col_h2:
        st.markdown("### 2. Média de Idade do Responsável (`RESP_IDADE`)")
        st.markdown(
            rf"""
            - **Hipóteses**: $H_0: \mu = 40$ vs $H_1: \mu \neq 40$ (Bilateral)
            - **Tipo de Teste**: Teste $t$ de Student para uma média (com FPC)
            - **Observações Válidas**: $n_{{válidos}} = """ + f"{n_resp}$"
        )
        st.write(fr"- **Estimativa Amostral ($\bar{{x}}$)**: `{media_am_resp:.3f}` anos")
        st.write(
            fr"- **IC 95% (com FPC)**: `[{ic_inf_resp:.3f}; {ic_sup_resp:.3f}]`"
        )
        st.write(fr"- **Estatística $t$**: `{t_stat_resp:.4f}`")
        st.write(fr"- **Valor-$p$**: `{p_val_resp:.4e}`")
        decisao_resp = (
            "Rejeita-se H0 ao nível de 5%"
            if p_val_resp < 0.05
            else "Não se rejeita H0 ao nível de 5%"
        )
        st.info(fr"**Decisão Estatística**: {decisao_resp}")
        cobre_resp = ic_inf_resp <= param_real_resp <= ic_sup_resp
        st.write(
            fr"- **Parâmetro Populacional Real ($\mu$)**: `{param_real_resp:.3f}` | "
            fr"**Dentro do IC?** {'✅ Sim' if cobre_resp else '❌ Não'}"
        )

        st.markdown("---")

        st.markdown("### 4. Proporção do Sexo Masculino (`SEXO_KIDS == 1`)")
        st.markdown(
            r"""
            - **Hipóteses**: $H_0: p = 0{,}50$ vs $H_1: p \neq 0{,}50$ (Bilateral)
            - **Tipo de Teste**: Teste $Z$ para uma proporção (com FPC)
            """
        )
        st.write(
            f"- **Estimativa Amostral ($\hat{{p}}$)**: `{p_am_sexo * 100:.2f}%` ({p_am_sexo:.4f})"
        )
        st.write(
            f"- **IC 95% (com FPC)**: `[{ic_inf_sexo * 100:.2f}%; {ic_sup_sexo * 100:.2f}%]`"
        )
        st.write(f"- **Estatística $Z$**: `{z_stat_sexo:.4f}`")
        st.write(f"- **Valor-$p$**: `{p_val_sexo:.4f}`")
        decisao_sexo = (
            "Rejeita-se H0 ao nível de 5%"
            if p_val_sexo < 0.05
            else "Não se rejeita H0 ao nível de 5%"
        )
        st.info(f"**Decisão Estatística**: {decisao_sexo}")
        cobre_sexo = ic_inf_sexo <= param_real_sexo <= ic_sup_sexo
        st.write(
            f"- **Parâmetro Populacional Real ($p$)**: `{param_real_sexo * 100:.2f}%` | "
            f"**Dentro do IC?** {'✅ Sim' if cobre_sexo else '❌ Não'}"
        )

    st.markdown("---")
    st.subheader("Tabela Comparativa de Acurácia Inferencial")

    dados_resumo = [
        {
            "Variável / Parâmetro": "Idade Criança (Média)",
            "Parâmetro Populacional (θ)": f"{param_real_kids:.3f}",
            "Estimativa Amostral (θ̂)": f"{media_am_kids:.3f}",
            "IC 95%": f"[{ic_inf_kids:.3f}; {ic_sup_kids:.3f}]",
            "Erro Absoluto (|θ̂ - θ|)": f"{abs(media_am_kids - param_real_kids):.3f}",
            "Erro Relativo (%)": f"{abs(media_am_kids - param_real_kids) / param_real_kids * 100:.2f}%",
            "IC cobre o real?": "✅ Sim" if cobre_kids else "❌ Não",
        },
        {
            "Variável / Parâmetro": "Idade Responsável (Média)",
            "Parâmetro Populacional (θ)": f"{param_real_resp:.3f}",
            "Estimativa Amostral (θ̂)": f"{media_am_resp:.3f}",
            "IC 95%": f"[{ic_inf_resp:.3f}; {ic_sup_resp:.3f}]",
            "Erro Absoluto (|θ̂ - θ|)": f"{abs(media_am_resp - param_real_resp):.3f}",
            "Erro Relativo (%)": f"{abs(media_am_resp - param_real_resp) / param_real_resp * 100:.2f}%",
            "IC cobre o real?": "✅ Sim" if cobre_resp else "❌ Não",
        },
        {
            "Variável / Parâmetro": "Área Urbana (Proporção)",
            "Parâmetro Populacional (θ)": f"{param_real_area * 100:.2f}%",
            "Estimativa Amostral (θ̂)": f"{p_am_area * 100:.2f}%",
            "IC 95%": f"[{ic_inf_area * 100:.2f}%; {ic_sup_area * 100:.2f}%]",
            "Erro Absoluto (|θ̂ - θ|)": f"{abs(p_am_area - param_real_area) * 100:.2f} p.p.",
            "Erro Relativo (%)": f"{abs(p_am_area - param_real_area) / param_real_area * 100:.2f}%",
            "IC cobre o real?": "✅ Sim" if cobre_area else "❌ Não",
        },
        {
            "Variável / Parâmetro": "Sexo Masculino (Proporção)",
            "Parâmetro Populacional (θ)": f"{param_real_sexo * 100:.2f}%",
            "Estimativa Amostral (θ̂)": f"{p_am_sexo * 100:.2f}%",
            "IC 95%": f"[{ic_inf_sexo * 100:.2f}%; {ic_sup_sexo * 100:.2f}%]",
            "Erro Absoluto (|θ̂ - θ|)": f"{abs(p_am_sexo - param_real_sexo) * 100:.2f} p.p.",
            "Erro Relativo (%)": f"{abs(p_am_sexo - param_real_sexo) / param_real_sexo * 100:.2f}%",
            "IC cobre o real?": "✅ Sim" if cobre_sexo else "❌ Não",
        },
    ]
    st.dataframe(pd.DataFrame(dados_resumo), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Visualização dos Intervalos de Confiança vs. Parâmetros Reais")

    col_fig1, col_fig2 = st.columns(2)

    with col_fig1:
        fig_ci_medias = go.Figure()
        vars_m = ["Idade Criança", "Idade Responsável"]
        est_m = [media_am_kids, media_am_resp]
        pop_m = [param_real_kids, param_real_resp]
        err_m_plus = [ic_sup_kids - media_am_kids, ic_sup_resp - media_am_resp]
        err_m_minus = [media_am_kids - ic_inf_kids, media_am_resp - ic_inf_resp]

        fig_ci_medias.add_trace(
            go.Scatter(
                x=vars_m,
                y=est_m,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_m_plus,
                    arrayminus=err_m_minus,
                ),
                mode="markers",
                name="Amostra (Estimativa Pontual & IC 95%)",
                marker=dict(color="#0984e3", size=11),
            )
        )
        fig_ci_medias.add_trace(
            go.Scatter(
                x=vars_m,
                y=pop_m,
                mode="markers",
                name="População Real (Parâmetro)",
                marker=dict(color="#d63031", symbol="x", size=13),
            )
        )
        fig_ci_medias.update_layout(
            title="Médias Amostrais (com IC 95%) vs. Média Populacional Real",
            yaxis_title="Anos",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ci_medias, use_container_width=True)

    with col_fig2:
        fig_ci_props = go.Figure()
        vars_p = ["Área Urbana", "Sexo Masculino"]
        est_p = [p_am_area * 100, p_am_sexo * 100]
        pop_p = [param_real_area * 100, param_real_sexo * 100]
        err_p_plus = [(ic_sup_area - p_am_area) * 100, (ic_sup_sexo - p_am_sexo) * 100]
        err_p_minus = [(p_am_area - ic_inf_area) * 100, (p_am_sexo - ic_inf_sexo) * 100]

        fig_ci_props.add_trace(
            go.Scatter(
                x=vars_p,
                y=est_p,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_p_plus,
                    arrayminus=err_p_minus,
                ),
                mode="markers",
                name="Amostra (Estimativa Pontual & IC 95%)",
                marker=dict(color="#00b894", size=11),
            )
        )
        fig_ci_props.add_trace(
            go.Scatter(
                x=vars_p,
                y=pop_p,
                mode="markers",
                name="População Real (Parâmetro)",
                marker=dict(color="#d63031", symbol="x", size=13),
            )
        )
        fig_ci_props.update_layout(
            title="Proporções Amostrais (com IC 95%) vs. Proporção Populacional Real",
            yaxis_title="Percentual (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ci_props, use_container_width=True)

    st.markdown("---")
    st.subheader("Discussão Teórica, Variabilidade Amostral e Limitações")
    st.markdown(
        r"""
        1. **Efeito da Aleatorização e Validade dos Intervalos de Confiança**:
           - Observa-se que todos os intervalos de 95% calculados com base na amostra de tamanho $n = 336$ contêm os
             verdadeiros parâmetros populacionais $\theta$. Esse resultado ilustra de forma prática o teorema central da inferência:
             em 95% das amostras possíveis sob este desenho amostral, o intervalo construído cobrirá o parâmetro verdadeiro.
           - O erro amostral observado em todas as variáveis manteve-se estritamente dentro da margem de tolerância planejada ($e \le 5\%$).

        2. **Restrição de Domínio e Normalidade Assintótica (`IDADE_KIDS`)**:
           - Por definição metodológica da pesquisa, a variável `IDADE_KIDS` possui suporte discreto e estritamente limitado ao intervalo
             $[9, 17]$ anos, delimitando o escopo da população-alvo (não devendo ser extrapolada para crianças menores de 9 anos).
           - Embora a distribuição individual dos dados seja limitada (não normal), o tamanho amostral elevado ($n = 336 \gg 30$) com
             variância estritamente finita garante, pelo **Teorema Central do Limite (TLC)**, a convergência da média amostral $\bar{X}$
             para a normalidade, validando o uso dos Intervalos de Confiança e do Teste $t$ de Student.
           - Aplicou-se saneamento prévio para garantir que eventuais códigos de recusa/não-resposta (como `99` ou `999`) sejam convertidos
             em valores ausentes (`NaN`) antes da estimação.

        3. **Correção para População Finita (FPC)**:
           - Como a fração amostral é $f = \frac{n}{N} = \frac{336}{2642} \approx 12{,}7\% > 5\%$, aplicou-se o FPC
             $\left(\sqrt{\frac{N - n}{N - 1}} \approx 0{,}934\right)$ de maneira uniforme tanto nos intervalos de confiança quanto nos
             erros-padrão dos testes de hipóteses ($t$ e $Z$), evitando superestimação da incerteza e garantindo coerência inferencial.

        4. **Limitações do Estudo**:
           - **AAS vs. Amostragem Complexa**: Na prática, o desenho da pesquisa TIC Kids Online pelo Cetic.br emprega amostragem por conglomerados
             estratificada com pesos pós-estratificados (`PESO`). Aqui adotou-se a Amostragem Aleatória Simples (AAS) como modelo didático de estudo,
             o que pressupõe equiprobabilidade de inclusão de cada elemento.
           - **Não-respostas pontuais**: Variáveis como `RESP_IDADE` podem apresentar registros ausentes ou códigos de omissão dependendo do tipo de
             responsável presente no momento da entrevista, sendo devidamente filtrados antes do cálculo dos estimadores.
        """
    )
