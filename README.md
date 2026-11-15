# Inferência Estatística e Amostragem — TIC Kids Online Brasil 2025 (IA003)

Dashboard interativo desenvolvido em **Streamlit** para análise descritiva e inferencial sobre os microdados oficiais da pesquisa **TIC Kids Online Brasil 2025**. O estudo confronta os verdadeiros parâmetros populacionais ($N = 2.642$) com estimativas pontuais, intervalos de confiança de 95% e testes de hipóteses obtidos a partir de uma Amostra Aleatória Simples ($n = 336$)[cite: 5].

---

## 1. Fonte e Contextualização dos Dados

- **Pesquisa**: TIC Kids Online Brasil 2025[cite: 5]
- **Instituição Responsável**: Cetic.br / NIC.br (Comitê Gestor da Internet no Brasil — CGI.br)[cite: 5]
- **Tema**: Padrões de acesso, uso de tecnologia e bem-estar digital infantojuvenil no Brasil[cite: 5].
- **Link Oficial**: [https://cetic.br/pt/pesquisa/kids-online/](https://cetic.br/pt/pesquisa/kids-online/)[cite: 5]
- **Dimensão da Base**: 2.642 registros e 258 variáveis[cite: 5].
- **Arquivo Populacional**: `data/tic_kids_online_brasil_2025_base_de_microdados_v1.0.csv`[cite: 5]

---

## 2. População, Unidade Amostral e Variáveis

- **População de Interesse ($N$)**: Crianças e adolescentes de 9 a 17 anos na base consolidada ($N = 2.642$)[cite: 5].
- **Unidade Amostral**: Indivíduo respondente no domicílio (com os dados do respectivo responsável legal pareados na mesma linha)[cite: 5].

### Variáveis Selecionadas
1. **`IDADE_KIDS`** *(quantitativa discreta, domínio $[9, 17]$ anos)*: Idade da criança ou adolescente respondente[cite: 5].
2. **`RESP_IDADE`** *(quantitativa)*: Idade em anos completos do pai, mãe ou responsável legal presente[cite: 5].
3. **`AREA`** *(qualitativa binária)*: Localização do domicílio (`1` = Urbana, `2` = Rural)[cite: 5].
4. **`SEXO_KIDS`** *(qualitativa binária)*: Sexo da criança ou adolescente (`1` = Masculino, `2` = Feminino)[cite: 5].

---

## 3. Dimensionamento e Seleção da Amostra (AAS)

O plano amostral adotado é a **Amostragem Aleatória Simples (AAS) sem reposição**, dimensionado com base na Calculadora Amostral da USP/Bauru[cite: 5]:

$$n = \frac{N \cdot Z^2 \cdot p \cdot (1 - p)}{e^2 \cdot (N - 1) + Z^2 \cdot p \cdot (1 - p)}$$

- **Tamanho da População ($N$)**: 2.642[cite: 5]
- **Nível de Confiança ($1 - \alpha$)**: 95% ($Z = 1{,}96$)[cite: 5]
- **Margem de Erro ($e$)**: 5,0% ($0{,}05$)[cite: 5]
- **Proporção Conservadora ($p$)**: 0,50 (máxima variância)[cite: 5]

$$n \approx \frac{2642 \cdot (1{,}96)^2 \cdot 0{,}5 \cdot 0{,}5}{(0{,}05)^2 \cdot 2641 + (1{,}96)^2 \cdot 0{,}5 \cdot 0{,}5} \approx 335{,}4 \implies \mathbf{n = 336}$$

- **Reprodutibilidade**: A seleção equiprobável é executada com `random_state=42` e salva em `data/amostra_aas_kids_2025.csv`[cite: 5].
- **Correção para População Finita (FPC)**: Como a fração amostral $f = \frac{336}{2642} \approx 12{,}7\%$ supera 5%, aplica-se o fator $\text{FPC} = \sqrt{\frac{N - n}{N - 1}} \approx 0{,}9345$ tanto nos Intervalos de Confiança quanto nos erros-padrão dos testes de hipóteses[cite: 5].

---

## 4. Estrutura dos Painéis e Hipóteses Formuladas

A aplicação está organizada em três painéis interativos[cite: 5]:
- **Painel 1 — Análise Descritiva da População ($N = 2.642$)**: Parâmetros reais ($\mu$ e $p$), métricas de dispersão (DP, IQR, CV%) e distribuições[cite: 5].
- **Painel 2 — Análise Descritiva da Amostra ($n = 336$)**: Estatísticas e gráficos calculados exclusivamente sobre a amostra selecionada[cite: 5].
- **Painel 3 — Inferência Estatística & Comparação**: Testes de hipóteses ($\alpha = 0{,}05$), intervalos de confiança de 95%, comparação direta com os parâmetros reais e discussão teórica[cite: 5]:

1. **Média de Idade da Criança (`IDADE_KIDS`)**: $H_0: \mu \le 12{,}5$ vs. $H_1: \mu > 12{,}5$ *(Teste $t$ unilateral à direita)*[cite: 5].
2. **Média de Idade do Responsável (`RESP_IDADE`)**: $H_0: \mu = 40$ vs. $H_1: \mu \neq 40$ *(Teste $t$ bilateral)*[cite: 5].
3. **Proporção em Área Urbana (`AREA == 1`)**: $H_0: p \le 0{,}75$ vs. $H_1: p > 0{,}75$ *(Teste $Z$ unilateral à direita)*[cite: 5].
4. **Proporção do Sexo Masculino (`SEXO_KIDS == 1`)**: $H_0: p = 0{,}50$ vs. $H_1: p \neq 0{,}50$ *(Teste $Z$ bilateral)*[cite: 5].

---

## 5. Instalação e Execução Local

### Pré-requisitos
- Python 3.10+
- Base de microdados presente em `data/tic_kids_online_brasil_2025_base_de_microdados_v1.0.csv`[cite: 5]

### Passo a Passo

1. **Instalar as dependências**:
   ```bash
   pip install -r requirements.txt