from pathlib import Path
import pandas as pd


def main():
    base_dir = Path(__file__).resolve().parent
    caminho_dados = base_dir / "data" / "tic_kids_online_brasil_2025_base_de_microdados_v1.0.csv"
    caminho_amostra = base_dir / "data" / "amostra_aas_kids_2025.csv"

    if not caminho_dados.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {caminho_dados}")

    print(f"Carregando dados de: {caminho_dados}")
    df_populacao = pd.read_csv(caminho_dados, sep=";")
    pop_size = len(df_populacao)

    n_amostra = 336
    df_amostra = df_populacao.sample(n=n_amostra, random_state=42)

    df_amostra.to_csv(caminho_amostra, sep=";", index=False)

    print(f"Confirmação:")
    print(f"- Tamanho da população (N): {pop_size}")
    print(f"- Tamanho da amostra gerada (n): {len(df_amostra)}")
    print(f"Amostra salva com sucesso em: {caminho_amostra}")


if __name__ == "__main__":
    main()
