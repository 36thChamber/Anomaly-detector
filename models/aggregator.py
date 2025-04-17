"""
Agrega as classificações, calcula grau de certeza e salva duas versões do dataset:
um com classificação final, outro com entradas de baixa certeza para classificação manual.
"""

import pandas as pd
import argparse

def agregador(df, n_manual=1000):
    """Agrega classificações e calcula grau de certeza."""
    df['certeza'] = df.apply(lambda row: row.value_counts().max() / len(row), axis=1)
    df['classificacao'] = df.mode(axis=1)[0].map({-1: 'anomalia', 1: 'normal'})
    df.sort_values(by='certeza', ascending=False).to_json('agregado.json')
    df.sort_values(by='certeza').head(n_manual).to_json('manual.json')

def main():
    parser = argparse.ArgumentParser(description="Agrega classificações e calcula certeza.")
    parser.add_argument("--input", default="classificacoes.json", help="Arquivo com classificações.")
    parser.add_argument("--manual", type=int, default=1000, help="Número de exemplos para revisão manual.")
    args = parser.parse_args()

    df = pd.read_json(args.input)
    agregador(df, args.manual)

if __name__ == "__main__":
    main()
