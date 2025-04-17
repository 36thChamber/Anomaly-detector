"""
Treina um modelo de regressão logística usando as classificações agregadas.
Salva o modelo e os resultados em arquivos.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser(description="Classificação supervisionada com regressão logística.")
    parser.add_argument("--input", default="agregado.json", help="Arquivo de entrada com dados agregados.")
    parser.add_argument("--output", default="final.json", help="Arquivo JSON com classificações finais.")
    parser.add_argument("--model", default="modelo.pkl", help="Arquivo para salvar o modelo treinado.")
    args = parser.parse_args()

    df = pd.read_json(args.input)
    X = df.drop(['classificacao'], axis=1)
    y = df['classificacao'].map({'anomalia': 1, 'normal': 0})

    model = LogisticRegression()
    model.fit(X, y)

    df['nivel1'] = model.predict(X)
    df.to_json(args.output)

    with open(args.model, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
