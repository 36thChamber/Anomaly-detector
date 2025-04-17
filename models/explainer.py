"""
Utiliza SHAP para explicar classificações feitas por regressão logística.
Gera gráfico waterfall da primeira instância e o exibe.
"""

import shap
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Explicação de modelo usando SHAP.")
    parser.add_argument("--input", default="agregado.json", help="Dados de entrada.")
    parser.add_argument("--model", default="modelo.pkl", help="Modelo treinado serializado.")
    parser.add_argument("--output", default="waterfall.png", help="Imagem com explicação SHAP.")
    args = parser.parse_args()

    df = pd.read_json(args.input)
    X = df.drop(['classificacao'], axis=1)

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(args.output)
    print(f"Gráfico salvo em {args.output}")

if __name__ == "__main__":
    main()
