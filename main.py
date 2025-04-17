# main.py
"""
Script principal que orquestra a execução completa do sistema de detecção de anomalias em logs HTTP.
"""
import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa pipeline completa do detector de anomalias")
    parser.add_argument("--input", required=True, help="Caminho para arquivo .txt contendo os logs HTTP")
    parser.add_argument("--output-dir", required=True, help="Diretório para salvar os arquivos processados")
    parser.add_argument("--n", type=int, default=2, help="Valor de n para n-gramas (default=2)")
    parser.add_argument("--manual", type=int, default=1000, help="Número de entradas para classificação manual (default=1000)")
    parser.add_argument("--sample", type=int, default=35, help="Porcentagem de pré-classificações usadas no nível 1 (default=35)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subprocess.run([
        "python", "preprocessing/extract_features.py",
        "--input", args.input,
        "--output", f"{args.output_dir}/features.json",
        "--n", str(args.n)
    ])

    subprocess.run([
        "python", "models/level0.py",
        "--input", f"{args.output_dir}/features.json",
        "--output", f"{args.output_dir}/level0.json"
    ])

    subprocess.run([
        "python", "models/aggregator.py",
        "--input", f"{args.output_dir}/level0.json",
        "--output", f"{args.output_dir}/aggregated.json",
        "--manual", str(args.manual)
    ])

    subprocess.run([
        "python", "models/level1.py",
        "--input", f"{args.output_dir}/aggregated.json",
        "--model-out", f"{args.output_dir}/logreg.pkl",
        "--output", f"{args.output_dir}/level1_output.json",
        "--sample", str(args.sample)
    ])

    subprocess.run([
        "python", "models/explainer.py",
        "--model", f"{args.output_dir}/logreg.pkl",
        "--data", f"{args.output_dir}/aggregated.json",
        "--out", f"{args.output_dir}/shap_visuals"
    ])
