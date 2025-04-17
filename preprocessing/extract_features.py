"""
Módulo de exploração e extração de features do dataset CSIC2010.

Este script permite carregar requisições HTTP normais e anômalas, exibir estatísticas básicas
sobre os dados e extrair bigramas como features para uso posterior em modelos de detecção de anomalias.

Funcionalidades:
- Leitura de arquivos de log contendo requisições HTTP.
- Exibição de estatísticas simples (número de requisições, tamanho médio, etc).
- Extração de bigramas via CountVectorizer do sklearn.
- Salvamento das features e labels em arquivos compactos para uso futuro.

Uso:
python explore_dataset.py --normal normal.txt --anomalous anomalo.txt --summary --output_dir features

Argumentos:
--normal: caminho para arquivo de requisições normais
--anomalous: caminho para arquivo de requisições anômalas
--summary: mostra estatísticas básicas das requisições
--output_dir: diretório para salvar arquivos de saída
"""
import argparse
import os
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import save_npz

nltk.download('punkt')


def load_logs(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.readlines()


def basic_stats(logs):
    lengths = [len(log.strip()) for log in logs]
    print(f"Total requests: {len(logs)}")
    print(f"Average request length: {np.mean(lengths):.2f} characters")
    print(f"Max request length: {np.max(lengths)}")
    print(f"Min request length: {np.min(lengths)}")
    print("\nSample requests:")
    for i in range(min(5, len(logs))):
        print(f"{i + 1}: {logs[i].strip()}")


def extract_features(logs):
    cleaned = [re.sub(r"[^a-zA-Z0-9\s]", "", log.lower()) for log in logs]
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), stop_words='english')
    features = vectorizer.fit_transform(cleaned)
    return features, vectorizer


def save_features(features, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_npz(os.path.join(output_dir, 'features.npz'), features)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)


def main():
    parser = argparse.ArgumentParser(description="Extrai features e/ou resume o dataset CSIC2010.")
    parser.add_argument("--normal", required=True, help="Caminho para o arquivo com requisições normais.")
    parser.add_argument("--anomalous", required=True, help="Caminho para o arquivo com requisições anômalas.")
    parser.add_argument("--output_dir", required=False, default="features", help="Diretório de saída.")
    parser.add_argument("--summary", action="store_true", help="Se fornecido, exibe estatísticas básicas do dataset.")
    args = parser.parse_args()

    normal_logs = load_logs(args.normal)
    anomalous_logs = load_logs(args.anomalous)

    if args.summary:
        print("-- Normal Logs --")
        basic_stats(normal_logs)
        print("\n-- Anomalous Logs --")
        basic_stats(anomalous_logs)

    logs = normal_logs + anomalous_logs
    labels = [0] * len(normal_logs) + [1] * len(anomalous_logs)

    print("\nExtraindo features...")
    features, vectorizer = extract_features(logs)
    save_features(features, labels, args.output_dir)
    print(f"Features salvas em: {args.output_dir}")


if __name__ == "__main__":
    main()
