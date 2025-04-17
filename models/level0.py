"""
Carrega features em JSON e aplica três algoritmos de detecção de anomalia:
Isolation Forest, K-Means e PCA com Isolation Forest.
Os resultados são salvos em JSON.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from threading import Thread
import argparse

def classificar(df):
    """Classifica usando Isolation Forest, KMeans e PCA com Isolation Forest."""
    results = {}

    def iso_forest():
        clf = IsolationForest(contamination=0.4)
        results['isolation'] = clf.fit_predict(df)

    def k_means():
        clf = KMeans(n_clusters=43)
        results['kmeans'] = clf.fit_predict(df)

    def pca():
        n_comp = max(1, int(df.shape[1] * 0.1))
        reduced = PCA(n_components=n_comp).fit_transform(df)
        clf = IsolationForest()
        results['pca'] = clf.fit_predict(reduced)

    threads = [Thread(target=fn) for fn in (iso_forest, k_means, pca)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Classifica features com modelos não supervisionados.")
    parser.add_argument("--input", default="features.json", help="JSON com features extraídas.")
    parser.add_argument("--output", default="classificacoes.json", help="Arquivo de saída com as classificações.")
    args = parser.parse_args()

    df = pd.read_json(args.input)
    df_feat = pd.json_normalize(df['features'])
    resultado = classificar(df_feat)
    resultado.to_json(args.output)

if __name__ == "__main__":
    main()
