# Anomaly Detector - Logs HTTP

Este projeto implementa um sistema de detecção de anomalias em logs HTTP utilizando ML e explicabilidade com SHAP.

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
//Para extraçao de features
python preprocessing/extract_features.py --input data/input/logs.txt --output data/processed/features.json --n 2

//Para aplicação do modelo em nível 0
python models/level0.py --input data/processed/features.json --output data/processed/level0.json

//Para procesamento
python models/aggregator.py --input data/processed/level0.json --output data/processed/aggregated.json

//Para aplicação do nível 1
python models/level1.py --input data/processed/aggregated.json --model-out models/logreg.pkl --output data/results/level1_output.json

