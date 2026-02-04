# RDKit Practice
This repository is a practice project for molecular property prediction using RDKit and machine learning models. 

### Description
I train an xgboost model to predict whether a molecule is an inhibitor of BACE1 (related to Alzheimer's disease) based on its SMILES representation.

The smiles data is processed to generate molecular fingerprints, which are then used as features. Good hyperparameters are found using bayesian optimization (using Optuna). 

The trained model can be used for inference on new molecular data, either through a command-line interface or a FastAPI server. The project also includes a Dockerfile for containerized deployment.

## Install
1. Clone the repository:
```bash
git clone
cd rdkit-practice
```
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage
### File Inference

To run inference on a dataset, use the following command:
```bash
python infer.py --model_path --input path/to/data.smi --batch_size 64 --top_k 5
```
**Arguments:**
- `--input`: Path to the input data file (SMILES format).
- `--batch_size`: Number of samples to process in each batch (default: 0, meaning process all samples at once).
- `--top_k`: Number of top predictions to display (default: 0, meaning display all predictions).

## FastAPI Server
You can also run a FastAPI server for real-time inference:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
- predict endpoint: `POST /predict` with JSON payload containing the list of smiles `{"smiles": ["CCO", "CCN"]}`

### Docker
There is a Dockerfile included for containerized deployment. To build and run the Docker container:
```bash
docker build -t rdkit-practice .
docker run -p 8000:8000 rdkit-practice
```