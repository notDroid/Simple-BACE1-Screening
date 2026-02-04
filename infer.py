import os
import time
import scipy
import heapq
import argparse
import numpy as np
from tqdm import tqdm

import joblib

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

MODEL_PATH = os.getenv('MODEL_PATH', 'models/xgb_bace_model.pkl')
MG_RADIUS = int(os.getenv('MG_RADIUS', '2'))
MG_FPSIZE = int(os.getenv('MG_FPSIZE', '2048'))

model = joblib.load(MODEL_PATH)
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=MG_RADIUS, fpSize=MG_FPSIZE)

def process_batch(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_mols = [m for m in mols if m is not None]
    
    if not valid_mols:
        return []

    fps = [mfpgen.GetFingerprintAsNumPy(m) for m in valid_mols]
    X_batch = np.array(fps, dtype=np.float32)
    X_batch_sparse = scipy.sparse.csr_matrix(X_batch)
    
    probs = model.predict_proba(X_batch_sparse)
    return probs

def infer(input_data):
    start_time = time.time()
    probs = process_batch(input_data)[:, 1]
    preds = (probs >= 0.5).astype(bool)
    end_time = time.time()

    duration = end_time - start_time
    throughput = len(input_data) / duration  
    return probs, preds, duration, throughput

def read_in_chunks(file_path, batch_size=1024):
    with open(file_path, 'r') as f:
        batch = []
        for line in f:
            smi = line.strip()
            if not smi: continue
            batch.append(smi)

            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

class TopKTracker:
    def __init__(self, k):
        self.k = k
        self.heap = []

    def update(self, smiles, probability):
        if len(self.heap) < self.k or self.k == 0:
            heapq.heappush(self.heap, (probability, smiles))
        elif probability > self.heap[0][0]:
            heapq.heapreplace(self.heap, (probability, smiles))

    def get_results(self):
        return sorted(self.heap, key=lambda x: x[0], reverse=True)

def batched_infer(file_path, batch_size, k=0):
    chunk_iterator = read_in_chunks(file_path, batch_size)
    tracker = TopKTracker(k=k)

    start_time = time.time()
    total_mols = 0
    for batch in tqdm(chunk_iterator, desc="Processing Batches"):
        probs = process_batch(batch)[:, 1]
        for smi, prob in zip(batch, probs):
            tracker.update(smi, prob)
        total_mols += len(batch)
    end_time = time.time()

    duration = end_time - start_time
    throughput = total_mols / duration
    
    results = tracker.get_results()
    probs = [prob for prob, smi in results]
    preds = [prob >= 0.5 for prob in probs]
    smiles = [smi for prob, smi in results]

    return total_mols, smiles, probs, preds, duration, throughput

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer BACE1 inhibition for a batch of molecules.")
    parser.add_argument('--input', type=str, required=True, help="Path to input file containing SMILES strings, one per line.")
    parser.add_argument('--batch_size', type=int, default=0, help="Number of molecules to process in each batch, 0 for all at once.")
    parser.add_argument('--top_k', type=int, default=0, help="Number of top predictions to track, 0 for all.")
    args = parser.parse_args()

    file_path = args.input
    batch_size = args.batch_size
    top_k = args.top_k

    # Single batch inference
    if batch_size == 0:
        # Load input data
        with open(args.input, 'r') as f:
            smiles = [line.strip() for line in f if line.strip()]
        
        total_mols = len(smiles)
        print(f"Loaded {total_mols} molecules for inference (Consider using --batch_size for large datasets).")

        # Infer in a single batch
        probs, preds, duration, throughput = infer(smiles)

        # If top_k is specified, filter results
        if top_k > 0:
            combined = list(zip(smiles, probs, preds))
            combined.sort(key=lambda x: x[1], reverse=True)
            combined = combined[:top_k]
            smiles, probs, preds = zip(*combined)
            total_mols = len(smiles)
        else:
            # Sort all results if top_k is 0
            combined = list(zip(smiles, probs, preds))
            combined.sort(key=lambda x: x[1], reverse=True)
            smiles, probs, preds = zip(*combined)
    else:
        # Batched inference
        total_mols, smiles, probs, preds, duration, throughput = batched_infer(file_path, batch_size, k=top_k)

    print("\n\nInference Results:")
    print(f"------------------------------------------------")
    print(f"Molecules:          {total_mols} molecules")
    print(f"Total Time:         {duration:.4f} seconds")
    print(f"Throughput:         {throughput:.2f} molecules/sec")
    print(f"Latency per mol:    {(duration/total_mols)*1000:.2f} ms")
    print(f"------------------------------------------------\n")

    print("Top Predictions:")
    for smi, prob, pred in zip(smiles, probs, preds):
        print(f"SMILES: {smi} | Probability: {prob:.4f} | Predicted Inhibitor: {pred}")