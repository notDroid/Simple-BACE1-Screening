import time
from infer import infer

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI(title="BACE1 Inhibitor Screening API")

class MoleculeRequest(BaseModel):
    smiles: list[str]

class PredictionResponse(BaseModel):
    is_inhibitor: list[bool]
    confidence: list[float]
    processing_time_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_molecule(request: MoleculeRequest):
    probs, preds, duration, _ = infer(request.smiles)
    
    return {
        "is_inhibitor": preds.tolist(),
        "confidence": probs.tolist(),
        "processing_time_ms": round(duration*1000, 2)
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the BACE1 Inhibitor Screening API. Use the /predict endpoint to get predictions."}