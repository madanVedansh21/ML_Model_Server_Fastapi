import os
import sys
import glob
from typing import Optional
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from python_services.utils.json_sanitizer import sanitize

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_model_paths():
    model_root = os.path.join(ROOT_DIR, "model", "scada")
    # SCADA model seen as .h5 in repo list
    model_path: Optional[str] = None
    h5 = glob.glob(os.path.join(model_root, "*.h5"))
    if h5:
        model_path = h5[0]
    if model_path is None:
        raise FileNotFoundError("Couldd not locate SCADA model (.h5) in model/scada")
    scaler_path = os.path.join(model_root, "scada_scaler.pkl")
    encoder_path = os.path.join(model_root, "scada_fault_encoder.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("Missing SCADA scaler or encoder in model/scada")
    return model_path, scaler_path, encoder_path

# Import SCADA pipeline (same package)
from .scada_pipeline import SCADATransformerPipeline

router = APIRouter(prefix="/scada", tags=["SCADA"])
SCADA_PIPELINE = None

def init_pipeline():
    global SCADA_PIPELINE
    if SCADA_PIPELINE is None:
        model_path, scaler_path, encoder_path = _resolve_model_paths()
        SCADA_PIPELINE = SCADATransformerPipeline(
            model_path=model_path,
            scaler_path=scaler_path,
            encoder_path=encoder_path,
        )

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict-json")
async def predict_json(payload: dict):
    try:
        logging.info(f"Received payload: {payload}")
        # Accept only JSON object representing SCADA measurements; the pipeline handles dict input
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid JSON payload for SCADA. Provide a JSON object of SCADA measurements.")
        results = SCADA_PIPELINE.run_from_json(payload, use_tta=True)
        logging.info(f"Pipeline results: {results}")
        if results is None:
            raise HTTPException(status_code=500, detail="SCADA pipeline failed")
        # Sanitize results to ensure JSON compliance (remove NaN/Inf, convert numpy types)
        safe = sanitize(results)
        return JSONResponse(content=safe)
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
