import os
import sys
import glob
import json
import importlib.util
import pandas as pd
import numpy as np
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Ensure project root for resolving model paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the FRA pipeline dynamically by file path (supports names with spaces or duplicates)
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_cls = None
FRA_PIPELINE = None

def _import_fra_pipeline():
    global pipeline_cls
    if pipeline_cls is not None:
        return pipeline_cls
    # Accept typical names, including the provided "industrial_fra_pipeline (1).py"
    candidates = [
        os.path.join(PIPELINE_DIR, "industrial_fra_pipeline.py"),
        os.path.join(PIPELINE_DIR, "industrial_fra_pipeline (1).py"),
    ]
    files = [p for p in candidates if os.path.exists(p)]
    if not files:
        files = glob.glob(os.path.join(PIPELINE_DIR, "industrial_fra_pipeline*.py"))
    for path in files:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("industrial_fra_pipeline", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                pipeline_cls = getattr(module, "IndustrialFRAPipeline")
                return pipeline_cls
    raise ImportError("IndustrialFRAPipeline not found in fra service directory")

def _resolve_model_paths():
    """Resolve FRA model, scaler, encoder, metadata paths under model/fra"""
    model_root = os.path.join(ROOT_DIR, "model", "fra")
    # Model may be .h5 or Keras SavedModel directory. Prefer .h5 or .keras if present.
    model_path: Optional[str] = None
    # Prefer .h5, then .keras (directory), then any .keras file
    h5 = glob.glob(os.path.join(model_root, "*.h5"))
    if h5:
        model_path = h5[0]
    else:
        keras_dirs = glob.glob(os.path.join(model_root, "*.keras"))
        if keras_dirs:
            # Keras model as directory path
            model_path = keras_dirs[0]
    scaler_path = os.path.join(model_root, "fra_scaler.pkl")
    encoder_path = os.path.join(model_root, "fra_fault_encoder.pkl")
    metadata_path = os.path.join(model_root, "fra_model_metadata.json")
    if not os.path.exists(scaler_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("Missing FRA scaler or encoder in model/fra")
    if model_path is None:
        raise FileNotFoundError("Could not locate FRA model file (.h5 or .keras) in model/fra")
    return model_path, scaler_path, encoder_path, metadata_path

router = APIRouter(prefix="/fra", tags=["FRA"])

def init_pipeline():
    # Initialize and keep a global instance
    global FRA_PIPELINE
    if FRA_PIPELINE is None:
        Pipeline = _import_fra_pipeline()
        model_path, scaler_path, encoder_path, metadata_path = _resolve_model_paths()
        FRA_PIPELINE = Pipeline(
            model_path=model_path,
            scaler_path=scaler_path,
            encoder_path=encoder_path,
            metadata_path=metadata_path,
        )

def _get_feature_columns():
    cols = None
    try:
        if getattr(FRA_PIPELINE, "metadata", None):
            cols = FRA_PIPELINE.metadata.get("model_info", {}).get("feature_columns")
    except Exception:
        cols = None
    return cols or []

def _is_features_csv(csv_path: str) -> bool:
    try:
        # Read only the header
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        feature_cols = set(_get_feature_columns())
        if not feature_cols:
            return False
        return feature_cols.issubset(set(header))
    except Exception:
        return False

def _predict_from_features_df(df: pd.DataFrame):
    feature_cols = _get_feature_columns()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing feature columns: {missing}")
    outputs = []
    for _, row in df.iterrows():
        X = np.array([row[feature_cols].astype(float).tolist()], dtype=float)
        X_scaled = FRA_PIPELINE.scaler.transform(X)
        raw_preds = FRA_PIPELINE.predict(X_scaled)
        diagnosis = FRA_PIPELINE.interpret_results(raw_preds)
        recs = FRA_PIPELINE.generate_recommendations(diagnosis)
        outputs.append({
            "diagnosis": diagnosis,
            "recommendations": recs,
        })
    summary = {
        "n_samples": len(outputs)
    }
    return {"results": outputs, "summary": summary}

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    # Expect CSV content for FRA
    try:
        # Save to a temp file because pipeline expects a path
        import tempfile
        suffix = os.path.splitext(file.filename or "")[1] or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        # Detect whether this is a raw FRA CSV or a precomputed features CSV
        if _is_features_csv(tmp_path):
            df = pd.read_csv(tmp_path)
            results = _predict_from_features_df(df)
        else:
            results = FRA_PIPELINE.run_pipeline(csv_path=tmp_path, transformer_meta=None, use_tta=True)
        os.unlink(tmp_path)
        if results is None:
            raise HTTPException(status_code=500, detail="FRA pipeline failed")
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict-json")
async def predict_json(payload: dict):
    # Accept pre-parsed FRA arrays or a JSON with inline CSV content
    # Expected keys: frequencies, magnitude_db, phase_deg, transformer_meta (optional)
    try:
        # If arrays are provided, run partial pipeline manually using public methods
        if all(k in payload for k in ("frequencies", "magnitude_db", "phase_deg")):
            freqs = np.array(payload["frequencies"], dtype=float)
            mags = np.array(payload["magnitude_db"], dtype=float)
            phases = np.array(payload["phase_deg"], dtype=float)
            transformer_meta = payload.get("transformer_meta")
            X_scaled, _ = FRA_PIPELINE.extract_features(freqs, mags, phases, transformer_meta)
            raw_preds = FRA_PIPELINE.predict(X_scaled)
            diagnosis = FRA_PIPELINE.interpret_results(raw_preds)
            recs = FRA_PIPELINE.generate_recommendations(diagnosis)
            return JSONResponse(content={
                "diagnosis": diagnosis,
                "recommendations": recs,
            })
        # If features dict or list is provided, use metadata feature columns
        if "features" in payload:
            features = payload["features"]
            if isinstance(features, dict):
                df = pd.DataFrame([features])
            elif isinstance(features, list):
                df = pd.DataFrame(features)
            else:
                raise HTTPException(status_code=400, detail="'features' must be an object or array of objects")
            results = _predict_from_features_df(df)
            return JSONResponse(content=results)
        # Otherwise require a CSV-like text content
        if "csv" in payload:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as tmp:
                tmp.write(payload["csv"]) 
                tmp_path = tmp.name
            transformer_meta = payload.get("transformer_meta")
            # Detect features vs raw CSV
            if _is_features_csv(tmp_path):
                df = pd.read_csv(tmp_path)
                results = _predict_from_features_df(df)
            else:
                results = FRA_PIPELINE.run_pipeline(csv_path=tmp_path, transformer_meta=transformer_meta, use_tta=True)
            os.unlink(tmp_path)
            if results is None:
                raise HTTPException(status_code=500, detail="FRA pipeline failed")
            return JSONResponse(content=results)
        raise HTTPException(status_code=400, detail="Invalid JSON schema. Provide arrays (frequencies, magnitude_db, phase_deg) or csv content.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
