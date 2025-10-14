import os
import sys
import json
import glob
import importlib.util
import pandas as pd

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
MODEL_DIR = os.path.join(ROOT_DIR, 'model', 'fra')

def import_pipeline_cls():
    candidates = [
        os.path.join(CURRENT_DIR, 'industrial_fra_pipeline.py'),
        os.path.join(CURRENT_DIR, 'industrial_fra_pipeline (1).py'),
    ]
    files = [p for p in candidates if os.path.exists(p)]
    if not files:
        files = glob.glob(os.path.join(CURRENT_DIR, 'industrial_fra_pipeline*.py'))
    for path in files:
        spec = importlib.util.spec_from_file_location('industrial_fra_pipeline_dyn', path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, 'IndustrialFRAPipeline')
    raise ImportError('IndustrialFRAPipeline not found')


def main(csv_path: str, limit: int = 10):
    model_path = None
    # prefer .h5, else .keras directory
    h5 = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    if h5:
        model_path = os.path.join(MODEL_DIR, h5[0])
    else:
        dirs = [d for d in os.listdir(MODEL_DIR) if d.endswith('.keras')]
        if dirs:
            model_path = os.path.join(MODEL_DIR, dirs[0])
    scaler_path = os.path.join(MODEL_DIR, 'fra_scaler.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'fra_fault_encoder.pkl')
    metadata_path = os.path.join(MODEL_DIR, 'fra_model_metadata.json')

    Pipeline = import_pipeline_cls()
    pipe = Pipeline(
        model_path=model_path,
        scaler_path=scaler_path,
        encoder_path=encoder_path,
        metadata_path=metadata_path,
    )

    # If CSV looks like features, treat accordingly
    df_head = pd.read_csv(csv_path, nrows=1)
    feature_cols = pipe.metadata.get('model_info', {}).get('feature_columns', []) if pipe.metadata else []
    if feature_cols and set(feature_cols).issubset(set(df_head.columns)):
        print('Detected feature CSV; running row-wise predictions...')
        results = []
        df_iter = pd.read_csv(csv_path, chunksize=max(1, limit))
        df = next(df_iter)
        for _, row in df.head(limit).iterrows():
            X = row[feature_cols].astype(float).values.reshape(1, -1)
            X_scaled = pipe.scaler.transform(X)
            raw = pipe.predict(X_scaled)
            diag = pipe.interpret_results(raw)
            rec = pipe.generate_recommendations(diag)
            results.append({
                'diagnosis': diag,
                'recommendations': rec,
            })
        print(json.dumps({'count': len(results), 'sample': results[0] if results else None}, indent=2))
    else:
        print('Assuming raw FRA CSV; running full pipeline...')
        out = pipe.run_pipeline(csv_path=csv_path, transformer_meta=None, use_tta=True)
        print(json.dumps(out, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    main(args.csv, limit=args.limit)
