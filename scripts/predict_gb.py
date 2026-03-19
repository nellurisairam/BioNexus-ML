#!/usr/bin/env python3
"""
Predict script for Bioreactor ML — Product Titer (g/L)
Model: GradientBoostingRegressor inside an sklearn Pipeline with median imputation.

Subcommands:
  train     - Train a GBR pipeline and save model + schema
  predict   - Predict Product_Titer_gL for a new CSV (unlabeled)
  benchmark - Evaluate a saved model on a labeled CSV (R^2, MAE, RMSE)

Examples:
  python predict.py train --data bioreactor_ml_dataset.csv --target Product_Titer_gL --out outputs/model_gbr.joblib
  python predict.py predict --model outputs/model_gbr.joblib --schema outputs/feature_schema.json --input new_samples.csv --output outputs/predictions.csv
  python predict.py benchmark --model outputs/model_gbr.joblib --schema outputs/feature_schema.json --data bioreactor_ml_dataset.csv --target Product_Titer_gL
"""

import argparse
import json
import os
import math
import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DEFAULT_MODEL = 'outputs/model_gbr.joblib'
DEFAULT_SCHEMA = 'outputs/feature_schema.json'


def load_schema(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schema not found: {path}. Train first or provide the correct path.")
    with open(path, 'r') as f:
        return json.load(f)


def align_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    feats = schema['features']
    # Add missing as NaN; drop extras; enforce order
    for c in feats:
        if c not in df.columns:
            df[c] = np.nan
    df = df[feats]
    # Coerce to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def compute_rmse(y_true, y_pred):
    # Version-safe RMSE: sqrt(MSE)
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def cmd_train(args):
    data = pd.read_csv(args.data)
    if args.target not in data.columns:
        raise ValueError(f"Target column '{args.target}' not in data")
    X = data.drop(columns=[args.target])
    y = data[args.target]

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingRegressor(random_state=42))
    ])

    pipe.fit(X, y)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dump(pipe, args.out)

    schema = {
        'algo': 'GradientBoostingRegressor',
        'target': args.target,
        'features': X.columns.tolist(),
        'dtypes': {c: str(data[c].dtype) for c in X.columns},
        'trained_rows': int(data.shape[0]),
        'trained_cols': int(data.shape[1])
    }
    schema_path = args.schema if args.schema else DEFAULT_SCHEMA
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"Model saved to: {args.out}")
    print(f"Schema saved to: {schema_path}")


def cmd_predict(args):
    model_path = args.model or DEFAULT_MODEL
    schema_path = args.schema or DEFAULT_SCHEMA
    clf = load(model_path)
    schema = load_schema(schema_path)

    df = pd.read_csv(args.input)
    X = align_columns(df.copy(), schema)
    preds = clf.predict(X)

    out_df = df.copy()
    out_df['Pred_Product_Titer_gL'] = preds

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        out_df.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
    else:
        print(out_df.head())


def cmd_benchmark(args):
    model_path = args.model or DEFAULT_MODEL
    schema_path = args.schema or DEFAULT_SCHEMA
    clf = load(model_path)
    schema = load_schema(schema_path)

    data = pd.read_csv(args.data)
    if args.target not in data.columns:
        raise ValueError(f"Target column '{args.target}' not in data")
    X = align_columns(data.drop(columns=[args.target]).copy(), schema)
    y = data[args.target]

    pred = clf.predict(X)
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = compute_rmse(y, pred)
    print({'r2': float(r2), 'mae': float(mae), 'rmse': float(rmse)})


def build_parser():
    p = argparse.ArgumentParser(description='Bioreactor ML deploy script (GBR model): train / predict / benchmark')
    sub = p.add_subparsers(dest='cmd', required=True)

    # train
    p_train = sub.add_parser('train', help='Train a GBR pipeline and save artifact')
    p_train.add_argument('--data', required=True, help='Training CSV path')
    p_train.add_argument('--target', required=True, help='Target column name')
    p_train.add_argument('--out', default=DEFAULT_MODEL, help='Output model path (joblib)')
    p_train.add_argument('--schema', default=DEFAULT_SCHEMA, help='Output schema json path')
    p_train.set_defaults(func=cmd_train)

    # predict
    p_pred = sub.add_parser('predict', help='Predict on new CSV using a saved model')
    p_pred.add_argument('--model', default=DEFAULT_MODEL, help='Model path (joblib)')
    p_pred.add_argument('--schema', default=DEFAULT_SCHEMA, help='Schema json path')
    p_pred.add_argument('--input', required=True, help='Input CSV with feature columns')
    p_pred.add_argument('--output', default='outputs/predictions.csv', help='Output CSV path for predictions')
    p_pred.set_defaults(func=cmd_predict)

    # benchmark
    p_bench = sub.add_parser('benchmark', help='Evaluate a saved model on a labeled dataset')
    p_bench.add_argument('--model', default=DEFAULT_MODEL, help='Model path (joblib)')
    p_bench.add_argument('--schema', default=DEFAULT_SCHEMA, help='Schema json path')
    p_bench.add_argument('--data', required=True, help='CSV with features + target')
    p_bench.add_argument('--target', required=True, help='Target column name')
    p_bench.set_defaults(func=cmd_benchmark)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == '__main__':
    main()
