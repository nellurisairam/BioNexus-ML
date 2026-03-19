#!/usr/bin/env python3
"""
Bioreactor ML — Product Titer (g/L)

Capabilities:
  • Train a RidgeCV pipeline and save the model artifact
  • Predict on new CSV data using a saved artifact
  • Benchmark on a labeled CSV to print metrics
  • Shared preprocessing ensures new samples match training schema
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load, dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DEFAULT_MODEL = 'outputs/model_ridgecv.joblib'
DEFAULT_SCHEMA = 'outputs/feature_schema.json'


# --- Shared preprocessing function ---
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna()

    # Derived features
    df["Glucose_Consumption_Rate"] = df["Glucose_gL"].diff()
    df["DO_Change"] = df["Dissolved_Oxygen_percent"].diff()
    df["Specific_Productivity"] = df["Product_Titer_gL"] / df["Cell_Viability_percent"]

    # Normalize agitation
    scaler = StandardScaler()
    df["Agitation_Normalized"] = scaler.fit_transform(df[["Agitation_RPM"]])

    # Flags
    df["High_Titer_Flag"] = (df["Product_Titer_gL"] > 1).astype(int)
    df["Low_Viability_Flag"] = (df["Cell_Viability_percent"] < 98).astype(int)

    return df


def load_schema(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schema not found: {path}. Train first or provide the correct path.")
    with open(path, 'r') as f:
        return json.load(f)


def align_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()
    feats = schema['features']
    for c in feats:
        if c not in df.columns:
            df[c] = np.nan
    df = df[feats]
    for c in df.columns:
        df.loc[:, c] = pd.to_numeric(df[c], errors='coerce')
    return df


def cmd_train(args):
    raw_df = pd.read_csv(args.data)
    if args.target not in raw_df.columns:
        raise ValueError(f"Target column '{args.target}' not in data")

    # Backup original dataset
    backup_path = Path(args.data).with_name("bioreactor_ml_dataset_Original.csv")
    raw_df.to_csv(backup_path, index=False)
    print(f"📂 Original dataset saved as: {backup_path.resolve()}")

    # Preprocess and override
    df = preprocess(raw_df)
    df.to_csv(args.data, index=False)
    print(f"✅ Cleaned dataset saved (overwritten): {Path(args.data).resolve()}")

    # Train model
    X = df.drop(columns=[args.target])
    y = df[args.target]

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
    ])
    pipe.fit(X, y)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dump(pipe, args.out)

    schema = {
        'target': args.target,
        'features': X.columns.tolist(),
        'dtypes': {c: str(df[c].dtype) for c in X.columns},
        'trained_rows': int(df.shape[0]),
        'trained_cols': int(df.shape[1])
    }
    schema_path = args.schema if args.schema else DEFAULT_SCHEMA
    schema_dir = os.path.dirname(schema_path)
    if schema_dir:
        os.makedirs(schema_dir, exist_ok=True)
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"Model saved to: {args.out}")
    print(f"Schema saved to: {schema_path}")


def cmd_predict(args):
    model_path = args.model or DEFAULT_MODEL
    schema_path = args.schema or DEFAULT_SCHEMA
    clf = load(model_path)
    schema = load_schema(schema_path)

    # Load and preprocess new samples
    df = pd.read_csv(args.input)
    df = preprocess(df)

    # Align with training schema
    X = align_columns(df.copy(), schema)
    preds = clf.predict(X)

    # Save predictions
    out_df = df.copy()
    out_df['Pred_Product_Titer_gL'] = preds

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")


def cmd_benchmark(args):
    model_path = args.model or DEFAULT_MODEL
    schema_path = args.schema or DEFAULT_SCHEMA
    clf = load(model_path)
    schema = load_schema(schema_path)

    data = pd.read_csv(args.data)
    data = preprocess(data)
    if args.target not in data.columns:
        raise ValueError(f"Target column '{args.target}' not in data")
    X = align_columns(data.drop(columns=[args.target]).copy(), schema)
    y = data[args.target]

    pred = clf.predict(X)
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    print({'r2': float(r2), 'mae': float(mae), 'rmse': float(rmse)})


def build_parser():
    p = argparse.ArgumentParser(description='Bioreactor ML deploy script (train/predict/benchmark)')
    sub = p.add_subparsers(dest='cmd', required=True)

    # train
    p_train = sub.add_parser('train', help='Train a RidgeCV pipeline and save artifact')
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
    p_pred.add_argument('--output', default='predictions.csv', help='Output CSV path for predictions')
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
