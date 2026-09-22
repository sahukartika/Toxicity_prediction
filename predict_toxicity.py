#!/usr/bin/env python
"""
predict_toxicity.py
--------------------
Predict Ames toxicity (mutagenicity) for new chemical samples using a model
trained with the nested-CV pipeline (see training repo/notebook).

Command-line usage:
    python predict_toxicity.py --input new_samples.csv --output predictions.csv

Input file requirements:
    - First column : sample / chemical name (any header name is fine).
    - Remaining columns : the SAME named feature set used to train the model.
      Column order does not matter -- columns are re-aligned automatically
      using feature_names.json. Missing feature columns are median-imputed
      by the model's own imputer; unrecognized extra columns are ignored.
      Both cases are reported as warnings so you can sanity-check results.

Output CSV columns:
    sample_name, predicted_label (Toxic/Non-toxic),
    predicted_probability_toxic (or decision_score if the model has no
    predict_proba), n_missing_features (diagnostic per-row count).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

LABEL_MAP = {0: "Non-toxic", 1: "Toxic"}


def load_artifacts(model_path: str, features_path: str):
    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    return model, feature_names


def load_input_table(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def align_features(df: pd.DataFrame, feature_names: list, sample_col: str):
    X_raw = df.drop(columns=[sample_col], errors="ignore")

    present = set(X_raw.columns)
    expected = set(feature_names)
    missing = expected - present
    extra = present - expected

    if missing:
        msg = f"{len(missing)} expected feature column(s) not found in input"
        if len(missing) <= 20:
            msg += f": {sorted(missing)}"
        print(f"WARNING: {msg}. These will be median-imputed by the model.")
    if extra:
        msg = f"{len(extra)} unexpected column(s) in input will be ignored"
        if len(extra) <= 20:
            msg += f": {sorted(extra)}"
        print(f"NOTE: {msg}.")

    # Reindex guarantees exact training column order; missing cols become NaN
    X_aligned = X_raw.reindex(columns=feature_names)

    # Coerce to numeric; anything unparsable becomes NaN (imputed downstream)
    X_aligned = X_aligned.apply(pd.to_numeric, errors="coerce")

    n_missing_per_sample = X_aligned.isna().sum(axis=1).values
    return X_aligned, n_missing_per_sample


def predict_toxicity(input_path: str, output_path: str = "predictions.csv",
                      model_path: str = "final_toxicity_model.pkl",
                      features_path: str = "feature_names.json",
                      threshold: float = 0.5) -> pd.DataFrame:
    model, feature_names = load_artifacts(model_path, features_path)

    df = load_input_table(input_path)
    sample_col = df.columns[0]
    sample_names = df[sample_col].astype(str).values

    X_aligned, n_missing = align_features(df, feature_names, sample_col)

    if hasattr(model, "n_features_in_") and model.n_features_in_ != X_aligned.shape[1]:
        raise ValueError(
            f"Feature count mismatch: model expects {model.n_features_in_} "
            f"features, got {X_aligned.shape[1]} after alignment. Check that "
            f"feature_names.json matches the model file you loaded."
        )

    X_values = X_aligned.values

    if hasattr(model, "predict_proba"):
        proba_toxic = model.predict_proba(X_values)[:, 1]
        pred_label = (proba_toxic >= threshold).astype(int)
        score_col_name = "predicted_probability_toxic"
        score_values = proba_toxic
    else:
        score_values = model.decision_function(X_values)
        pred_label = (score_values >= 0).astype(int)
        score_col_name = "decision_score"
        print("NOTE: underlying model has no predict_proba; using the raw decision "
              "score and its default (0) boundary. --threshold is ignored.")

    result = pd.DataFrame({
        "sample_name": sample_names,
        "predicted_label": [LABEL_MAP[v] for v in pred_label],
        score_col_name: np.round(score_values, 4),
        "n_missing_features": n_missing,
    })

    result.to_csv(output_path, index=False)
    print(f"\nSaved predictions for {len(result)} samples to: {output_path}")
    print(result["predicted_label"].value_counts().rename("count").to_string())
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict Ames toxicity for new samples.")
    parser.add_argument("--input", required=True, help="CSV/XLSX file: sample name + features")
    parser.add_argument("--output", default="predictions.csv", help="Where to save predictions")
    parser.add_argument("--model", default="final_toxicity_model.pkl", help="Path to saved model (.pkl)")
    parser.add_argument("--features", default="feature_names.json", help="Path to feature_names.json")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Probability cutoff for calling a sample 'Toxic' (default 0.5)")
    args = parser.parse_args()

    try:
        predict_toxicity(args.input, args.output, args.model, args.features, args.threshold)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()