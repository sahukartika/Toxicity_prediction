# Ames Toxicity (Mutagenicity) Prediction Model

A machine-learning model that predicts Ames-test mutagenicity (toxic vs.
non-toxic) from a set of 999 chemical/molecular features. Trained with a
leakage-free **nested cross-validation** pipeline — outer 5-fold CV for
unbiased performance estimation, inner 3-fold CV for hyperparameter tuning —
comparing Logistic Regression, Random Forest, SVM (RBF), k-NN, and XGBoost.

## Model performance (nested CV, mean ± std over 5 outer folds)

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|---|
| **XGBoost (deployed)** | 0.897 ± 0.008 | 0.821 ± 0.012 | 0.842 ± 0.010 |


XGBoost was auto-selected (highest mean MCC) and retrained on the full
5,362-sample dataset (2,346 non-toxic / 3,016 toxic) for deployment.

## Repo contents

- `final_toxicity_model.pkl` — trained scikit-learn `Pipeline` (median
  imputation → zero-variance filter → standard scaling → mutual-information
  feature selection → XGBoost classifier)
- `feature_names.json` — ordered list of the 999 feature columns the model
  expects
- `predict_toxicity.py` — prediction script (CLI + importable function)
- `requirements.txt` — Python dependencies

## Usage

```bash
pip install -r requirements.txt
python predict_toxicity.py --input your_samples.csv --output predictions.csv
```

### Input file format
- First column: sample / chemical name (any header name).
- Remaining columns: the same ~999 named feature columns used in training.
  **Column order doesn't matter** — they're re-aligned by name against
  `feature_names.json`. Missing columns are median-imputed by the model;
  unrecognized extra columns are ignored. Both cases print a warning so you
  can sanity-check results before trusting them.

### Output
A CSV with:
- `sample_name`
- `predicted_label` (`Toxic` / `Non-toxic`)
- `predicted_probability_toxic`
- `n_missing_features` — diagnostic count of expected features that were
  absent for that row (high values mean that prediction is less trustworthy)

### Custom decision threshold
```bash
python predict_toxicity.py --input your_samples.csv --threshold 0.35
```
Lowering the threshold below 0.5 increases sensitivity (catches more true
toxicants) at the cost of more false positives — useful when missing a
toxicant is costlier than a false alarm.

### As a Python function
```python
from predict_toxicity import predict_toxicity
df = predict_toxicity("your_samples.csv")
```

## Limitations
- Inputs must use the **same feature-computation method** (same descriptors,
  same units/scale) as the training set. Same-named features computed with a
  different tool/settings are not guaranteed to be comparable.
- Predictions for chemicals well outside the training set's structural space
  (applicability domain) may be unreliable.
- This model predicts Ames mutagenicity specifically — not general toxicity.

