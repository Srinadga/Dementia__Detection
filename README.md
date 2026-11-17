# Dementia Detection

This project explores supervised machine learning for classifying dementia status from a structured clinical dataset. The Jupyter notebook covers data preparation, exploratory visualizations, model training, and evaluation.

> **Disclaimer:** This is an educational machine-learning project, not a medical diagnostic tool. Predictions must not be used to diagnose or guide treatment.

## Project contents

```text
.
|-- Dementia.ipynb
|-- dementia_dataset.csv
`-- README.md
```

## Workflow

The notebook:

1. Loads `dementia_dataset.csv`.
2. Removes the subject and MRI identifier columns.
3. Imputes missing values in `SES` and `MMSE`.
4. Encodes categorical columns (`M/F`, `Hand`, and `Group`).
5. Visualizes age, MMSE by group, feature correlations, and class counts.
6. Splits data into stratified training and test sets (80/20).
7. Trains and evaluates Random Forest, XGBoost, Gradient Boosting, stacking, blending, and voting classifiers.
8. Reports accuracy, weighted precision and recall, confusion matrices, classification reports, and ROC/AUC visualizations.

The target column is `Group`, with the notebook's encoded classes documented as `0 = Converted`, `1 = Demented`, and `2 = Nondemented`.

## Requirements

- Python
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost

Install the libraries in your Python environment:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost notebook
```

## Run the notebook

Keep `Dementia.ipynb` and `dementia_dataset.csv` in the same folder, then start Jupyter:

```bash
jupyter notebook
```

Open `Dementia.ipynb` and run the cells in order. The notebook reads the dataset using the relative filename `dementia_dataset.csv`.

## Evaluation

The notebook evaluates model performance on a held-out test set and prints classification metrics. Results depend on the dataset and environment; no specific accuracy or best-performing model is guaranteed.
