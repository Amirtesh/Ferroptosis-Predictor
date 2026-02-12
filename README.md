# Ferroptosis-Predictor

## Interpretable QSAR Framework for Ferroptosis Modulator Classification

This repository contains the code, trained model, and supporting files for an interpretable QSAR framework developed to classify ferroptosis inducers and inhibitors.

The workflow includes data exploration, model training, interpretability analysis, and prediction utilities.

---

## Repository Structure

* **model_training.ipynb**
  Notebook for model development, cross-validation, scaffold-based evaluation, and performance analysis.

* **fp_exploration.ipynb**
  Morgan fingerprint analysis and functional group enrichment for interpretability.

* **eda.py**
  Script for exploratory data analysis of molecular descriptors and class distributions.

* **predict.py**
  Script to generate predictions for new compounds using the trained ensemble model.

* **ensemble_model.pkl**
  Serialized trained soft-voting ensemble classifier.

* **feature_info.json**
  Metadata describing descriptor and fingerprint feature ordering.

* **ferroptosis_features.csv**
  Precomputed feature matrix used for training and evaluation.

* **Model-interpretation.docx**
  Supplementary interpretability notes and analysis documentation.

* **README.md**
  Project documentation (this file).

---

Complete it cleanly and professionally like this:

---

## Recommended environment:

* Python 3.9+
* scikit-learn
* xgboost
* rdkit
* pandas
* numpy
* shap
* matplotlib
* seaborn

To install all dependencies:

```
pip install -r requirements.txt
```

To run the notebooks:

1. Clone the repository:

```
git clone https://github.com/Amirtesh/Ferroptosis-Predictor.git
cd Ferroptosis-Predictor
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:

```
jupyter notebook
```

4. Open:

* `model_training.ipynb` to reproduce training and evaluation
* `fp_exploration.ipynb` for fingerprint interpretation and enrichment analysis
* `Model-Interpretation.ipynb` to reproduce the interpretability analysis

---

## Usage

### 1. Model Training

To reproduce model development and evaluation:

Open:

```
model_training.ipynb
```

This notebook performs:

* Data loading
* Cross-validation
* Scaffold-based evaluation
* Performance metric calculation

---

### 2. Fingerprint Interpretation

To reproduce functional group enrichment and fingerprint analysis:

Open:

```
fp_exploration.ipynb
```

---

### 3. Predicting New Compounds

To generate predictions using the trained ensemble model on either a single SMILES string or a txt file containing one SMILES per line:

```
python3 predict.py --smiles CC(=O)O

python3 predict.py --smiles_file smiles.txt
```

`input.csv` should contain precomputed features in the same format as `ferroptosis_features.csv`.

---

## Notes

* The ensemble model is trained using a combination of Random Forest and XGBoost classifiers.
* Feature ordering must match the structure defined in `feature_info.json`.
* Heuristic screening rules described in the associated manuscript are not implemented as standalone classifiers; prediction relies on the trained ensemble model.

---

