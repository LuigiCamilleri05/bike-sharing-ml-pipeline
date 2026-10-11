# Bike Sharing Demand Prediction: A Machine Learning Pipeline

Predicting hourly bike rental demand with **CART decision trees and random forests implemented from scratch in Python**. This project compares individual trees, ensemble learning, and two hyperparameter search strategies on the Bike Sharing dataset.

Developed as coursework for Machine Learning and Fuzzy Logic, it focuses on understanding how tree-based regression works, from choosing splits to averaging predictions across a forest. NumPy powers the model implementations; scikit-learn provides data splitting, estimator interfaces, and cross-validation tools.

## What the project includes

- A recursive CART regression tree with MSE, MAE, and sum of squared residuals (SSR) splitting criteria.
- A random forest with bootstrap sampling, random feature selection at each split, and averaged predictions.
- A reproducible training, validation, and test split.
- Grid search and random search with three-fold cross-validation.
- Evaluation using MSE, RMSE, MAE, and R².
- Feature importance, prediction, and model comparison plots.

## Dataset

The project uses `dataset/hour.csv`: **17,379 hourly observations** from Capital Bikeshare in Washington, D.C., covering **2011–2012**. The target, `cnt`, is the total number of rentals in an hour.

The 12 predictors describe calendar and weather conditions:

| Group | Features |
| --- | --- |
| Calendar | `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday` |
| Weather | `weathersit`, `temp`, `atemp`, `hum`, `windspeed` |

`instant` and `dteday` are excluded. `casual` and `registered` are also excluded to prevent target leakage, since their sum equals `cnt`. The original dataset's normalised weather values and numeric category codes are used directly.

The daily dataset, `dataset/day.csv`, is included but is not used by the main experiment. See [the dataset documentation](dataset/README.txt) for field definitions and attribution.

## Getting started

Use **Python 3.11 or later** and run the following commands from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/main.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/main.py
```

**Case-sensitive filesystems:** before running, change both `'Dataset'` path components in `src/main.py` to `'dataset'` to match the included directory.

Dependencies are NumPy, pandas, Matplotlib, and scikit-learn; minimum versions are listed in [requirements.txt](requirements.txt).

The full run trains the baseline models, performs both searches, prints evaluation metrics, and saves results. Hyperparameter tuning can take substantial time because the custom trees examine candidate thresholds in Python and the searches run sequentially. For a smaller experiment, reduce the parameter grid or `n_iter_random` in `src/main.py`.

## Experiment design

Data is shuffled and split approximately **70% / 15% / 15%** into training, validation, and test sets, using random seed **50**. Baseline models are evaluated on all three sets. Hyperparameter search uses only the training set with three-fold cross-validation, selects models by R², and evaluates the refitted models on the test set.

| Baseline | Configuration |
| --- | --- |
| Decision tree | Maximum depth 10, minimum 5 samples per leaf, MSE criterion |
| Random forest | 50 trees, maximum depth 10, minimum 5 samples per leaf, square-root feature sampling, MSE criterion, seed 50 |

Both search methods use the following parameter space:

| Parameter | Values |
| --- | --- |
| `n_estimators` | 10, 30 |
| `max_depth` | 5, 10 |
| `min_samples_leaf` | 2, 5, 10 |
| `criterion` | `mse`, `mae`, `ssr` |

Grid search evaluates all **36 combinations**. Random search samples **10 combinations** with seed 50. Both retain the forest's default square-root feature sampling.

## Recorded results

The table below reproduces the existing local `results/hyperparameter_tuning.txt` output; it is not a newly executed benchmark. Higher R² and lower error values indicate better performance. RMSE and MAE are measured in hourly rental counts.

| Search method | Best CV R² | Test R² | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: | ---: |
| Grid search | 0.9225 | 0.9248 | 50.31 | 30.21 |
| Random search | 0.9104 | 0.9108 | 54.78 | 33.85 |

The best recorded grid search configuration used **30 trees**, **depth 10**, **2 samples per leaf**, and **SSR**. The best recorded random search configuration used **10 trees**, **depth 10**, **5 samples per leaf**, and **MSE**.

These results describe performance on a shuffled holdout from the historical dataset. Since observations are time-dependent, they should not be interpreted as a chronological forecasting benchmark. A future forecasting experiment should use a time-ordered split and time-aware cross-validation. Dependency versions are not pinned, so results may differ between environments.

## Generated outputs

Each complete run writes or replaces:

```text
dataset/splits/
├── train.csv
├── validation.csv
└── test.csv

results/
├── hyperparameter_tuning.txt
└── plots/
    ├── feature_importance.png
    ├── predictions_vs_actual.png
    └── model_comparison.png
```

The feature importance and prediction plots compare the baseline tree and forest. The model comparison plot includes both baselines and both tuned forests. Results are ignored by Git and are generated locally rather than linked as images here.

## Project structure

```text
├── dataset/
│   ├── README.txt                # Original dataset documentation
│   ├── hour.csv                  # Hourly data used by the experiment
│   └── day.csv                   # Daily data
├── src/
│   ├── main.py                   # Complete experiment pipeline
│   ├── data_split.py             # Feature selection and data splitting
│   ├── decision_tree.py          # Custom CART regressor
│   ├── random_forest.py          # Custom random forest regressor
│   ├── hyperparameter_tuning.py  # Grid search and random search
│   ├── evaluation.py             # Metrics and feature importance
│   └── visualisation.py          # Result files and plots
├── requirements.txt
└── README.md
```

To change the baseline models or tuning settings, edit `src/main.py`. To change the split proportions or seed, edit the defaults in `src/data_split.py` or pass arguments to `split_data`.

## Dataset attribution

The original dataset documentation requests citation of:

Hadi Fanaee-T and João Gama. *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence (2013). DOI: [10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3).

Dataset attribution is separate from the project's source code licensing. No source code licence is currently included in this repository.
