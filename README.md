# eKYC Fraud Detection — Unsupervised Anomaly Detection

This repository contains notebooks for detecting fraud in eKYC (electronic Know Your Customer) authentication data using unsupervised anomaly detection. The dataset (`fraud_data_encoded_v2.csv`) contains labeled records with two classes: `unknown` (unlabeled, used for training) and `fraud` (confirmed fraud, used for evaluation).

---

## Repository Structure

```
.
├── eda.ipynb                    # Exploratory Data Analysis
├── kmeans.ipynb                 # K-Means anomaly detection (full analysis)
├── kmeans_fraud_detection.ipynb # K-Means anomaly detection (clean pipeline)
├── ocsvm.ipynb                  # One-Class SVM anomaly detection (full analysis)
├── ocsvm_fraud_detection.ipynb  # One-Class SVM anomaly detection (clean pipeline)
├── fraud_data_encoded_v2.csv    # Dataset (required, not included)
└── README.md
```

---

## Dataset

Place `fraud_data_encoded_v2.csv` in the root directory before running any notebook.

The dataset contains eKYC session records with the following key columns:

| Column | Description |
|---|---|
| `fraud` | Label — `"fraud"` or `"unknown"` |
| `tokenid` | Unique request token (ID, dropped before modeling) |
| `ocr_cccd` | Raw OCR text from national ID card (high cardinality, dropped) |
| `finger_print` | Device fingerprint (unique per device, dropped) |
| `ipaddress` | IP address of the session |
| `client_session` | Client session string (unique identifier, dropped) |
| `ocr_tampering_flag` | Document tampering flag — `"yes"` / `"no"` |
| `front_liveness_fakeprint` | Front image liveness check — `"true"` / `"false"` |
| `back_liveness_fakeprint` | Back image liveness check — `"true"` / `"false"` |
| `front_object_liveness` | Front object liveness result — `"success"` / `"failure"` |
| `back_object_liveness` | Back object liveness result — `"success"` / `"failure"` |
| `liveness3d_object_liveness` | 3D liveness result — `"success"` / `"failure"` |
| `tampering_flag` | Numeric document tampering indicator |
| `req_path` | API request path |
| `year` / `month` / `day` | Timestamp components |
| `latitude` / `longitude` / `country` | Geolocation (dropped — noisy) |

**Label convention used across all notebooks:**

| Split | Label | Role |
|---|---|---|
| Train | `unknown` | Fit the model — labels never passed to `.fit()` |
| Test | `fraud` | Evaluate recall on confirmed fraud cases |

> `unknown ≠ normal`. The training set may contain hidden fraud, which slightly blurs the model's view of normal behavior.

---

## Shared Preprocessing Pipeline

All four model notebooks share an identical preprocessing pipeline:

**1. Flag encoding** — Six string flag columns are mapped to 0/1:

| Column | Mapping |
|---|---|
| `ocr_tampering_flag` | `yes → 1`, `no → 0` |
| `front/back_liveness_fakeprint` | `true → 1`, `false → 0` |
| `front/back_object_liveness` | `success → 1`, `failure → 0` |
| `liveness3d_object_liveness` | `success → 1`, `failure → 0` |

**2. Feature selection** — Columns are dropped by the following rules (applied in order):

| Rule | Reason |
|---|---|
| Label columns (`fraud`, `target`) | Not a feature |
| Manual drop list | ID-like or redundant (`tokenid`, `ocr_cccd`, `finger_print`, etc.) |
| Name keyword rule | Columns containing `uuid`, `hash`, `embedding`, `image`, `img`, etc. |
| Single-value columns | Zero variance — no signal |
| `>= 98%` missing | Nearly empty |
| High-cardinality categoricals | `> 30` unique values or `> 5%` unique ratio |

**3. sklearn ColumnTransformer pipeline:**
- **Numeric:** median imputation → `StandardScaler`
- **Categorical:** fill `"missing"` → `OneHotEncoder(handle_unknown="ignore")`

**4. Dimensionality reduction** — If the OHE output is sparse, `TruncatedSVD` (up to 50 components) is applied followed by a second `StandardScaler` to produce a dense matrix suitable for both K-Means and One-Class SVM.

---

## Notebook 1 — Exploratory Data Analysis (`eda.ipynb`)

A focused EDA on the fraud dataset structured into 6 sections.

**What it covers:**

**Section 1 — Quick Look** — Dtype summary, label distribution, and verification that the binary `target` column (fraud=1, unknown=0) was created correctly.

**Section 2 — Missing Values and Cardinality** — A full column summary table showing missing count, missing percentage, unique count, and a sample value per column. Plots the top 15 columns by missing percentage and separates columns by cardinality level (low ≤ 10 unique, high ≥ 1000 unique).

**Section 3 — Small-Category Columns** — Iterates through all low-cardinality columns (≤ 10 unique values), printing value counts and a bar chart for each. Useful for spotting status fields, hidden binary flags, and formatting issues.

**Section 4 — Repeated Keys** — Analyzes five key-like columns (`tokenid`, `ocr_cccd`, `finger_print`, `ipaddress`, `client_session`) for repetition. Builds a summary table with unique count, number of repeated keys, rows in repeated keys, and repeat ratio. Also computes fraud rate per IP address, filtered to IPs with at least 2 records to suppress noisy one-off cases.

**Section 5 — Feature vs Label** — For each of eight business-facing features (`req_path`, `ocr_tampering_flag`, `front_liveness_fakeprint`, `front_object_liveness`, `back_liveness_fakeprint`, `back_object_liveness`, `liveness3d_object_liveness`, `tampering_flag`): prints a count crosstab, a row-normalized rate table, and a fraud-rate bar chart for each.

**Section 6 — Notes Before Modeling** — Flags ID-like columns that should be dropped or reviewed before modeling by scanning for keywords (`uuid`, `hash`, `timestamp`, `image`, `embedding`). Also displays all confirmed fraud rows in full for manual inspection.

**Key helper functions:**

| Function | Purpose |
|---|---|
| `preprocess_data()` | Encodes flag columns, casts numeric columns, creates `target` |
| `build_column_summary()` | Returns per-column dtype, missing count/%, cardinality, sample value |
| `label_distribution()` | Count and percentage breakdown of the label column |
| `build_duplicate_summary()` | Repetition stats for key-like columns |
| `crosstab_count_and_rate()` | Count and row-normalized crosstab of any feature vs label |
| `plot_positive_rate()` | Bar chart of fraud rate per category value |

---

## Notebook 2 — K-Means Anomaly Detection (`kmeans.ipynb`)

Full K-Means pipeline with extended feature importance and anomaly profile analysis.

**Approach:** Fit K-Means on `unknown` (unlabeled) records. Use each sample's distance to its nearest centroid as an anomaly score. Flag the top 1% farthest points as suspected fraud.

**Key hyperparameters:**

| Parameter | Value | Description |
|---|---|---|
| `KMEANS_K` | 3 | Number of clusters |
| `KMEANS_DISTANCE_QUANTILE` | 0.99 | Top 1% farthest points flagged as anomalies |
| `SVD_MAX_COMPONENTS` | 50 | Max TruncatedSVD components for sparse OHE output |
| `MAX_MISSING_RATE` | 0.98 | Drop columns with ≥ 98% missing values |
| `MAX_CATEGORICAL_LEVELS` | 30 | Drop categoricals with > 30 unique values |

**Pipeline sections:**

1. **Load & Split** — Separates `unknown` rows (train) from `fraud` rows (test). Labels are never passed to `.fit()`.
2. **Feature Selection** — Applies the conservative drop rules described above.
3. **Preprocessing** — Builds the `ColumnTransformer` pipeline; applies `TruncatedSVD` if output is sparse.
4. **K-Means Training** — Fits on train set only. Anomaly score = `min(distance to all centroids)`.
5. **Cluster Visualization** — PCA 2D projection of training data colored by cluster assignment.
6. **Threshold & Predictions** — Threshold = 99th percentile of train distances. No test labels used.
7. **Evaluation** — Reports recall: how many of the `fraud` test rows were flagged.
8. **Anomaly Score Distribution** — Overlaid histogram of train vs test distances with threshold line.
9. **Export** — Saves flagged training rows to `train_flagged.csv`.

**Feature Importance Analysis (Section 11):**

- **Centroid contribution** — Per-feature squared distance to nearest centroid: `contribution_j = (x_j − centroid_j)²`. Aggregated over all fraud test rows and flagged-only rows separately. OHE-expanded features are collapsed back to original column names.
- **Feature distribution: train vs test** — Numeric: mean/median/std comparison sorted by absolute mean difference %. Categorical: top-3 value frequencies sorted by max frequency difference.
- **Ablation test** — Retrains K-Means with the liveness & tampering feature group removed. Reports threshold, flagged count, recall, and recall drop vs baseline.

**Anomaly Profile Analysis (Section 12):**

- PCA 2D scatter: flagged anomalies (crimson) vs normal train points (grey).
- Boxplots comparing timestamp feature distributions (`req_timestamplog`, `liveness_timestamplog_front`, `facecompare_timestamplog`, etc.) between flagged and non-flagged training rows.

**Output file:** `train_flagged.csv`

---

## Notebook 3 — K-Means Fraud Detection (`kmeans_fraud_detection.ipynb`)

A clean, production-oriented version of the K-Means pipeline. Identical logic to `kmeans.ipynb` through Section 11 (feature importance + ablation), without the Section 12 anomaly profiling plots.

The only functional difference is the export filename: flagged training rows are saved to `k_means_train_flagged.csv`.

**Output file:** `k_means_train_flagged.csv`

---

## Notebook 4 — One-Class SVM Anomaly Detection (`ocsvm.ipynb`)

Full One-Class SVM pipeline with extended feature importance and anomaly profile analysis.

**Approach:** Fit One-Class SVM on `unknown` records. Use the **negated decision function** as the anomaly score — higher score means further outside the learned boundary, i.e., more anomalous. Flag the top 1% most anomalous points as suspected fraud.

**Score sign convention:**
- sklearn `decision_function` returns positive for inliers (inside boundary) and negative for outliers.
- Negating it aligns with the K-Means convention: **higher score = more anomalous**.

**Key hyperparameters:**

| Parameter | Value | Description |
|---|---|---|
| `OCSVM_KERNEL` | `"rbf"` | RBF kernel for nonlinear boundary |
| `OCSVM_NU` | 0.01 | Upper bound on fraction of training outliers (~1%) |
| `OCSVM_GAMMA` | `"scale"` | Kernel coefficient |
| `OCSVM_ANOMALY_QUANTILE` | 0.99 | Top 1% most anomalous flagged |

**Pipeline sections:**

1. **Load & Split** — Same label-based split as K-Means.
2. **Feature Selection** — Same conservative drop rules.
3. **Preprocessing** — Same `ColumnTransformer` + optional `TruncatedSVD`.
4. **One-Class SVM Training** — Fits on train set only. Score = `-decision_function(X)`.
5. **PCA 2D Visualization** — Train points colored by anomaly score (red-yellow-green colormap).
6. **Threshold & Predictions** — 99th percentile of train scores. No test labels used.
7. **Evaluation** — Recall on confirmed fraud test rows.
8. **Anomaly Score Distribution** — Overlaid histogram of train vs test scores with threshold line.
9. **Export** — Saves all scored rows (train and test) with `ocsvm_anomaly_score` and `ocsvm_flagged` columns to `train_scored_ocsvm.csv` and `test_scored_ocsvm.csv`.

**Feature Importance Analysis (Section 11):**

- **Permutation-based anomaly impact** — For each feature, shuffles it 5 times and measures the mean drop in group anomaly score vs the unshuffled baseline: `impact_j = mean_score_baseline − mean_score_when_feature_j_shuffled`. Larger drop = feature contributes more to the anomaly score. Computed separately for all fraud rows and flagged-only rows, then collapsed back to original column names.
- **Feature distribution: train vs test** — Same numeric and categorical comparisons as K-Means.
- **Ablation test** — Retrains OCSVM with liveness & tampering features removed. Reports recall drop vs baseline.

**Anomaly Profile Analysis (Section 12):**

- PCA 2D scatter: flagged anomalies (crimson) vs normal train (grey).
- Boxplots of timestamp features comparing flagged vs non-flagged training rows.

**Output files:** `train_scored_ocsvm.csv`, `test_scored_ocsvm.csv`

---

## Notebook 5 — One-Class SVM Fraud Detection (`ocsvm_fraud_detection.ipynb`)

A clean, production-oriented version of the OCSVM pipeline. Identical logic to `ocsvm.ipynb` through Section 11 (feature importance + ablation), without the Section 12 anomaly profiling plots.

The only functional difference is the export: only the flagged training rows are saved (without per-row anomaly scores) to `ocsvm_train_flagged.csv`.

**Output file:** `ocsvm_train_flagged.csv`

---

## Key Design Decisions

**Why use `unknown` as train and `fraud` as test?**
There are no labeled `normal` records. `unknown` is the best available proxy for normal behavior, but it may contain hidden fraud — the model learns a slightly imperfect view of normality as a result.

**Why use the 99th percentile as the anomaly threshold?**
The threshold is derived purely from the train set distribution without touching any test labels. Flagging approximately 1% of training records as anomalous also matches the `nu=0.01` setting in OCSVM, keeping the false-positive rate manageable for an operations team to review.

**Why TruncatedSVD before K-Means / OCSVM?**
One-Hot Encoding categorical columns produces a sparse, high-dimensional matrix. Both K-Means (distance-based) and OCSVM (kernel-based) benefit from a dense, lower-dimensional representation. TruncatedSVD compresses the sparse OHE output to at most 50 components, followed by a StandardScaler.

**Why drop high-cardinality columns like `ocr_cccd`, `finger_print`, and `tokenid`?**
These columns are essentially unique identifiers per row. They carry no generalizable signal for anomaly detection and would dominate distance and kernel calculations if included.

**Why permutation impact for OCSVM instead of centroid contribution?**
OCSVM does not have explicit cluster centers so centroid contribution is not applicable. Permutation-based feature importance is a model-agnostic alternative: shuffling a feature breaks its relationship with the anomaly boundary, and the resulting score drop measures how much that feature contributed to the anomaly signal.

---

## Setup & Requirements

```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

Python 3.8+ recommended. Run notebooks in this order:

```bash
jupyter notebook eda.ipynb
jupyter notebook kmeans.ipynb                # full analysis
jupyter notebook kmeans_fraud_detection.ipynb # clean pipeline
jupyter notebook ocsvm.ipynb                 # full analysis
jupyter notebook ocsvm_fraud_detection.ipynb  # clean pipeline
```
