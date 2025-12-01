# POWER GRID FORECASTING (TIME SERIES - REGRESSION PROBLEM)

## Table of contents (Index)
1. Project Overview
2. Key Results (at a glance)
3. Dataset (summary)
4. Environment & Setup
5. Reproducible Artifacts (what I produced)
6. Workflow — Step-by-step (chronological)
   - Initial inspection
   - Datetime handling & indexing
   - Target visualization and outlier removal
   - EDA and pattern discovery
   - Time-feature extraction
   - Boxplots and distribution checks
   - Bivariate analysis and correlation
   - Decomposition & stationarity tests
   - Solar imputation & rolling proxy (shift+rolling)
   - Final cleanup and column pruning
   - Train / test split (time-based)
   - Baseline model
   - Initial XGBoost training
   - Hyperparameter tuning (RandomizedSearchCV)
   - Feature importance and leakage discovery
   - Final predictions & visualization
7. Evaluation metrics & numbers (concrete)
8. Key findings, interpretation & operational caveats
9. Reproducibility & how to re-run (notes)
10. Short “What to show in a meeting” list
11. Next steps and recommended experiments
12. Appendix: quick references and explanations

---

## 1. Project Overview
- Project title: POWER GRID FORECASTING (Time Series Regression)
- Goal: Build and evaluate hourly demand forecasting models using historical power-grid data (2015–2025). Produce a reproducible analysis, highlight data-quality issues, and identify valid pathways to a deployable forecasting model.

---

## 2. Key Results (at a glance)
- Cleaned dataset used for modeling: **92,186** rows (after outlier removal and dropping rolling-window NaNs).
- Train / test split (time-based):
  - Train: **69,767** rows (~76%) — all data before 2023-01-01
  - Test: **22,419** rows (~24%) — 2023-01-01 onward
- Baseline (naive persistence, demand_t = demand_t−1): **MAE ≈ 399.05 MW**
- Initial XGBoost (untuned): **MAE ≈ 187.20 MW**
- Tuned model (RandomizedSearchCV reported CV MAE): **56.37 MW** (diagnostic)
- Tuned model test RMSE (computed on test predictions): **≈ 76.84 MW**
- Major caveat: The tuned model relies overwhelmingly on `generation_mw` (feature importance ≈ 95.6%), a contemporaneous / reactive variable — this is data leakage. The tuned MAE should not be taken as an honest forecasting metric without removing leaked features.

---

## 3. Dataset (summary)
- File used: `power_grid_corp_power_demand.csv`
- Original shape: **92,218 rows × 14 columns**
- Key columns:
  - datetime, generation_mw, demand_mw (target), load_shedding, gas, liquid_fuel, coal, hydro, solar, wind, india_bheramara_hvdc, india_tripura, india_adani, nepal
- Missingness (raw counts):
  - solar: 21,888 missing
  - wind: 73,562 missing
  - india_adani: 84,886 missing
  - nepal: 86,871 missing
- Outliers: some extremely large values discovered (scientific notation like 6.45e+07 revealed physically impossible spikes).

---

## 4. Environment & Setup
- Core stack used: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, statsmodels
- Reproducibility notes:
  - Use a stable environment (requirements.txt or conda env)
  - Set a global random seed for deterministic outcomes during experiments
  - Save all intermediate artifacts (cleaned CSVs, model picks, plots)

---

## 5. Reproducible Artifacts (what I produced)
- Cleaned dataset snapshot (after outlier removal and dropping initial rolling NaNs)
- Notebook with EDA, feature engineering, modeling & tuning steps
- Trained model artifacts (initial and best_estimator from tuning)
- Plots: missingness heatmap, demand timeseries (before/after cleaning), hourly/day-of-week averages, correlation heatmap, decomposition, feature-importance
- Final predictions file (test set predictions) and plotted comparisons for sample week (2024-01-01 to 2024-01-08)

---

## 6. Workflow — Step-by-step (chronological)

The following lists the exact steps executed in the order performed. Each step notes the action, reason, and brief result.

### 6.1 Initial inspection
- Action: Load CSV, view head/tail, shape, dtypes, null counts, descriptive statistics.
- Why: Verify format, timestamp range, presence of missing values, and detect gross data problems.
- Result: Data ranges from 2015-04-19 to 2025-06-17; heavy missingness in `solar`, `wind`, `india_adani`, `nepal`.

### 6.2 Datetime handling & index
- Action: Converted `datetime` to pandas datetime and set as index.
- Why: Required for all time-based operations (lag/rolling/decompose/splits).
- Result: Index now supports hourly operations and slicing.

### 6.3 Target visualization and outlier removal
- Action: Plotted `demand_mw` as scatter dots to expose point-level anomalies; selected threshold = 25,000 MW.
- Why: Visual detection of physically implausible spikes that would distort models.
- Result: 8 rows removed (demand > 25,000). Shape changed from 92,218 → 92,210 (after outlier removal).

### 6.4 High-level EDA and pattern discovery
- Action: Re-plot cleaned demand and inspect time series.
- Why: Identify trend, seasonality, variance changes to guide feature engineering.
- Result: Observed increasing long-term trend, strong yearly seasonality (summer peak), and increasing variance — multiplicative behavior.

### 6.5 Time-based feature extraction
- Action: Create `hour`, `dayofweek`, `month`, `year`, `dayofyear`, `weekofyear`.
- Why: These encode daily/weekly/annual cycles important for forecasting.
- Result: Hour and dayofweek found to be predictive upon aggregation.

### 6.6 Boxplots and distribution inspection
- Action: Boxplots of demand by month, hour, dayofweek, year.
- Why: Inspect distributional changes and where variance grows.
- Result: Summer months (Apr–Sep) show higher and more variable demand; evening hours highest spread.

### 6.7 Bivariate scatter inspections and correlation matrix
- Action: Sampled 10k rows for scatter plots and computed correlation heatmap for candidate variables.
- Why: Identify candidate predictors and detect potential leakage.
- Result: `generation_mw` and `demand_mw` appear virtually perfectly correlated (1.00). Coal, `india_adani`, and gas show strong positive correlation. Solar/hydro/wind weakly correlated.

### 6.8 Decomposition & stationarity checks
- Action: Multiplicative seasonal_decompose (period=24) + ADF and KPSS tests.
- Why: Understand trend/seasonal structure; test stationarity assumptions.
- Result: Trend + daily seasonality confirmed. ADF suggested stationarity; KPSS suggested non-stationarity (contradiction explained by strong daily cycle + long-term trend). Practical conclusion: non-stationary with strong seasonality — include time features or detrend for models requiring stationarity.

### 6.9 Solar missing-value handling & rolling proxy
- Action:
  - Imputed missing solar as 0 (defensible: solar is zero during night; many NaNs are night/missing sensor).
  - Engineered `rolling_mean_solar_24h = solar.shift(1).rolling(24).mean()` — past-only 24-hour mean.
- Why:
  - Use solar as a proxy for weather/insolation.
  - Shift ensures no leakage (use only past values); rolling(24) summarizes the last day.
- Result: Rolling feature has NaNs for first 24 rows (lack of full history).

### 6.10 Dropping rows with NaNs created by rolling & final cleanup
- Action: Dropped first 24 rows produced by the rolling computation.
- Why: Those rows lack a full 24-hour history; minimal data lost relative to overall dataset.
- Result: Row count after dropna: **92,186**. Verified no remaining NaNs in df_cleaned.

### 6.11 Column pruning (why certain predictors were removed)
- Action: Removed `wind`, `india_adani`, `nepal` prior to final modeling; `generation_mw` kept initially for a diagnostic experiment but flagged as leakage.
- Why:
  - `wind`, `nepal`: extremely high missingness and weak predictive signal — imputation would be speculative and could add noise or bias.
  - `india_adani`: high missingness and reactive behavior (leakage candidate).
  - `generation_mw`: reactive/cheater variable — used in a diagnostic model to prove leakage; **must be excluded** for an honest forecasting model.
- Result: Final feature set excludes very sparse and noisy columns.

### 6.12 Train / test split (time-based)
- Action: Train on data before 2023-01-01; test from 2023-01-01 onward.
- Why: Simulate real forecasting (train on past, test on future) and prevent random-split leakage.
- Result: Train = **69,767** rows; Test = **22,419** rows.

### 6.13 Baseline model
- Action: Implemented naive persistence baseline (predict demand_t = demand_t−1).
- Why: Provide a simple, interpretable benchmark.
- Result: Baseline MAE on test ≈ **399.05 MW**.

### 6.14 Initial XGBoost training
- Action: Trained XGBoost regressor (n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01) with eval_set.
- Why: Test a powerful tree-based regressor that handles nonlinearities well.
- Result: Initial test MAE ≈ **187.20 MW**. Validation_1-rmse logged near 581.706 in training logs.

### 6.15 Hyperparameter tuning (RandomizedSearchCV with weighting)
- Action:
  - Performed RandomizedSearchCV to tune `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`.
  - Used `sample_weights = np.arange(1, len(X_train)+1)` to prioritize recent data (tilt for nonstationarity).
  - Used `cv=3` (note: standard CV used; TimeSeriesSplit recommended for time series).
- Result:
  - Best params found: `{'subsample': 1.0, 'n_estimators': 800, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 1.0}`
  - RandomizedSearchCV reported a best CV MAE ≈ **56.37**.

### 6.16 Feature importance and leakage discovery
- Action: Extracted `feature_importances_` from tuned model and inspected ranking.
- Why: Check which features the model relies on and detect leakage.
- Result: `generation_mw` importance ≈ **0.956** (95.6%). This confirms the tuned model's excellent metrics are driven by a leaked, contemporaneous feature — a fatal problem for real forecasting.

### 6.17 Final predictions & visualization
- Action: Predicted on test set with the tuned model and plotted sample week (2024-01-01 → 2024-01-08).
- Why: Visual check of daily tracking and error patterns (weekend lows observed).
- Result: Tuned predictions tracked actuals very closely (consistent with leakage-driven performance). Reported tuned RMSE on test ≈ **76.84** MW.

---

## 7. Evaluation metrics & numbers (concrete)
- Baseline (persistence) MAE: **399.05 MW**
- Initial XGBoost MAE (untuned): **187.20 MW**
- Tuned model CV MAE (RandomizedSearchCV report): **56.37 MW**
- Initial model test RMSE (from logs): **≈ 581.71 MW**
- Tuned model test RMSE (computed): **≈ 76.84 MW**
- Outliers removed: **8** rows (demand > 25,000)
- Rows removed due to rolling NaNs: **24** rows
- Final cleaned rows used for modeling: **92,186** (logged; be aware of the small discrepancy noted in the notebook comments — always refer to the artifact CSV snapshot)

> Note: The tuned CV MAE is a cross-validated *training* metric; the tuned model’s extremely low error is driven by leakage (see section 8). The honest forecasting metric must be computed after removing leaked features.

---

## 8. Key findings, interpretation & operational caveats
- The dataset exhibits strong daily seasonality and long-term upward trend (multiplicative behavior).
- Several columns have very high missingness (wind, india_adani, nepal); imputing them would be speculative.
- The tuned XGBoost model achieves excellent numeric performance, but feature-importance analysis proves it is dominated by `generation_mw` (≈95.6% importance). This indicates severe data leakage: `generation_mw` is reactive to demand and therefore not a valid predictor for future demand unless you can independently forecast generation.
- Operational consequence: do **not** deploy the tuned model that uses `generation_mw`. Instead, either:
  - Remove `generation_mw` and retrain; or
  - Build a separate, reliable forecast for `generation_mw` and feed that into a downstream demand model.
- The baseline persistence model remains a useful reference (MAE ≈ 399 MW).

---

## 9. Reproducibility & how to re-run (notes)
- Recommended minimal steps to reproduce:
  1. Create a Python environment with exact package versions (save `requirements.txt`).
  2. Load `power_grid_corp_power_demand.csv`.
  3. Convert `datetime` to datetime dtype and set as index.
  4. Remove demand outliers > 25,000 (8 rows).
  5. Create time features: hour, dayofweek, month, year, dayofyear, weekofyear.
  6. Impute `solar` missing values as 0; create `solar.shift(1).rolling(24).mean()` and drop the first 24 rows.
  7. Drop `wind`, `india_adani`, `nepal`.
  8. Split train/test by date (train < 2023-01-01, test ≥ 2023-01-01).
  9. Train baseline naive predictor and XGBoost; tune with RandomizedSearchCV as documented.
- Notes / caveats:
  - If you re-run tuning, use `TimeSeriesSplit` for CV to be strictly time-aware.
  - If you retrain an “honest” model, permanently exclude `generation_mw` (and any other contemporaneous reactive features such as contemporaneous load_shedding unless a forecast for them is available).

---

## 10. Short “What to show in a meeting” list
- Slide 1: Project goal, dataset range, and key artifacts.
- Slide 2: Time series plot (before and after outlier removal) — show 8 removed rows.
- Slide 3: Correlation heatmap highlighting generation vs demand.
- Slide 4: Feature importance bar chart (show `generation_mw` dominating).
- Slide 5: Metric table — Baseline MAE (399), Initial XGB MAE (187), Tuned CV MAE (56) — and explain leakage caveat.
- Slide 6: Recommendation & next-step timeline (retrain honest model within 48 hours).

---

## 11. Next steps and recommended experiments
1. **Immediate (24–48 hours)**:
   - Retrain models after excluding `generation_mw` (and exclude contemporaneous `load_shedding`) and report honest MAE/RMSE on the test set.
   - Use `TimeSeriesSplit` for hyperparameter tuning to respect time ordering.
2. **Short-term (1 week)**:
   - Acquire and incorporate exogenous weather forecasts (temperature, humidity) to improve honest model performance.
   - Run SHAP analysis on the honest model to identify genuinely predictive features.
3. **Medium-term**:
   - Create probabilistic forecasts (quantiles) to provide uncertainty bounds.
   - Implement a reproducible pipeline (preprocessing + model artifact) and a monitoring dashboard for rolling MAE and input drift.
4. **Optional experiments**:
   - Add multiple rolling windows (24h, 72h, 168h) for solar and test which windows are most predictive.
   - Benchmark other models (RandomForest, LightGBM, SARIMAX with a few exogenous features, or LSTM if you invest in more engineering).

---

## 12. Appendix: quick references and explanations
- `solar.shift(1).rolling(24).mean()` — computes the mean solar over the 24 hours prior to the current timestamp (uses only past information thanks to `shift(1)`); first 24 rows are NaN.
- Why `MAE`? — Mean Absolute Error is intuitive for stakeholders (average MW error). RMSE is also reported because it penalizes large errors more.
- Why Time-based split? — Simulates real forecasting: train on past, test on future to avoid temporal leakage.
- Why is `generation_mw` leakage? — Generation is set to meet demand at the same timestamps; it is not an exogenous predictor and therefore cannot be used for forward forecasts unless generation itself is forecast separately.

---

## Contact / ownership
- Analysis author: Tarun Sirohi (notebook and artifacts available in the project folder).

--- 
