# POWER GRID FORECASTING (Time Series - Regression Problem)

## Table of contents
1. [Project Overview](#project-overview)  
2. [Key Results](#key-results)  
3. [Dataset Summary](#dataset-summary)  
4. [Environment and Setup](#environment-and-setup)  
5. [Reproducible Artifacts](#reproducible-artifacts)  
6. [Workflow - Step by step](#workflow---step-by-step)  
   6.1 [Initial inspection](#initial-inspection)  
   6.2 [Datetime handling and index](#datetime-handling-and-index)  
   6.3 [Target visualization and outlier removal](#target-visualization-and-outlier-removal)  
   6.4 [High-level EDA and pattern discovery](#high-level-eda-and-pattern-discovery)  
   6.5 [Time-based feature extraction](#time-based-feature-extraction)  
   6.6 [Boxplots and distribution inspection](#boxplots-and-distribution-inspection)  
   6.7 [Bivariate scatter and correlation matrix](#bivariate-scatter-and-correlation-matrix)  
   6.8 [Decomposition and stationarity tests](#decomposition-and-stationarity-tests)  
   6.9 [Solar imputation and rolling proxy](#solar-imputation-and-rolling-proxy)  
   6.10 [Dropping rolling NaNs and final cleanup](#dropping-rolling-nans-and-final-cleanup)  
   6.11 [Column pruning](#column-pruning)  
   6.12 [Train and test split (time based)](#train-and-test-split-time-based)  
   6.13 [Baseline model](#baseline-model)  
   6.14 [Initial XGBoost training](#initial-xgboost-training)  
   6.15 [Hyperparameter tuning](#hyperparameter-tuning)  
   6.16 [Feature importance and leakage discovery](#feature-importance-and-leakage-discovery)  
   6.17 [Final predictions and visualization](#final-predictions-and-visualization)  
7. [Evaluation metrics and numbers](#evaluation-metrics-and-numbers)  
8. [Key findings, interpretation and operational caveats](#key-findings-interpretation-and-operational-caveats)  
9. [Reproducibility and how to re-run](#reproducibility-and-how-to-re-run)  

---

## Project Overview
Project title: POWER GRID FORECASTING (Time Series Regression)  
Goal: Build and evaluate hourly demand forecasting models using historical grid data (2015 to 2025). Deliver a reproducible analysis, highlight data quality issues, and recommend a path to a deployable forecasting model.

## Key Results
- Cleaned dataset used for modeling: 92,186 rows (after outlier removal and dropping rolling NaNs).  
- Train / test split (time based): Train = 69,767 rows (before 2023-01-01), Test = 22,419 rows (2023-01-01 onward).  
- Baseline (persistence, lag-1): MAE ≈ 399.05 MW.  
- Initial XGBoost (untuned): MAE ≈ 187.20 MW.  
- Tuned model (RandomizedSearchCV reported CV MAE): 56.37 MW (diagnostic).  
- Tuned model test RMSE (computed): ≈ 76.84 MW.  
- Major caveat: tuned performance is driven by generation_mw (feature importance ≈ 95.6%), indicating data leakage. Do not deploy the tuned model that uses generation_mw without addressing leakage.

## Dataset Summary
- Source file: power_grid_corp_power_demand.csv  
- Original shape: 92,218 rows × 14 columns.  
- Key columns: datetime, generation_mw, demand_mw, load_shedding, gas, liquid_fuel, coal, hydro, solar, wind, india_bheramara_hvdc, india_tripura, india_adani, nepal.  
- Missingness (raw): solar 21,888 missing; wind 73,562 missing; india_adani 84,886 missing; nepal 86,871 missing.  
- Outliers: some extremely large values discovered in describe() (scientific notation indicated implausible values).

## Environment and Setup
- Core libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, statsmodels.  
- Reproducibility: use a pinned environment or requirements file, set a random seed, and save all artifacts.

## Reproducible Artifacts
- Cleaned dataset snapshot.  
- Notebook with EDA, feature engineering, modeling, and tuning steps.  
- Trained model artifacts (initial and best_estimator).  
- Plots: missingness heatmap, demand time series before and after cleaning, hourly and weekday averages, correlation heatmap, decomposition plots, feature importance chart.  
- Final predictions file for the test set and sample-week comparison plots.

---

## Workflow - Step by step

### Initial inspection
- Action: Load CSV, view head/tail, check shape, dtypes, null counts, and describe.  
- Reason: Confirm timestamp range and detect missing values and gross data issues.  
- Result: Confirmed data spans 2015-04-19 to 2025-06-17. Noted heavy missingness in solar and wind and extremely large values in describe().

### Datetime handling and index
- Action: Convert datetime column to pandas datetime and set index.  
- Reason: Enable time-based slicing, rolling, and decomposition.  
- Result: Index supports hourly operations and slicing.

### Target visualization and outlier removal
- Action: Plot demand_mw as scatter to reveal spikes. Chose threshold = 25,000 MW.  
- Result: 8 rows removed where demand_mw > 25,000. Data shape changed accordingly.

### High-level EDA and pattern discovery
- Action: Re-plot cleaned demand and inspect for trend and seasonality.  
- Observations: Long-term upward trend, strong yearly seasonality (summer peaks), increasing variance with trend. Concluded multiplicative behavior.

### Time-based feature extraction
- Action: Create hour, dayofweek, month, year, dayofyear, weekofyear.  
- Reason: Capture daily, weekly, and seasonal cycles.  
- Result: Hour and dayofweek found predictive on aggregation.

### Boxplots and distribution inspection
- Action: Boxplots of demand by month, hour, dayofweek, year.  
- Observations: Summer months show higher demand and variance; evening hours show peak spread; yearly boxes grow taller.

### Bivariate scatter and correlation matrix
- Action: Sample 10k rows for scatter plots; compute correlation matrix for candidate variables.  
- Observations: generation_mw and demand_mw are nearly perfectly correlated. Coal, india_adani, gas show strong correlation. Solar, hydro, wind weak.

### Decomposition and stationarity tests
- Action: Run multiplicative seasonal_decompose with period=24. Run ADF and KPSS tests.  
- Result: Trend and daily seasonality confirmed. ADF suggests stationarity; KPSS suggests non-stationarity. Practical conclusion: series is non-stationary with strong seasonality and long-term trend.

### Solar imputation and rolling proxy
- Action: Impute missing solar with 0. Create rolling_mean_solar_24h using solar.shift(1).rolling(24).mean().  
- Reason: Use solar as a weather proxy while avoiding leakage by shifting first.  
- Note: first 24 rows of this feature are NaN.

### Dropping rolling NaNs and final cleanup
- Action: Drop first 24 rows produced by rolling.  
- Result: Row count after dropna: 92,186. Verified no remaining NaNs.

### Column pruning
- Action: Remove wind, india_adani, nepal due to extreme missingness and weak signal. Keep generation_mw in a diagnostic run but mark it as leakage.  
- Reason: Avoid speculative imputation and noisy predictors.

### Train and test split (time based)
- Action: Train on data before 2023-01-01 and test from 2023-01-01 onward.  
- Result: Train 69,767 rows; Test 22,419 rows. This simulates forecasting on future data.

### Baseline model
- Action: Persistence naive baseline predict demand_t = demand_t-1.  
- Result: Baseline MAE ≈ 399.05 MW.

### Initial XGBoost training
- Action: Train XGBRegressor with n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01 using an eval_set.  
- Result: Initial test MAE ≈ 187.20 MW. Training logs showed validation_1-rmse near 581.706.

### Hyperparameter tuning
- Action: RandomizedSearchCV over n_estimators, learning_rate, max_depth, subsample, colsample_bytree with sample weights = np.arange(1, N+1) to emphasize recent data. Used cv=3.  
- Result: Best params: n_estimators=800, learning_rate=0.05, max_depth=3, subsample=1.0, colsample_bytree=1.0. RandomizedSearchCV reported CV MAE ≈ 56.37.

### Feature importance and leakage discovery
- Action: Inspect feature_importances_ on tuned model.  
- Result: generation_mw importance ≈ 0.956. This confirms leakage: the model achieves low error by relying on a contemporaneous reactive feature.

### Final predictions and visualization
- Action: Predict on test set and plot sample week (2024-01-01 to 2024-01-08).  
- Result: Tuned predictions track actuals closely; tuned test RMSE ≈ 76.84 MW. However, close tracking is attributable to leakage.

---

## Evaluation metrics and numbers
- Baseline (persistence) MAE: 399.05 MW.  
- Initial XGBoost MAE: 187.20 MW.  
- Tuned model CV MAE: 56.37 MW (diagnostic).  
- Initial model test RMSE: ≈ 581.71 MW.  
- Tuned model test RMSE: ≈ 76.84 MW.  
- Outliers removed: 8 rows.  
- Rows removed due to rolling NaNs: 24 rows.  
- Final cleaned rows: 92,186 (keep the cleaned CSV snapshot for exact reproducibility).

---

## Key findings, interpretation and operational caveats
- The tuned model's excellent numeric performance is driven almost entirely by generation_mw, which is reactive to demand. This is data leakage and invalidates the tuned model for forward forecasting.  
- The correct production approach is to remove generation_mw and similar contemporaneous features or to obtain independent forecasts for those variables and use them as exogenous inputs.  
- The naive persistence baseline remains a useful operational benchmark.  
- Columns with extremely high missingness were removed to avoid speculative imputation.

---

## Reproducibility and how to re-run
- Steps to reproduce:
  1. Create environment and install required packages.  
  2. Load CSV, convert datetime to index, and run the notebook top-to-bottom.  
  3. Remove demand outliers above 25,000 MW.  
  4. Create time features and rolling solar proxy as documented.  
  5. Drop sparse columns (wind, india_adani, nepal).  
  6. Split by date at 2023-01-01 and train models as shown.  
- Notes: Use TimeSeriesSplit for time-aware CV in a production retrain. Save model artifacts and the final cleaned CSV.

---
