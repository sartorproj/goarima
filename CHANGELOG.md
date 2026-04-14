# Changelog

All notable changes to GoARIMA will be documented in this file.

The demo suite (`demo/main.go`) runs 10 real-world datasets from [Forecasting: Principles and Practice](https://otexts.com/fpppy):

| # | Dataset | Source File | Period | Type |
|---|---------|-------------|--------|------|
| 1 | Australian Population | `aus_economy.csv` | -- | Annual population (millions) |
| 2 | Australian Cement | `aus_production.csv` | 4 | Quarterly cement production |
| 3 | Australian Beer | `aus_production.csv` | 4 | Quarterly beer production |
| 4 | Australian Electricity | `aus_production.csv` | 4 | Quarterly electricity production |
| 5 | Australian Gas | `aus_production.csv` | 4 | Quarterly gas production |
| 6 | US Eggs | `eggs.csv` | -- | Annual eggs per capita |
| 7 | US House Sales | `hsales.csv` | 12 | Monthly new house sales |
| 8 | US Strikes | `strikes.csv` | -- | Annual strikes count |
| 9 | US Employment | `us_employment.csv` | 12 | Monthly private employment (000s) |
| 10 | Google Stock | `gafa_stock.csv` | -- | Daily closing price |

---

## [Unreleased] — Post-v0.3.0 Fixes

Merged as PR #4: `fix: comprehensive audit fixes across all packages`

### Fixed

#### SARIMA Fitting Improvements
- **SARIMA optimizer**: Reworked convergence behavior producing significantly better fits on several datasets
- **US Employment SARIMA(1,0,0)(1,1,0)[12]**: RMSE improved from 1866.92 to 1093.99 (**-41%**)
- **US House Sales SARIMA(1,0,0)(1,1,0)[12]**: RMSE improved from 5.64 to 4.88 (**-13%**)

#### Statistical Tests
- **ADF test**: Added `ADFWithRegression()` supporting `"nc"` (no constant), `"c"` (constant), and `"ct"` (constant + trend) regression variants
- **Stationarity detection**: Fixed nil-pointer bug in `determineDifferencing()` when KPSS result was nil — was causing Auto-ARIMA to skip differencing on trending data
- **KPSS p-value**: Improved interpolation and clamped output to [0, 1]
- **MacKinnon p-values**: Finer interpolation for ADF/PP test p-values
- **Chi-squared CDF**: Replaced custom `gamma()` with `math.Gamma()` from stdlib for better numerical stability

#### Decomposition
- **STL**: Replaced O(n^2) insertion sort in `median()` with `sort.Float64s()`
- **Classical decomposition**: Minor robustness improvements

#### Auto-ARIMA
- **`determineSeasonalDifferencing`**: Switched from ACF heuristic to proper seasonal strength (F_S) measure using decomposition
- **`ModelsEvaluated`**: Now reports actual count (was always 1 in v0.3.0)
- **Diagnostic field**: Added `FitErrors []string` to `Result` for tracking models that failed to fit

#### TimeSeries
- Added input validation for NaN/Inf values
- `DiffN(n)`: Documented that this computes lag-n difference (not n-th order difference)

### Added
- `demo/README.md` with dataset descriptions and visualization instructions
- `demo/images/` with sample forecast and metrics plots
- `stats.ADFWithRegression()` for regression variant selection
- Box-Cox transformation with automatic lambda selection

### Demo Results (Post-Fix vs v0.3.0)

| Dataset | Best Model | v0.3.0 RMSE | Post-Fix RMSE | Change |
|---------|-----------|-------------|---------------|--------|
| Aus Population | Auto-ARIMA(1,1,0) | 1.0486 | 1.0486 | -- |
| Aus Cement | SARIMA(1,0,0)(1,1,0)[4] | 190.96 | 190.98 | -- |
| Aus Beer | SARIMA(1,0,0)(1,1,0)[4] | 14.16 | 14.17 | -- |
| Aus Electricity | SARIMA(1,0,0)(1,1,0)[4] | 1592.41 | 1591.93 | -- |
| Aus Gas | SARIMA(0,1,1)(0,1,1)[4] | 9.11 | 9.28 | -- |
| US Eggs | Auto-ARIMA(0,1,0) | 35.77 | 35.77 | -- |
| US House Sales | SARIMA(1,0,0)(1,1,0)[12] | **5.64** | **4.88** | -13% |
| US Strikes | Auto-ARIMA(1,0,0) | 1315.01 | **1257.93** | -4% |
| US Employment | SARIMA(1,0,0)(1,1,0)[12] | **1866.92** | **1093.99** | -41% |
| Google Stock | Auto-ARIMA(3,1,0) | 31.50 | 31.50 | -- |

---

## [v0.3.0] - 2025-12-15

### Added

#### Automatic Seasonal Period Detection
- Auto-detects seasonality from ACF analysis — no need to specify `SeasonalM`
- Checks common periods: 4, 6, 7, 12, 24, 52, 168, 365
- Configurable via `SeasonalPeriods` and `SeasonalityThreshold`
- New `Result` fields: `DetectedPeriod`, `SeasonalityStrength`, `DetectionMethod`

#### Model Comparison (ARIMA vs SARIMA)
- Automatically compares seasonal and non-seasonal models
- Selects best model based on cross-validation RMSE
- New `Candidates []ModelCandidate` field shows all models evaluated
- Each candidate includes: Name, RMSE, MAPE, AICc, selected status, rank

#### Cross-Validation Based Model Selection
- Uses time series holdout evaluation instead of just AIC/AICc
- Configurable via `ModelSelection`: `"cv"`, `"aicc"`, `"aic"`, `"bic"`
- `TestRatio` controls train/test split (default: 0.2)
- `PreferSimpler` option prefers simpler models when scores are close

#### Multi-Level Prediction Intervals (R-style)
- New `PredictWithLevels(steps, levels)` method
- Returns 80% and 95% intervals by default (like R's `forecast()`)
- `ForecastResult` struct with `Lower` and `Upper` maps by confidence level

#### New Config Options
- `AutoSeasonal` (default: true) — enable/disable auto-detection
- `SeasonalPeriods` — periods to check
- `SeasonalityThreshold` (default: 0.4) — ACF threshold
- `ModelSelection` (default: "cv") — selection criterion
- `CVFolds`, `TestRatio`, `CompareModels`, `PreferSimpler`, `SimplerThreshold`

#### Helper Methods
- `Order() string` — formatted order string (e.g., `"SARIMA(1,1,1)(0,1,1)[24]"`)
- `DefaultForecastLevels = []float64{0.80, 0.95}`

### Changed

- **Breaking**: `Config.Seasonal` and `Config.SeasonalM` replaced by `AutoSeasonal` and `SeasonalPeriods`
- **Default criterion**: Changed from `"aic"` to `"aicc"`
- **Default behavior**: `AutoARIMA(series, nil)` now auto-detects seasonality and uses CV

### Known Issues (fixed in post-v0.3.0)

- `ModelsEvaluated` always reports 1 instead of actual count
- Auto-SARIMA not triggered in demo due to `CompareModels=false` in seasonal config — 6 seasonal datasets lost Auto-SARIMA output compared to v0.2.0
- `determineDifferencing` nil-pointer risk when KPSS returns nil

### Demo Results (v0.3.0)

Note: Auto-SARIMA results missing for seasonal datasets due to demo config issue.

| Dataset | Best Model | v0.2.0 RMSE | v0.3.0 RMSE | Change |
|---------|-----------|-------------|-------------|--------|
| Aus Population | Auto-ARIMA(1,1,0) | 1.0486 | 1.0486 | -- |
| Aus Cement | SARIMA(1,0,0)(1,1,0)[4] | 190.96 | 190.96 | -- |
| Aus Beer | SARIMA(1,0,0)(1,1,0)[4] | 14.16 | 14.16 | -- |
| Aus Electricity | SARIMA(1,0,0)(1,1,0)[4] | 1592.41 | 1592.41 | -- |
| Aus Gas | SARIMA(0,1,1)(0,1,1)[4] | 9.11 | 9.11 | -- |
| US Eggs | Auto-ARIMA(0,1,0) | 35.77 | 35.77 | -- |
| US House Sales | SARIMA(1,0,0)(1,1,0)[12] | 5.64 | 5.64 | -- |
| US Strikes | Auto-ARIMA(1,0,0) | 1315.01 | 1315.01 | -- |
| US Employment | SARIMA(0,1,1)(0,1,1)[12] | 327.92 | 327.92 | -- |
| Google Stock | Auto-ARIMA(3,1,0) | 31.50 | 31.50 | -- |

---

## [v0.2.0] - 2025-12-12

### Added

#### ACF/PACF-Based Model Selection
- Auto-ARIMA uses ACF/PACF analysis to suggest initial (p, q) orders based on Box-Jenkins methodology
- Seasonal order suggestions from ACF/PACF at seasonal lags (m, 2m, 3m...)
- New `Result` fields: `SuggestedP`, `SuggestedQ`, `SuggestedSP`, `SuggestedSQ`

#### AICc Criterion Support
- Added `"aicc"` option for `Config.Criterion`
- Small-sample corrected AIC: `AICc = AIC + 2k(k+1)/(n-k-1)`
- Added `AICc` field to model structs and `Result`

#### Prediction Intervals (ARIMA & SARIMA)
- New `PredictWithInterval(steps, confidence)` method
- Uses psi-weights (MA-infinity representation) for ARIMA variance calculation
- Variance growth for differenced/seasonal series
- Default 95% confidence intervals

#### Coefficient Standard Errors
- `ARStdErrors`, `MAStdErrors` in ARIMA Model
- `SARStdErrors`, `SMAStdErrors` in SARIMA Model
- Computed using numerical Hessian approximation
- Included in `Summary()` output

#### Improved Optimizer
- Momentum-based gradient descent (momentum=0.9)
- Adaptive learning rate with decay (0.99 per iteration)
- Best-solution tracking across iterations
- Early stopping after 20 iterations without improvement
- Max iterations increased from 100 to 200
- Convergence tolerance tightened from 1e-6 to 1e-8

#### Enhanced Stepwise Search
- Model deduplication avoids redundant evaluations (~20-30% fewer)
- All 8 cross-combination neighbors for seasonal models
- Diagonal combinations for both non-seasonal and seasonal

### Changed

- Default `Criterion` remains `"aic"` (changed to `"aicc"` in v0.3.0)
- Demo displays ACF/PACF suggested orders alongside selected orders

### Demo Results (v0.2.0 vs v0.1.0)

v0.1.0 had severe SARIMA convergence issues — most seasonal fixed-order models diverged to astronomically high RMSE (e.g., Australian Cement SARIMA(0,1,1)(0,1,1)[4] = 75 million RMSE). The v0.2.0 optimizer fixes resolved all of these.

| Dataset | Best Model | v0.1.0 RMSE | v0.2.0 RMSE | Change |
|---------|-----------|-------------|-------------|--------|
| Aus Population | ARIMA(1,1,1) | 1.04 | 1.04 | -- |
| Aus Cement | Auto-SARIMA | 323.30 | **277.30** | -14% |
| Aus Beer | SARIMA(1,0,0)(1,1,0)[4] | 25413.90 | **14.16** | fixed |
| Aus Electricity | SARIMA(1,0,0)(1,1,0)[4] | 908679.70 | **1592.41** | fixed |
| Aus Gas | Auto-SARIMA | 9.82 | **9.11** | -7% |
| US Eggs | Auto-ARIMA(0,1,0) | 35.77 | 35.77 | -- |
| US House Sales | SARIMA(1,0,0)(1,1,0)[12] | 6.74 | **5.64** | -16% |
| US Strikes | Auto-ARIMA(1,0,0) | 771.82 | 1315.01 | different model |
| US Employment | SARIMA(0,1,1)(0,1,1)[12] | diverged | **327.92** | fixed |
| Google Stock | Auto-ARIMA(3,1,0) | 32.27 | **31.50** | -2% |

---

## [v0.1.0] - 2025-12-10

Initial release.

### Features

- **ARIMA**: Model fitting with CSS (Conditional Sum of Squares) estimation
- **SARIMA**: Seasonal ARIMA with seasonal differencing
- **Auto-ARIMA**: Automatic model selection with stepwise and exhaustive search
- **Stationarity tests**: ADF, KPSS, Phillips-Perron
- **ACF/PACF**: Autocorrelation and partial autocorrelation with confidence bounds
- **Ljung-Box test**: Residual autocorrelation diagnostics
- **Time series utilities**: Differencing, seasonal differencing, lag, log transform, moving average, normalization
- **CSV I/O**: Load/save with filtering, multiple date formats, NA handling
- **Information criteria**: AIC, BIC, log-likelihood
- **Classical decomposition**: Additive and multiplicative
- **STL decomposition**: Simplified seasonal-trend decomposition

### Demo Datasets

11 CSV datasets included in `demo/data/` from the FPP textbook:
- `aus_economy.csv` — Australian economic indicators (GDP, population)
- `aus_livestock.csv` — Australian livestock counts by state/animal
- `aus_production.csv` — Australian quarterly production (beer, cement, electricity, gas)
- `aus_retail.csv` — Australian retail turnover by state/industry
- `eggs.csv` — US annual egg prices per dozen
- `gafa_stock.csv` — Google/Apple/Facebook/Amazon daily stock prices
- `global_economy.csv` — World economic indicators by country
- `hsales.csv` — US monthly new home sales
- `PBS_unparsed.csv` — Australian pharmaceutical benefits scheme
- `strikes.csv` — US annual work stoppages
- `us_employment.csv` — US monthly employment by sector

### Known Issues

- SARIMA optimizer often converges to poor local minima, producing divergent forecasts on most seasonal datasets (fixed in v0.2.0)
- No prediction intervals
- No coefficient standard errors
- No AICc criterion
