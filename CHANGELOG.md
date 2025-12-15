# Changelog

All notable changes to GoARIMA will be documented in this file.

## [v0.3.0] - 2025-12-15

### 🎯 Major Feature: Automatic Seasonality Detection

This release transforms goarima into a true "auto" library that automatically detects and handles seasonal patterns without user configuration.

### Added

#### Automatic Seasonal Period Detection
- **Auto-detects seasonality from ACF analysis** - no need to specify `SeasonalM`
- Checks common periods: 4, 6, 7, 12, 24, 52, 168, 365
- Configurable via `SeasonalPeriods` and `SeasonalityThreshold`
- New fields in `Result`:
  - `DetectedPeriod` - auto-detected seasonal period (0 if none)
  - `SeasonalityStrength` - ACF value at detected period
  - `DetectionMethod` - "acf" or "none"

#### Model Comparison (ARIMA vs SARIMA)
- **Automatically compares seasonal and non-seasonal models**
- Selects best model based on cross-validation RMSE
- New `Candidates []ModelCandidate` field shows all models evaluated
- Each candidate includes: Name, RMSE, MAPE, AICc, selected status

#### Cross-Validation Based Model Selection
- **Uses time series cross-validation** instead of just AIC/AICc
- Configurable via `ModelSelection`: "cv", "aicc", "aic", "bic"
- `TestRatio` controls train/test split (default: 0.2)
- `PreferSimpler` option prefers simpler models when scores are close

#### Multi-Level Prediction Intervals (R-style)
- New `PredictWithLevels(steps, levels)` method
- Returns 80% and 95% intervals by default (like R's `forecast()`)
- `ForecastResult` struct with `Lower` and `Upper` maps by confidence level
- Example: `fc.Lower[0.95]` for 95% lower bound

#### New Config Options
```go
type Config struct {
    // Seasonality detection (NEW)
    AutoSeasonal         bool      // Auto-detect seasonality (default: true)
    SeasonalPeriods      []int     // Periods to check
    SeasonalityThreshold float64   // ACF threshold (default: 0.4)
    MinSeasonalPeriod    int       // Minimum period (default: 4)
    MaxSeasonalPeriod    int       // Maximum period (default: 168)

    // Model selection (NEW)
    ModelSelection   string   // "cv", "aicc", "aic", "bic" (default: "cv")
    CVFolds          int      // CV folds (default: 5)
    TestRatio        float64  // Test set ratio (default: 0.2)
    CompareModels    bool     // Compare seasonal vs non-seasonal (default: true)
    PreferSimpler    bool     // Prefer simpler if close (default: true)
    SimplerThreshold float64  // Threshold for simpler preference (default: 0.05)
}
```

#### Helper Methods
- `Order() string` - Returns formatted order string (e.g., "SARIMA(1,1,1)(0,1,1)[24]")
- `DefaultForecastLevels = []float64{0.80, 0.95}` - Default confidence levels

### Changed

- **Default behavior**: Auto-seasonality detection is now ON by default
- **Better defaults**: Cross-validation is now the default model selection method
- **Simpler API**: `AutoARIMA(series, nil)` now "just works" for most cases

### Comparison with R's auto.arima

| Feature | R auto.arima | goarima v0.2.0 | goarima v0.3.0 |
|---------|-------------|----------------|----------------|
| Auto-detect period | ✅ (from ts freq) | ❌ | ✅ (from ACF) |
| Compare seasonal/non-seasonal | ✅ | ❌ | ✅ |
| Cross-validation | ❌ (uses AICc) | ❌ | ✅ |
| Multi-level intervals | ✅ (80%, 95%) | ❌ | ✅ |
| Return all candidates | ❌ | ❌ | ✅ |

### Example Usage

```go
// Fully automatic - detects seasonality and selects best model
result, _ := autoarima.AutoARIMA(series, nil)

// Result includes:
// - result.DetectedPeriod = 24 (auto-detected)
// - result.IsSeasonal = true
// - result.Order() = "SARIMA(3,0,1)(0,1,0)[24]"
// - result.RMSE, result.MAPE = cross-validated metrics
// - result.Candidates = all models compared

// Multi-level prediction intervals (R-style)
fc, _ := result.PredictWithLevels(24, nil)  // 80% and 95%
fc.Lower[0.95]  // 95% lower bound
fc.Upper[0.80]  // 80% upper bound
```

### Validation Results

**sample-app-4 Memory Metrics (168 hourly data points)**

| Model | CV RMSE | CV MAPE | Auto-Selected |
|-------|---------|---------|---------------|
| ARIMA(4,0,0) | 0.000019 | 15.87% | ❌ |
| **SARIMA(3,0,1)(0,1,0)[24]** | **0.000007** | **7.07%** | ✅ |

**Result**: SARIMA automatically selected, **2.5x more accurate** than ARIMA.

---

## [v0.2.0] - 2025-12-12

### Added

#### ACF/PACF-Based Model Selection
- **Data-driven starting points**: Auto-ARIMA now uses ACF/PACF analysis to suggest initial (p, q) orders based on the Box-Jenkins methodology
- **Seasonal order suggestions**: For SARIMA models, ACF/PACF analysis at seasonal lags (m, 2m, 3m...) suggests seasonal AR and MA orders
- New functions in `autoarima`:
  - `suggestOrdersFromACF()` - Analyzes ACF/PACF patterns to suggest non-seasonal orders
  - `suggestSeasonalOrdersFromACF()` - Suggests seasonal orders from correlations at seasonal lags
  - `countSignificantLags()` - Helper to detect "tailing off" vs "cutting off" patterns
- **Result struct enhancements**: Added `SuggestedP`, `SuggestedQ`, `SuggestedSP`, `SuggestedSQ` fields to track ACF/PACF recommendations

#### AICc Criterion Support
- Added `aicc` option for the `Criterion` config field
- Small-sample corrected AIC (AICc = AIC + 2k(k+1)/(n-k-1)) for better model selection with limited data
- Added `AICc` field to `Result` struct

#### Enhanced Stepwise Search
- **Model deduplication**: Tracks evaluated models to avoid redundant computations
- **Complete cross-combinations**: Added all 8 cross-combination neighbors for seasonal models:
  - (p±1, sp±1) and (q±1, sq±1) combinations
- **Diagonal combinations**: Both non-seasonal (p±1, q±1) and seasonal (sp±1, sq±1) diagonal moves

#### Prediction Intervals (ARIMA & SARIMA)
- New `PredictWithInterval(steps, confidence)` method returns forecasts with lower/upper bounds
- Uses psi-weights (MA∞ representation) for accurate variance calculation in ARIMA
- Accounts for variance growth in differenced/seasonal series
- Default 95% confidence intervals

#### Coefficient Standard Errors (ARIMA & SARIMA)
- New `ARStdErrors`, `MAStdErrors` fields in Model struct
- New `SARStdErrors`, `SMAStdErrors` fields in SARIMA Model
- Computed using numerical Hessian approximation
- Included in `Summary()` output for statistical inference

#### Improved Optimizer (Major Performance Improvement)
- **Momentum-based gradient descent**: Added momentum (0.9) for faster convergence
- **Adaptive learning rate**: Learning rate decay (0.99 per iteration) for stability
- **Best solution tracking**: Keeps track of best parameters found during optimization
- **Early stopping**: Stops if no improvement for 20 iterations
- **More iterations**: Increased max iterations from 100 to 200 for better convergence
- **Tighter convergence tolerance**: Changed from 1e-6 to 1e-8 for better parameter estimates

### Changed

- **More efficient search**: Model deduplication reduces redundant evaluations by ~20-30%
  - Non-seasonal example: 14 models (down from 17)
  - Seasonal example: 24 models (down from 33)
- **Better exploration**: ACF/PACF suggested models added to starting set improves convergence to optimal

### Demo Updates

- Demo now displays ACF/PACF suggested orders alongside selected orders
- JSON output includes `suggested_order` and `models_evaluated` fields
- Visualization shows suggested orders and evaluation count in metrics table

---

## Performance Comparison: v0.1.0 vs v0.2.0

The optimizer improvements in v0.2.0 dramatically improve SARIMA model accuracy. The v0.1.0 basic gradient descent often got stuck in poor local minima, especially for seasonal models.

### Benchmark Results (v0.2.0)

| Dataset | Model Type | Best Model | RMSE | Models Evaluated |
|---------|------------|------------|------|------------------|
| Australian Population | ARIMA | ARIMA(1,1,1) | 1.04 | 11 |
| Australian Cement | SARIMA | SARIMA(1,0,0)(1,1,0)[4] | 190.96 | 25 |
| Australian Beer | SARIMA | SARIMA(1,0,0)(1,1,0)[4] | 14.16 | 25 |
| Australian Electricity | SARIMA | SARIMA(1,0,0)(1,1,0)[4] | 1,592.41 | 26 |
| Australian Gas | SARIMA | SARIMA(0,1,1)(0,1,1)[4] | 9.11 | 41 |
| US Eggs | ARIMA | ARIMA(0,1,0) | 35.77 | 11 |
| US House Sales | SARIMA | SARIMA(1,0,0)(1,1,0)[12] | 5.64 | 47 |
| US Strikes | ARIMA | Auto-ARIMA(1,0,0) | 1,315.01 | 13 |
| US Employment | SARIMA | SARIMA(0,1,1)(0,1,1)[12] | 327.92 | 26 |
| Google Stock | ARIMA | Auto-ARIMA(3,1,0) | 31.50 | 13 |

### Key Performance Improvements

| Improvement Area | v0.1.0 | v0.2.0 | Impact |
|------------------|--------|--------|--------|
| Optimizer | Fixed learning rate (0.01) | Momentum + decay | **Converges to better minima** |
| Max iterations | 100 | 200 | Better parameter estimates |
| Tolerance | 1e-6 | 1e-8 | Finer convergence |
| SARIMA convergence | Often stuck in local minima | Consistently finds good solutions | **Major accuracy improvement** |
| Solution tracking | None | Best-solution memory | Avoids regression during optimization |

### Feature Comparison

| Feature | v0.1.0 | v0.2.0 |
|---------|--------|--------|
| Starting models | Fixed set | Fixed + ACF/PACF suggested |
| Criterion options | `aic`, `bic` | `aic`, `aicc`, `bic` |
| Model deduplication | No | Yes |
| Cross-combinations | 2 partial | 8 complete |
| Prediction intervals | No | Yes |
| Coefficient std errors | No | Yes |
| Suggested orders in output | No | Yes |
| Typical models evaluated (seasonal) | ~33 | ~24 |

### Example Output Comparison

**v0.1.0:**
```
Auto-SARIMA(0,1,2)(1,1,1)[4]: RMSE=9.8185 (37 models)
```

**v0.2.0:**
```
Auto-SARIMA(0,1,0)(1,1,1)[4]: RMSE=13.40 (41 models, ACF/PACF suggested: (1,1,1)(2,1,1)[4])
```

The new version provides:
1. Better optimizer convergence for accurate parameter estimates
2. Prediction intervals for uncertainty quantification
3. Coefficient standard errors for statistical inference
4. Transparency into model selection via ACF/PACF suggestions

---

## [v0.1.0] - Initial Release

### Features

- ARIMA model fitting with CSS (Conditional Sum of Squares) estimation
- SARIMA (Seasonal ARIMA) support with seasonal differencing
- Auto-ARIMA with stepwise and exhaustive search
- Stationarity tests: ADF, KPSS, Phillips-Perron
- ACF/PACF computation with confidence bounds
- Ljung-Box test for residual diagnostics
- Time series utilities: differencing, seasonal differencing, CSV loading
- Information criteria: AIC, BIC, Log-Likelihood
