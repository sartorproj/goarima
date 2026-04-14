# Plan 3: ARIMAX / SARIMAX (Exogenous Variables)

**Branch:** `feat/arimax`
**Effort:** Medium (~350 lines)
**Dependencies:** Benefits from Kalman MLE (Plan 2) but can work with CSS

## Goal

Support external regressors (exogenous variables) in ARIMA and SARIMA models,
enabling ARIMAX(p,d,q) and SARIMAX(p,d,q)(P,D,Q)[m] models.

Model: `y_t = β'·X_t + η_t` where `φ(B)·(1-B)^d·η_t = θ(B)·ε_t`

## Design

### Approach: Regression with ARIMA Errors

This is the standard approach (R's `Arima(xreg=...)`, statsmodels' `SARIMAX`).
NOT transfer function models — just linear regression + ARIMA errors.

### New types:

```go
// In arima/arima.go
type ModelX struct {
    *Model
    ExogCoeffs    []float64 // β coefficients for exogenous variables
    ExogStdErrors []float64
    ExogNames     []string  // optional column names
    nExog         int
}

func NewX(p, d, q, nExog int) *ModelX
func (m *ModelX) Fit(series *timeseries.Series, exog [][]float64) error
func (m *ModelX) Predict(steps int, futureExog [][]float64) ([]float64, error)
func (m *ModelX) PredictWithInterval(steps int, futureExog [][]float64, confidence float64) (forecasts, lower, upper []float64, err error)
```

Similar `ModelX` in `sarima/` package.

### Fit algorithm (CSS-based):

1. **Initialize β via OLS:** regress y on X, get residuals
2. **Fit ARIMA to residuals:** get initial AR/MA coefficients
3. **Joint optimization:** minimize CSS over [β, φ, θ] simultaneously
   - For each parameter set, compute: `η_t = y_t - β'·X_t`
   - Then compute ARIMA residuals from η_t
   - SSE = Σ ε_t²

With Kalman MLE (Plan 2): β enters the observation equation directly:
```
y_t = Z'·x_t + β'·X_t
```
And the Kalman filter handles it naturally.

### Predict algorithm:

1. Compute exogenous contribution: `β'·X_{future}`
2. Forecast ARIMA errors: standard ARIMA prediction on η_t
3. Combine: `ŷ_{t+h} = β'·X_{t+h} + η̂_{t+h}`
4. Prediction intervals: from ARIMA error intervals (exog is deterministic)

### Validation:

- `len(exog)` must equal `series.Len()`
- All exog columns same length
- `len(futureExog)` must equal `steps`
- No NaN/Inf in exog
- Warn if exog columns are highly collinear (VIF > 10)

### Auto-ARIMAX in autoarima:

```go
// Extended AutoARIMA config
type Config struct {
    // ... existing fields ...
    Exogenous [][]float64 // exogenous regressors for fitting
}

// Extended Result
type Result struct {
    // ... existing fields ...
    ExogCoeffs []float64
}

// Predict with future exogenous values
func (r *Result) PredictWithExog(steps int, futureExog [][]float64) ([]float64, error)
```

### Files touched:
- NEW `arima/arimax.go` — ModelX type, Fit, Predict
- NEW `arima/arimax_test.go`
- NEW `sarima/sarimax.go` — ModelX type, Fit, Predict
- NEW `sarima/sarimax_test.go`
- `autoarima/autoarima.go` — add Exogenous field to Config/Result
- `timeseries/series.go` — optional: helper to align series with exog matrix

### Test cases:
- ARIMAX(1,0,0) with single exog: recover known β and φ₁
- ARIMAX(0,1,0) with trend exog: equivalent to ARIMA with drift
- SARIMAX(1,0,0)(1,0,0)[12] with temperature exog
- Prediction: futureExog shifts forecast by β'·X
- Error: mismatched exog/series lengths
- Error: NaN in exog
- Roundtrip: fit → predict → residuals should be white noise
