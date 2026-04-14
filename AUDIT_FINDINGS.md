# GoARIMA Audit Findings & Fix Plan

## High Severity

### H1: MA Gradient Is Biased in CSS Optimizer
- **Files**: `arima/arima.go:205-215`, `sarima/sarima.go:257-283`
- **Problem**: The gradient for MA coefficients treats past residuals as constants (`∂residuals[t-i-1]/∂θ = 0`), but residuals depend on θ. This makes the gradient biased, not just approximate. All models with q>0 or SQ>0 are affected.
- **Expected behavior**: Proper CSS gradient must propagate through the residual chain (recursive derivatives), or use a derivative-free optimizer (Nelder-Mead, L-BFGS with numerical gradient).
- **Fix**: Replace the hand-rolled gradient descent with either:
  - (a) Nelder-Mead simplex (no gradients needed, standard for ARIMA in R), or
  - (b) L-BFGS with finite-difference gradients, or
  - (c) At minimum, compute the full recursive MA gradient correctly.

### H2: SARIMA Missing Multiplicative Cross-Terms
- **File**: `sarima/sarima.go:201-229`
- **Problem**: A true SARIMA(p,d,q)(P,D,Q)[m] has multiplicative AR/MA polynomials: `φ(B)Φ(Bᵐ)` and `θ(B)Θ(Bᵐ)`. The current code sums non-seasonal and seasonal components independently — it never computes cross-terms like `φ₁·Φ₁` at lag `m+1`. The model being fit is mathematically different from (and simpler than) a real SARIMA.
- **Expected behavior**: Expand the product polynomials `φ(B)·Φ(Bᵐ)` and `θ(B)·Θ(Bᵐ)` to get the full set of AR/MA lags and coefficients, then fit those.
- **Fix**: Implement polynomial multiplication for AR and MA sides. Use the expanded coefficient vector during fitting and prediction.

### H3: Prediction Intervals Double-Count Variance Growth (ARIMA)
- **File**: `arima/arima.go:461-490`
- **Problem**: Lines 462-469 compute cumulative psi-weight variance `σ²(1 + Σψᵢ²)` which already accounts for the integrated (d>0) process. Then lines 485-488 multiply SE by an additional `sqrt(h+1)` when `d>0`. This double-counts variance growth, producing intervals that are far too wide.
- **Fix**: Remove the `sqrt(h+1)` multiplier at line 486-488. The psi-weights already encode the correct variance growth for integrated series.

### H4: SARIMA Prediction Intervals Use Crude Heuristic Instead of Psi-Weights
- **File**: `sarima/sarima.go:487-504`
- **Problem**: Unlike the ARIMA package which computes psi-weights (MA-infinity representation), the SARIMA prediction intervals use `sqrt(h+1)` and `sqrt(seasonalCycles)` heuristics. This produces inaccurate interval widths, especially for models with MA components or mixed orders.
- **Fix**: Implement `computePsiWeights()` for SARIMA (using the full expanded polynomial from H2), then use the same `σ²(1 + Σψᵢ²)` formula as ARIMA.

---

## Medium Severity

### M1: d=2 Integration Produces Wrong Forecasts
- **File**: `arima/arima.go:543-564`
- **Problem**: For d=2, the second integration pass uses `original[len(original)-1-1]` (the second-to-last original value) as its starting point. It should use the last value of the *once-differenced* series instead.
- **Fix**: Compute intermediate differenced series and use their last values as starting points for each integration level, rather than indexing backwards into the original.

### M2: No Proper Stationarity/Invertibility Enforcement
- **Files**: `arima/arima.go:222-223`, `sarima/sarima.go:289-304`
- **Problem**: Coefficients are individually clamped to `[-0.99, 0.99]`, but this does not ensure the AR characteristic polynomial has all roots outside the unit circle. For example, AR(2) with `(0.9, 0.9)` passes the clamp but represents a non-stationary process.
- **Fix**: After optimization, check the roots of the AR polynomial `1 - φ₁z - φ₂z² - ...`. If any root has modulus ≤ 1, project coefficients onto the stationary region (or use the Jones reparameterization to ensure stationarity by construction).

### M3: MacKinnon p-Value Is a Crude 6-Bin Lookup
- **File**: `stats/stationarity.go:455-477`
- **Problem**: The ADF p-value uses a coarse step function (6 bins) that ignores sample size entirely (the `nobs` parameter is unused/discarded). It cannot distinguish p=0.001 from p=0.009, and has wide gaps. This affects automatic differencing decisions in `autoarima`.
- **Fix**: Implement MacKinnon (1994) response surface regression: `p = Φ(β∞ + β₁/T + β₂/T²)` with the published coefficient tables. At minimum, use finer interpolation with sample-size adjustment.

### M4: KPSS p-Value Can Exceed 1.0
- **File**: `stats/stationarity.go:497-508`
- **Problem**: The linear extrapolation `0.10 + (0.347 - stat) * 0.5` for small test statistics can produce p-values greater than 1.0, which is nonsensical.
- **Fix**: Clamp the output to `[0, 1]` at minimum. Better: use proper interpolation from the KPSS critical value tables (Kwiatkowski et al. 1992).

### M5: Cross-Validation Is Actually a Single Holdout Split
- **File**: `autoarima/autoarima.go:677-716`
- **Problem**: Despite `CVFolds` being configurable (default 5), `evaluateWithCV()` performs a single train/test split (last 20%). This is holdout evaluation, not cross-validation, and is sensitive to where the split falls.
- **Fix**: Implement time series cross-validation with expanding window (rolling origin):
  ```
  fold 1: train=[1..60], test=[61..70]
  fold 2: train=[1..70], test=[71..80]
  fold 3: train=[1..80], test=[81..90]
  ...
  ```
  Average RMSE/MAE/MAPE across folds.

### M6: No Input Validation for NaN/Inf Values
- **Files**: All `Fit()` methods, `timeseries/series.go`
- **Problem**: If the input series contains NaN or Inf values, the fitting silently produces garbage results. No validation or warning is provided.
- **Fix**: Add validation in `Fit()` methods (and optionally in `timeseries.New()`) to reject or flag series containing NaN/Inf. Optionally provide an interpolation utility for handling missing values.

### M7: SARIMA Integration Breaks for sd>1
- **File**: `sarima/sarima.go:554-569`
- **Problem**: The seasonal integration loop modifies `result` in-place across iterations. For `sd>1`, the second pass re-uses already-integrated values without recomputing the intermediate series, producing incorrect results.
- **Fix**: For each seasonal integration pass, work on a fresh copy or recompute the reference series correctly.

---

## Low Severity

### L1: Custom Gamma Function Instead of stdlib
- **File**: `stats/ljungbox.go:119-145`
- **Problem**: Uses a hand-rolled Lanczos-approximation `gamma()` with recursion via the reflection formula. For edge-case inputs this is less robust than `math.Gamma()` from Go's standard library, which handles special cases (negative integers, large values) properly.
- **Fix**: Replace `gamma(z)` calls with `math.Gamma(z)`. Similarly, `lowerIncompleteGamma` could use or cross-check against `math.Lgamma`.

### L2: `countModelsEvaluated` Undercounts
- **File**: `autoarima/autoarima.go:887-890`
- **Problem**: Returns `len(candidates)` (2 at most: one ARIMA, one SARIMA), but many models are evaluated internally during stepwise search. The field `ModelsEvaluated` is misleading.
- **Fix**: Pass a counter through `fitBestARIMA`/`fitBestSARIMA` and return the actual count.

### L3: `STL` Is a Simplified Approximation
- **File**: `stats/decomposition.go:195-326`
- **Problem**: The STL implementation uses weighted moving average instead of actual LOESS (locally weighted regression). It is labeled "Seasonal and Trend using Loess" but does not perform LOESS. Results will differ from proper STL implementations.
- **Fix**: Either implement true LOESS smoothing or rename the function to clarify it's an approximation (e.g., `STLApprox` or document the limitation).

### L4: `median()` Uses Insertion Sort
- **File**: `stats/decomposition.go:329-353`
- **Problem**: O(n^2) insertion sort. Not a correctness issue, but could be slow for large series in the robust-fitting loop of STL.
- **Fix**: Use `sort.Float64s()` (stdlib quicksort) or `slices.Sort()`.

### L5: Duplicated `normalQuantile` Function
- **Files**: `arima/arima.go:522-540`, `sarima/sarima.go:510-523`
- **Problem**: Identical function defined in both packages. Should be in a shared location.
- **Fix**: Move to `stats` package and export as `stats.NormalQuantile()`.

### L6: `DiffN` Computes Lag-n Difference, Not n-th Order Difference
- **File**: `timeseries/series.go:128-148`
- **Problem**: `DiffN(2)` computes `y[i] - y[i-2]` (lag-2 difference), not the second-order difference `(y[i] - y[i-1]) - (y[i-1] - y[i-2])`. The docstring says "n-th order difference" but the implementation does lag-n. This is not used internally (the code calls `Diff()` in a loop), but it's a public API that would confuse users.
- **Fix**: Either fix the implementation to apply `Diff()` n times, or rename to `DiffLag(n)` and document correctly.

---

## Missing Features (Enhancements)

### E1: Proper MLE Optimizer
Replace CSS gradient descent with exact or conditional MLE using Nelder-Mead or L-BFGS. This is the single highest-impact improvement and would also resolve H1.

### E2: ARIMAX / SARIMAX (Exogenous Variables)
Support for external regressors is critical for real-world forecasting.

### E3: Box-Cox Transformation
Automatic variance-stabilizing transformation (R's `auto.arima` `lambda` parameter).

### E4: Model Serialization (Save/Load)
JSON or gob encoding for fitted models, enabling persist-and-reload workflows.

### E5: Concurrent Model Fitting in Auto-ARIMA
Use goroutines to fit candidate models in parallel during auto-selection.

### E6: Spectral Analysis for Seasonality Detection
The `DetectionMethod` field has "spectral" as a value but is never implemented. Periodogram-based detection would be more robust than ACF alone.

### E7: Residual Diagnostics Bundle
A single method returning residual ACF, PACF, Ljung-Box at multiple lags, Durbin-Watson, normality test — bundled for easy inspection.

---

## Suggested Fix Order

1. **H3** (prediction interval double-count) — one-line fix, high impact
2. **H2 + H4** (SARIMA cross-terms + psi-weights) — coupled, fixes the core SARIMA model
3. **H1 / E1** (optimizer) — most complex but highest overall impact
4. **M1** (d=2 integration) — straightforward fix
5. **M4** (KPSS p-value clamp) — one-line fix
6. **M2** (stationarity enforcement) — moderate complexity
7. **M3** (MacKinnon p-values) — moderate complexity
8. **M5** (true CV) — moderate complexity
9. **M6** (NaN validation) — straightforward
10. **L1-L6** (cleanup) — low effort, low risk
11. **E2-E7** (enhancements) — larger feature work
