# GoARIMA Audit Plan

Comprehensive findings from a 38-agent code review, grouped by severity.
Each item includes the file, line numbers, and a description of the fix needed.

---

## P0 - CRITICAL BUGS (Produce Wrong Results)

### P0-1: ADF Test Uses Wrong Lagged Variable Index
- **File:** `stats/stationarity.go:58`
- **Problem:** Uses `series.Values[t]` instead of `series.Values[t-1]`. The ADF regression equation is `delta_y_t = alpha + gamma * y_{t-1} + ...` but the code uses `y_t` (contemporaneous level) instead of `y_{t-1}` (lagged level).
- **Impact:** ADF test statistics and p-values are meaningless. All stationarity decisions based on ADF are unreliable.
- **Fix:** Change `x[i][1] = series.Values[t]` to `x[i][1] = series.Values[t-1]`.

### P0-2: SARIMA Uses Additive Instead of Multiplicative Seasonal Formula
- **File:** `sarima/sarima.go:201-228`
- **Problem:** Seasonal AR and MA terms are simply added to non-seasonal terms. The correct SARIMA(p,d,q)(P,D,Q)m model requires multiplicative interaction between seasonal and non-seasonal polynomials: `phi(B) * Phi(B^m) * nabla^d * nabla_m^D * y_t = theta(B) * Theta(B^m) * e_t`.
- **Impact:** SARIMA model coefficients and forecasts are mathematically incorrect for any seasonal model.
- **Fix:** Implement proper multiplicative polynomial expansion for the combined AR and MA operators.

### P0-3: Psi Weights Computation Is Broken
- **File:** `arima/arima.go:497-519`
- **Problem:** The MA(infinity) representation has multiple errors: (1) When `j < q`, it directly assigns `psi[j] = MACoeffs[j]` then adds AR contributions, double-counting. (2) `psi[0]` is never set to 1 (required by definition). (3) The recursive formula `psi_j = theta_j + phi_1*psi_{j-1} + ... + phi_p*psi_{j-p}` is not correctly implemented.
- **Impact:** Prediction intervals are systematically wrong (too narrow or too wide).
- **Fix:** Rewrite to: `psi[0] = 1.0; for j := 1..maxLag: psi[j] = theta_j + sum(phi_i * psi[j-i-1])`.

### P0-4: Integration Logic Flawed for d > 1
- **File:** `arima/arima.go:553-560`
- **Problem:** For multi-step integration (d > 1), uses `original[len(original)-1-i]` to get the base value for each integration step. The second integration should use the last value of the previously integrated series, not the original data.
- **Impact:** Forecasts for ARIMA models with d=2 or higher are incorrect.
- **Fix:** Accumulate integrations sequentially, using the last value of the previously integrated series as the base for the next.

### P0-5: Phillips-Perron Correction Formula Is Incorrect
- **File:** `stats/stationarity.go:304-309`
- **Problem:** Contains an extra `math.Sqrt(float64(nObs))` factor not present in the standard Phillips-Perron (1988) formula. The PP correction should be: `t_pp = sqrt(gamma0/lambda2) * t_stat - 0.5 * (lambda2 - gamma0) / sqrt(sumXDev2) / sqrt(lambda2)`.
- **Impact:** PP test statistics are wrong; stationarity decisions using PP are unreliable.
- **Fix:** Remove the spurious `math.Sqrt(float64(nObs))` and match the standard formula.

### P0-6: Prediction Interval Variance Is Incorrect (ARIMA)
- **File:** `arima/arima.go:460-469, 485-487`
- **Problem:** (1) The cumulative psi-weight variance sum skips `psi[0]^2 = 1` on the first step. (2) For integrated series (d > 0), applies an ad-hoc `sqrt(h+1)` multiplier instead of properly propagating variance through the integration. The standard formula is `Var(e_{n+h}) = sigma^2 * sum(psi_j^2, j=0..h-1)`.
- **Impact:** Confidence intervals are incorrect, especially at longer forecast horizons.
- **Fix:** Start cumulative sum with `psi[0]^2 = 1`, and propagate variance through integration steps analytically.

### P0-7: Prediction Intervals Completely Wrong for SARIMA
- **File:** `sarima/sarima.go:487-504`
- **Problem:** No psi weights are computed at all. Uses arbitrary heuristics: `sqrt(h+1)` for non-seasonal growth and `sqrt(h/period + 1)` for seasonal growth. These have no mathematical basis.
- **Impact:** All SARIMA prediction intervals are meaningless.
- **Fix:** Compute psi weights for the full SARIMA model (accounting for multiplicative seasonal/non-seasonal polynomial interaction) and use the standard cumulative variance formula.

### P0-8: Multiplicative Decomposition Uses Arithmetic Mean Instead of Geometric Mean
- **File:** `stats/decomposition.go:74-82`
- **Problem:** Seasonal factors are normalized by dividing by their arithmetic mean. For multiplicative decomposition (Y = T * S * R), seasonal factors should have a geometric mean of 1, not an arithmetic mean of 1.
- **Impact:** Multiplicative seasonal decomposition produces biased seasonal factors and residuals.
- **Fix:** Compute geometric mean: `geoMean = (product of factors)^(1/period)` and divide each factor by it.

---

## P1 - HIGH SEVERITY (Significant Correctness or Safety Issues)

### P1-1: No Polynomial Root Checking for Stationarity/Invertibility
- **Files:** `arima/arima.go:222,229`, `sarima/sarima.go:289,294,299,304`
- **Problem:** AR and MA coefficients are clamped to [-0.99, 0.99] individually. Stationarity requires all roots of the AR characteristic polynomial to lie outside the unit circle. Example: AR(2) with phi_1=0.6, phi_2=0.6 passes individual bounds but is non-stationary (sum > 1).
- **Fix:** Implement companion matrix eigenvalue checking or Jones (1980) reparametrization during optimization.

### P1-2: SARIMA Does Not Compute Standard Errors
- **File:** `sarima/sarima.go` (missing entirely)
- **Problem:** The `estimateStdErrors()` method exists in ARIMA but is completely absent from SARIMA. Fields `SARStdErrors`, `SMAStdErrors`, `ARStdErrors`, `MAStdErrors` are declared but never populated.
- **Fix:** Port the numerical Hessian approach from ARIMA, extending it to cover seasonal coefficients.

### P1-3: ARIMA Standard Error Formula Has Unexplained Factor of 2
- **File:** `arima/arima.go:344,362`
- **Problem:** Uses `math.Sqrt(2 * m.Variance / hessianDiag)`. The standard formula for CSS-based standard errors is `SE = sqrt(sigma^2 * (H^{-1})_ii)` where H is the Hessian of SSE. The factor 2 is unexplained.
- **Fix:** Verify against reference implementation and remove or justify the factor.

### P1-4: MacKinnon P-Values Ignore Sample Size
- **File:** `stats/stationarity.go:452-477`
- **Problem:** The function accepts `nobs` and `regression` parameters but ignores both. Uses hard-coded asymptotic critical values in a switch statement. MacKinnon (1994, 2010) provides response surface regressions that depend on sample size and regression type.
- **Impact:** ADF and PP p-values are only approximately correct for very large samples.
- **Fix:** Implement MacKinnon response surface coefficients for finite-sample adjustment.

### P1-5: Negative Model Orders Cause Panic
- **Files:** `arima/arima.go:42-47`, `sarima/sarima.go:51-62`
- **Problem:** `New(p, d, q)` calls `make([]float64, p)` which panics if p < 0. No input validation.
- **Fix:** Add validation: `if p < 0 || d < 0 || q < 0 { return nil, errors.New("orders must be non-negative") }`. Change return type to `(*Model, error)`.

### P1-6: Silent Error Swallowing in Auto-ARIMA Model Fitting
- **File:** `autoarima/autoarima.go:466-467, 501-502, 599-600, 639-640`
- **Problem:** When a model fails to fit during the search, the error is silently discarded with `if err := model.Fit(series); err != nil { continue }`. Users cannot diagnose why models fail.
- **Fix:** Collect errors and expose them in the Result struct, or add a Trace/Verbose mode that logs fitting failures.

### P1-7: No NaN/Inf Input Validation Anywhere
- **Files:** Library-wide (arima, sarima, autoarima, stats, timeseries)
- **Problem:** No function checks for NaN or Inf in input data. NaN values silently propagate through all calculations producing garbage results.
- **Fix:** Add input validation at entry points (Fit, AutoARIMA, statistical test functions). Return error if NaN/Inf detected.

### P1-8: ACF Returns nil for Constant Series
- **File:** `stats/acf.go:28-29`
- **Problem:** When variance is 0 (constant series), returns `nil` instead of `[1.0, 0, 0, ...]`. Downstream consumers (Yule-Walker, auto-arima period detection) don't handle nil ACF gracefully.
- **Fix:** Return `[1.0, 0, 0, ...]` for constant series, or return `([]float64, error)`.

### P1-9: SARIMA Seasonal Integration Logic Is Flawed
- **File:** `sarima/sarima.go:528-589`
- **Problem:** The integration logic for reversing seasonal + non-seasonal differencing has index errors. Line 562 uses `nonSeasonalDiff[idx]` but `nonSeasonalDiff` is rebuilt incorrectly. For d > 1, uses `original[n-1]` repeatedly instead of the previously integrated series' last value.
- **Fix:** Implement integration in the correct order: undo seasonal differences first (using non-seasonally-differenced series), then undo regular differences.

### P1-10: Information Criteria Don't Count Variance Parameter
- **Files:** `arima/arima.go:370`, `sarima/sarima.go:373`
- **Problem:** Parameter count k = P + Q + 1 (intercept only). Standard ARIMA IC implementations count k = P + Q + 2 to include the variance parameter sigma^2.
- **Fix:** Add 1 to k for the variance parameter, or document the convention used.

### P1-11: PredictWithInterval() Is Never Tested
- **Files:** `arima/arima_test.go`, `sarima/sarima_test.go`
- **Problem:** The primary confidence interval method is not tested in either package. No assertions on interval width, coverage probability, or growth with forecast horizon.
- **Fix:** Add tests that verify intervals widen with horizon, contain known values, and match reference implementations.

### P1-12: No Reference-Validated Tests
- **Files:** All test files
- **Problem:** No test compares outputs against R's forecast package or Python's statsmodels. Test tolerances are extremely loose (30% for AR coefficients at `arima_test.go:51`, 50% for intercept at line 258).
- **Fix:** Create test cases using known datasets (e.g., Box-Jenkins airline data) with expected values from R/statsmodels.

### P1-13: Stats Functions Return nil Without Error Context
- **Files:** `stats/acf.go`, `stats/decomposition.go`, `stats/stationarity.go`
- **Problem:** Functions like `ACF()`, `PACF()`, `Decompose()`, `ADF()` return `nil` for invalid inputs instead of `(result, error)`. Callers cannot distinguish between "no result because invalid input" and programming bugs.
- **Fix:** Change signatures to return `(result, error)` or at minimum document when nil is returned.

### P1-14: ~250+ Lines Duplicated Between arima and sarima
- **Files:** `arima/arima.go`, `sarima/sarima.go`
- **Problem:** `normalQuantile()` (19 lines identical), `Residuals()` and `FittedValues()` (18 lines identical), `calculateIC()` (~25 lines near-identical), `Predict()` wrapper (4 lines identical), optimization loop structure (~170 lines structurally identical).
- **Fix:** Extract shared utilities into an `internal/` package: normalQuantile, calculateIC, clamp, Residuals/FittedValues base implementations.

### P1-15: Excessive Memory Allocations in Optimization Hot Loops
- **Files:** `arima/arima.go:166,202-203`, `sarima/sarima.go:198,252-255`
- **Problem:** `residuals`, `arGrad`, `maGrad` slices are allocated inside the optimization loop on every iteration (up to 200 iterations). For n=1000 points, this means 200+ allocations of 8KB+ each.
- **Fix:** Pre-allocate all working arrays before the loop and reuse them via zeroing.

---

## P2 - MEDIUM SEVERITY (Incomplete Features, Design Issues)

### P2-1: No Drift Term for Differenced Models
- **Files:** `arima/arima.go`, `sarima/sarima.go`, `autoarima/autoarima.go`
- **Problem:** The library always includes an intercept (mean of differenced series) but has no separate drift term. R's `include.drift=TRUE` adds a linear trend to differenced series. Auto-ARIMA doesn't test with/without constant.
- **Fix:** Add `IncludeMean` and `IncludeDrift` options. Auto-ARIMA should test both configurations.

### P2-2: CSS Only - No Kalman Filter / Exact MLE
- **Files:** `arima/arima.go:133-286`, `sarima/sarima.go:162-368`
- **Problem:** Uses Conditional Sum of Squares with gradient descent. No exact MLE via Kalman filter. MA coefficient SEs are underestimated 15-25%. Loses first max(p,q) observations.
- **Fix:** Implement state-space representation and Kalman filter for exact MLE. Consider as a long-term improvement.

### P2-3: ADF Test Only Supports Constant-Only Variant
- **File:** `stats/stationarity.go:73-82`
- **Problem:** Only the "constant" regression variant is implemented. Missing "no constant" and "constant + trend" variants. Critical values are hard-coded for constant-only.
- **Fix:** Add `regression` parameter supporting "nc", "c", "ct" with appropriate critical values for each.

### P2-4: Chi-Squared CDF Precision Loss
- **File:** `stats/ljungbox.go:158`
- **Problem:** `gamma(a) - gammaIncCF(a, x)` loses precision through catastrophic cancellation when gamma(a) is large. Test tolerances at `stats_test.go:364-385` are suspiciously wide (0.93-0.98 instead of near 0.95).
- **Fix:** Compute regularized form directly: `1 - (gammaIncCF(a, x) / gamma(a))`. Tighten test tolerances.

### P2-5: determineSeasonalDifferencing Ignores maxSD Parameter
- **File:** `autoarima/autoarima.go:393`
- **Problem:** Second parameter `maxSD` is silently discarded with `_`. Always returns 0 or 1 regardless of configuration. Doesn't use the existing `stats.NSDiffs()` function which has proper seasonal strength detection.
- **Fix:** Use `stats.NSDiffs()` or respect the maxSD parameter.

### P2-6: autoarima Differencing Test Logic Is Inconsistent
- **File:** `autoarima/autoarima.go:366-376`
- **Problem:** Mixes KPSS and ADF results together without clear logic. Returns stationary if both agree OR if "KPSS stationary AND pValue > 0.1". This inconsistency can give false positives.
- **Fix:** Use KPSS alone as default (matching R's auto.arima) or clearly document the combined logic.

### P2-7: No Model Serialization (Save/Load)
- **Files:** `arima/arima.go`, `sarima/sarima.go`
- **Problem:** No way to serialize a fitted model to disk and reload it. The `fitted` bool and `data`/`diffData`/`residuals` fields are unexported, preventing standard JSON marshaling.
- **Fix:** Implement `MarshalJSON`/`UnmarshalJSON` methods that include unexported state, or add explicit `Save()`/`Load()` methods.

### P2-8: No Exogenous Variable Support (ARIMAX/SARIMAX)
- **Files:** Library-wide
- **Problem:** Only univariate time series supported. No way to include external regressors.
- **Fix:** Create ARIMAX model that extends ARIMA with exogenous regression coefficients. Modify Fit/Predict signatures to accept exogenous data.

### P2-9: No Box-Cox Transformation
- **File:** `timeseries/series.go`
- **Problem:** Only manual `Log()` available. No Box-Cox with automatic lambda selection. No inverse transformation for forecasts. Naive back-transformation of prediction intervals is mathematically incorrect.
- **Fix:** Implement `BoxCox(lambda)`, `InverseBoxCox(lambda)`, and automatic lambda selection via profile likelihood.

### P2-10: NaN Handling Broken in TimeSeries Aggregations
- **File:** `timeseries/series.go:48-71`
- **Problem:** `Mean()` includes NaN in sum without checking. `Variance()` doesn't skip NaN. `Min()`/`Max()` fail if first value is NaN. `MovingAverage()` doesn't handle NaN windows.
- **Fix:** Add NaN-aware versions of all aggregation methods.

### P2-11: CSV Parsing Can Desynchronize Timestamps and Values
- **File:** `timeseries/csv.go:136-142, 177-184`
- **Problem:** When a value fails to parse as float, it's silently skipped, but the timestamp from that row may still be added. Final check at lines 177-181 creates series with auto-generated timestamps if counts don't match.
- **Fix:** Skip both timestamp and value when either fails to parse, or track row-level success.

### P2-12: Missing Normality Tests for Residuals
- **File:** `stats/` (missing)
- **Problem:** No Jarque-Bera, Shapiro-Wilk, or Anderson-Darling tests. Cannot validate the normality assumption underlying prediction intervals.
- **Fix:** Implement at least Jarque-Bera (simplest: based on skewness and kurtosis of residuals).

### P2-13: No Inverse Differencing Functions
- **File:** `stats/differencing.go`, `timeseries/series.go`
- **Problem:** The stats package has no undifferencing/inverse differencing capability. This is handled ad-hoc in the integrate() methods of arima/sarima, leading to bugs (see P0-4, P1-9).
- **Fix:** Add `InverseDiff()`, `InverseSeasonalDiff()` to the timeseries package.

### P2-14: Cross-Validation Is Single Train/Test Split
- **File:** `autoarima/autoarima.go:676-716`
- **Problem:** Despite having a `CVFolds` config parameter (default 5), the actual implementation performs a single train/test split. No rolling origin or expanding window CV.
- **Fix:** Implement proper time series CV with multiple folds.

### P2-15: No Benchmark Tests
- **Files:** No `*_bench_test.go` files exist
- **Problem:** Cannot measure performance regressions. Critical for optimization-heavy code.
- **Fix:** Add benchmarks for `Fit()`, `Predict()`, `AutoARIMA()`, `ACF()`, `ADF()`.

### P2-16: Seasonal Strength NaN Bias in Decomposition
- **File:** `stats/differencing.go:101-106`
- **Problem:** When decomposition produces NaN at edges (from trend MA calculation), the seasonal+residual array gets zeros for NaN indices. This artificially lowers variance, biasing the seasonal strength metric.
- **Fix:** Exclude NaN positions from variance calculation or use proper NaN-aware statistics.

### P2-17: PACF Denominator Zero Doesn't Update Phi Matrix
- **File:** `stats/acf.go:81-83`
- **Problem:** When denominator is zero, sets `pacf[k] = 0` and continues but doesn't update `phi[k][j]` values. Causes cascading errors in subsequent lags.
- **Fix:** Set all `phi[k][:]` appropriately when denominator is zero, or return error.

### P2-18: Decomposition Returns nil Instead of Error
- **File:** `stats/decomposition.go:24-26`
- **Problem:** When `n < 2*period`, returns nil silently. Also allows `n == 2*period` which produces all-NaN trend (loop range becomes empty).
- **Fix:** Return `(*DecompositionResult, error)` and require `n >= 3*period` for meaningful results.

### P2-19: Missing Type-Level Doc Comments on Key Exported Types
- **Files:** `arima/arima.go:19`, `sarima/sarima.go:25`, `autoarima/autoarima.go:31,102,131`
- **Problem:** `arima.Model`, `sarima.Model`, `autoarima.Config`, `autoarima.ModelCandidate`, `autoarima.Result` all lack type-level godoc comments despite having field comments.
- **Fix:** Add godoc comment above each type declaration.

---

## P3 - LOW SEVERITY (Code Quality, Minor Issues)

### P3-1: Custom itoa Instead of strconv.Itoa
- **File:** `autoarima/autoarima.go:901-909`
- **Fix:** Replace with `strconv.Itoa()`.

### P3-2: Custom gamma Instead of math.Gamma
- **File:** `stats/ljungbox.go:119-145`
- **Fix:** Consider using `math.Gamma()` from stdlib (available since Go 1.0).

### P3-3: Local min Function Shadows Go 1.21+ Builtin
- **File:** `arima/arima_test.go:324-329`
- **Fix:** Remove the local `min` function; use the builtin.

### P3-4: Unchecked os.WriteFile Error in Demo
- **File:** `demo/main.go:107`
- **Fix:** Check and log the error.

### P3-5: Outdated doc.go References Non-Existent Config Fields
- **File:** `autoarima/doc.go`
- **Problem:** References `config.Seasonal` and `config.SeasonalM` which don't exist. Actual fields are `AutoSeasonal` and `SeasonalPeriods`.
- **Fix:** Update doc.go to match current API.

### P3-6: Exported HourlyPeriods/DailyPeriods Never Used
- **File:** `autoarima/autoarima.go:24-27`
- **Fix:** Either reference in docs/examples or unexport.

### P3-7: Inconsistent Learning Rates Between ARIMA and SARIMA
- **Files:** `arima/arima.go:150` (0.01), `sarima/sarima.go:172` (0.005)
- **Fix:** Standardize or document the rationale for different rates.

### P3-8: No Interfaces for Core Abstractions
- **Files:** Library-wide
- **Problem:** No `Fitter`, `Predictor`, or `StatTest` interfaces. Both ARIMA and SARIMA have identical Fit/Predict/Residuals/FittedValues/Summary method signatures.
- **Fix:** Define interfaces in a shared package for extensibility.

### P3-9: God Objects - Model Structs Are Too Large
- **Files:** `arima/arima.go:20-39`, `sarima/sarima.go:25-48`
- **Problem:** 17-18 fields mixing parameters, data, fitted values, and errors in a single struct.
- **Fix:** Consider composition: separate Order, Coefficients, and Diagnostics structs.

### P3-10: Deep Nesting in optimizeCSS
- **Files:** `arima/arima.go:164-239`, `sarima/sarima.go:196-314`
- **Problem:** 4 levels of nesting. Gradient calculation spans many lines with nested loops.
- **Fix:** Extract gradient calculation into a helper method.

### P3-11: Convergence Check Uses Absolute Tolerance on SSE
- **Files:** `arima/arima.go:236`, `sarima/sarima.go:311`
- **Problem:** `math.Abs(currentSSE - bestSSE) < 1e-8` uses absolute tolerance. For large SSE values, this is too tight; for small SSE, too loose.
- **Fix:** Use relative tolerance: `math.Abs((currentSSE - bestSSE) / bestSSE) < tolerance`.

### P3-12: No Thread-Safety Documentation
- **Files:** All doc.go files
- **Problem:** The library is not thread-safe (no mutexes, exported mutable fields, mutable global slices) but this is not documented.
- **Fix:** Add a note to package docs: "Model instances are not safe for concurrent use."

### P3-13: O(n^2) PACF Allocates Unnecessary 2D Matrix
- **File:** `stats/acf.go:64-67`
- **Problem:** Allocates `(maxLag+1)^2` elements but only uses O(maxLag). For maxLag=200: 40K elements allocated, ~400 used.
- **Fix:** Use rolling vector instead of 2D matrix in Durbin-Levinson algorithm.

### P3-14: No Spectral Analysis for Period Detection
- **File:** `autoarima/autoarima.go:287-350`
- **Problem:** Only ACF-based period detection. Spectral analysis (FFT/periodogram) would be more robust for discovering non-obvious periods.
- **Fix:** Add periodogram-based detection as complementary method.

### P3-15: Missing Accuracy Metrics (MASE, sMAPE)
- **File:** `autoarima/autoarima.go:718-735`
- **Problem:** Only RMSE, MAE, MAPE available. MASE is scale-invariant and better for cross-series comparison.
- **Fix:** Add MASE and sMAPE to the metrics calculation.

### P3-16: README Missing Limitations Section
- **File:** `README.md`
- **Problem:** No documentation of known limitations, minimum data requirements, or when ARIMA is inappropriate.
- **Fix:** Add a "Limitations" section covering CSS-only estimation, no exogenous variables, and dataset requirements.

---

## Implementation Order Recommendation

**Phase 1 - Fix Critical Bugs (P0):**
All P0 items should be fixed before any release. Start with P0-1 (ADF index) and P0-3 (psi weights) as they are single-line or small-scope fixes.

**Phase 2 - Address Safety Issues (P1-1 through P1-7):**
Input validation, polynomial root checking, and error handling improvements. These prevent silent failures.

**Phase 3 - Fix Test Suite (P1-11, P1-12):**
Add reference-validated tests and PredictWithInterval tests. This creates a safety net before refactoring.

**Phase 4 - Refactor Duplication (P1-14):**
Extract shared code between arima/sarima. This simplifies all subsequent work.

**Phase 5 - Feature Gaps (P2):**
Prioritize: drift term (P2-1), model serialization (P2-7), normality tests (P2-12), inverse differencing (P2-13).

**Phase 6 - Long-term (P2-2, P2-8, P2-9):**
Kalman filter, ARIMAX, Box-Cox are significant features requiring careful design.
