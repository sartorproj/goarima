# Plan 1: Box-Cox Transformation

**Branch:** `feat/box-cox`
**Effort:** Small (~150 lines)
**Dependencies:** None — fully independent

## Goal

Add variance-stabilizing Box-Cox transformation with automatic lambda selection,
matching R's `forecast::BoxCox()` and `BoxCox.lambda()`.

## Design

### New functions in `timeseries/series.go`:

```go
// BoxCox applies the Box-Cox transformation with parameter lambda.
// lambda=0 → log(y), lambda=1 → y-1, general → (y^lambda - 1)/lambda
func (s *Series) BoxCox(lambda float64) *Series

// InverseBoxCox reverses the Box-Cox transformation.
func (s *Series) InverseBoxCox(lambda float64) *Series

// BoxCoxLambda finds optimal lambda via profile log-likelihood.
// Searches lambda ∈ [-1, 2] by default. Requires all values > 0.
func BoxCoxLambda(series *Series) (float64, error)
```

### Implementation details:

1. **BoxCox transform:**
   - `lambda == 0`: `log(y)` (limit case)
   - `lambda != 0`: `(y^lambda - 1) / lambda`
   - Returns error/NaN for `y <= 0` when `lambda < 1`

2. **InverseBoxCox:**
   - `lambda == 0`: `exp(y)`
   - `lambda != 0`: `(lambda*y + 1)^(1/lambda)`
   - Bias correction for prediction intervals: `E[InvBoxCox(X)] ≠ InvBoxCox(E[X])`
     Apply: `adjusted = invBC(mean) * (1 + sigma²*(1-lambda)/(2*(invBC(mean))^(2*lambda)))`

3. **Lambda selection (profile likelihood):**
   - Grid search: lambda from -1 to 2, step 0.01
   - For each lambda, compute Box-Cox transformed series
   - Log-likelihood ∝ `(lambda-1) * Σlog(y_i) - n/2 * log(var(transformed))`
   - Return lambda that maximizes this
   - Rounded to nearest 0.5 if within tolerance (prefer interpretable values)

### Files touched:
- `timeseries/series.go` — add BoxCox, InverseBoxCox, BoxCoxLambda
- `timeseries/series_test.go` — tests for transform, inverse, lambda selection
- `autoarima/autoarima.go` — optional: add `BoxCoxLambda` config field, auto-transform before fit

### Test cases:
- BoxCox(0) ≡ log for positive series
- BoxCox(1) ≡ identity shift (y-1)
- InverseBoxCox(BoxCox(y, λ), λ) ≡ y (roundtrip)
- Lambda ≈ 0 for exponentially growing data
- Lambda ≈ 1 for already-stable-variance data
- Lambda ≈ 0.5 for square-root-like variance growth
- Error on series with non-positive values
