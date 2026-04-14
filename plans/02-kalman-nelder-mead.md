# Plan 2: Kalman Filter + Nelder-Mead MLE

**Branch:** `feat/kalman-mle`
**Effort:** Large (~700 lines)
**Dependencies:** None, but Box-Cox complements it

## Goal

Replace CSS gradient descent with exact MLE using state-space form + Kalman filter,
optimized via Nelder-Mead simplex. This fixes the MA gradient bias (AUDIT_FINDINGS H1)
and is the single highest-impact improvement.

## Design

### Phase 1: Nelder-Mead Optimizer (`internal/optimize/`)

```go
// NelderMead minimizes f(x) using the simplex method.
// Returns optimal x and f(x).
func NelderMead(f func([]float64) float64, x0 []float64, opts Options) ([]float64, float64)

type Options struct {
    MaxIter  int     // default 1000
    Tol      float64 // default 1e-8
    Alpha    float64 // reflection: 1.0
    Gamma    float64 // expansion: 2.0
    Rho      float64 // contraction: 0.5
    Sigma    float64 // shrink: 0.5
}
```

Standard textbook Nelder-Mead. ~120 lines. No gradients needed.

### Phase 2: State-Space Representation (`internal/statespace/`)

ARIMA(p,d,q) in state-space form:

```
State equation:    x_{t+1} = T·x_t + R·η_t     (η_t ~ N(0, σ²))
Observation eq:    y_t = Z'·x_t + d_t
```

For ARMA(p,q) with r = max(p, q+1):
- State vector: x_t = [ξ_t, ξ_{t-1}, ..., ξ_{t-r+1}]'  (r × 1)
- T = companion matrix (r × r)
- Z = [1, 0, ..., 0]'  (r × 1)
- R = [1, θ₁, θ₂, ..., θ_{q}, 0, ..., 0]'  (r × 1)

For SARIMA: use expanded AR/MA polynomials (already implemented in sarima.go).

```go
type StateSpace struct {
    T [][]float64 // Transition matrix (r×r)
    Z []float64   // Observation vector (r)
    R []float64   // State noise loading (r)
    d float64     // Intercept
}

func NewARIMAStateSpace(ar, ma []float64, intercept float64) *StateSpace
func NewSARIMAStateSpace(ar, ma, sar, sma []float64, period int, intercept float64) *StateSpace
```

### Phase 3: Kalman Filter (`internal/statespace/`)

```go
type KalmanResult struct {
    LogLikelihood float64
    Residuals     []float64
    Variance      float64
}

// Filter runs the Kalman filter and returns the exact log-likelihood.
func (ss *StateSpace) Filter(y []float64) *KalmanResult
```

Kalman recursion (~80 lines):
```
Predict:  x_{t|t-1} = T·x_{t-1|t-1}
          P_{t|t-1} = T·P_{t-1|t-1}·T' + R·σ²·R'

Update:   v_t = y_t - Z'·x_{t|t-1}       (innovation)
          F_t = Z'·P_{t|t-1}·Z             (innovation variance)
          K_t = P_{t|t-1}·Z / F_t          (Kalman gain)
          x_{t|t} = x_{t|t-1} + K_t·v_t
          P_{t|t} = P_{t|t-1} - K_t·Z'·P_{t|t-1}

LogLik = -n/2·log(2π) - 1/2·Σ[log(F_t) + v_t²/F_t]
```

Initialize with diffuse prior: P_{0|0} = κ·I (κ large, e.g., 1e6).

### Phase 4: Integration into ARIMA/SARIMA

```go
func (m *Model) fitMLE() error {
    // 1. Use CSS result as initial parameter guess
    // 2. Pack parameters into vector: [φ₁..φ_p, θ₁..θ_q, μ]
    // 3. Objective: -loglik from Kalman filter
    // 4. Optimize with Nelder-Mead
    // 5. Unpack and store results
    // 6. Compute Hessian numerically for standard errors
}
```

Add `Method` field to Model: "css" (default, backward compat) or "mle".

### Files touched:
- NEW `internal/optimize/neldermead.go` + test
- NEW `internal/statespace/statespace.go` + test
- NEW `internal/statespace/kalman.go` + test
- `arima/arima.go` — add fitMLE(), Method field
- `sarima/sarima.go` — add fitMLE(), Method field
- `go.mod` — no new deps (pure Go)

### Test cases:
- Nelder-Mead: Rosenbrock function, quadratic
- Kalman: known AR(1) — loglik matches analytical formula
- Kalman: MA(1) — verify CSS vs MLE difference on MA coefficient
- ARIMA(1,1,1): MLE coefficients closer to true values than CSS
- SARIMA: verify expanded polynomial state-space matches direct computation
- Backward compat: Method="" defaults to CSS, existing tests unchanged
