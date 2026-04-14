// Package statespace implements the state-space representation of ARIMA models
// and the Kalman filter for exact maximum likelihood estimation.
package statespace

import (
	"math"
)

// Model represents an ARMA model in state-space form.
//
// State equation:   x_{t+1} = T·x_t + R·η_t     (η_t ~ N(0, σ²))
// Observation eq:   y_t = Z'·x_t + intercept
//
// For ARMA(p,q) with r = max(p, q+1):
//   - State dim r
//   - T is the companion matrix
//   - Z = [1, 0, ..., 0]
//   - R = [1, θ₁, θ₂, ..., θ_{q}, 0, ..., 0]
type Model struct {
	R      int       // State dimension = max(p, q+1)
	T      []float64 // Transition matrix (r×r), row-major
	Z      []float64 // Observation vector (r)
	RR     []float64 // State noise loading (r) — called RR to avoid clash with R field
	Interc float64   // Intercept (mean of differenced series)
}

// NewARMA creates a state-space model from ARMA coefficients.
// ar = [φ₁, φ₂, ..., φ_p], ma = [θ₁, θ₂, ..., θ_q].
func NewARMA(ar, ma []float64, intercept float64) *Model {
	p := len(ar)
	q := len(ma)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		r = 1
	}

	m := &Model{
		R:      r,
		T:      make([]float64, r*r),
		Z:      make([]float64, r),
		RR:     make([]float64, r),
		Interc: intercept,
	}

	// Z = [1, 0, 0, ...]
	m.Z[0] = 1.0

	// RR = [1, θ₁, θ₂, ..., θ_q, 0, ...]
	m.RR[0] = 1.0
	for i := 0; i < q && i+1 < r; i++ {
		m.RR[i+1] = ma[i]
	}

	// T = companion matrix (Harvey 1989 / Durbin-Koopman 2012 form):
	// [φ₁  1   0   ... 0]
	// [φ₂  0   1   ... 0]
	// [...]
	// [φ_r 0   0   ... 0]
	// First column: AR coefficients
	for i := 0; i < p && i < r; i++ {
		m.T[i*r] = ar[i] // first column of row i
	}
	// Superdiagonal: identity shift
	for i := 0; i < r-1; i++ {
		m.T[i*r+i+1] = 1.0 // superdiagonal of row i
	}

	return m
}

// KalmanResult holds the output of the Kalman filter.
type KalmanResult struct {
	LogLikelihood float64   // Exact log-likelihood (concentrated on σ²)
	Sigma2        float64   // MLE estimate of innovation variance σ²
	Innovations   []float64 // Standardized innovations v_t / sqrt(F_t)
	SSE           float64   // Sum of squared standardized innovations
}

// Filter runs the Kalman filter on observations y and returns the exact log-likelihood.
// Uses diffuse initialization (large P₀) for non-stationary models.
func (m *Model) Filter(y []float64) *KalmanResult {
	n := len(y)
	r := m.R

	// State vector and covariance
	state := make([]float64, r)
	P := make([]float64, r*r)

	// Diffuse initialization: P = κ·I
	kappa := 1e6
	for i := 0; i < r; i++ {
		P[i*r+i] = kappa
	}

	// Working arrays
	predState := make([]float64, r)
	predP := make([]float64, r*r)
	K := make([]float64, r)     // Kalman gain
	tmp := make([]float64, r*r) // scratch for matrix ops

	sumLogF := 0.0
	sumV2F := 0.0
	validObs := 0

	innovations := make([]float64, n)

	for t := 0; t < n; t++ {
		// --- Prediction step ---
		// predState = T · state
		for i := 0; i < r; i++ {
			predState[i] = 0
			for j := 0; j < r; j++ {
				predState[i] += m.T[i*r+j] * state[j]
			}
		}

		// predP = T · P · T' + RR · RR' (σ² factored out)
		// First: tmp = T · P
		for i := 0; i < r; i++ {
			for j := 0; j < r; j++ {
				tmp[i*r+j] = 0
				for k := 0; k < r; k++ {
					tmp[i*r+j] += m.T[i*r+k] * P[k*r+j]
				}
			}
		}
		// predP = tmp · T'
		for i := 0; i < r; i++ {
			for j := 0; j < r; j++ {
				predP[i*r+j] = 0
				for k := 0; k < r; k++ {
					predP[i*r+j] += tmp[i*r+k] * m.T[j*r+k] // T' => T[j][k]
				}
				// Add RR·RR' (σ²=1 for concentrated likelihood)
				predP[i*r+j] += m.RR[i] * m.RR[j]
			}
		}

		// --- Update step ---
		// Innovation: v_t = y_t - Z' · predState - intercept
		yPred := m.Interc
		for i := 0; i < r; i++ {
			yPred += m.Z[i] * predState[i]
		}
		v := y[t] - yPred

		// Innovation variance: F_t = Z' · predP · Z
		F := 0.0
		for i := 0; i < r; i++ {
			for j := 0; j < r; j++ {
				F += m.Z[i] * predP[i*r+j] * m.Z[j]
			}
		}

		if F <= 0 || math.IsNaN(F) || math.IsInf(F, 0) {
			F = 1e-10
		}

		// Kalman gain: K = predP · Z / F
		for i := 0; i < r; i++ {
			K[i] = 0
			for j := 0; j < r; j++ {
				K[i] += predP[i*r+j] * m.Z[j]
			}
			K[i] /= F
		}

		// Update state: state = predState + K · v
		for i := 0; i < r; i++ {
			state[i] = predState[i] + K[i]*v
		}

		// Update P: P[i][j] = predP[i][j] - K[i] * sum_k(Z[k] * predP[k][j])
		for i := 0; i < r; i++ {
			zPredPJ := make([]float64, r)
			for j := 0; j < r; j++ {
				for k := 0; k < r; k++ {
					zPredPJ[j] += m.Z[k] * predP[k*r+j]
				}
			}
			for j := 0; j < r; j++ {
				P[i*r+j] = predP[i*r+j] - K[i]*zPredPJ[j]
			}
		}

		// Accumulate log-likelihood components
		// Skip early observations during diffuse phase
		if F < kappa*0.5 {
			sumLogF += math.Log(F)
			sumV2F += v * v / F
			validObs++
			innovations[t] = v / math.Sqrt(F)
		}
	}

	if validObs == 0 {
		return &KalmanResult{LogLikelihood: math.Inf(-1)}
	}

	// Concentrated log-likelihood:
	// logL = -n/2·log(2π) - 1/2·Σlog(F_t) - n/2·log(σ²) - n/2
	// where σ² = (1/n)·Σ(v_t²/F_t)
	sigma2 := sumV2F / float64(validObs)
	nf := float64(validObs)
	logLik := -nf/2*math.Log(2*math.Pi) - 0.5*sumLogF - nf/2*math.Log(sigma2) - nf/2

	return &KalmanResult{
		LogLikelihood: logLik,
		Sigma2:        sigma2,
		Innovations:   innovations,
		SSE:           sumV2F,
	}
}
