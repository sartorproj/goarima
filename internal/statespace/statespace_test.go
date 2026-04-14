package statespace

import (
	"math"
	"math/rand"
	"testing"
)

func TestNewARMA_AR1(t *testing.T) {
	// AR(1) with φ = 0.8
	m := NewARMA([]float64{0.8}, nil, 0)

	if m.R != 1 {
		t.Errorf("Expected state dim 1, got %d", m.R)
	}
	if m.T[0] != 0.8 {
		t.Errorf("Expected T[0][0] = 0.8, got %f", m.T[0])
	}
	if m.Z[0] != 1.0 {
		t.Errorf("Expected Z[0] = 1, got %f", m.Z[0])
	}
	if m.RR[0] != 1.0 {
		t.Errorf("Expected RR[0] = 1, got %f", m.RR[0])
	}
}

func TestNewARMA_MA1(t *testing.T) {
	// MA(1) with θ = 0.6
	m := NewARMA(nil, []float64{0.6}, 0)

	if m.R != 2 {
		t.Errorf("Expected state dim 2, got %d", m.R)
	}
	// T should be [0, 1; 0, 0] for MA(1) (superdiagonal identity)
	if m.T[0] != 0 || m.T[1] != 1 {
		t.Errorf("Expected T first row = [0, 1], got [%f, %f]", m.T[0], m.T[1])
	}
	if m.RR[0] != 1.0 || m.RR[1] != 0.6 {
		t.Errorf("Expected RR = [1, 0.6], got [%f, %f]", m.RR[0], m.RR[1])
	}
}

func TestNewARMA_ARMA11(t *testing.T) {
	// ARMA(1,1) with φ=0.7, θ=0.3
	m := NewARMA([]float64{0.7}, []float64{0.3}, 0)

	if m.R != 2 {
		t.Errorf("Expected state dim 2, got %d", m.R)
	}
	// T = [0.7, 0; 1, 0]
	if math.Abs(m.T[0]-0.7) > 1e-10 {
		t.Errorf("T[0][0] = %f, want 0.7", m.T[0])
	}
	// RR = [1, 0.3]
	if math.Abs(m.RR[1]-0.3) > 1e-10 {
		t.Errorf("RR[1] = %f, want 0.3", m.RR[1])
	}
}

func TestKalmanFilter_AR1_LogLik(t *testing.T) {
	// Generate AR(1) data with known parameters
	rng := rand.New(rand.NewSource(42))
	phi := 0.7
	sigma := 1.0
	n := 500

	y := make([]float64, n)
	y[0] = rng.NormFloat64() * sigma
	for i := 1; i < n; i++ {
		y[i] = phi*y[i-1] + rng.NormFloat64()*sigma
	}

	// Filter with true parameters
	m := NewARMA([]float64{phi}, nil, 0)
	result := m.Filter(y)

	if math.IsInf(result.LogLikelihood, -1) {
		t.Fatal("Log-likelihood is -Inf")
	}
	if result.Sigma2 <= 0 {
		t.Fatalf("Sigma2 = %f, expected positive", result.Sigma2)
	}

	t.Logf("AR(1) φ=0.7: loglik=%.2f, σ²=%.4f (true σ²=1.0)", result.LogLikelihood, result.Sigma2)

	// σ² should be close to 1.0
	if math.Abs(result.Sigma2-1.0) > 0.2 {
		t.Errorf("Sigma2 = %.4f, expected ~1.0", result.Sigma2)
	}

	// Filter with wrong parameters should give worse loglik
	mWrong := NewARMA([]float64{0.1}, nil, 0)
	resultWrong := mWrong.Filter(y)

	if resultWrong.LogLikelihood >= result.LogLikelihood {
		t.Errorf("Wrong parameters gave better loglik: %.2f >= %.2f",
			resultWrong.LogLikelihood, result.LogLikelihood)
	}
}

func TestKalmanFilter_WhiteNoise(t *testing.T) {
	// White noise: ARMA(0,0), should estimate σ² ≈ variance of data
	rng := rand.New(rand.NewSource(123))
	n := 1000
	y := make([]float64, n)
	for i := range y {
		y[i] = rng.NormFloat64() * 2.0 // σ = 2, σ² = 4
	}

	m := NewARMA(nil, nil, 0)
	result := m.Filter(y)

	if math.Abs(result.Sigma2-4.0) > 0.5 {
		t.Errorf("Sigma2 = %.4f, expected ~4.0", result.Sigma2)
	}
	t.Logf("White noise σ=2: σ²=%.4f (expected 4.0)", result.Sigma2)
}

func TestKalmanFilter_MA1(t *testing.T) {
	// Generate MA(1) data
	rng := rand.New(rand.NewSource(99))
	theta := 0.5
	n := 500

	eps := make([]float64, n+1)
	for i := range eps {
		eps[i] = rng.NormFloat64()
	}
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		y[i] = eps[i+1] + theta*eps[i]
	}

	// Filter with true parameters
	m := NewARMA(nil, []float64{theta}, 0)
	result := m.Filter(y)

	t.Logf("MA(1) θ=0.5: loglik=%.2f, σ²=%.4f", result.LogLikelihood, result.Sigma2)

	// σ² should be in reasonable range (diffuse init can inflate slightly)
	if result.Sigma2 < 0.5 || result.Sigma2 > 2.0 {
		t.Errorf("Sigma2 = %.4f, expected in [0.5, 2.0]", result.Sigma2)
	}

	// True parameters should give better loglik than wrong ones
	mWrong := NewARMA(nil, []float64{-0.5}, 0)
	resultWrong := mWrong.Filter(y)
	if resultWrong.LogLikelihood >= result.LogLikelihood {
		t.Errorf("Wrong θ gave better loglik: %.2f >= %.2f",
			resultWrong.LogLikelihood, result.LogLikelihood)
	}
}
