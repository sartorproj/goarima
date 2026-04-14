// Package sarima implements Seasonal ARIMA (SARIMA) models.
package sarima

import (
	"errors"
	"math"

	"github.com/sartorproj/goarima/stats"
	"github.com/sartorproj/goarima/timeseries"
)

// Order represents SARIMA model order (p, d, q) x (P, D, Q, m).
type Order struct {
	P int // Non-seasonal AR order
	D int // Non-seasonal differencing order
	Q int // Non-seasonal MA order
	// Seasonal components
	SP int // Seasonal AR order
	SD int // Seasonal differencing order
	SQ int // Seasonal MA order
	M  int // Seasonal period (e.g., 12 for monthly data with yearly seasonality)
}

// Model represents a SARIMA(p,d,q)(P,D,Q)[m] model with multiplicative seasonal terms.
// Use New() to create and Fit() to estimate parameters from data.
type Model struct {
	Order      Order
	ARCoeffs   []float64 // Non-seasonal AR coefficients
	MACoeffs   []float64 // Non-seasonal MA coefficients
	SARCoeffs  []float64 // Seasonal AR coefficients
	SMACoeffs  []float64 // Seasonal MA coefficients
	Intercept  float64
	Variance   float64
	AIC        float64
	AICc       float64 // Corrected AIC for small sample sizes
	BIC        float64
	LogLik     float64
	fitted     bool
	data       *timeseries.Series
	diffData   *timeseries.Series
	residuals  []float64
	fittedVals []float64

	// Standard errors for coefficients
	ARStdErrors  []float64
	MAStdErrors  []float64
	SARStdErrors []float64
	SMAStdErrors []float64
}

// New creates a new SARIMA model with the specified order.
// Returns nil if any order is negative or period is non-positive.
func New(p, d, q, sp, sd, sq, m int) *Model {
	if p < 0 || d < 0 || q < 0 || sp < 0 || sd < 0 || sq < 0 || m <= 0 {
		return nil
	}
	return &Model{
		Order: Order{
			P: p, D: d, Q: q,
			SP: sp, SD: sd, SQ: sq, M: m,
		},
		ARCoeffs:  make([]float64, p),
		MACoeffs:  make([]float64, q),
		SARCoeffs: make([]float64, sp),
		SMACoeffs: make([]float64, sq),
	}
}

// Fit fits the SARIMA model to the given time series data.
func (m *Model) Fit(series *timeseries.Series) error {
	minLen := m.Order.P + m.Order.Q + m.Order.D +
		m.Order.SP*m.Order.M + m.Order.SD*m.Order.M + m.Order.SQ*m.Order.M + 20

	if series.Len() < minLen {
		return errors.New("insufficient data points for the specified order")
	}

	// Validate input for NaN/Inf
	for _, v := range series.Values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("input series contains NaN or Inf values")
		}
	}

	m.data = series

	// Apply non-seasonal differencing
	diffSeries := series
	for i := 0; i < m.Order.D; i++ {
		diffSeries = diffSeries.Diff()
		if diffSeries.Len() == 0 {
			return errors.New("differencing resulted in empty series")
		}
	}

	// Apply seasonal differencing
	for i := 0; i < m.Order.SD; i++ {
		diffSeries = diffSeries.SeasonalDiff(m.Order.M)
		if diffSeries.Len() == 0 {
			return errors.New("seasonal differencing resulted in empty series")
		}
	}

	m.diffData = diffSeries

	// Fit the model
	err := m.fitCSS()
	if err != nil {
		return err
	}

	// Calculate information criteria
	m.calculateIC()

	m.fitted = true
	return nil
}

// fitCSS fits the model using Conditional Sum of Squares estimation.
func (m *Model) fitCSS() error {
	y := m.diffData.Values
	n := len(y)
	p := m.Order.P
	sp := m.Order.SP
	period := m.Order.M

	// Calculate mean
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(n)
	m.Intercept = mean

	// Initialize AR coefficients using ACF
	if p > 0 {
		acf := stats.ACF(m.diffData, p)
		if acf != nil {
			m.ARCoeffs = initARCoeffs(acf, p)
		}
	}

	// Initialize seasonal AR coefficients
	if sp > 0 {
		acf := stats.ACF(m.diffData, sp*period)
		if acf != nil {
			for i := 0; i < sp; i++ {
				idx := (i + 1) * period
				if idx < len(acf) {
					m.SARCoeffs[i] = acf[idx] * 0.5
				}
			}
		}
	}

	// Initialize MA and SMA coefficients
	for i := range m.MACoeffs {
		m.MACoeffs[i] = 0.1
	}
	for i := range m.SMACoeffs {
		m.SMACoeffs[i] = 0.1
	}

	// Optimize using iterative method
	err := m.optimizeCSS(y)
	if err != nil {
		return err
	}

	return nil
}

// predict computes the one-step prediction at time t using the multiplicative SARIMA formula.
// φ(B)·Φ(B^m)·z_t = θ(B)·Θ(B^m)·ε_t with cross-product terms.
func (m *Model) predict(t int, y, residuals []float64, intercept float64) float64 {
	p := m.Order.P
	q := m.Order.Q
	sp := m.Order.SP
	sq := m.Order.SQ
	period := m.Order.M

	pred := intercept

	// Non-seasonal AR: +φ_i * (y_{t-i} - μ)
	for i := 0; i < p; i++ {
		lag := i + 1
		if t-lag >= 0 {
			pred += m.ARCoeffs[i] * (y[t-lag] - intercept)
		}
	}

	// Seasonal AR: +Φ_j * (y_{t-jm} - μ)
	for j := 0; j < sp; j++ {
		lag := (j + 1) * period
		if t-lag >= 0 {
			pred += m.SARCoeffs[j] * (y[t-lag] - intercept)
		}
	}

	// Cross-product AR: -φ_i * Φ_j * (y_{t-i-jm} - μ)
	for i := 0; i < p; i++ {
		for j := 0; j < sp; j++ {
			lag := (i + 1) + (j+1)*period
			if t-lag >= 0 {
				pred -= m.ARCoeffs[i] * m.SARCoeffs[j] * (y[t-lag] - intercept)
			}
		}
	}

	// Non-seasonal MA: +θ_i * ε_{t-i}
	for i := 0; i < q; i++ {
		lag := i + 1
		if t-lag >= 0 {
			pred += m.MACoeffs[i] * residuals[t-lag]
		}
	}

	// Seasonal MA: +Θ_j * ε_{t-jm}
	for j := 0; j < sq; j++ {
		lag := (j + 1) * period
		if t-lag >= 0 {
			pred += m.SMACoeffs[j] * residuals[t-lag]
		}
	}

	// Cross-product MA: +θ_i * Θ_j * ε_{t-i-jm}
	for i := 0; i < q; i++ {
		for j := 0; j < sq; j++ {
			lag := (i + 1) + (j+1)*period
			if t-lag >= 0 {
				pred += m.MACoeffs[i] * m.SMACoeffs[j] * residuals[t-lag]
			}
		}
	}

	return pred
}

// optimizeCSS optimizes SARIMA parameters with adaptive learning and momentum.
func (m *Model) optimizeCSS(y []float64) error {
	n := len(y)
	p := m.Order.P
	q := m.Order.Q
	sp := m.Order.SP
	sq := m.Order.SQ
	period := m.Order.M

	maxIter := 200
	tolerance := 1e-8
	learningRate := 0.005 // Lower than ARIMA (0.01) due to more parameters and cross-product terms
	momentum := 0.9
	decay := 0.99

	// Momentum terms
	arMomentum := make([]float64, p)
	maMomentum := make([]float64, q)
	sarMomentum := make([]float64, sp)
	smaMomentum := make([]float64, sq)

	// Start index must accommodate maximum lag from expanded polynomials
	maxARLag := p + sp*period
	maxMALag := q + sq*period
	startIdx := max(maxARLag, maxMALag)
	if startIdx >= n-10 {
		startIdx = max(max(p, q), max(sp*period, sq*period))
	}
	if startIdx >= n-10 {
		startIdx = 0
	}

	// Track best solution
	bestSSE := math.Inf(1)
	bestARCoeffs := make([]float64, p)
	bestMACoeffs := make([]float64, q)
	bestSARCoeffs := make([]float64, sp)
	bestSMACoeffs := make([]float64, sq)
	noImproveCount := 0

	for iter := 0; iter < maxIter; iter++ {
		// Calculate residuals with current parameters
		residuals := make([]float64, n)
		currentSSE := 0.0

		for t := startIdx; t < n; t++ {
			pred := m.predict(t, y, residuals, m.Intercept)
			residuals[t] = y[t] - pred
			currentSSE += residuals[t] * residuals[t]
		}

		// Track best solution
		if currentSSE < bestSSE {
			bestSSE = currentSSE
			copy(bestARCoeffs, m.ARCoeffs)
			copy(bestMACoeffs, m.MACoeffs)
			copy(bestSARCoeffs, m.SARCoeffs)
			copy(bestSMACoeffs, m.SMACoeffs)
			noImproveCount = 0
		} else {
			noImproveCount++
		}

		// Early stopping
		if noImproveCount > 20 {
			break
		}

		// Calculate gradients (with cross-product terms)
		arGrad := make([]float64, p)
		maGrad := make([]float64, q)
		sarGrad := make([]float64, sp)
		smaGrad := make([]float64, sq)

		for t := startIdx; t < n; t++ {
			// AR gradients: ∂pred/∂φ_i = (y_{t-i} - μ) - Σ_j Φ_j*(y_{t-i-jm} - μ)
			for i := 0; i < p; i++ {
				lag := i + 1
				if t-lag >= 0 {
					grad := y[t-lag] - m.Intercept
					for j := 0; j < sp; j++ {
						crossLag := lag + (j+1)*period
						if t-crossLag >= 0 {
							grad -= m.SARCoeffs[j] * (y[t-crossLag] - m.Intercept)
						}
					}
					arGrad[i] -= 2 * residuals[t] * grad
				}
			}

			// SAR gradients: ∂pred/∂Φ_j = (y_{t-jm} - μ) - Σ_i φ_i*(y_{t-i-jm} - μ)
			for j := 0; j < sp; j++ {
				lag := (j + 1) * period
				if t-lag >= 0 {
					grad := y[t-lag] - m.Intercept
					for i := 0; i < p; i++ {
						crossLag := (i + 1) + lag
						if t-crossLag >= 0 {
							grad -= m.ARCoeffs[i] * (y[t-crossLag] - m.Intercept)
						}
					}
					sarGrad[j] -= 2 * residuals[t] * grad
				}
			}

			// MA gradients: ∂pred/∂θ_i = ε_{t-i} + Σ_j Θ_j*ε_{t-i-jm}
			for i := 0; i < q; i++ {
				lag := i + 1
				if t-lag >= 0 {
					grad := residuals[t-lag]
					for j := 0; j < sq; j++ {
						crossLag := lag + (j+1)*period
						if t-crossLag >= 0 {
							grad += m.SMACoeffs[j] * residuals[t-crossLag]
						}
					}
					maGrad[i] -= 2 * residuals[t] * grad
				}
			}

			// SMA gradients: ∂pred/∂Θ_j = ε_{t-jm} + Σ_i θ_i*ε_{t-i-jm}
			for j := 0; j < sq; j++ {
				lag := (j + 1) * period
				if t-lag >= 0 {
					grad := residuals[t-lag]
					for i := 0; i < q; i++ {
						crossLag := (i + 1) + lag
						if t-crossLag >= 0 {
							grad += m.MACoeffs[i] * residuals[t-crossLag]
						}
					}
					smaGrad[j] -= 2 * residuals[t] * grad
				}
			}
		}

		// Update parameters with momentum
		for i := 0; i < p; i++ {
			arMomentum[i] = momentum*arMomentum[i] + learningRate*arGrad[i]/float64(n)
			m.ARCoeffs[i] -= arMomentum[i]
			m.ARCoeffs[i] = clamp(m.ARCoeffs[i], -0.99, 0.99)
		}
		for i := 0; i < sp; i++ {
			sarMomentum[i] = momentum*sarMomentum[i] + learningRate*sarGrad[i]/float64(n)
			m.SARCoeffs[i] -= sarMomentum[i]
			m.SARCoeffs[i] = clamp(m.SARCoeffs[i], -0.99, 0.99)
		}
		for i := 0; i < q; i++ {
			maMomentum[i] = momentum*maMomentum[i] + learningRate*maGrad[i]/float64(n)
			m.MACoeffs[i] -= maMomentum[i]
			m.MACoeffs[i] = clamp(m.MACoeffs[i], -0.99, 0.99)
		}
		for i := 0; i < sq; i++ {
			smaMomentum[i] = momentum*smaMomentum[i] + learningRate*smaGrad[i]/float64(n)
			m.SMACoeffs[i] -= smaMomentum[i]
			m.SMACoeffs[i] = clamp(m.SMACoeffs[i], -0.99, 0.99)
		}

		// Decay learning rate
		learningRate *= decay

		// Convergence check (relative tolerance)
		if iter > 0 && bestSSE > 0 && math.Abs(currentSSE-bestSSE)/bestSSE < tolerance {
			break
		}
	}

	// Restore best solution
	copy(m.ARCoeffs, bestARCoeffs)
	copy(m.MACoeffs, bestMACoeffs)
	copy(m.SARCoeffs, bestSARCoeffs)
	copy(m.SMACoeffs, bestSMACoeffs)

	// Calculate final residuals and fitted values
	m.residuals = make([]float64, n)
	m.fittedVals = make([]float64, n)

	for t := 0; t < n; t++ {
		pred := m.predict(t, y, m.residuals, m.Intercept)
		m.fittedVals[t] = pred
		m.residuals[t] = y[t] - pred
	}

	// Calculate variance
	sse := 0.0
	count := 0
	for t := startIdx; t < n; t++ {
		sse += m.residuals[t] * m.residuals[t]
		count++
	}

	numParams := p + q + sp + sq + 1
	if count > numParams {
		m.Variance = sse / float64(count-numParams)
	} else {
		m.Variance = sse / float64(count)
	}

	// Estimate coefficient standard errors
	m.estimateStdErrors(y)

	return nil
}

// estimateStdErrors estimates standard errors for all coefficients using numerical Hessian.
func (m *Model) estimateStdErrors(y []float64) {
	n := len(y)
	p := m.Order.P
	q := m.Order.Q
	sp := m.Order.SP
	sq := m.Order.SQ

	if p+q+sp+sq == 0 {
		return
	}

	eps := 1e-5

	computeSSE := func(arC, maC, sarC, smaC []float64) float64 {
		residuals := make([]float64, n)
		sse := 0.0
		// Save/restore coefficients
		origAR, origMA := m.ARCoeffs, m.MACoeffs
		origSAR, origSMA := m.SARCoeffs, m.SMACoeffs
		m.ARCoeffs, m.MACoeffs = arC, maC
		m.SARCoeffs, m.SMACoeffs = sarC, smaC
		startIdx := max(p+sp*m.Order.M, q+sq*m.Order.M)
		if startIdx >= n {
			startIdx = 0
		}
		for t := startIdx; t < n; t++ {
			pred := m.predict(t, y, residuals, m.Intercept)
			residuals[t] = y[t] - pred
			sse += residuals[t] * residuals[t]
		}
		m.ARCoeffs, m.MACoeffs = origAR, origMA
		m.SARCoeffs, m.SMACoeffs = origSAR, origSMA
		return sse
	}

	baseSSE := computeSSE(m.ARCoeffs, m.MACoeffs, m.SARCoeffs, m.SMACoeffs)

	perturbAndCompute := func(coeffs []float64, idx int, isSeasonal bool, isAR bool) float64 {
		plus := make([]float64, len(coeffs))
		minus := make([]float64, len(coeffs))
		copy(plus, coeffs)
		copy(minus, coeffs)
		plus[idx] += eps
		minus[idx] -= eps

		var ssePlus, sseMinus float64
		switch {
		case isAR && !isSeasonal:
			ssePlus = computeSSE(plus, m.MACoeffs, m.SARCoeffs, m.SMACoeffs)
			sseMinus = computeSSE(minus, m.MACoeffs, m.SARCoeffs, m.SMACoeffs)
		case !isAR && !isSeasonal:
			ssePlus = computeSSE(m.ARCoeffs, plus, m.SARCoeffs, m.SMACoeffs)
			sseMinus = computeSSE(m.ARCoeffs, minus, m.SARCoeffs, m.SMACoeffs)
		case isAR && isSeasonal:
			ssePlus = computeSSE(m.ARCoeffs, m.MACoeffs, plus, m.SMACoeffs)
			sseMinus = computeSSE(m.ARCoeffs, m.MACoeffs, minus, m.SMACoeffs)
		default:
			ssePlus = computeSSE(m.ARCoeffs, m.MACoeffs, m.SARCoeffs, plus)
			sseMinus = computeSSE(m.ARCoeffs, m.MACoeffs, m.SARCoeffs, minus)
		}

		hessianDiag := (ssePlus - 2*baseSSE + sseMinus) / (eps * eps)
		if hessianDiag > 0 {
			return math.Sqrt(2 * m.Variance / hessianDiag)
		}
		return 0
	}

	m.ARStdErrors = make([]float64, p)
	for i := 0; i < p; i++ {
		m.ARStdErrors[i] = perturbAndCompute(m.ARCoeffs, i, false, true)
	}

	m.MAStdErrors = make([]float64, q)
	for i := 0; i < q; i++ {
		m.MAStdErrors[i] = perturbAndCompute(m.MACoeffs, i, false, false)
	}

	m.SARStdErrors = make([]float64, sp)
	for i := 0; i < sp; i++ {
		m.SARStdErrors[i] = perturbAndCompute(m.SARCoeffs, i, true, true)
	}

	m.SMAStdErrors = make([]float64, sq)
	for i := 0; i < sq; i++ {
		m.SMAStdErrors[i] = perturbAndCompute(m.SMACoeffs, i, true, false)
	}
}

// calculateIC calculates AIC, AICc, and BIC.
func (m *Model) calculateIC() {
	n := len(m.residuals)
	k := m.Order.P + m.Order.Q + m.Order.SP + m.Order.SQ + 2 // +2 for intercept and variance

	sse := 0.0
	for _, r := range m.residuals {
		sse += r * r
	}

	if m.Variance > 0 {
		m.LogLik = -float64(n)/2*math.Log(2*math.Pi) - float64(n)/2*math.Log(m.Variance) - sse/(2*m.Variance)
	} else {
		m.LogLik = math.Inf(-1)
	}

	m.AIC = -2*m.LogLik + 2*float64(k)

	// AICc = AIC + 2*k*(k+1)/(n-k-1) - corrected AIC for small sample sizes
	kf := float64(k)
	nf := float64(n)
	if nf-kf-1 > 0 {
		m.AICc = m.AIC + 2*kf*(kf+1)/(nf-kf-1)
	} else {
		m.AICc = math.Inf(1)
	}

	m.BIC = -2*m.LogLik + float64(k)*math.Log(float64(n))
}

// Predict generates forecasts for the specified number of steps ahead.
func (m *Model) Predict(steps int) ([]float64, error) {
	forecasts, _, _, err := m.PredictWithInterval(steps, 0.95)
	return forecasts, err
}

// PredictWithInterval generates forecasts with prediction intervals.
// Returns point forecasts, lower bounds, and upper bounds at the given confidence level.
func (m *Model) PredictWithInterval(steps int, confidence float64) (forecasts, lower, upper []float64, err error) {
	if !m.fitted {
		return nil, nil, nil, errors.New("model must be fitted before prediction")
	}

	if steps < 1 {
		return nil, nil, nil, errors.New("steps must be at least 1")
	}

	if confidence <= 0 || confidence >= 1 {
		confidence = 0.95
	}

	d := m.Order.D
	sd := m.Order.SD
	period := m.Order.M

	y := m.diffData.Values
	n := len(y)

	// Extended arrays for recursive forecasting
	extY := make([]float64, n+steps)
	copy(extY, y)

	extResiduals := make([]float64, n+steps)
	copy(extResiduals, m.residuals)

	// Generate forecasts using multiplicative SARIMA formula
	for h := 0; h < steps; h++ {
		t := n + h
		// For forecasting, future residuals are 0 (already zeroed in extResiduals)
		pred := m.predict(t, extY, extResiduals, m.Intercept)
		extY[t] = pred
	}

	forecasts = make([]float64, steps)
	copy(forecasts, extY[n:])

	// Compute psi weights for full model (on differenced scale)
	psi := m.computePsiWeights(steps)

	// Propagate psi weights through non-seasonal integration (cumsum for each d)
	for i := 0; i < d; i++ {
		for j := 1; j < len(psi); j++ {
			psi[j] += psi[j-1]
		}
	}

	// Propagate psi weights through seasonal integration
	for i := 0; i < sd; i++ {
		for j := period; j < len(psi); j++ {
			psi[j] += psi[j-period]
		}
	}

	// Calculate prediction variance: Var(e_h) = σ² * Σ_{j=0}^{h-1} Ψ_j²
	predVariance := make([]float64, steps)
	cumPsiSq := 0.0
	for h := 0; h < steps; h++ {
		cumPsiSq += psi[h] * psi[h]
		predVariance[h] = m.Variance * cumPsiSq
	}

	// Integrate forecasts back to original scale
	forecasts = m.integrate(forecasts)

	// Calculate prediction intervals
	z := normalQuantile((1 + confidence) / 2)

	lower = make([]float64, steps)
	upper = make([]float64, steps)

	for h := 0; h < steps; h++ {
		se := math.Sqrt(predVariance[h])
		lower[h] = forecasts[h] - z*se
		upper[h] = forecasts[h] + z*se
	}

	return forecasts, lower, upper, nil
}

// computePsiWeights computes the MA(∞) psi weights for the full SARIMA model.
// Uses the multiplicative expansion of AR and MA polynomials.
// Returns psi[0..steps] where psi[0] = 1 (ψ_0).
func (m *Model) computePsiWeights(steps int) []float64 {
	p := m.Order.P
	q := m.Order.Q
	sp := m.Order.SP
	sq := m.Order.SQ
	period := m.Order.M

	// Compute max lag of expanded polynomials
	maxARLag := p + sp*period
	maxMALag := q + sq*period

	// Expanded AR coefficients: a_k where Π(B) = 1 - Σ a_k·B^k
	// (a_k > 0 means positive contribution to prediction)
	ar := make([]float64, maxARLag+1)
	for i := 0; i < p; i++ {
		ar[i+1] += m.ARCoeffs[i]
	}
	for j := 0; j < sp; j++ {
		ar[(j+1)*period] += m.SARCoeffs[j]
	}
	for i := 0; i < p; i++ {
		for j := 0; j < sp; j++ {
			ar[(i+1)+(j+1)*period] -= m.ARCoeffs[i] * m.SARCoeffs[j]
		}
	}

	// Expanded MA coefficients: b_k where Θ(B) = 1 + Σ b_k·B^k
	ma := make([]float64, maxMALag+1)
	for i := 0; i < q; i++ {
		ma[i+1] += m.MACoeffs[i]
	}
	for j := 0; j < sq; j++ {
		ma[(j+1)*period] += m.SMACoeffs[j]
	}
	for i := 0; i < q; i++ {
		for j := 0; j < sq; j++ {
			ma[(i+1)+(j+1)*period] += m.MACoeffs[i] * m.SMACoeffs[j]
		}
	}

	// Recursive computation: ψ_0 = 1, ψ_j = b_j + Σ_{k=1}^{j} a_k·ψ_{j-k}
	psi := make([]float64, steps)
	psi[0] = 1.0
	for j := 1; j < steps; j++ {
		if j < len(ma) {
			psi[j] = ma[j]
		}
		for k := 1; k < len(ar) && k <= j; k++ {
			psi[j] += ar[k] * psi[j-k]
		}
	}

	return psi
}

// normalQuantile wraps stats.NormalQuantile.
func normalQuantile(p float64) float64 {
	return stats.NormalQuantile(p)
}

// integrate undoes differencing to return forecasts on original scale.
// Differencing in Fit() is: first non-seasonal (d times), then seasonal (sd times).
// Integration order: first undo seasonal, then undo non-seasonal.
func (m *Model) integrate(forecasts []float64) []float64 {
	d := m.Order.D
	sd := m.Order.SD
	period := m.Order.M
	original := m.data.Values

	result := make([]float64, len(forecasts))
	copy(result, forecasts)

	// Compute intermediate non-seasonally differenced series
	// nsLevels[0] = original, nsLevels[i] = i-th non-seasonal diff
	nsLevels := make([][]float64, d+1)
	nsLevels[0] = original
	for i := 1; i <= d; i++ {
		prev := nsLevels[i-1]
		if len(prev) <= 1 {
			break
		}
		diff := make([]float64, len(prev)-1)
		for j := 0; j < len(diff); j++ {
			diff[j] = prev[j+1] - prev[j]
		}
		nsLevels[i] = diff
	}

	// The fully non-seasonally-differenced series
	nsDiffed := nsLevels[d]

	// Compute intermediate seasonally-differenced series
	// sLevels[0] = nsDiffed, sLevels[i] = i-th seasonal diff of nsDiffed
	sLevels := make([][]float64, sd+1)
	sLevels[0] = nsDiffed
	for i := 1; i <= sd; i++ {
		prev := sLevels[i-1]
		if len(prev) <= period {
			break
		}
		sdiff := make([]float64, len(prev)-period)
		for j := period; j < len(prev); j++ {
			sdiff[j-period] = prev[j] - prev[j-period]
		}
		sLevels[i] = sdiff
	}

	// Step 1: Undo seasonal differencing (innermost to outermost)
	for i := sd - 1; i >= 0; i-- {
		refSeries := sLevels[i]
		nRef := len(refSeries)
		for j := 0; j < len(result); j++ {
			if j < period {
				idx := nRef - period + j
				if idx >= 0 && idx < nRef {
					result[j] += refSeries[idx]
				}
			} else {
				result[j] += result[j-period]
			}
		}
	}

	// Step 2: Undo non-seasonal differencing (innermost to outermost)
	for i := d - 1; i >= 0; i-- {
		lastVal := nsLevels[i][len(nsLevels[i])-1]
		for j := 0; j < len(result); j++ {
			if j == 0 {
				result[j] += lastVal
			} else {
				result[j] += result[j-1]
			}
		}
	}

	return result
}

// Residuals returns the model residuals.
func (m *Model) Residuals() []float64 {
	if !m.fitted {
		return nil
	}
	result := make([]float64, len(m.residuals))
	copy(result, m.residuals)
	return result
}

// FittedValues returns the fitted values.
func (m *Model) FittedValues() []float64 {
	if !m.fitted {
		return nil
	}
	result := make([]float64, len(m.fittedVals))
	copy(result, m.fittedVals)
	return result
}

// Summary represents a model summary.
type Summary struct {
	Order        Order
	ARCoeffs     []float64
	MACoeffs     []float64
	SARCoeffs    []float64
	SMACoeffs    []float64
	ARStdErrors  []float64 // Standard errors for AR coefficients
	MAStdErrors  []float64 // Standard errors for MA coefficients
	SARStdErrors []float64 // Standard errors for seasonal AR coefficients
	SMAStdErrors []float64 // Standard errors for seasonal MA coefficients
	Intercept    float64
	Variance     float64
	AIC          float64
	AICc         float64 // Corrected AIC
	BIC          float64
	LogLik       float64
	NObs         int
	LjungBox     *stats.LjungBoxResult
}

// Summary returns a summary of the fitted model.
func (m *Model) Summary() *Summary {
	if !m.fitted {
		return nil
	}

	residSeries := timeseries.New(m.residuals)
	lb := stats.LjungBox(residSeries, 10, m.Order.P+m.Order.Q+m.Order.SP+m.Order.SQ)

	return &Summary{
		Order:        m.Order,
		ARCoeffs:     m.ARCoeffs,
		MACoeffs:     m.MACoeffs,
		SARCoeffs:    m.SARCoeffs,
		SMACoeffs:    m.SMACoeffs,
		ARStdErrors:  m.ARStdErrors,
		MAStdErrors:  m.MAStdErrors,
		SARStdErrors: m.SARStdErrors,
		SMAStdErrors: m.SMAStdErrors,
		Intercept:    m.Intercept,
		Variance:     m.Variance,
		AIC:          m.AIC,
		AICc:         m.AICc,
		BIC:          m.BIC,
		LogLik:       m.LogLik,
		NObs:         len(m.data.Values),
		LjungBox:     lb,
	}
}

// initARCoeffs initializes AR coefficients from ACF.
func initARCoeffs(acf []float64, order int) []float64 {
	coeffs := make([]float64, order)
	for i := 0; i < order && i+1 < len(acf); i++ {
		coeffs[i] = acf[i+1] * 0.5
	}
	return coeffs
}

func clamp(v, lower, upper float64) float64 { //nolint:unparam // lower is always -0.99 currently but may vary
	if v < lower {
		return lower
	}
	if v > upper {
		return upper
	}
	return v
}
