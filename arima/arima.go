// Package arima implements ARIMA (AutoRegressive Integrated Moving Average) models.
package arima

import (
	"errors"
	"math"

	"github.com/sartorproj/goarima/stats"
	"github.com/sartorproj/goarima/timeseries"
)

// Order represents ARIMA model order (p, d, q).
type Order struct {
	P int // AR order (number of autoregressive terms)
	D int // Differencing order
	Q int // MA order (number of moving average terms)
}

// Model represents an ARIMA model.
type Model struct {
	Order      Order
	ARCoeffs   []float64 // AR coefficients (phi)
	MACoeffs   []float64 // MA coefficients (theta)
	Intercept  float64
	Variance   float64 // Residual variance
	AIC        float64
	AICc       float64 // Corrected AIC for small sample sizes
	BIC        float64
	LogLik     float64
	fitted     bool
	data       *timeseries.Series
	diffData   *timeseries.Series
	residuals  []float64
	fittedVals []float64

	// Standard errors for coefficients (computed during fit)
	ARStdErrors []float64
	MAStdErrors []float64
}

// New creates a new ARIMA model with the specified order.
func New(p, d, q int) *Model {
	return &Model{
		Order:    Order{P: p, D: d, Q: q},
		ARCoeffs: make([]float64, p),
		MACoeffs: make([]float64, q),
	}
}

// Fit fits the ARIMA model to the given time series data.
func (m *Model) Fit(series *timeseries.Series) error {
	if series.Len() < m.Order.P+m.Order.Q+m.Order.D+10 {
		return errors.New("insufficient data points for the specified order")
	}

	m.data = series

	// Apply differencing
	diffSeries := series
	for i := 0; i < m.Order.D; i++ {
		diffSeries = diffSeries.Diff()
		if diffSeries.Len() == 0 {
			return errors.New("differencing resulted in empty series")
		}
	}
	m.diffData = diffSeries

	// Fit using Conditional Sum of Squares (CSS) method
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
	q := m.Order.Q

	if p == 0 && q == 0 {
		// Just a white noise model
		mean := 0.0
		for _, v := range y {
			mean += v
		}
		m.Intercept = mean / float64(n)
		m.Variance = 0
		for _, v := range y {
			diff := v - m.Intercept
			m.Variance += diff * diff
		}
		m.Variance /= float64(n - 1)
		m.residuals = make([]float64, n)
		m.fittedVals = make([]float64, n)
		for i, v := range y {
			m.residuals[i] = v - m.Intercept
			m.fittedVals[i] = m.Intercept
		}
		return nil
	}

	// Initialize parameters
	if p > 0 {
		// Use Yule-Walker for initial AR estimates
		acf := stats.ACF(m.diffData, p)
		if acf != nil {
			m.ARCoeffs = yuleWalker(acf, p)
		}
	}

	// Initialize MA coefficients to small values
	for i := range m.MACoeffs {
		m.MACoeffs[i] = 0.1
	}

	// Use iterative optimization (simplified gradient descent)
	err := m.optimizeCSS(y)
	if err != nil {
		return err
	}

	return nil
}

// optimizeCSS optimizes parameters using conditional sum of squares with adaptive learning.
func (m *Model) optimizeCSS(y []float64) error {
	n := len(y)
	p := m.Order.P
	q := m.Order.Q

	// Calculate mean
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(n)
	m.Intercept = mean

	// Adaptive gradient descent with momentum
	maxIter := 200
	tolerance := 1e-8
	learningRate := 0.01
	momentum := 0.9
	decay := 0.99 // Learning rate decay

	// Momentum terms
	arMomentum := make([]float64, p)
	maMomentum := make([]float64, q)

	startIdx := max(p, q)
	bestSSE := math.Inf(1)
	bestARCoeffs := make([]float64, p)
	bestMACoeffs := make([]float64, q)
	noImproveCount := 0

	for iter := 0; iter < maxIter; iter++ {
		// Calculate residuals
		residuals := make([]float64, n)
		currentSSE := 0.0

		for t := startIdx; t < n; t++ {
			pred := m.Intercept

			// AR component
			for i := 0; i < p && t-i-1 >= 0; i++ {
				pred += m.ARCoeffs[i] * (y[t-i-1] - m.Intercept)
			}

			// MA component
			for i := 0; i < q && t-i-1 >= 0; i++ {
				pred += m.MACoeffs[i] * residuals[t-i-1]
			}

			residuals[t] = y[t] - pred
			currentSSE += residuals[t] * residuals[t]
		}

		// Track best solution
		if currentSSE < bestSSE {
			bestSSE = currentSSE
			copy(bestARCoeffs, m.ARCoeffs)
			copy(bestMACoeffs, m.MACoeffs)
			noImproveCount = 0
		} else {
			noImproveCount++
		}

		// Early stopping if no improvement
		if noImproveCount > 20 {
			break
		}

		// Calculate gradients
		arGrad := make([]float64, p)
		maGrad := make([]float64, q)

		for t := startIdx; t < n; t++ {
			// Gradient for AR coefficients
			for i := 0; i < p && t-i-1 >= 0; i++ {
				arGrad[i] -= 2 * residuals[t] * (y[t-i-1] - m.Intercept)
			}

			// Gradient for MA coefficients
			for i := 0; i < q && t-i-1 >= 0; i++ {
				maGrad[i] -= 2 * residuals[t] * residuals[t-i-1]
			}
		}

		// Update with momentum
		for i := 0; i < p; i++ {
			arMomentum[i] = momentum*arMomentum[i] + learningRate*arGrad[i]/float64(n)
			m.ARCoeffs[i] -= arMomentum[i]
			// Constrain for stationarity
			m.ARCoeffs[i] = math.Max(-0.99, math.Min(0.99, m.ARCoeffs[i]))
		}

		for i := 0; i < q; i++ {
			maMomentum[i] = momentum*maMomentum[i] + learningRate*maGrad[i]/float64(n)
			m.MACoeffs[i] -= maMomentum[i]
			// Constrain for invertibility
			m.MACoeffs[i] = math.Max(-0.99, math.Min(0.99, m.MACoeffs[i]))
		}

		// Decay learning rate
		learningRate *= decay

		// Convergence check
		if iter > 0 && math.Abs(currentSSE-bestSSE) < tolerance {
			break
		}
	}

	// Restore best solution
	copy(m.ARCoeffs, bestARCoeffs)
	copy(m.MACoeffs, bestMACoeffs)

	// Calculate final residuals and variance
	m.residuals = make([]float64, n)
	m.fittedVals = make([]float64, n)

	startIdx = max(p, q)
	for t := 0; t < n; t++ {
		if t < startIdx {
			m.fittedVals[t] = m.Intercept
			m.residuals[t] = y[t] - m.fittedVals[t]
			continue
		}

		pred := m.Intercept
		for i := 0; i < p && t-i-1 >= 0; i++ {
			pred += m.ARCoeffs[i] * (y[t-i-1] - m.Intercept)
		}
		for i := 0; i < q && t-i-1 >= 0; i++ {
			pred += m.MACoeffs[i] * m.residuals[t-i-1]
		}

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
	if count > p+q+1 {
		m.Variance = sse / float64(count-p-q-1)
	} else {
		m.Variance = sse / float64(count)
	}

	// Estimate coefficient standard errors using numerical approximation
	m.estimateStdErrors(y)

	return nil
}

// estimateStdErrors estimates standard errors for AR and MA coefficients.
// Uses a numerical approximation of the Hessian matrix.
func (m *Model) estimateStdErrors(y []float64) {
	n := len(y)
	p := m.Order.P
	q := m.Order.Q
	startIdx := max(p, q)

	if p == 0 && q == 0 {
		return
	}

	// Small perturbation for numerical differentiation
	eps := 1e-5

	// Function to compute SSE given coefficients
	computeSSE := func(arCoeffs, maCoeffs []float64) float64 {
		residuals := make([]float64, n)
		sse := 0.0
		for t := startIdx; t < n; t++ {
			pred := m.Intercept
			for i := 0; i < p && t-i-1 >= 0; i++ {
				pred += arCoeffs[i] * (y[t-i-1] - m.Intercept)
			}
			for i := 0; i < q && t-i-1 >= 0; i++ {
				pred += maCoeffs[i] * residuals[t-i-1]
			}
			residuals[t] = y[t] - pred
			sse += residuals[t] * residuals[t]
		}
		return sse
	}

	// Compute approximate standard errors using second derivatives
	m.ARStdErrors = make([]float64, p)
	m.MAStdErrors = make([]float64, q)

	baseSSE := computeSSE(m.ARCoeffs, m.MACoeffs)

	// AR coefficient standard errors
	for i := 0; i < p; i++ {
		// Perturb coefficient
		arPlus := make([]float64, p)
		arMinus := make([]float64, p)
		copy(arPlus, m.ARCoeffs)
		copy(arMinus, m.ARCoeffs)
		arPlus[i] += eps
		arMinus[i] -= eps

		ssePlus := computeSSE(arPlus, m.MACoeffs)
		sseMinus := computeSSE(arMinus, m.MACoeffs)

		// Second derivative approximation: d²SSE/dθ² ≈ (f(θ+ε) - 2f(θ) + f(θ-ε)) / ε²
		hessianDiag := (ssePlus - 2*baseSSE + sseMinus) / (eps * eps)
		if hessianDiag > 0 {
			// SE = sqrt(2 * σ² / H_ii)
			m.ARStdErrors[i] = math.Sqrt(2 * m.Variance / hessianDiag)
		}
	}

	// MA coefficient standard errors
	for i := 0; i < q; i++ {
		maPlus := make([]float64, q)
		maMinus := make([]float64, q)
		copy(maPlus, m.MACoeffs)
		copy(maMinus, m.MACoeffs)
		maPlus[i] += eps
		maMinus[i] -= eps

		ssePlus := computeSSE(m.ARCoeffs, maPlus)
		sseMinus := computeSSE(m.ARCoeffs, maMinus)

		hessianDiag := (ssePlus - 2*baseSSE + sseMinus) / (eps * eps)
		if hessianDiag > 0 {
			m.MAStdErrors[i] = math.Sqrt(2 * m.Variance / hessianDiag)
		}
	}
}

// calculateIC calculates AIC, AICc, and BIC.
func (m *Model) calculateIC() {
	n := len(m.residuals)
	k := m.Order.P + m.Order.Q + 1 // number of parameters (AR + MA + intercept)

	// Log-likelihood (assuming Gaussian errors)
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

	p := m.Order.P
	q := m.Order.Q
	d := m.Order.D

	// Get the differenced series and residuals
	y := m.diffData.Values
	n := len(y)

	// Extended arrays for forecasting
	extY := make([]float64, n+steps)
	copy(extY, y)

	extResiduals := make([]float64, n+steps)
	copy(extResiduals, m.residuals)

	// Compute psi weights (MA representation coefficients) for variance calculation
	psiWeights := m.computePsiWeights(steps)

	// Generate forecasts for differenced series
	for h := 0; h < steps; h++ {
		t := n + h
		pred := m.Intercept

		// AR component
		for i := 0; i < p && t-i-1 >= 0; i++ {
			pred += m.ARCoeffs[i] * (extY[t-i-1] - m.Intercept)
		}

		// MA component (future residuals are 0)
		for i := 0; i < q && t-i-1 >= 0 && t-i-1 < n; i++ {
			pred += m.MACoeffs[i] * extResiduals[t-i-1]
		}

		extY[t] = pred
		extResiduals[t] = 0 // Expected residual is 0
	}

	// Get forecasts on differenced scale
	forecasts = make([]float64, steps)
	copy(forecasts, extY[n:])

	// Calculate prediction variance for each horizon
	// Var(e_{n+h}) = σ² * (1 + ψ₁² + ψ₂² + ... + ψ_{h-1}²)
	predVariance := make([]float64, steps)
	cumPsiSq := 0.0
	for h := 0; h < steps; h++ {
		if h > 0 && h-1 < len(psiWeights) {
			cumPsiSq += psiWeights[h-1] * psiWeights[h-1]
		}
		predVariance[h] = m.Variance * (1 + cumPsiSq)
	}

	// Integrate forecasts back to original scale
	if d > 0 {
		forecasts = m.integrate(forecasts)
	}

	// Calculate prediction intervals
	// z-value for confidence level (approximate)
	z := normalQuantile((1 + confidence) / 2)

	lower = make([]float64, steps)
	upper = make([]float64, steps)
	for h := 0; h < steps; h++ {
		se := math.Sqrt(predVariance[h])
		// For integrated series, variance grows with horizon
		if d > 0 {
			se *= math.Sqrt(float64(h + 1))
		}
		lower[h] = forecasts[h] - z*se
		upper[h] = forecasts[h] + z*se
	}

	return forecasts, lower, upper, nil
}

// computePsiWeights computes the MA(∞) representation weights (psi weights).
// These are used for calculating prediction interval widths.
func (m *Model) computePsiWeights(maxLag int) []float64 {
	p := m.Order.P
	q := m.Order.Q

	psi := make([]float64, maxLag)

	for j := 0; j < maxLag; j++ {
		if j < q {
			psi[j] = m.MACoeffs[j]
		}

		// Add AR contributions: ψ_j = φ₁ψ_{j-1} + φ₂ψ_{j-2} + ... + θ_j
		for i := 0; i < p && i <= j; i++ {
			if j-i-1 >= 0 {
				psi[j] += m.ARCoeffs[i] * psi[j-i-1]
			} else if j-i-1 == -1 {
				psi[j] += m.ARCoeffs[i] // ψ_0 = 1 implicitly
			}
		}
	}

	return psi
}

// normalQuantile returns the z-value for a given probability using approximation.
func normalQuantile(p float64) float64 {
	// Rational approximation for the normal quantile function
	// Abramowitz and Stegun approximation
	if p <= 0 || p >= 1 {
		return 0
	}

	if p < 0.5 {
		return -normalQuantile(1 - p)
	}

	t := math.Sqrt(-2 * math.Log(1-p))

	// Coefficients for approximation
	c0, c1, c2 := 2.515517, 0.802853, 0.010328
	d1, d2, d3 := 1.432788, 0.189269, 0.001308

	return t - (c0+c1*t+c2*t*t)/(1+d1*t+d2*t*t+d3*t*t*t)
}

// integrate undoes differencing to return forecasts on original scale.
func (m *Model) integrate(forecasts []float64) []float64 {
	d := m.Order.D
	original := m.data.Values

	result := make([]float64, len(forecasts))
	copy(result, forecasts)

	// We need to integrate d times
	for i := 0; i < d; i++ {
		// The last value before differencing
		lastVal := original[len(original)-1-i]
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

// Summary returns a summary of the fitted model.
type Summary struct {
	Order       Order
	ARCoeffs    []float64
	MACoeffs    []float64
	ARStdErrors []float64 // Standard errors for AR coefficients
	MAStdErrors []float64 // Standard errors for MA coefficients
	Intercept   float64
	Variance    float64
	AIC         float64
	AICc        float64 // Corrected AIC
	BIC         float64
	LogLik      float64
	NObs        int
	LjungBox    *stats.LjungBoxResult
}

// Summary returns a summary of the fitted model.
func (m *Model) Summary() *Summary {
	if !m.fitted {
		return nil
	}

	residSeries := timeseries.New(m.residuals)
	lb := stats.LjungBox(residSeries, 10, m.Order.P+m.Order.Q)

	return &Summary{
		Order:       m.Order,
		ARCoeffs:    m.ARCoeffs,
		MACoeffs:    m.MACoeffs,
		ARStdErrors: m.ARStdErrors,
		MAStdErrors: m.MAStdErrors,
		Intercept:   m.Intercept,
		Variance:    m.Variance,
		AIC:         m.AIC,
		AICc:        m.AICc,
		BIC:         m.BIC,
		LogLik:      m.LogLik,
		NObs:        len(m.data.Values),
		LjungBox:    lb,
	}
}

// yuleWalker estimates AR coefficients using Yule-Walker equations.
func yuleWalker(acf []float64, order int) []float64 {
	if order <= 0 || len(acf) <= order {
		return nil
	}

	// Build Toeplitz matrix R and vector r
	R := make([][]float64, order)
	r := make([]float64, order)

	for i := 0; i < order; i++ {
		R[i] = make([]float64, order)
		r[i] = acf[i+1]
		for j := 0; j < order; j++ {
			idx := i - j
			if idx < 0 {
				idx = -idx
			}
			R[i][j] = acf[idx]
		}
	}

	// Solve R * phi = r using Levinson-Durbin algorithm
	phi := make([]float64, order)

	// Simple case for AR(1)
	if order == 1 {
		phi[0] = acf[1]
		return phi
	}

	// Levinson-Durbin recursion
	phi[0] = acf[1]
	v := 1 - phi[0]*phi[0]

	for i := 1; i < order; i++ {
		lambda := acf[i+1]
		for j := 0; j < i; j++ {
			lambda -= phi[j] * acf[i-j]
		}
		lambda /= v

		// Update phi
		newPhi := make([]float64, i+1)
		for j := 0; j < i; j++ {
			newPhi[j] = phi[j] - lambda*phi[i-1-j]
		}
		newPhi[i] = lambda
		copy(phi, newPhi)

		v *= (1 - lambda*lambda)
		if v <= 0 {
			break
		}
	}

	return phi
}
