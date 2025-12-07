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

// optimizeCSS optimizes parameters using conditional sum of squares.
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

	// Simple iterative refinement
	maxIter := 100
	tolerance := 1e-6
	learningRate := 0.01

	for iter := 0; iter < maxIter; iter++ {
		// Calculate residuals
		residuals := make([]float64, n)
		prevSSE := 0.0

		startIdx := max(p, q)
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
			prevSSE += residuals[t] * residuals[t]
		}

		// Calculate gradients and update parameters
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

		// Update parameters
		for i := 0; i < p; i++ {
			m.ARCoeffs[i] -= learningRate * arGrad[i] / float64(n)
			// Constrain for stationarity (simple bounds)
			m.ARCoeffs[i] = math.Max(-0.99, math.Min(0.99, m.ARCoeffs[i]))
		}

		for i := 0; i < q; i++ {
			m.MACoeffs[i] -= learningRate * maGrad[i] / float64(n)
			// Constrain for invertibility
			m.MACoeffs[i] = math.Max(-0.99, math.Min(0.99, m.MACoeffs[i]))
		}

		// Recalculate SSE
		newSSE := 0.0
		for t := startIdx; t < n; t++ {
			pred := m.Intercept
			for i := 0; i < p && t-i-1 >= 0; i++ {
				pred += m.ARCoeffs[i] * (y[t-i-1] - m.Intercept)
			}
			for i := 0; i < q && t-i-1 >= 0; i++ {
				pred += m.MACoeffs[i] * residuals[t-i-1]
			}
			residuals[t] = y[t] - pred
			newSSE += residuals[t] * residuals[t]
		}

		if math.Abs(prevSSE-newSSE) < tolerance {
			break
		}
	}

	// Calculate final residuals and variance
	m.residuals = make([]float64, n)
	m.fittedVals = make([]float64, n)

	startIdx := max(p, q)
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

	return nil
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

	// AIC = -2*loglik + 2*k
	m.AIC = -2*m.LogLik + 2*float64(k)

	// AICc = AIC + 2*k*(k+1)/(n-k-1) - corrected AIC for small sample sizes
	kf := float64(k)
	nf := float64(n)
	if nf-kf-1 > 0 {
		m.AICc = m.AIC + 2*kf*(kf+1)/(nf-kf-1)
	} else {
		m.AICc = math.Inf(1)
	}

	// BIC = -2*loglik + k*log(n)
	m.BIC = -2*m.LogLik + float64(k)*math.Log(float64(n))
}

// Predict generates forecasts for the specified number of steps ahead.
func (m *Model) Predict(steps int) ([]float64, error) {
	if !m.fitted {
		return nil, errors.New("model must be fitted before prediction")
	}

	if steps < 1 {
		return nil, errors.New("steps must be at least 1")
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
	forecasts := extY[n:]

	// Integrate back to original scale
	if d > 0 {
		forecasts = m.integrate(forecasts)
	}

	return forecasts, nil
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
				result[j] = result[j] + lastVal
			} else {
				result[j] = result[j] + result[j-1]
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
	Order     Order
	ARCoeffs  []float64
	MACoeffs  []float64
	Intercept float64
	Variance  float64
	AIC       float64
	AICc      float64 // Corrected AIC
	BIC       float64
	LogLik    float64
	NObs      int
	LjungBox  *stats.LjungBoxResult
}

// Summary returns a summary of the fitted model.
func (m *Model) Summary() *Summary {
	if !m.fitted {
		return nil
	}

	residSeries := timeseries.New(m.residuals)
	lb := stats.LjungBox(residSeries, 10, m.Order.P+m.Order.Q)

	return &Summary{
		Order:     m.Order,
		ARCoeffs:  m.ARCoeffs,
		MACoeffs:  m.MACoeffs,
		Intercept: m.Intercept,
		Variance:  m.Variance,
		AIC:       m.AIC,
		AICc:      m.AICc,
		BIC:       m.BIC,
		LogLik:    m.LogLik,
		NObs:      len(m.data.Values),
		LjungBox:  lb,
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
	var v float64 = 1 - phi[0]*phi[0]

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
