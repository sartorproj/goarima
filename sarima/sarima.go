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

// Model represents a SARIMA model.
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
func New(p, d, q, sp, sd, sq, m int) *Model {
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
	learningRate := 0.005
	momentum := 0.9
	decay := 0.99

	// Momentum terms
	arMomentum := make([]float64, p)
	maMomentum := make([]float64, q)
	sarMomentum := make([]float64, sp)
	smaMomentum := make([]float64, sq)

	// Start index to avoid boundary issues
	startIdx := max(max(p, q), max(sp*period, sq*period))
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
			pred := m.Intercept

			// Non-seasonal AR component
			for i := 0; i < p && t-i-1 >= 0; i++ {
				pred += m.ARCoeffs[i] * (y[t-i-1] - m.Intercept)
			}

			// Seasonal AR component
			for i := 0; i < sp; i++ {
				lag := (i + 1) * period
				if t-lag >= 0 {
					pred += m.SARCoeffs[i] * (y[t-lag] - m.Intercept)
				}
			}

			// Non-seasonal MA component
			for i := 0; i < q && t-i-1 >= 0; i++ {
				pred += m.MACoeffs[i] * residuals[t-i-1]
			}

			// Seasonal MA component
			for i := 0; i < sq; i++ {
				lag := (i + 1) * period
				if t-lag >= 0 {
					pred += m.SMACoeffs[i] * residuals[t-lag]
				}
			}

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

		// Calculate gradients
		arGrad := make([]float64, p)
		maGrad := make([]float64, q)
		sarGrad := make([]float64, sp)
		smaGrad := make([]float64, sq)

		for t := startIdx; t < n; t++ {
			// AR gradients
			for i := 0; i < p && t-i-1 >= 0; i++ {
				arGrad[i] -= 2 * residuals[t] * (y[t-i-1] - m.Intercept)
			}

			// SAR gradients
			for i := 0; i < sp; i++ {
				lag := (i + 1) * period
				if t-lag >= 0 {
					sarGrad[i] -= 2 * residuals[t] * (y[t-lag] - m.Intercept)
				}
			}

			// MA gradients
			for i := 0; i < q && t-i-1 >= 0; i++ {
				maGrad[i] -= 2 * residuals[t] * residuals[t-i-1]
			}

			// SMA gradients
			for i := 0; i < sq; i++ {
				lag := (i + 1) * period
				if t-lag >= 0 {
					smaGrad[i] -= 2 * residuals[t] * residuals[t-lag]
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

		// Convergence check
		if iter > 0 && math.Abs(currentSSE-bestSSE) < tolerance {
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
		pred := m.Intercept

		for i := 0; i < p && t-i-1 >= 0; i++ {
			pred += m.ARCoeffs[i] * (y[t-i-1] - m.Intercept)
		}
		for i := 0; i < sp; i++ {
			lag := (i + 1) * period
			if t-lag >= 0 {
				pred += m.SARCoeffs[i] * (y[t-lag] - m.Intercept)
			}
		}
		for i := 0; i < q && t-i-1 >= 0; i++ {
			pred += m.MACoeffs[i] * m.residuals[t-i-1]
		}
		for i := 0; i < sq; i++ {
			lag := (i + 1) * period
			if t-lag >= 0 {
				pred += m.SMACoeffs[i] * m.residuals[t-lag]
			}
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

	numParams := p + q + sp + sq + 1
	if count > numParams {
		m.Variance = sse / float64(count-numParams)
	} else {
		m.Variance = sse / float64(count)
	}

	return nil
}

// calculateIC calculates AIC, AICc, and BIC.
func (m *Model) calculateIC() {
	n := len(m.residuals)
	k := m.Order.P + m.Order.Q + m.Order.SP + m.Order.SQ + 1

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
	sp := m.Order.SP
	sq := m.Order.SQ
	d := m.Order.D
	sd := m.Order.SD
	period := m.Order.M

	y := m.diffData.Values
	n := len(y)

	// Extended arrays
	extY := make([]float64, n+steps)
	copy(extY, y)

	extResiduals := make([]float64, n+steps)
	copy(extResiduals, m.residuals)

	// Generate forecasts
	for h := 0; h < steps; h++ {
		t := n + h
		pred := m.Intercept

		// Non-seasonal AR
		for i := 0; i < p && t-i-1 >= 0; i++ {
			pred += m.ARCoeffs[i] * (extY[t-i-1] - m.Intercept)
		}

		// Seasonal AR
		for i := 0; i < sp; i++ {
			lag := (i + 1) * period
			if t-lag >= 0 {
				pred += m.SARCoeffs[i] * (extY[t-lag] - m.Intercept)
			}
		}

		// Non-seasonal MA (only past residuals, future = 0)
		for i := 0; i < q && t-i-1 >= 0 && t-i-1 < n; i++ {
			pred += m.MACoeffs[i] * extResiduals[t-i-1]
		}

		// Seasonal MA
		for i := 0; i < sq; i++ {
			lag := (i + 1) * period
			if t-lag >= 0 && t-lag < n {
				pred += m.SMACoeffs[i] * extResiduals[t-lag]
			}
		}

		extY[t] = pred
		extResiduals[t] = 0
	}

	forecasts = make([]float64, steps)
	copy(forecasts, extY[n:])

	// Integrate back
	forecasts = m.integrate(forecasts)

	// Calculate prediction intervals
	// Approximate: variance grows with horizon for integrated series
	z := normalQuantile((1 + confidence) / 2)

	lower = make([]float64, steps)
	upper = make([]float64, steps)

	for h := 0; h < steps; h++ {
		// Base standard error from residual variance
		se := math.Sqrt(m.Variance)

		// Variance grows with horizon for integrated/seasonal-integrated series
		growthFactor := 1.0
		if d > 0 {
			growthFactor *= math.Sqrt(float64(h + 1))
		}
		if sd > 0 && period > 0 {
			seasonalCycles := float64(h/period + 1)
			growthFactor *= math.Sqrt(seasonalCycles)
		}

		se *= growthFactor
		lower[h] = forecasts[h] - z*se
		upper[h] = forecasts[h] + z*se
	}

	return forecasts, lower, upper, nil
}

// normalQuantile returns the z-value for a given probability.
func normalQuantile(p float64) float64 {
	if p <= 0 || p >= 1 {
		return 0
	}
	if p < 0.5 {
		return -normalQuantile(1 - p)
	}

	t := math.Sqrt(-2 * math.Log(1-p))
	c0, c1, c2 := 2.515517, 0.802853, 0.010328
	d1, d2, d3 := 1.432788, 0.189269, 0.001308

	return t - (c0+c1*t+c2*t*t)/(1+d1*t+d2*t*t+d3*t*t*t)
}

// integrate undoes differencing to return forecasts on original scale.
// Differencing in Fit() is: first non-seasonal (d times), then seasonal (sd times).
// Integration order: first undo seasonal, then undo non-seasonal.
func (m *Model) integrate(forecasts []float64) []float64 {
	d := m.Order.D
	sd := m.Order.SD
	period := m.Order.M
	original := m.data.Values
	n := len(original)

	result := make([]float64, len(forecasts))
	copy(result, forecasts)

	// Compute the non-seasonally differenced series (needed for seasonal integration)
	nonSeasonalDiff := original
	for i := 0; i < d; i++ {
		if len(nonSeasonalDiff) <= 1 {
			break
		}
		newDiff := make([]float64, len(nonSeasonalDiff)-1)
		for j := 1; j < len(nonSeasonalDiff); j++ {
			newDiff[j-1] = nonSeasonalDiff[j] - nonSeasonalDiff[j-1]
		}
		nonSeasonalDiff = newDiff
	}

	// Step 1: Undo seasonal differencing
	// Seasonal diff: z_t = y_t - y_{t-m}, so y_t = z_t + y_{t-m}
	// We need the last 'period' values from the non-seasonally differenced series
	if sd > 0 && period > 0 {
		nDiff := len(nonSeasonalDiff)
		for i := 0; i < sd; i++ {
			for j := 0; j < len(result); j++ {
				if j < period {
					// Use non-seasonally differenced original data
					idx := nDiff - period + j
					if idx >= 0 && idx < nDiff {
						result[j] += nonSeasonalDiff[idx]
					}
				} else {
					// Use earlier integrated forecast
					result[j] += result[j-period]
				}
			}
		}
	}

	// Step 2: Undo non-seasonal differencing
	// Non-seasonal diff: y'_t = y_t - y_{t-1}, so y_t = y'_t + y_{t-1}
	// We need to cumsum starting from the last value of original
	for i := 0; i < d; i++ {
		lastVal := original[n-1]
		// For multiple diffs, we need the last value at each integration level
		// Both cases use the same logic: cumsum starting from lastVal
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
