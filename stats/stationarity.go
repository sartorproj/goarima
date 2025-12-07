package stats

import (
	"math"

	"github.com/sartorproj/goarima/timeseries"
)

// ADFResult represents the result of an Augmented Dickey-Fuller test.
type ADFResult struct {
	Statistic    float64
	PValue       float64
	Lags         int
	NObs         int
	CriticalVals map[string]float64 // Critical values at 1%, 5%, 10%
	IsStationary bool
}

// ADF performs the Augmented Dickey-Fuller test for unit root.
// The null hypothesis is that the series has a unit root (is non-stationary).
// If p-value < 0.05, we reject the null and conclude the series is stationary.
func ADF(series *timeseries.Series, maxLag int) *ADFResult {
	n := series.Len()
	if n < 10 {
		return nil
	}

	// Use default lag selection (floor of (n-1)^(1/3))
	if maxLag <= 0 {
		maxLag = int(math.Floor(math.Pow(float64(n-1), 1.0/3.0)))
	}
	if maxLag >= n-1 {
		maxLag = n - 2
	}

	// Calculate first difference
	diff := series.Diff()

	// Build regression: delta_y_t = alpha + beta*y_{t-1} + sum(gamma_i * delta_y_{t-i}) + epsilon
	// We're testing if beta = 0 (unit root) vs beta < 0 (stationary)

	nObs := n - maxLag - 1
	if nObs < 10 {
		return nil
	}

	// Prepare matrices for OLS
	y := make([]float64, nObs)
	x := make([][]float64, nObs)

	for i := 0; i < nObs; i++ {
		t := i + maxLag
		y[i] = diff.Values[t]

		// x[i] = [1, y_{t-1}, delta_y_{t-1}, ..., delta_y_{t-maxLag}]
		x[i] = make([]float64, 2+maxLag)
		x[i][0] = 1                // constant
		x[i][1] = series.Values[t] // lagged level
		for j := 1; j <= maxLag; j++ {
			x[i][1+j] = diff.Values[t-j] // lagged differences
		}
	}

	// Perform OLS regression
	coeffs, se := olsRegression(x, y)
	if coeffs == nil || se == nil || len(coeffs) < 2 || len(se) < 2 {
		return nil
	}

	// Test statistic is t-stat for the lagged level coefficient
	tStat := coeffs[1] / se[1]

	// Critical values for ADF test (with constant, no trend)
	// These are approximate critical values
	criticalVals := map[string]float64{
		"1%":  -3.43,
		"5%":  -2.86,
		"10%": -2.57,
	}

	// Approximate p-value using MacKinnon approximation
	pValue := mackinnonPValue(tStat, n, "c")

	isStationary := pValue < 0.05

	return &ADFResult{
		Statistic:    tStat,
		PValue:       pValue,
		Lags:         maxLag,
		NObs:         nObs,
		CriticalVals: criticalVals,
		IsStationary: isStationary,
	}
}

// KPSSResult represents the result of a KPSS test.
type KPSSResult struct {
	Statistic    float64
	PValue       float64
	Lags         int
	CriticalVals map[string]float64
	IsStationary bool
}

// KPSS performs the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
// The null hypothesis is that the series is stationary.
// If p-value < 0.05, we reject the null and conclude the series is non-stationary.
func KPSS(series *timeseries.Series, regression string, nlags int) *KPSSResult {
	n := series.Len()
	if n < 10 {
		return nil
	}

	// Default lag selection
	if nlags <= 0 {
		nlags = int(math.Ceil(12 * math.Pow(float64(n)/100, 0.25)))
	}

	// Detrend: remove mean (or trend)
	mean := series.Mean()
	residuals := make([]float64, n)

	if regression == "ct" {
		// Remove constant and trend
		// Simple linear detrending: y = a + b*t + residual
		sumT := 0.0
		sumY := 0.0
		sumTY := 0.0
		sumT2 := 0.0
		for i, v := range series.Values {
			t := float64(i)
			sumT += t
			sumY += v
			sumTY += t * v
			sumT2 += t * t
		}
		nf := float64(n)
		b := (nf*sumTY - sumT*sumY) / (nf*sumT2 - sumT*sumT)
		a := (sumY - b*sumT) / nf

		for i, v := range series.Values {
			residuals[i] = v - a - b*float64(i)
		}
	} else {
		// Just remove mean (constant only)
		for i, v := range series.Values {
			residuals[i] = v - mean
		}
	}

	// Calculate partial sums
	cumSum := make([]float64, n)
	cumSum[0] = residuals[0]
	for i := 1; i < n; i++ {
		cumSum[i] = cumSum[i-1] + residuals[i]
	}

	// Calculate s2 (long-run variance estimator using Newey-West)
	s2 := 0.0
	for _, r := range residuals {
		s2 += r * r
	}
	s2 /= float64(n)

	// Add autocovariance terms with Bartlett weights
	for l := 1; l <= nlags; l++ {
		cov := 0.0
		for i := l; i < n; i++ {
			cov += residuals[i] * residuals[i-l]
		}
		cov /= float64(n)
		weight := 1.0 - float64(l)/float64(nlags+1)
		s2 += 2 * weight * cov
	}

	if s2 <= 0 {
		s2 = 1e-10 // Prevent division by zero
	}

	// Calculate KPSS statistic
	etaSq := 0.0
	for _, cs := range cumSum {
		etaSq += cs * cs
	}
	kpssStat := etaSq / (float64(n) * float64(n) * s2)

	// Critical values depend on regression type
	var criticalVals map[string]float64
	if regression == "ct" {
		criticalVals = map[string]float64{
			"10%": 0.119,
			"5%":  0.146,
			"1%":  0.216,
		}
	} else {
		criticalVals = map[string]float64{
			"10%": 0.347,
			"5%":  0.463,
			"1%":  0.739,
		}
	}

	// Approximate p-value
	pValue := kpssPValue(kpssStat, regression)

	// For KPSS, null is stationary, so stationary if we don't reject null
	isStationary := pValue >= 0.05

	return &KPSSResult{
		Statistic:    kpssStat,
		PValue:       pValue,
		Lags:         nlags,
		CriticalVals: criticalVals,
		IsStationary: isStationary,
	}
}

// PhillipsPerronResult represents the result of a Phillips-Perron test.
type PhillipsPerronResult struct {
	Statistic    float64
	PValue       float64
	Lags         int
	CriticalVals map[string]float64
	IsStationary bool
}

// PhillipsPerron performs the Phillips-Perron test for unit root.
// Similar to ADF but handles serial correlation differently.
func PhillipsPerron(series *timeseries.Series, nlags int) *PhillipsPerronResult {
	n := series.Len()
	if n < 10 {
		return nil
	}

	// Default lag selection
	if nlags <= 0 {
		nlags = int(math.Floor(4 * math.Pow(float64(n)/100, 0.25)))
	}

	// Calculate first difference
	diff := series.Diff()

	// Run OLS: delta_y_t = alpha + beta * y_{t-1} + epsilon
	nObs := n - 1
	y := diff.Values
	x := make([][]float64, nObs)
	for i := 0; i < nObs; i++ {
		x[i] = []float64{1, series.Values[i]}
	}

	coeffs, se := olsRegression(x, y)
	if coeffs == nil || se == nil || len(coeffs) < 2 || len(se) < 2 {
		return nil
	}

	// Calculate residuals
	residuals := make([]float64, nObs)
	for i := 0; i < nObs; i++ {
		residuals[i] = y[i] - coeffs[0] - coeffs[1]*x[i][1]
	}

	// Calculate s^2 (variance of residuals)
	s2 := 0.0
	for _, r := range residuals {
		s2 += r * r
	}
	s2 /= float64(nObs - 2)

	// Calculate lambda^2 (long-run variance)
	gamma0 := 0.0
	for _, r := range residuals {
		gamma0 += r * r
	}
	gamma0 /= float64(nObs)

	lambda2 := gamma0
	for l := 1; l <= nlags; l++ {
		gammaL := 0.0
		for i := l; i < nObs; i++ {
			gammaL += residuals[i] * residuals[i-l]
		}
		gammaL /= float64(nObs)
		weight := 1.0 - float64(l)/float64(nlags+1)
		lambda2 += 2 * weight * gammaL
	}

	// Calculate PP statistic
	tStat := coeffs[1] / se[1]

	// Calculate sum of squared x_{t-1} deviations
	xMean := 0.0
	for i := 0; i < nObs; i++ {
		xMean += x[i][1]
	}
	xMean /= float64(nObs)

	sumXDev2 := 0.0
	for i := 0; i < nObs; i++ {
		diff := x[i][1] - xMean
		sumXDev2 += diff * diff
	}

	// PP correction
	correction := 0.0
	if lambda2 > 0 && s2 > 0 {
		correction = (lambda2 - gamma0) * math.Sqrt(float64(nObs)) / (2 * math.Sqrt(lambda2) * math.Sqrt(sumXDev2))
	}

	ppStat := math.Sqrt(gamma0/lambda2)*tStat - correction

	criticalVals := map[string]float64{
		"1%":  -3.43,
		"5%":  -2.86,
		"10%": -2.57,
	}

	pValue := mackinnonPValue(ppStat, n, "c")
	isStationary := pValue < 0.05

	return &PhillipsPerronResult{
		Statistic:    ppStat,
		PValue:       pValue,
		Lags:         nlags,
		CriticalVals: criticalVals,
		IsStationary: isStationary,
	}
}

// olsRegression performs ordinary least squares regression.
// Returns coefficients and their standard errors.
func olsRegression(x [][]float64, y []float64) (coeffs, stdErrors []float64) {
	n := len(y)
	if n == 0 || len(x) != n {
		return nil, nil
	}

	k := len(x[0]) // number of regressors

	// Build X'X and X'y
	xtx := make([][]float64, k)
	for i := range xtx {
		xtx[i] = make([]float64, k)
	}

	xty := make([]float64, k)

	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
			xty[j] += x[i][j] * y[i]
			for l := 0; l < k; l++ {
				xtx[j][l] += x[i][j] * x[i][l]
			}
		}
	}

	// Invert X'X using Cholesky decomposition or direct inverse for small matrices
	xtxInv := invertMatrix(xtx)
	if xtxInv == nil {
		return nil, nil
	}

	// Calculate coefficients: beta = (X'X)^-1 X'y
	coeffs = make([]float64, k)
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			coeffs[i] += xtxInv[i][j] * xty[j]
		}
	}

	// Calculate residuals and SSE
	sse := 0.0
	for i := 0; i < n; i++ {
		pred := 0.0
		for j := 0; j < k; j++ {
			pred += coeffs[j] * x[i][j]
		}
		residual := y[i] - pred
		sse += residual * residual
	}

	// Calculate standard errors
	if n <= k {
		return coeffs, nil
	}

	s2 := sse / float64(n-k)
	stdErrors = make([]float64, k)
	for i := 0; i < k; i++ {
		stdErrors[i] = math.Sqrt(s2 * xtxInv[i][i])
	}

	return coeffs, stdErrors
}

// invertMatrix inverts a square matrix using Gauss-Jordan elimination.
func invertMatrix(m [][]float64) [][]float64 {
	n := len(m)
	if n == 0 {
		return nil
	}

	// Create augmented matrix [A|I]
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		copy(aug[i][:n], m[i])
		aug[i][n+i] = 1
	}

	// Forward elimination
	for i := 0; i < n; i++ {
		// Find pivot
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(aug[k][i]) > math.Abs(aug[maxRow][i]) {
				maxRow = k
			}
		}
		aug[i], aug[maxRow] = aug[maxRow], aug[i]

		if math.Abs(aug[i][i]) < 1e-10 {
			return nil // Singular matrix
		}

		// Scale pivot row
		pivot := aug[i][i]
		for j := 0; j < 2*n; j++ {
			aug[i][j] /= pivot
		}

		// Eliminate column
		for k := 0; k < n; k++ {
			if k != i {
				factor := aug[k][i]
				for j := 0; j < 2*n; j++ {
					aug[k][j] -= factor * aug[i][j]
				}
			}
		}
	}

	// Extract inverse
	result := make([][]float64, n)
	for i := 0; i < n; i++ {
		result[i] = make([]float64, n)
		copy(result[i], aug[i][n:])
	}

	return result
}

// mackinnonPValue approximates p-value for ADF/PP test using MacKinnon response surface.
// The nobs and regression parameters are reserved for future improvements using
// MacKinnon's response surface regression for finite sample corrections.
func mackinnonPValue(stat float64, _ int, _ string) float64 {
	// Simplified approximation based on MacKinnon (1994)
	// For "c" (constant only) regression

	// Asymptotic critical values interpolation
	switch {
	case stat < -3.96:
		return 0.001
	case stat < -3.43:
		return 0.01
	case stat < -2.86:
		return 0.05
	case stat < -2.57:
		return 0.10
	case stat < -1.94:
		return 0.25
	case stat < -1.62:
		return 0.50
	default:
		// Linear interpolation towards 1
		return math.Min(0.5+(stat+1.62)*0.25, 0.99)
	}
}

// kpssPValue approximates p-value for KPSS test.
func kpssPValue(stat float64, regression string) float64 {
	// Critical values for level stationarity (c)
	// 10%: 0.347, 5%: 0.463, 2.5%: 0.574, 1%: 0.739

	if regression == "ct" {
		// Trend stationarity
		switch {
		case stat > 0.216:
			return 0.01
		case stat > 0.146:
			return 0.05
		case stat > 0.119:
			return 0.10
		default:
			return 0.10 + (0.119-stat)*2
		}
	}

	// Level stationarity
	switch {
	case stat > 0.739:
		return 0.01
	case stat > 0.463:
		return 0.05
	case stat > 0.347:
		return 0.10
	default:
		return 0.10 + (0.347-stat)*0.5
	}
}
