// Package stats provides statistical tests and functions for time series analysis.
package stats

import (
	"math"

	"github.com/sartorproj/goarima/timeseries"
)

// ACF calculates the Autocorrelation Function for the given series.
// Returns ACF values for lags 0 to maxLag.
func ACF(series *timeseries.Series, maxLag int) []float64 {
	n := series.Len()
	if maxLag >= n {
		maxLag = n - 1
	}
	if maxLag < 0 {
		return nil
	}

	mean := series.Mean()
	variance := 0.0
	for _, v := range series.Values {
		diff := v - mean
		variance += diff * diff
	}

	if variance == 0 {
		return nil
	}

	acf := make([]float64, maxLag+1)
	for k := 0; k <= maxLag; k++ {
		sum := 0.0
		for i := k; i < n; i++ {
			sum += (series.Values[i] - mean) * (series.Values[i-k] - mean)
		}
		acf[k] = sum / variance
	}

	return acf
}

// PACF calculates the Partial Autocorrelation Function using the Durbin-Levinson algorithm.
// Returns PACF values for lags 1 to maxLag.
func PACF(series *timeseries.Series, maxLag int) []float64 {
	n := series.Len()
	if maxLag >= n {
		maxLag = n - 1
	}
	if maxLag < 1 {
		return nil
	}

	acf := ACF(series, maxLag)
	if acf == nil {
		return nil
	}

	pacf := make([]float64, maxLag+1)
	pacf[0] = 1.0 // PACF at lag 0 is always 1

	// Durbin-Levinson algorithm
	phi := make([][]float64, maxLag+1)
	for i := range phi {
		phi[i] = make([]float64, maxLag+1)
	}

	phi[1][1] = acf[1]
	pacf[1] = acf[1]

	for k := 2; k <= maxLag; k++ {
		// Calculate phi[k][k]
		num := acf[k]
		den := 1.0
		for j := 1; j < k; j++ {
			num -= phi[k-1][j] * acf[k-j]
			den -= phi[k-1][j] * acf[j]
		}

		if den == 0 {
			pacf[k] = 0
			continue
		}

		phi[k][k] = num / den
		pacf[k] = phi[k][k]

		// Update phi[k][j] for j < k
		for j := 1; j < k; j++ {
			phi[k][j] = phi[k-1][j] - phi[k][k]*phi[k-1][k-j]
		}
	}

	return pacf
}

// ACFResult represents the result of ACF analysis.
type ACFResult struct {
	Lags       []int
	Values     []float64
	ConfBounds float64 // 95% confidence bounds (±1.96/sqrt(n))
}

// ACFWithConfidence calculates ACF with confidence bounds.
func ACFWithConfidence(series *timeseries.Series, maxLag int) *ACFResult {
	acf := ACF(series, maxLag)
	if acf == nil {
		return nil
	}

	lags := make([]int, len(acf))
	for i := range lags {
		lags[i] = i
	}

	confBound := 1.96 / math.Sqrt(float64(series.Len()))

	return &ACFResult{
		Lags:       lags,
		Values:     acf,
		ConfBounds: confBound,
	}
}

// PACFResult represents the result of PACF analysis.
type PACFResult struct {
	Lags       []int
	Values     []float64
	ConfBounds float64
}

// PACFWithConfidence calculates PACF with confidence bounds.
func PACFWithConfidence(series *timeseries.Series, maxLag int) *PACFResult {
	pacf := PACF(series, maxLag)
	if pacf == nil {
		return nil
	}

	lags := make([]int, len(pacf))
	for i := range lags {
		lags[i] = i
	}

	confBound := 1.96 / math.Sqrt(float64(series.Len()))

	return &PACFResult{
		Lags:       lags,
		Values:     pacf,
		ConfBounds: confBound,
	}
}

// SignificantLags returns the lags where ACF/PACF values exceed confidence bounds.
func SignificantLags(values []float64, confBound float64) []int {
	var significant []int
	for i := 1; i < len(values); i++ { // Skip lag 0
		if math.Abs(values[i]) > confBound {
			significant = append(significant, i)
		}
	}
	return significant
}
