package stats

import (
	"math"

	"github.com/sartorproj/goarima/timeseries"
)

// DecompositionResult represents the decomposition of a time series.
type DecompositionResult struct {
	Original *timeseries.Series
	Trend    *timeseries.Series
	Seasonal *timeseries.Series
	Residual *timeseries.Series
	Period   int
	Type     string // "additive" or "multiplicative"
}

// Decompose performs seasonal decomposition of a time series.
// Uses classical decomposition with moving average for trend.
// Type can be "additive" (Y = T + S + R) or "multiplicative" (Y = T * S * R).
func Decompose(series *timeseries.Series, period int, decompositionType string) *DecompositionResult {
	n := series.Len()
	if n < 2*period {
		return nil
	}

	if decompositionType != "additive" && decompositionType != "multiplicative" {
		decompositionType = "additive"
	}

	// Step 1: Calculate trend using centered moving average
	trend := calculateTrend(series, period)

	// Step 2: Detrend the series
	detrended := make([]float64, n)
	if decompositionType == "multiplicative" {
		for i := 0; i < n; i++ {
			if !math.IsNaN(trend.Values[i]) && trend.Values[i] != 0 {
				detrended[i] = series.Values[i] / trend.Values[i]
			} else {
				detrended[i] = math.NaN()
			}
		}
	} else {
		for i := 0; i < n; i++ {
			if !math.IsNaN(trend.Values[i]) {
				detrended[i] = series.Values[i] - trend.Values[i]
			} else {
				detrended[i] = math.NaN()
			}
		}
	}

	// Step 3: Calculate seasonal component by averaging within each period
	seasonalPattern := make([]float64, period)
	counts := make([]int, period)

	for i := 0; i < n; i++ {
		if !math.IsNaN(detrended[i]) {
			seasonIdx := i % period
			seasonalPattern[seasonIdx] += detrended[i]
			counts[seasonIdx]++
		}
	}

	for i := 0; i < period; i++ {
		if counts[i] > 0 {
			seasonalPattern[i] /= float64(counts[i])
		}
	}

	// Normalize seasonal component
	if decompositionType == "multiplicative" {
		sum := 0.0
		for _, v := range seasonalPattern {
			sum += v
		}
		mean := sum / float64(period)
		for i := range seasonalPattern {
			seasonalPattern[i] /= mean
		}
	} else {
		sum := 0.0
		for _, v := range seasonalPattern {
			sum += v
		}
		mean := sum / float64(period)
		for i := range seasonalPattern {
			seasonalPattern[i] -= mean
		}
	}

	// Extend seasonal pattern to full series length
	seasonal := make([]float64, n)
	for i := 0; i < n; i++ {
		seasonal[i] = seasonalPattern[i%period]
	}

	// Step 4: Calculate residual
	residual := make([]float64, n)
	if decompositionType == "multiplicative" {
		for i := 0; i < n; i++ {
			if !math.IsNaN(trend.Values[i]) && trend.Values[i] != 0 && seasonal[i] != 0 {
				residual[i] = series.Values[i] / (trend.Values[i] * seasonal[i])
			} else {
				residual[i] = math.NaN()
			}
		}
	} else {
		for i := 0; i < n; i++ {
			if !math.IsNaN(trend.Values[i]) {
				residual[i] = series.Values[i] - trend.Values[i] - seasonal[i]
			} else {
				residual[i] = math.NaN()
			}
		}
	}

	return &DecompositionResult{
		Original: series,
		Trend: &timeseries.Series{
			Values:     trend.Values,
			Timestamps: series.Timestamps,
			Name:       "trend",
		},
		Seasonal: &timeseries.Series{
			Values:     seasonal,
			Timestamps: series.Timestamps,
			Name:       "seasonal",
		},
		Residual: &timeseries.Series{
			Values:     residual,
			Timestamps: series.Timestamps,
			Name:       "residual",
		},
		Period: period,
		Type:   decompositionType,
	}
}

// calculateTrend calculates trend using centered moving average.
func calculateTrend(series *timeseries.Series, period int) *timeseries.Series {
	n := series.Len()
	trend := make([]float64, n)

	// Initialize with NaN
	for i := range trend {
		trend[i] = math.NaN()
	}

	halfPeriod := period / 2

	if period%2 == 0 {
		// Even period: use 2xperiod MA (centered)
		for i := halfPeriod; i < n-halfPeriod; i++ {
			sum := 0.0
			// First and last values get half weight
			sum += series.Values[i-halfPeriod] * 0.5
			sum += series.Values[i+halfPeriod] * 0.5
			for j := i - halfPeriod + 1; j < i+halfPeriod; j++ {
				sum += series.Values[j]
			}
			trend[i] = sum / float64(period)
		}
	} else {
		// Odd period: simple centered MA
		for i := halfPeriod; i < n-halfPeriod; i++ {
			sum := 0.0
			for j := i - halfPeriod; j <= i+halfPeriod; j++ {
				sum += series.Values[j]
			}
			trend[i] = sum / float64(period)
		}
	}

	return &timeseries.Series{
		Values:     trend,
		Timestamps: series.Timestamps,
		Name:       "trend",
	}
}

// STLResult represents the result of STL decomposition.
type STLResult struct {
	Original *timeseries.Series
	Trend    *timeseries.Series
	Seasonal *timeseries.Series
	Residual *timeseries.Series
	Period   int
}

// STL performs Seasonal and Trend decomposition using Loess.
// This is a simplified implementation of the STL algorithm.
func STL(series *timeseries.Series, period int, robustIters int) *STLResult {
	n := series.Len()
	if n < 2*period {
		return nil
	}

	if robustIters < 1 {
		robustIters = 2
	}

	// Initialize components
	trend := make([]float64, n)
	seasonal := make([]float64, n)
	residual := make([]float64, n)
	weights := make([]float64, n)

	for i := range weights {
		weights[i] = 1.0
	}

	// Iterative fitting
	for iter := 0; iter < robustIters; iter++ {
		// Step 1: Detrend
		detrended := make([]float64, n)
		for i := 0; i < n; i++ {
			detrended[i] = series.Values[i] - trend[i]
		}

		// Step 2: Compute seasonal by averaging
		seasonalPattern := make([]float64, period)
		counts := make([]float64, period)
		for i := 0; i < n; i++ {
			idx := i % period
			seasonalPattern[idx] += detrended[i] * weights[i]
			counts[idx] += weights[i]
		}
		for i := 0; i < period; i++ {
			if counts[i] > 0 {
				seasonalPattern[i] /= counts[i]
			}
		}

		// Center seasonal
		meanSeasonal := 0.0
		for _, v := range seasonalPattern {
			meanSeasonal += v
		}
		meanSeasonal /= float64(period)
		for i := range seasonalPattern {
			seasonalPattern[i] -= meanSeasonal
		}

		// Extend seasonal
		for i := 0; i < n; i++ {
			seasonal[i] = seasonalPattern[i%period]
		}

		// Step 3: Deseasonalize and compute trend
		deseasonalized := make([]float64, n)
		for i := 0; i < n; i++ {
			deseasonalized[i] = series.Values[i] - seasonal[i]
		}

		// Smooth deseasonalized data for trend (using weighted moving average)
		trendWindow := period
		if trendWindow%2 == 0 {
			trendWindow++
		}
		halfWindow := trendWindow / 2

		for i := 0; i < n; i++ {
			sum := 0.0
			weightSum := 0.0
			for j := -halfWindow; j <= halfWindow; j++ {
				idx := i + j
				if idx >= 0 && idx < n {
					w := weights[idx] * (1 - math.Abs(float64(j))/float64(halfWindow+1))
					sum += deseasonalized[idx] * w
					weightSum += w
				}
			}
			if weightSum > 0 {
				trend[i] = sum / weightSum
			}
		}

		// Step 4: Compute residual
		for i := 0; i < n; i++ {
			residual[i] = series.Values[i] - trend[i] - seasonal[i]
		}

		// Update weights for robust fitting
		if iter < robustIters-1 {
			// Calculate MAD of residuals
			absResiduals := make([]float64, n)
			for i, r := range residual {
				absResiduals[i] = math.Abs(r)
			}
			h := 6 * median(absResiduals)
			if h > 0 {
				for i := 0; i < n; i++ {
					u := math.Abs(residual[i]) / h
					if u < 1 {
						weights[i] = (1 - u*u) * (1 - u*u)
					} else {
						weights[i] = 0
					}
				}
			}
		}
	}

	return &STLResult{
		Original: series,
		Trend: &timeseries.Series{
			Values:     trend,
			Timestamps: series.Timestamps,
			Name:       "trend",
		},
		Seasonal: &timeseries.Series{
			Values:     seasonal,
			Timestamps: series.Timestamps,
			Name:       "seasonal",
		},
		Residual: &timeseries.Series{
			Values:     residual,
			Timestamps: series.Timestamps,
			Name:       "residual",
		},
		Period: period,
	}
}

// median calculates the median of a slice.
func median(data []float64) float64 {
	n := len(data)
	if n == 0 {
		return 0
	}

	sorted := make([]float64, n)
	copy(sorted, data)

	// Simple insertion sort for small arrays
	for i := 1; i < n; i++ {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] > key {
			sorted[j+1] = sorted[j]
			j--
		}
		sorted[j+1] = key
	}

	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}
