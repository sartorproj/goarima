// Package stats provides statistical tests and analysis functions for time series.
//
// This package includes stationarity tests, autocorrelation functions, and
// diagnostic tests for ARIMA model validation.
//
// # Stationarity Tests
//
// Test whether a time series is stationary:
//
//	// Augmented Dickey-Fuller test
//	// H0: Series has unit root (non-stationary)
//	adf := stats.ADF(series, 0)
//	fmt.Printf("ADF: stat=%.4f, p=%.4f, stationary=%v\n",
//	    adf.Statistic, adf.PValue, adf.IsStationary)
//
//	// KPSS test (recommended)
//	// H0: Series is stationary
//	kpss := stats.KPSS(series, "c", 0)
//	fmt.Printf("KPSS: stat=%.4f, p=%.4f, stationary=%v\n",
//	    kpss.Statistic, kpss.PValue, kpss.IsStationary)
//
//	// Phillips-Perron test
//	pp := stats.PhillipsPerron(series, 0)
//
// # Differencing Analysis
//
// Determine optimal differencing orders:
//
//	// Number of first differences needed
//	d := stats.NDiffs(series, 2, "kpss")
//
//	// Number of seasonal differences needed (for seasonal data)
//	sd := stats.NSDiffs(series, 12, 1)  // period=12 for monthly data
//
// # Autocorrelation Functions
//
// Analyze autocorrelation patterns:
//
//	// Autocorrelation Function
//	acf := stats.ACF(series, 20)
//
//	// Partial Autocorrelation Function
//	pacf := stats.PACF(series, 20)
//
//	// ACF with confidence bounds
//	acfResult := stats.ACFWithConfidence(series, 20)
//	significant := stats.SignificantLags(acfResult.Values, acfResult.ConfBounds)
//
// # Residual Diagnostics
//
// Test residuals for autocorrelation:
//
//	// Ljung-Box test for autocorrelation
//	lb := stats.LjungBox(residuals, 10, p+q)
//	if lb.PValue > 0.05 {
//	    // Residuals are white noise (good)
//	}
//
//	// Box-Pierce test
//	bp := stats.BoxPierce(residuals, 10, p+q)
//
//	// Durbin-Watson test
//	dw := stats.DurbinWatson(residuals.Values)
//
// # Time Series Decomposition
//
// Decompose time series into components:
//
//	// Classical decomposition
//	decomp := stats.Decompose(series, 12, "additive")
//	// decomp.Trend, decomp.Seasonal, decomp.Residual
//
//	// STL decomposition (more robust)
//	stl := stats.STL(series, 12, 2)
package stats
