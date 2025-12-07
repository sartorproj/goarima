# GoARIMA

[![CI](https://github.com/sartorproj/goarima/actions/workflows/ci.yml/badge.svg)](https://github.com/sartorproj/goarima/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/sartorproj/goarima)](https://goreportcard.com/report/github.com/sartorproj/goarima)
[![GoDoc](https://pkg.go.dev/badge/github.com/sartorproj/goarima)](https://pkg.go.dev/github.com/sartorproj/goarima)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Go Version](https://img.shields.io/github/go-mod/go-version/sartorproj/goarima)](https://go.dev/)

A comprehensive Go package implementing ARIMA, SARIMA, and Auto-ARIMA time series models with statistical tests, following the methodology from [Forecasting: Principles and Practice](https://otexts.com/fpppy/nbs/09-arima.html).

> **Note**: For general time series analysis, Python libraries like `statsmodels`, `pmdarima`, and `prophet` offer more mature ecosystems with extensive documentation and community support. This Go implementation is designed for scenarios where ARIMA forecasting needs to run natively in Go environments—such as Kubernetes operators, Go microservices, or embedded systems—without the overhead of calling Python or managing inter-process communication.

## Features

- **ARIMA Models**: AutoRegressive Integrated Moving Average models
- **SARIMA Models**: Seasonal ARIMA for data with periodic patterns
- **Auto-ARIMA**: Automatic model selection using information criteria (AIC/AICc/BIC)
- **Statistical Tests**: 
  - Augmented Dickey-Fuller (ADF) test for unit root
  - KPSS test for stationarity
  - Phillips-Perron test for unit root
  - Ljung-Box test for autocorrelation
  - Box-Pierce test for autocorrelation
  - Durbin-Watson test for autocorrelation
- **Differencing Analysis**:
  - `ndiffs()` - Determine optimal number of first differences
  - `nsdiffs()` - Determine optimal number of seasonal differences
- **Correlation Analysis**:
  - Autocorrelation Function (ACF)
  - Partial Autocorrelation Function (PACF)
- **Decomposition**:
  - Classical decomposition (additive/multiplicative)
  - STL decomposition (Seasonal and Trend using Loess)
- **Information Criteria**:
  - AIC (Akaike Information Criterion)
  - AICc (Corrected AIC for small sample sizes)
  - BIC (Bayesian Information Criterion)
- **Data Loading**:
  - CSV file loading with filtering support
  - Multiple date format parsing
  - NA/NaN value handling

## Installation

```bash
go get github.com/sartorproj/goarima
```

## Demo

Run the comprehensive demonstration with real data:

```bash
cd demo

# Run Go demo (fits models on 10 datasets, exports JSON)
go run .

# Visualize with Python Plotly (optional)
python3 -m venv venv
source venv/bin/activate
pip install plotly
python visualize.py

# Open generated charts (one per dataset)
open charts/*.html
```

### Sample Data

The demo includes datasets from the forecasting literature (`demo/data/`):

| Dataset | Frequency | Description |
|---------|-----------|-------------|
| `aus_economy.csv` | Annual | Australian population |
| `aus_production.csv` | Quarterly | Beer, Cement, Electricity, Gas |
| `us_employment.csv` | Monthly | US employment by industry |
| `gafa_stock.csv` | Daily | GAFA stock prices |
| `hsales.csv` | Monthly | US house sales |
| `eggs.csv` | Annual | US eggs per capita |
| `strikes.csv` | Annual | US strikes count |

## Quick Start

### Loading CSV Data

```go
package main

import (
    "fmt"
    "github.com/sartorproj/goarima/timeseries"
)

func main() {
    // Load a specific column from CSV
    series, err := timeseries.LoadCSVColumn("demo/data/aus_production.csv", "Cement")
    if err != nil {
        panic(err)
    }
    
    // Load with filtering (e.g., specific country/series)
    series, err = timeseries.LoadCSVFiltered(
        "demo/data/aus_economy.csv",
        "unique_id", "Australia", "y",
    )
    
    fmt.Printf("Loaded %d observations\n", series.Len())
}
```

### Stationarity Analysis (ndiffs/nsdiffs)

```go
package main

import (
    "fmt"
    "github.com/sartorproj/goarima/stats"
    "github.com/sartorproj/goarima/timeseries"
)

func main() {
    series := timeseries.New(values)
    
    // Determine number of differences needed for stationarity
    // Uses KPSS test by default
    d := stats.NDiffs(series, 2, "kpss")
    fmt.Printf("Recommended first differences: %d\n", d)
    
    // For seasonal data, determine seasonal differencing
    // Uses seasonal strength measure (F_S >= 0.64 suggests differencing)
    sd := stats.NSDiffs(series, 12, 1) // period=12 for monthly data
    fmt.Printf("Recommended seasonal differences: %d\n", sd)
}
```

### Basic ARIMA Model

```go
package main

import (
    "fmt"
    "github.com/sartorproj/goarima/arima"
    "github.com/sartorproj/goarima/timeseries"
)

func main() {
    // Create time series data
    values := []float64{100, 102, 105, 103, 108, 110, 112, 115, 113, 118}
    series := timeseries.New(values)

    // Fit ARIMA(1,1,0) model
    model := arima.New(1, 1, 0)
    err := model.Fit(series)
    if err != nil {
        panic(err)
    }

    // Get model summary with AIC, AICc, BIC
    summary := model.Summary()
    fmt.Printf("AIC: %.2f, AICc: %.2f, BIC: %.2f\n", 
        summary.AIC, summary.AICc, summary.BIC)
    fmt.Printf("AR coefficients: %v\n", summary.ARCoeffs)

    // Make predictions
    forecasts, _ := model.Predict(5)
    fmt.Printf("Forecasts: %v\n", forecasts)
}
```

### SARIMA for Seasonal Data

```go
package main

import (
    "fmt"
    "math"
    "github.com/sartorproj/goarima/sarima"
    "github.com/sartorproj/goarima/timeseries"
)

func main() {
    // Generate monthly data with yearly seasonality
    values := make([]float64, 120) // 10 years
    for i := range values {
        trend := float64(i) * 0.5
        seasonal := 20 * math.Sin(2*math.Pi*float64(i)/12)
        values[i] = 100 + trend + seasonal
    }
    series := timeseries.New(values)

    // Fit SARIMA(1,0,0)(1,1,0)[12] model
    // As recommended in Forecasting: Principles and Practice
    model := sarima.New(1, 0, 0, 1, 1, 0, 12)
    err := model.Fit(series)
    if err != nil {
        panic(err)
    }

    // Predict next 12 months
    forecasts, _ := model.Predict(12)
    fmt.Printf("Next year forecasts: %v\n", forecasts)
}
```

### Automatic Model Selection

```go
package main

import (
    "fmt"
    "github.com/sartorproj/goarima/autoarima"
    "github.com/sartorproj/goarima/timeseries"
)

func main() {
    values := []float64{/* your data */}
    series := timeseries.New(values)

    // Configure auto-ARIMA
    config := autoarima.DefaultConfig()
    config.MaxP = 3
    config.MaxQ = 3
    config.Criterion = "aicc" // Use corrected AIC (recommended)
    config.Stepwise = true    // Use Hyndman-Khandakar stepwise search

    // Find best model
    result, _ := autoarima.AutoARIMA(series, config)
    
    fmt.Printf("Best model: ARIMA(%d,%d,%d)\n", result.P, result.D, result.Q)
    fmt.Printf("AIC: %.2f, Models evaluated: %d\n", result.AIC, result.ModelsEvaluated)

    // Make predictions
    forecasts, _ := result.Predict(5)
    fmt.Println("Forecasts:", forecasts)
}
```

### Seasonal Auto-ARIMA

```go
config := autoarima.DefaultConfig()
config.Seasonal = true
config.SeasonalM = 12  // Monthly data with yearly seasonality
config.MaxSP = 2
config.MaxSQ = 2
config.Criterion = "aicc"

result, _ := autoarima.AutoARIMA(series, config)
fmt.Printf("Best model: SARIMA(%d,%d,%d)(%d,%d,%d)[%d]\n",
    result.P, result.D, result.Q,
    result.SP, result.SD, result.SQ, result.M)
```

## Statistical Tests

### Stationarity Testing

```go
package main

import (
    "fmt"
    "github.com/sartorproj/goarima/stats"
    "github.com/sartorproj/goarima/timeseries"
)

func main() {
    series := timeseries.New(values)

    // Augmented Dickey-Fuller test
    // H0: Series has unit root (non-stationary)
    adf := stats.ADF(series, 0)
    fmt.Printf("ADF Statistic: %.4f, P-Value: %.4f\n", adf.Statistic, adf.PValue)
    fmt.Printf("Is Stationary: %v\n", adf.IsStationary)

    // KPSS test (recommended in Forecasting: Principles and Practice)
    // H0: Series is stationary
    kpss := stats.KPSS(series, "c", 5) // "c" = constant, "ct" = constant + trend
    fmt.Printf("KPSS Statistic: %.4f, P-Value: %.4f\n", kpss.Statistic, kpss.PValue)

    // Phillips-Perron test
    pp := stats.PhillipsPerron(series, 0)
    fmt.Printf("PP Statistic: %.4f, P-Value: %.4f\n", pp.Statistic, pp.PValue)
}
```

### ACF and PACF Analysis

```go
// Calculate ACF with confidence bounds
acf := stats.ACFWithConfidence(series, 20)
fmt.Printf("ACF values: %v\n", acf.Values)
fmt.Printf("95%% Confidence: ±%.4f\n", acf.ConfBounds)

// Find significant lags
significant := stats.SignificantLags(acf.Values, acf.ConfBounds)
fmt.Printf("Significant lags: %v\n", significant)

// Calculate PACF
pacf := stats.PACFWithConfidence(series, 20)
fmt.Printf("PACF values: %v\n", pacf.Values)
```

### Residual Diagnostics

```go
// Ljung-Box test for autocorrelation in residuals
residuals := model.Residuals()
residSeries := timeseries.New(residuals)
lb := stats.LjungBox(residSeries, 10, p+q) // fitdf = p + q

fmt.Printf("Ljung-Box Q: %.4f, P-Value: %.4f\n", lb.Statistic, lb.PValue)
if lb.PValue > 0.05 {
    fmt.Println("No significant autocorrelation in residuals (white noise)")
}

// Durbin-Watson test
dw := stats.DurbinWatson(residuals)
fmt.Printf("Durbin-Watson: %.4f\n", dw.Statistic)
// d ≈ 2: no autocorrelation
// d < 2: positive autocorrelation
// d > 2: negative autocorrelation
```

## Time Series Decomposition

```go
// Classical decomposition
decomp := stats.Decompose(series, 12, "additive") // or "multiplicative"
fmt.Printf("Trend component: %v\n", decomp.Trend.Values)
fmt.Printf("Seasonal component: %v\n", decomp.Seasonal.Values)
fmt.Printf("Residual component: %v\n", decomp.Residual.Values)

// STL decomposition (more robust)
stl := stats.STL(series, 12, 2) // period=12, robustIters=2
fmt.Printf("STL Trend: %v\n", stl.Trend.Values)
```

## Time Series Utilities

```go
import "github.com/sartorproj/goarima/timeseries"

series := timeseries.New(values)

// Basic statistics
mean := series.Mean()
std := series.Std()
variance := series.Variance()
min := series.Min()
max := series.Max()
median := series.Median()

// Transformations
diff := series.Diff()                  // First difference
diff2 := series.DiffN(2)               // Second difference
seasonalDiff := series.SeasonalDiff(12)  // Seasonal difference
logged := series.Log()                 // Log transform
normalized := series.Normalize()       // Z-score normalization
ma := series.MovingAverage(7)          // Moving average

// Slicing and lagging
sliced := series.Slice(10, 50)
lagged := series.Lag(1)
copied := series.Copy()
```

## Model Order Selection Guide

Based on ACF/PACF patterns (from [Forecasting: Principles and Practice](https://otexts.com/fpppy/nbs/09-arima.html)):

| ACF Pattern | PACF Pattern | Suggested Model |
|-------------|--------------|-----------------|
| Cuts off after lag q | Tails off | MA(q) |
| Tails off | Cuts off after lag p | AR(p) |
| Tails off | Tails off | ARMA(p,q) |

For seasonal patterns, examine lags at multiples of the seasonal period.

## Information Criteria

- **AIC (Akaike Information Criterion)**: Balances model fit and complexity
- **AICc (Corrected AIC)**: AIC + 2k(k+1)/(n-k-1), recommended for small samples
- **BIC (Bayesian Information Criterion)**: Penalizes complexity more than AIC

The AICc is preferred for model selection as it corrects for small sample sizes and converges to AIC as n → ∞.

## References

- [Forecasting: Principles and Practice, the Pythonic Way - Chapter 9: ARIMA models](https://otexts.com/fpppy/nbs/09-arima.html)
- [7 Statistical Tests to validate and help to fit ARIMA model](https://towardsdatascience.com/7-statistical-tests-to-validate-and-help-to-fit-arima-model-33c5853e2e93/)
- Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control
- Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice
- Hyndman, R.J., & Khandakar, Y. (2008). Automatic Time Series Forecasting: The forecast Package for R

## License

MIT License
