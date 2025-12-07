// Package goarima provides ARIMA, SARIMA, and Auto-ARIMA time series modeling.
//
// GoARIMA is a comprehensive Go package for time series analysis and forecasting
// using AutoRegressive Integrated Moving Average (ARIMA) models and their seasonal
// variants (SARIMA). It follows the methodology from "Forecasting: Principles and Practice".
//
// # Features
//
//   - ARIMA models for non-seasonal time series
//   - SARIMA models for seasonal time series
//   - Automatic model selection (Auto-ARIMA) using information criteria
//   - Statistical tests for stationarity (ADF, KPSS, Phillips-Perron)
//   - Autocorrelation analysis (ACF, PACF)
//   - Differencing analysis (ndiffs, nsdiffs)
//   - Time series decomposition (classical and STL)
//
// # Quick Start
//
// Fit an ARIMA model:
//
//	series := timeseries.New(values)
//	model := arima.New(1, 1, 0)  // ARIMA(1,1,0)
//	model.Fit(series)
//	forecasts, _ := model.Predict(10)
//
// Use Auto-ARIMA for automatic model selection:
//
//	config := autoarima.DefaultConfig()
//	result, _ := autoarima.AutoARIMA(series, config)
//	forecasts, _ := result.Predict(10)
//
// # Packages
//
// The library is organized into the following packages:
//
//   - arima: Non-seasonal ARIMA models
//   - sarima: Seasonal ARIMA (SARIMA) models
//   - autoarima: Automatic model selection
//   - stats: Statistical tests and analysis functions
//   - timeseries: Time series data structures and utilities
//
// # References
//
//   - Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice
//   - Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control
package goarima
