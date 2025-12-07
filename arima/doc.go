// Package arima implements AutoRegressive Integrated Moving Average (ARIMA) models.
//
// ARIMA models are used for analyzing and forecasting time series data. An ARIMA(p,d,q)
// model combines:
//   - AR(p): AutoRegressive component with p lags
//   - I(d): Integration (differencing) of order d
//   - MA(q): Moving Average component with q lags
//
// # Basic Usage
//
// Create and fit an ARIMA model:
//
//	// Create ARIMA(1,1,0) model
//	model := arima.New(1, 1, 0)
//
//	// Fit the model to data
//	err := model.Fit(series)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Get model summary
//	summary := model.Summary()
//	fmt.Printf("AIC: %.2f, BIC: %.2f\n", summary.AIC, summary.BIC)
//
//	// Generate forecasts
//	forecasts, _ := model.Predict(10)
//
// # Model Selection
//
// Use information criteria (AIC, AICc, BIC) to compare models:
//
//	model1 := arima.New(1, 1, 0)
//	model2 := arima.New(1, 1, 1)
//	model1.Fit(series)
//	model2.Fit(series)
//
//	// Lower AICc is better
//	if model1.AICc < model2.AICc {
//	    // Use model1
//	}
//
// # Residual Analysis
//
// Analyze model residuals to check model adequacy:
//
//	residuals := model.Residuals()
//	// Use stats.LjungBox to test for autocorrelation in residuals
//
// For seasonal data, use the sarima package instead.
// For automatic model selection, use the autoarima package.
package arima
