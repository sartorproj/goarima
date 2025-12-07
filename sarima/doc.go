// Package sarima implements Seasonal ARIMA (SARIMA) models for time series with seasonality.
//
// SARIMA models extend ARIMA to handle seasonal patterns. A SARIMA(p,d,q)(P,D,Q)[m] model includes:
//   - Non-seasonal components: AR(p), I(d), MA(q)
//   - Seasonal components: SAR(P), SI(D), SMA(Q) at seasonal period m
//
// # Basic Usage
//
// Create and fit a SARIMA model for quarterly data (m=4):
//
//	// SARIMA(1,0,0)(1,1,0)[4]
//	model := sarima.New(1, 0, 0, 1, 1, 0, 4)
//
//	err := model.Fit(series)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Generate forecasts for next 8 quarters
//	forecasts, _ := model.Predict(8)
//
// # Common Models
//
// Popular SARIMA configurations:
//
//	// Airline Model: SARIMA(0,1,1)(0,1,1)[12] for monthly data
//	model := sarima.New(0, 1, 1, 0, 1, 1, 12)
//
//	// Quarterly with seasonal AR: SARIMA(1,0,0)(1,1,0)[4]
//	model := sarima.New(1, 0, 0, 1, 1, 0, 4)
//
// # Seasonal Periods
//
// Common seasonal periods:
//   - Monthly data with yearly seasonality: m = 12
//   - Quarterly data: m = 4
//   - Weekly data with yearly seasonality: m = 52
//   - Daily data with weekly seasonality: m = 7
//
// # Model Selection
//
// Use information criteria to select the best model:
//
//	// Compare AICc values (lower is better)
//	fmt.Printf("AIC: %.2f, AICc: %.2f, BIC: %.2f\n",
//	    model.AIC, model.AICc, model.BIC)
//
// For automatic seasonal model selection, use autoarima with Seasonal=true.
package sarima
