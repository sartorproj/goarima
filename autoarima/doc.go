// Package autoarima implements automatic ARIMA model selection.
//
// Auto-ARIMA automatically selects the best ARIMA or SARIMA model by searching
// through combinations of model orders and selecting based on information criteria.
//
// # Basic Usage
//
// Automatic non-seasonal model selection:
//
//	config := autoarima.DefaultConfig()
//	result, err := autoarima.AutoARIMA(series, config)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	fmt.Printf("Best model: ARIMA(%d,%d,%d)\n",
//	    result.P, result.D, result.Q)
//	fmt.Printf("AIC: %.2f, Models evaluated: %d\n",
//	    result.AIC, result.ModelsEvaluated)
//
//	forecasts, _ := result.Predict(10)
//
// # Seasonal Model Selection
//
// For seasonal data, enable seasonality in the configuration:
//
//	config := autoarima.DefaultConfig()
//	config.Seasonal = true
//	config.SeasonalM = 12  // Monthly data with yearly seasonality
//
//	result, _ := autoarima.AutoARIMA(series, config)
//	fmt.Printf("Best model: SARIMA(%d,%d,%d)(%d,%d,%d)[%d]\n",
//	    result.P, result.D, result.Q,
//	    result.SP, result.SD, result.SQ, result.M)
//
// # Configuration Options
//
// Customize the search with Config:
//
//	config := &autoarima.Config{
//	    MaxP:        3,        // Maximum AR order
//	    MaxD:        2,        // Maximum differencing order
//	    MaxQ:        3,        // Maximum MA order
//	    MaxSP:       2,        // Maximum seasonal AR order
//	    MaxSD:       1,        // Maximum seasonal differencing
//	    MaxSQ:       2,        // Maximum seasonal MA order
//	    Seasonal:    true,     // Enable seasonal search
//	    SeasonalM:   12,       // Seasonal period
//	    Criterion:   "aicc",   // "aic", "aicc", or "bic"
//	    Stepwise:    true,     // Use stepwise search
//	    StationTest: "kpss",   // "adf" or "kpss"
//	}
//
// # Search Methods
//
// Two search methods are available:
//   - Stepwise (default): Fast search using Hyndman-Khandakar algorithm
//   - Grid: Exhaustive search over all combinations (set Stepwise=false)
//
// Stepwise search is recommended for most use cases as it's faster while
// typically finding a good model.
package autoarima
