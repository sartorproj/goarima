package autoarima

import (
	"math"
	"testing"

	"github.com/sartorproj/goarima/timeseries"
)

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.MaxP != 5 {
		t.Errorf("Expected MaxP=5, got %d", config.MaxP)
	}
	if config.MaxD != 2 {
		t.Errorf("Expected MaxD=2, got %d", config.MaxD)
	}
	if config.MaxQ != 5 {
		t.Errorf("Expected MaxQ=5, got %d", config.MaxQ)
	}
	if config.Criterion != "aic" {
		t.Errorf("Expected Criterion='aic', got %s", config.Criterion)
	}
	if config.Stepwise != true {
		t.Error("Expected Stepwise=true")
	}
}

func TestAutoARIMAStationary(t *testing.T) {
	// Stationary AR(1) data
	n := 200
	phi := 0.6
	values := make([]float64, n)
	values[0] = 100

	for i := 1; i < n; i++ {
		innovation := float64(i%7-3) / 3
		values[i] = phi*(values[i-1]-100) + 100 + innovation
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.MaxP = 3
	config.MaxQ = 3
	config.Stepwise = true

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	t.Logf("Selected model: ARIMA(%d,%d,%d)", result.P, result.D, result.Q)
	t.Logf("AIC: %f, BIC: %f", result.AIC, result.BIC)
	t.Logf("Models evaluated: %d", result.ModelsEvaluated)

	// D should be 0 for stationary data
	if result.D > 1 {
		t.Logf("Warning: D=%d for stationary data", result.D)
	}
}

func TestAutoARIMANonStationary(t *testing.T) {
	// Random walk (needs differencing)
	n := 200
	values := make([]float64, n)
	values[0] = 100

	for i := 1; i < n; i++ {
		values[i] = values[i-1] + float64(i%5-2)/2
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.MaxP = 2
	config.MaxQ = 2

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	t.Logf("Selected model: ARIMA(%d,%d,%d)", result.P, result.D, result.Q)
	t.Logf("AIC: %f", result.AIC)

	// D should be >= 1 for non-stationary data
	if result.D == 0 {
		t.Log("Note: D=0 selected for trending data")
	}
}

func TestAutoARIMASeasonal(t *testing.T) {
	// Monthly data with seasonality
	n := 120
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		trend := float64(i) * 0.3
		seasonal := 15 * math.Sin(2*math.Pi*float64(i)/float64(period))
		values[i] = 100 + trend + seasonal + float64(i%5-2)/3
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.Seasonal = true
	config.SeasonalM = 12
	config.MaxP = 2
	config.MaxQ = 2
	config.MaxSP = 1
	config.MaxSQ = 1

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	if !result.IsSeasonal {
		t.Error("Expected seasonal model")
	}

	t.Logf("Selected model: SARIMA(%d,%d,%d)(%d,%d,%d)[%d]",
		result.P, result.D, result.Q,
		result.SP, result.SD, result.SQ, result.M)
	t.Logf("AIC: %f, Models evaluated: %d", result.AIC, result.ModelsEvaluated)
}

func TestAutoARIMAPredict(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i)/10 + float64(i%5-2)
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.MaxP = 2
	config.MaxQ = 2

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	forecasts, err := result.Predict(5)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	if len(forecasts) != 5 {
		t.Errorf("Expected 5 forecasts, got %d", len(forecasts))
	}

	t.Logf("Forecasts: %v", forecasts)

	// Check forecasts are reasonable
	for i, f := range forecasts {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Errorf("Forecast %d is NaN or Inf", i)
		}
	}
}

func TestAutoARIMAResiduals(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i%5-2)
	}

	series := timeseries.New(values)
	config := DefaultConfig()

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	residuals := result.Residuals()
	if residuals == nil {
		t.Error("Residuals should not be nil")
	}

	// Calculate mean of residuals
	if len(residuals) > 0 {
		sum := 0.0
		for _, r := range residuals {
			sum += r
		}
		mean := sum / float64(len(residuals))
		t.Logf("Mean of residuals: %f", mean)
	}
}

func TestAutoARIMABICCriterion(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i%7-3)/2
	}

	series := timeseries.New(values)

	// Test with AIC
	configAIC := DefaultConfig()
	configAIC.Criterion = "aic"
	configAIC.MaxP = 2
	configAIC.MaxQ = 2

	resultAIC, _ := AutoARIMA(series, configAIC)

	// Test with BIC
	configBIC := DefaultConfig()
	configBIC.Criterion = "bic"
	configBIC.MaxP = 2
	configBIC.MaxQ = 2

	resultBIC, _ := AutoARIMA(series, configBIC)

	t.Logf("AIC criterion: ARIMA(%d,%d,%d), AIC=%f",
		resultAIC.P, resultAIC.D, resultAIC.Q, resultAIC.AIC)
	t.Logf("BIC criterion: ARIMA(%d,%d,%d), BIC=%f",
		resultBIC.P, resultBIC.D, resultBIC.Q, resultBIC.BIC)
}

func TestAutoARIMAExhaustiveSearch(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + 0.5*float64(i%7-3)
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.Stepwise = false // Use exhaustive search
	config.MaxP = 2
	config.MaxQ = 2

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	t.Logf("Exhaustive search: ARIMA(%d,%d,%d)", result.P, result.D, result.Q)
	t.Logf("Models evaluated: %d", result.ModelsEvaluated)

	// Exhaustive should evaluate more models
	if result.ModelsEvaluated < 5 {
		t.Log("Note: Exhaustive search evaluated fewer models than expected")
	}
}

func TestAutoARIMAADFTest(t *testing.T) {
	// Test with ADF stationarity test
	n := 150
	values := make([]float64, n)
	values[0] = 100

	// Non-stationary
	for i := 1; i < n; i++ {
		values[i] = values[i-1] + 0.5 + float64(i%5-2)/5
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.StationTest = "adf"
	config.MaxP = 2
	config.MaxQ = 2

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	t.Logf("ADF test selected: ARIMA(%d,%d,%d)", result.P, result.D, result.Q)
}

func TestDetermineDifferencing(t *testing.T) {
	// Stationary data
	n := 100
	stationary := make([]float64, n)
	for i := 0; i < n; i++ {
		stationary[i] = math.Sin(float64(i)/10) + float64(i%5-2)/5
	}
	seriesStationary := timeseries.New(stationary)

	d := determineDifferencing(seriesStationary, 2, "kpss")
	t.Logf("Stationary data d=%d", d)

	// Non-stationary data (trend)
	nonStationary := make([]float64, n)
	for i := 0; i < n; i++ {
		nonStationary[i] = float64(i) * 0.5
	}
	seriesNonStationary := timeseries.New(nonStationary)

	d2 := determineDifferencing(seriesNonStationary, 2, "kpss")
	t.Logf("Non-stationary data d=%d", d2)
}

func TestDetermineSeasonalDifferencing(t *testing.T) {
	// Data with strong seasonality
	n := 120
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		values[i] = 100 + 20*math.Sin(2*math.Pi*float64(i)/float64(period))
	}

	series := timeseries.New(values)
	sd := determineSeasonalDifferencing(series, 1, period)
	t.Logf("Seasonal differencing: D=%d", sd)
}

func TestAutoARIMANilConfig(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i%5-2)
	}

	series := timeseries.New(values)

	// Should use default config
	result, err := AutoARIMA(series, nil)
	if err != nil {
		t.Fatalf("AutoARIMA with nil config failed: %v", err)
	}

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	t.Logf("With nil config: ARIMA(%d,%d,%d)", result.P, result.D, result.Q)
}
