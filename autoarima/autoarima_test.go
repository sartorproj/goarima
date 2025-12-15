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
	if config.Criterion != "aicc" {
		t.Errorf("Expected Criterion='aicc', got %s", config.Criterion)
	}
	if !config.Stepwise {
		t.Error("Expected Stepwise=true")
	}
	if !config.AutoSeasonal {
		t.Error("Expected AutoSeasonal=true by default")
	}
	if config.ModelSelection != "cv" {
		t.Errorf("Expected ModelSelection='cv', got %s", config.ModelSelection)
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
	config.AutoSeasonal = false // Disable for this test

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	t.Logf("Selected model: %s", result.Order())
	t.Logf("AIC: %f, BIC: %f", result.AIC, result.BIC)
	t.Logf("CV RMSE: %f, MAPE: %f%%", result.RMSE, result.MAPE)

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
	config.AutoSeasonal = false

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	t.Logf("Selected model: %s", result.Order())
	t.Logf("AIC: %f", result.AIC)

	// D should be >= 1 for non-stationary data
	if result.D == 0 {
		t.Log("Note: D=0 selected for trending data")
	}
}

func TestAutoARIMAAutoSeasonality(t *testing.T) {
	// Data with clear daily seasonality (period 24)
	n := 168 // 7 days of hourly data
	period := 24
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		trend := float64(i) * 0.01
		seasonal := 15 * math.Sin(2*math.Pi*float64(i)/float64(period))
		values[i] = 100 + trend + seasonal + float64(i%5-2)/3
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.AutoSeasonal = true
	config.SeasonalPeriods = []int{12, 24, 48, 168}
	config.MaxP = 2
	config.MaxQ = 2

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	t.Logf("Detected period: %d (strength: %.4f)", result.DetectedPeriod, result.SeasonalityStrength)
	t.Logf("Detection method: %s", result.DetectionMethod)
	t.Logf("Selected model: %s", result.Order())
	t.Logf("Is seasonal: %v", result.IsSeasonal)
	t.Logf("CV RMSE: %f, MAPE: %f%%", result.RMSE, result.MAPE)

	// Should detect period 24
	if result.DetectedPeriod != 24 {
		t.Logf("Warning: Expected period 24, got %d", result.DetectedPeriod)
	}

	// Should select seasonal model
	if !result.IsSeasonal {
		t.Log("Warning: Expected seasonal model to be selected")
	}
}

func TestAutoARIMAModelComparison(t *testing.T) {
	// Data with seasonality - test that both models are compared
	n := 168
	period := 24
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		seasonal := 10 * math.Sin(2*math.Pi*float64(i)/float64(period))
		values[i] = 100 + seasonal + float64(i%5-2)/3
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.CompareModels = true
	config.AutoSeasonal = true

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	t.Logf("Candidates evaluated: %d", len(result.Candidates))
	for _, c := range result.Candidates {
		selected := ""
		if c.Selected {
			selected = " [SELECTED]"
		}
		t.Logf("  %s: RMSE=%.6f, MAPE=%.2f%%, AICc=%.2f%s",
			c.Name, c.RMSE, c.MAPE, c.AICc, selected)
	}

	if len(result.Candidates) < 2 {
		t.Log("Warning: Expected at least 2 candidates (ARIMA + SARIMA)")
	}
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
	config.AutoSeasonal = false

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

func TestAutoARIMAPredictWithInterval(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i%7-3)
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.AutoSeasonal = false

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	forecast, lower, upper, err := result.PredictWithInterval(5, 0.95)
	if err != nil {
		t.Fatalf("PredictWithInterval failed: %v", err)
	}

	if len(forecast) != 5 {
		t.Errorf("Expected 5 forecasts, got %d", len(forecast))
	}

	t.Logf("Forecast: %v", forecast)
	t.Logf("Lower: %v", lower)
	t.Logf("Upper: %v", upper)

	// Lower should be <= forecast <= upper
	for i := 0; i < len(forecast); i++ {
		if lower != nil && lower[i] > forecast[i] {
			t.Errorf("Lower bound %f > forecast %f at %d", lower[i], forecast[i], i)
		}
		if upper != nil && upper[i] < forecast[i] {
			t.Errorf("Upper bound %f < forecast %f at %d", upper[i], forecast[i], i)
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
	config.AutoSeasonal = false

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
	configAIC.ModelSelection = "aic" // Use AIC for selection too
	configAIC.MaxP = 2
	configAIC.MaxQ = 2
	configAIC.AutoSeasonal = false

	resultAIC, _ := AutoARIMA(series, configAIC)

	// Test with BIC
	configBIC := DefaultConfig()
	configBIC.Criterion = "bic"
	configBIC.ModelSelection = "bic"
	configBIC.MaxP = 2
	configBIC.MaxQ = 2
	configBIC.AutoSeasonal = false

	resultBIC, _ := AutoARIMA(series, configBIC)

	t.Logf("AIC criterion: %s, AIC=%f",
		resultAIC.Order(), resultAIC.AIC)
	t.Logf("BIC criterion: %s, BIC=%f",
		resultBIC.Order(), resultBIC.BIC)
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
	config.AutoSeasonal = false

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	t.Logf("ADF test selected: %s", result.Order())
}

func TestDetectSeasonalPeriod(t *testing.T) {
	// Data with clear seasonality at period 12
	n := 120
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		values[i] = 100 + 20*math.Sin(2*math.Pi*float64(i)/float64(period))
	}

	series := timeseries.New(values)
	config := DefaultConfig()

	detectedPeriod, strength := detectSeasonalPeriod(series, config)

	t.Logf("Detected period: %d (expected: %d)", detectedPeriod, period)
	t.Logf("Strength: %.4f", strength)

	if detectedPeriod != period {
		t.Errorf("Expected period %d, got %d", period, detectedPeriod)
	}
	if strength < 0.5 {
		t.Errorf("Expected strength > 0.5, got %.4f", strength)
	}
}

func TestDetectSeasonalPeriodNoSeasonality(t *testing.T) {
	// Random data without seasonality
	n := 100
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i%7-3)
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.SeasonalityThreshold = 0.5

	detectedPeriod, strength := detectSeasonalPeriod(series, config)

	t.Logf("Detected period: %d, strength: %.4f", detectedPeriod, strength)

	// Should detect no significant seasonality (or low strength)
	if detectedPeriod > 0 && strength > 0.7 {
		t.Logf("Warning: Detected period %d with high strength for non-seasonal data", detectedPeriod)
	}
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

	// Should use default config with auto-seasonality
	result, err := AutoARIMA(series, nil)
	if err != nil {
		t.Fatalf("AutoARIMA with nil config failed: %v", err)
	}

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	t.Logf("With nil config: %s", result.Order())
	t.Logf("Detected period: %d", result.DetectedPeriod)
}

func TestAutoARIMAOrder(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i%5-2)
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.AutoSeasonal = false

	result, _ := AutoARIMA(series, config)

	order := result.Order()
	t.Logf("Order string: %s", order)

	// Should be in format ARIMA(p,d,q)
	if len(order) < 10 {
		t.Errorf("Order string too short: %s", order)
	}
}

func TestCrossValidationSelection(t *testing.T) {
	// Test that CV-based selection works
	n := 120
	period := 24
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		seasonal := 10 * math.Sin(2*math.Pi*float64(i)/float64(period))
		values[i] = 100 + seasonal + float64(i%5-2)/3
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.ModelSelection = "cv"
	config.TestRatio = 0.2
	config.CompareModels = true

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	t.Logf("Selected: %s", result.Order())
	t.Logf("CV RMSE: %f", result.RMSE)
	t.Logf("CV MAPE: %f%%", result.MAPE)

	// RMSE should be finite
	if math.IsInf(result.RMSE, 0) || math.IsNaN(result.RMSE) {
		t.Error("RMSE should be finite")
	}
}

func TestPredictWithLevels(t *testing.T) {
	// Test multi-level prediction intervals (like R's forecast)
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i%7-3)
	}

	series := timeseries.New(values)
	config := DefaultConfig()
	config.AutoSeasonal = false

	result, err := AutoARIMA(series, config)
	if err != nil {
		t.Fatalf("AutoARIMA failed: %v", err)
	}

	// Test with default levels (80%, 95%)
	fc, err := result.PredictWithLevels(5, nil)
	if err != nil {
		t.Fatalf("PredictWithLevels failed: %v", err)
	}

	t.Logf("Forecasts: %v", fc.Forecasts)
	t.Logf("Levels: %v", fc.Levels)
	t.Logf("80%% Lower: %v", fc.Lower[0.80])
	t.Logf("80%% Upper: %v", fc.Upper[0.80])
	t.Logf("95%% Lower: %v", fc.Lower[0.95])
	t.Logf("95%% Upper: %v", fc.Upper[0.95])

	// Check we have both levels
	if len(fc.Levels) != 2 {
		t.Errorf("Expected 2 levels, got %d", len(fc.Levels))
	}

	// 80% interval should be narrower than 95%
	if len(fc.Lower[0.80]) > 0 && len(fc.Lower[0.95]) > 0 {
		width80 := fc.Upper[0.80][0] - fc.Lower[0.80][0]
		width95 := fc.Upper[0.95][0] - fc.Lower[0.95][0]
		if width80 >= width95 {
			t.Errorf("80%% interval (%.4f) should be narrower than 95%% (%.4f)", width80, width95)
		}
		t.Logf("Interval widths: 80%%=%.4f, 95%%=%.4f", width80, width95)
	}

	// Test with custom levels
	customLevels := []float64{0.90, 0.99}
	fc2, err := result.PredictWithLevels(5, customLevels)
	if err != nil {
		t.Fatalf("PredictWithLevels with custom levels failed: %v", err)
	}

	if len(fc2.Levels) != 2 {
		t.Errorf("Expected 2 custom levels, got %d", len(fc2.Levels))
	}
	if _, ok := fc2.Lower[0.90]; !ok {
		t.Error("Missing 90% lower bound")
	}
	if _, ok := fc2.Upper[0.99]; !ok {
		t.Error("Missing 99% upper bound")
	}
}
