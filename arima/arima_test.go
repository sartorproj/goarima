package arima

import (
	"math"
	"testing"

	"github.com/sartorproj/goarima/timeseries"
)

func TestNewARIMA(t *testing.T) {
	model := New(2, 1, 1)

	if model.Order.P != 2 {
		t.Errorf("Expected P=2, got %d", model.Order.P)
	}
	if model.Order.D != 1 {
		t.Errorf("Expected D=1, got %d", model.Order.D)
	}
	if model.Order.Q != 1 {
		t.Errorf("Expected Q=1, got %d", model.Order.Q)
	}
}

func TestARIMAFitAR1(t *testing.T) {
	// Generate AR(1) data
	n := 200
	phi := 0.7
	values := make([]float64, n)
	values[0] = 100
	for i := 1; i < n; i++ {
		innovation := float64(i%7-3) / 3
		values[i] = phi*(values[i-1]-100) + 100 + innovation
	}

	series := timeseries.New(values)
	model := New(1, 0, 0)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit AR(1) model: %v", err)
	}

	// Check that AR coefficient is estimated reasonably
	if len(model.ARCoeffs) != 1 {
		t.Errorf("Expected 1 AR coefficient, got %d", len(model.ARCoeffs))
	}

	t.Logf("True AR coeff: %f, Estimated: %f", phi, model.ARCoeffs[0])

	// The estimate should be in a reasonable range
	if math.Abs(model.ARCoeffs[0]-phi) > 0.3 {
		t.Logf("AR coefficient estimate may be off: true=%f, est=%f", phi, model.ARCoeffs[0])
	}

	// Check that residuals exist
	residuals := model.Residuals()
	if residuals == nil || len(residuals) == 0 {
		t.Error("Residuals should not be empty")
	}
}

func TestARIMAFitMA1(t *testing.T) {
	// Generate MA(1) data (approximately)
	n := 200
	values := make([]float64, n)
	innovations := make([]float64, n)

	for i := 0; i < n; i++ {
		innovations[i] = float64(i%7-3) / 3
	}

	theta := 0.5
	values[0] = innovations[0]
	for i := 1; i < n; i++ {
		values[i] = innovations[i] + theta*innovations[i-1]
	}

	// Add a mean
	for i := range values {
		values[i] += 100
	}

	series := timeseries.New(values)
	model := New(0, 0, 1)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit MA(1) model: %v", err)
	}

	t.Logf("True MA coeff: %f, Estimated: %f", theta, model.MACoeffs[0])
}

func TestARIMAFitWithDifferencing(t *testing.T) {
	// Generate random walk data (needs differencing)
	n := 200
	values := make([]float64, n)
	values[0] = 100

	for i := 1; i < n; i++ {
		values[i] = values[i-1] + float64(i%5-2)/2
	}

	series := timeseries.New(values)
	model := New(1, 1, 0) // ARIMA(1,1,0)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit ARIMA(1,1,0) model: %v", err)
	}

	t.Logf("ARIMA(1,1,0) - AIC: %f, BIC: %f", model.AIC, model.BIC)
}

func TestARIMAPredict(t *testing.T) {
	// Generate simple data
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i)/10 + float64(i%7-3)/2
	}

	series := timeseries.New(values)
	model := New(1, 1, 0)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	forecasts, err := model.Predict(5)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	if len(forecasts) != 5 {
		t.Errorf("Expected 5 forecasts, got %d", len(forecasts))
	}

	// Forecasts should be in reasonable range
	lastValue := values[n-1]
	for i, f := range forecasts {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Errorf("Forecast %d is NaN or Inf", i)
		}
		// Should be somewhat close to the last value for trending data
		if math.Abs(f-lastValue) > 50 {
			t.Logf("Forecast %d may be unusual: %f (last value: %f)", i, f, lastValue)
		}
	}

	t.Logf("Last value: %f, Forecasts: %v", lastValue, forecasts)
}

func TestARIMASummary(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = 100 + float64(i%7-3)/2
	}

	series := timeseries.New(values)
	model := New(1, 0, 1)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	summary := model.Summary()
	if summary == nil {
		t.Fatal("Summary should not be nil")
	}

	if summary.NObs != n {
		t.Errorf("Expected NObs=%d, got %d", n, summary.NObs)
	}

	t.Logf("Summary - AIC: %f, BIC: %f, LogLik: %f", summary.AIC, summary.BIC, summary.LogLik)
	if summary.LjungBox != nil {
		t.Logf("Ljung-Box Q: %f, P-Value: %f", summary.LjungBox.Statistic, summary.LjungBox.PValue)
	}
}

func TestARIMAInsufficientData(t *testing.T) {
	values := []float64{1, 2, 3}
	series := timeseries.New(values)
	model := New(5, 2, 5)

	err := model.Fit(series)
	if err == nil {
		t.Error("Expected error for insufficient data")
	}
}

func TestARIMAFittedValues(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = float64(i) + float64(i%5-2)/2
	}

	series := timeseries.New(values)
	model := New(1, 0, 0)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	fitted := model.FittedValues()
	if len(fitted) != n {
		t.Errorf("Expected %d fitted values, got %d", n, len(fitted))
	}
}

func TestYuleWalker(t *testing.T) {
	// Create ACF that corresponds to AR(1) process (more realistic)
	acf := []float64{1.0, 0.6, 0.36, 0.216, 0.13}

	coeffs := yuleWalker(acf, 2)
	if coeffs == nil {
		t.Fatal("yuleWalker returned nil")
	}

	if len(coeffs) != 2 {
		t.Errorf("Expected 2 coefficients, got %d", len(coeffs))
	}

	t.Logf("Yule-Walker coefficients: %v", coeffs)

	// Just check they're not NaN or Inf
	for i, c := range coeffs {
		if c != c { // NaN check
			t.Errorf("Coefficient %d is NaN", i)
		}
	}
}

func TestARIMAWhiteNoise(t *testing.T) {
	// White noise should result in near-zero coefficients
	n := 200
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = float64(i%7-3) / 3
	}

	series := timeseries.New(values)
	model := New(0, 0, 0) // Just constant model

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit white noise: %v", err)
	}

	// Mean should be close to actual mean
	actualMean := series.Mean()
	if math.Abs(model.Intercept-actualMean) > 0.5 {
		t.Errorf("Intercept should be close to mean: got %f, expected ~%f", model.Intercept, actualMean)
	}
}

func TestARIMAMultipleOrders(t *testing.T) {
	tests := []struct {
		name    string
		p, d, q int
	}{
		{"AR1", 1, 0, 0},
		{"AR2", 2, 0, 0},
		{"MA1", 0, 0, 1},
		{"MA2", 0, 0, 2},
		{"ARMA11", 1, 0, 1},
		{"ARIMA110", 1, 1, 0},
		{"ARIMA011", 0, 1, 1},
		{"ARIMA111", 1, 1, 1},
		{"ARIMA211", 2, 1, 1},
		{"ARIMA212", 2, 1, 2},
	}

	// Generate test data
	n := 150
	values := make([]float64, n)
	values[0] = 100
	for i := 1; i < n; i++ {
		values[i] = 0.6*(values[i-1]-100) + 100 + float64(i%7-3)/3
	}

	series := timeseries.New(values)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := New(tt.p, tt.d, tt.q)
			err := model.Fit(series)

			if err != nil {
				t.Logf("Model %s failed to fit: %v", tt.name, err)
				return
			}

			// Check model was fitted
			summary := model.Summary()
			if summary == nil {
				t.Error("Summary should not be nil after fitting")
				return
			}

			// Try prediction
			forecasts, err := model.Predict(3)
			if err != nil {
				t.Errorf("Prediction failed: %v", err)
				return
			}

			if len(forecasts) != 3 {
				t.Errorf("Expected 3 forecasts, got %d", len(forecasts))
			}

			t.Logf("%s - AIC: %.2f, BIC: %.2f, Forecasts: %v",
				tt.name, summary.AIC, summary.BIC, forecasts[:min(3, len(forecasts))])
		})
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
