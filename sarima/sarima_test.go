package sarima

import (
	"math"
	"testing"

	"github.com/sartorproj/goarima/timeseries"
)

func TestNewSARIMA(t *testing.T) {
	model := New(1, 1, 1, 1, 1, 1, 12)

	if model.Order.P != 1 {
		t.Errorf("Expected P=1, got %d", model.Order.P)
	}
	if model.Order.D != 1 {
		t.Errorf("Expected D=1, got %d", model.Order.D)
	}
	if model.Order.Q != 1 {
		t.Errorf("Expected Q=1, got %d", model.Order.Q)
	}
	if model.Order.SP != 1 {
		t.Errorf("Expected SP=1, got %d", model.Order.SP)
	}
	if model.Order.SD != 1 {
		t.Errorf("Expected SD=1, got %d", model.Order.SD)
	}
	if model.Order.SQ != 1 {
		t.Errorf("Expected SQ=1, got %d", model.Order.SQ)
	}
	if model.Order.M != 12 {
		t.Errorf("Expected M=12, got %d", model.Order.M)
	}
}

func TestSARIMAFitMonthlyData(t *testing.T) {
	// Generate monthly data with yearly seasonality
	n := 120 // 10 years of monthly data
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		trend := float64(i) * 0.5
		seasonal := 20 * math.Sin(2*math.Pi*float64(i)/float64(period))
		noise := float64(i%5-2) / 2
		values[i] = 100 + trend + seasonal + noise
	}

	series := timeseries.New(values)
	model := New(1, 0, 0, 1, 0, 0, 12)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit SARIMA model: %v", err)
	}

	t.Logf("SARIMA(1,0,0)(1,0,0)[12] - AIC: %f, BIC: %f", model.AIC, model.BIC)
	t.Logf("AR coeffs: %v", model.ARCoeffs)
	t.Logf("SAR coeffs: %v", model.SARCoeffs)
}

func TestSARIMAWithDifferencing(t *testing.T) {
	// Generate data with trend and seasonality
	n := 144 // 12 years
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		trend := float64(i) * 0.3
		seasonal := 15 * math.Cos(2*math.Pi*float64(i)/float64(period))
		values[i] = 50 + trend + seasonal + float64(i%7-3)/3
	}

	series := timeseries.New(values)
	model := New(1, 1, 0, 1, 1, 0, 12)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit SARIMA(1,1,0)(1,1,0)[12]: %v", err)
	}

	t.Logf("SARIMA(1,1,0)(1,1,0)[12] - AIC: %f, BIC: %f", model.AIC, model.BIC)
}

func TestSARIMAPredict(t *testing.T) {
	n := 96 // 8 years
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		seasonal := 10 * math.Sin(2*math.Pi*float64(i)/float64(period))
		values[i] = 100 + seasonal + float64(i%5-2)/2
	}

	series := timeseries.New(values)
	model := New(0, 0, 0, 1, 0, 0, 12)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	forecasts, err := model.Predict(12) // Predict one year ahead
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	if len(forecasts) != 12 {
		t.Errorf("Expected 12 forecasts, got %d", len(forecasts))
	}

	// Forecasts should capture seasonal pattern
	t.Logf("Forecasts for next 12 periods: %v", forecasts)

	// Check forecasts are in reasonable range
	for i, f := range forecasts {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Errorf("Forecast %d is NaN or Inf", i)
		}
		if f < 50 || f > 150 {
			t.Logf("Forecast %d may be unusual: %f", i, f)
		}
	}
}

func TestSARIMASummary(t *testing.T) {
	n := 60
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		values[i] = 100 + 5*math.Sin(2*math.Pi*float64(i)/float64(period))
	}

	series := timeseries.New(values)
	model := New(1, 0, 1, 1, 0, 1, 12)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	summary := model.Summary()
	if summary == nil {
		t.Fatal("Summary should not be nil")
	}

	if summary.NObs != n {
		t.Errorf("Expected NObs=%d, got %d", n, summary.NObs)
	}

	t.Logf("Summary:")
	t.Logf("  Order: (%d,%d,%d)(%d,%d,%d)[%d]",
		summary.Order.P, summary.Order.D, summary.Order.Q,
		summary.Order.SP, summary.Order.SD, summary.Order.SQ, summary.Order.M)
	t.Logf("  AIC: %f, BIC: %f", summary.AIC, summary.BIC)
	t.Logf("  AR: %v, MA: %v", summary.ARCoeffs, summary.MACoeffs)
	t.Logf("  SAR: %v, SMA: %v", summary.SARCoeffs, summary.SMACoeffs)
}

func TestSARIMAResiduals(t *testing.T) {
	n := 60
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		values[i] = 100 + 5*math.Sin(2*math.Pi*float64(i)/float64(period))
	}

	series := timeseries.New(values)
	model := New(1, 0, 0, 1, 0, 0, 12)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	residuals := model.Residuals()
	if residuals == nil {
		t.Fatal("Residuals should not be nil")
	}

	if len(residuals) != n {
		t.Errorf("Expected %d residuals, got %d", n, len(residuals))
	}

	// Calculate mean of residuals (should be close to 0)
	sum := 0.0
	for _, r := range residuals {
		sum += r
	}
	mean := sum / float64(len(residuals))
	t.Logf("Mean of residuals: %f", mean)
}

func TestSARIMAFittedValues(t *testing.T) {
	n := 60
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		values[i] = 100 + 5*math.Sin(2*math.Pi*float64(i)/float64(period))
	}

	series := timeseries.New(values)
	model := New(1, 0, 0, 1, 0, 0, 12)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	fitted := model.FittedValues()
	if fitted == nil {
		t.Fatal("Fitted values should not be nil")
	}

	if len(fitted) != n {
		t.Errorf("Expected %d fitted values, got %d", n, len(fitted))
	}
}

func TestSARIMAMultipleOrders(t *testing.T) {
	// Generate test data with seasonality
	n := 96
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		trend := float64(i) * 0.2
		seasonal := 10 * math.Sin(2*math.Pi*float64(i)/float64(period))
		values[i] = 100 + trend + seasonal + float64(i%5-2)/3
	}

	series := timeseries.New(values)

	tests := []struct {
		name          string
		p, d, q       int
		sp, sd, sq, m int
	}{
		{"SARIMA(1,0,0)(1,0,0)12", 1, 0, 0, 1, 0, 0, 12},
		{"SARIMA(0,0,1)(0,0,1)12", 0, 0, 1, 0, 0, 1, 12},
		{"SARIMA(1,0,1)(1,0,1)12", 1, 0, 1, 1, 0, 1, 12},
		{"SARIMA(1,1,0)(1,1,0)12", 1, 1, 0, 1, 1, 0, 12},
		{"SARIMA(2,1,1)(1,0,1)12", 2, 1, 1, 1, 0, 1, 12},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := New(tt.p, tt.d, tt.q, tt.sp, tt.sd, tt.sq, tt.m)
			err := model.Fit(series)

			if err != nil {
				t.Logf("Model %s failed: %v", tt.name, err)
				return
			}

			// Try prediction
			forecasts, err := model.Predict(6)
			if err != nil {
				t.Errorf("Prediction failed: %v", err)
				return
			}

			summary := model.Summary()
			t.Logf("%s - AIC: %.2f, Forecasts: %v", tt.name, summary.AIC, forecasts)
		})
	}
}

func TestSARIMAWeeklyData(t *testing.T) {
	// Weekly data with weekly seasonality (period=7)
	n := 140 // 20 weeks
	period := 7
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		dayOfWeek := i % period
		// Weekend effect
		if dayOfWeek == 5 || dayOfWeek == 6 {
			values[i] = 150 + float64(i%5-2)
		} else {
			values[i] = 100 + float64(i%5-2)
		}
	}

	series := timeseries.New(values)
	model := New(1, 0, 0, 1, 0, 0, 7)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit weekly SARIMA: %v", err)
	}

	forecasts, _ := model.Predict(7)
	t.Logf("Weekly SARIMA forecasts: %v", forecasts)
}

func TestSARIMAQuarterlyData(t *testing.T) {
	// Quarterly data (period=4)
	n := 80 // 20 years of quarters
	period := 4
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		quarter := i % period
		var seasonal float64
		switch quarter {
		case 0:
			seasonal = -10 // Q1 low
		case 1:
			seasonal = 5 // Q2 medium
		case 2:
			seasonal = 15 // Q3 high (summer)
		case 3:
			seasonal = -5 // Q4 medium-low
		}
		trend := float64(i) * 0.5
		values[i] = 100 + trend + seasonal + float64(i%3-1)
	}

	series := timeseries.New(values)
	model := New(1, 0, 0, 1, 0, 0, 4)

	err := model.Fit(series)
	if err != nil {
		t.Fatalf("Failed to fit quarterly SARIMA: %v", err)
	}

	forecasts, _ := model.Predict(4)
	t.Logf("Quarterly SARIMA - AIC: %f, Forecasts: %v", model.AIC, forecasts)
}
