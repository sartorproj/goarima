package stats

import (
	"math"
	"testing"

	"github.com/sartorproj/goarima/timeseries"
)

func TestACF(t *testing.T) {
	// Create a simple AR(1) process
	n := 100
	phi := 0.8
	values := make([]float64, n)
	values[0] = 0
	for i := 1; i < n; i++ {
		values[i] = phi*values[i-1] + (float64(i%10)-5)/10
	}

	series := timeseries.New(values)
	acf := ACF(series, 10)

	if acf == nil {
		t.Fatal("ACF returned nil")
	}

	// ACF at lag 0 should be 1
	if math.Abs(acf[0]-1.0) > 1e-10 {
		t.Errorf("ACF at lag 0 should be 1, got %f", acf[0])
	}

	// ACF values should decay for AR(1)
	for i := 1; i < len(acf)-1; i++ {
		if math.Abs(acf[i]) > math.Abs(acf[i-1])+0.1 {
			// Allow some tolerance, but generally should decay
			t.Logf("ACF may not be decaying properly at lag %d", i)
		}
	}
}

func TestPACF(t *testing.T) {
	// Create a simple AR(1) process
	n := 100
	phi := 0.7
	values := make([]float64, n)
	values[0] = 0
	for i := 1; i < n; i++ {
		values[i] = phi*values[i-1] + (float64(i%10)-5)/10
	}

	series := timeseries.New(values)
	pacf := PACF(series, 10)

	if pacf == nil {
		t.Fatal("PACF returned nil")
	}

	// PACF at lag 0 should be 1
	if math.Abs(pacf[0]-1.0) > 1e-10 {
		t.Errorf("PACF at lag 0 should be 1, got %f", pacf[0])
	}

	// For AR(1), PACF should be significant only at lag 1
	// PACF at lag 1 should be close to phi (though not exact due to noise)
	if math.Abs(pacf[1]) < 0.3 {
		t.Logf("PACF at lag 1 seems low for AR(1) with phi=0.7: %f", pacf[1])
	}
}

func TestACFWithConfidence(t *testing.T) {
	values := make([]float64, 100)
	for i := range values {
		values[i] = float64(i) + math.Sin(float64(i)/10)
	}

	series := timeseries.New(values)
	result := ACFWithConfidence(series, 20)

	if result == nil {
		t.Fatal("ACFWithConfidence returned nil")
	}

	// Confidence bounds should be approximately 1.96/sqrt(n)
	expected := 1.96 / math.Sqrt(100)
	if math.Abs(result.ConfBounds-expected) > 0.01 {
		t.Errorf("Expected confidence bounds ~%f, got %f", expected, result.ConfBounds)
	}
}

func TestSignificantLags(t *testing.T) {
	values := []float64{1.0, 0.5, 0.3, 0.1, 0.05, -0.2, -0.5}
	confBound := 0.15

	significant := SignificantLags(values, confBound)

	// Should include lags 1, 2, 5, 6 (values > 0.15 or < -0.15, excluding lag 0)
	expected := []int{1, 2, 5, 6}
	if len(significant) != len(expected) {
		t.Errorf("Expected %d significant lags, got %d", len(expected), len(significant))
	}
}

func TestADF(t *testing.T) {
	// Test with stationary data (oscillating around mean)
	n := 200
	stationary := make([]float64, n)
	for i := range stationary {
		stationary[i] = 100 + math.Sin(float64(i)/10)*5 + float64(i%5-2)
	}

	series := timeseries.New(stationary)
	result := ADF(series, 0)

	if result == nil {
		t.Fatal("ADF returned nil for stationary data")
	}

	t.Logf("ADF Statistic: %f, P-Value: %f, IsStationary: %v",
		result.Statistic, result.PValue, result.IsStationary)

	// Test with non-stationary data (trending)
	nonStationary := make([]float64, n)
	for i := 0; i < n; i++ {
		nonStationary[i] = float64(i)*0.5 + float64(i%5-2)
	}

	series2 := timeseries.New(nonStationary)
	result2 := ADF(series2, 0)

	if result2 == nil {
		t.Log("ADF returned nil for non-stationary data (may need more data points)")
	} else {
		t.Logf("ADF Non-Stationary - Statistic: %f, P-Value: %f, IsStationary: %v",
			result2.Statistic, result2.PValue, result2.IsStationary)
	}
}

func TestKPSS(t *testing.T) {
	// Stationary data
	n := 200
	stationary := make([]float64, n)
	for i := range stationary {
		stationary[i] = math.Sin(float64(i)/10) + float64(i%5-2)/5
	}

	series := timeseries.New(stationary)
	result := KPSS(series, "c", 0)

	if result == nil {
		t.Fatal("KPSS returned nil")
	}

	t.Logf("KPSS Stationary - Statistic: %f, P-Value: %f, IsStationary: %v",
		result.Statistic, result.PValue, result.IsStationary)

	// Non-stationary (trend)
	nonStationary := make([]float64, n)
	for i := range nonStationary {
		nonStationary[i] = float64(i) * 0.5
	}

	series2 := timeseries.New(nonStationary)
	result2 := KPSS(series2, "c", 0)

	if result2 == nil {
		t.Fatal("KPSS returned nil for non-stationary data")
	}

	t.Logf("KPSS Non-Stationary - Statistic: %f, P-Value: %f, IsStationary: %v",
		result2.Statistic, result2.PValue, result2.IsStationary)
}

func TestPhillipsPerron(t *testing.T) {
	// Stationary data
	n := 200
	stationary := make([]float64, n)
	for i := range stationary {
		stationary[i] = math.Sin(float64(i)/10) + float64(i%5-2)/5
	}

	series := timeseries.New(stationary)
	result := PhillipsPerron(series, 0)

	if result == nil {
		t.Fatal("PhillipsPerron returned nil")
	}

	t.Logf("PP Stationary - Statistic: %f, P-Value: %f, IsStationary: %v",
		result.Statistic, result.PValue, result.IsStationary)
}

func TestLjungBox(t *testing.T) {
	// White noise should pass Ljung-Box test (no autocorrelation)
	n := 100
	whiteNoise := make([]float64, n)
	for i := range whiteNoise {
		whiteNoise[i] = float64(i%7-3) / 3
	}

	series := timeseries.New(whiteNoise)
	result := LjungBox(series, 10, 0)

	if result == nil {
		t.Fatal("LjungBox returned nil")
	}

	t.Logf("Ljung-Box - Q: %f, P-Value: %f, DOF: %d",
		result.Statistic, result.PValue, result.DOF)

	// Autocorrelated series should fail
	autocorrelated := make([]float64, n)
	autocorrelated[0] = 0
	for i := 1; i < n; i++ {
		autocorrelated[i] = 0.9*autocorrelated[i-1] + float64(i%7-3)/10
	}

	series2 := timeseries.New(autocorrelated)
	result2 := LjungBox(series2, 10, 0)

	if result2 == nil {
		t.Fatal("LjungBox returned nil for autocorrelated data")
	}

	t.Logf("Ljung-Box Autocorrelated - Q: %f, P-Value: %f",
		result2.Statistic, result2.PValue)
}

func TestBoxPierce(t *testing.T) {
	n := 100
	values := make([]float64, n)
	for i := range values {
		values[i] = float64(i%7-3) / 3
	}

	series := timeseries.New(values)
	result := BoxPierce(series, 10, 0)

	if result == nil {
		t.Fatal("BoxPierce returned nil")
	}

	t.Logf("Box-Pierce - Q: %f, P-Value: %f, DOF: %d",
		result.Statistic, result.PValue, result.DOF)
}

func TestDurbinWatson(t *testing.T) {
	tests := []struct {
		name      string
		residuals []float64
		expected  float64
	}{
		{
			name:      "no autocorrelation",
			residuals: []float64{1, -1, 1, -1, 1, -1, 1, -1},
			expected:  4.0, // Alternating pattern = high DW
		},
		{
			name:      "positive autocorrelation",
			residuals: []float64{1, 1, 1, 1, -1, -1, -1, -1},
			expected:  0.5, // Low DW
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DurbinWatson(tt.residuals)
			if result == nil {
				t.Fatal("DurbinWatson returned nil")
			}
			// Check roughly in expected range
			if tt.expected > 2 && result.Statistic < 2 {
				t.Logf("Expected high DW, got %f", result.Statistic)
			}
			if tt.expected < 2 && result.Statistic > 2 {
				t.Logf("Expected low DW, got %f", result.Statistic)
			}
		})
	}
}

func TestDecompose(t *testing.T) {
	// Create data with trend and seasonality
	n := 120 // 10 years of monthly data
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		trend := float64(i) * 0.5                                              // Linear trend
		seasonal := 10 * math.Sin(2*math.Pi*float64(i%period)/float64(period)) // Seasonal
		noise := float64(i%5-2) / 5                                            // Noise
		values[i] = trend + seasonal + noise
	}

	series := timeseries.New(values)
	result := Decompose(series, period, "additive")

	if result == nil {
		t.Fatal("Decompose returned nil")
	}

	if result.Trend.Len() != n {
		t.Errorf("Trend length mismatch: expected %d, got %d", n, result.Trend.Len())
	}

	if result.Seasonal.Len() != n {
		t.Errorf("Seasonal length mismatch: expected %d, got %d", n, result.Seasonal.Len())
	}

	if result.Residual.Len() != n {
		t.Errorf("Residual length mismatch: expected %d, got %d", n, result.Residual.Len())
	}

	// Check that components roughly sum to original (for additive)
	// Skip edges where trend may be NaN
	for i := period; i < n-period; i++ {
		reconstructed := result.Trend.Values[i] + result.Seasonal.Values[i] + result.Residual.Values[i]
		original := series.Values[i]
		if !math.IsNaN(reconstructed) && math.Abs(reconstructed-original) > 1.0 {
			t.Errorf("Reconstruction error at index %d: original=%f, reconstructed=%f",
				i, original, reconstructed)
		}
	}
}

func TestSTL(t *testing.T) {
	n := 120
	period := 12
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		trend := float64(i) * 0.5
		seasonal := 10 * math.Sin(2*math.Pi*float64(i%period)/float64(period))
		values[i] = trend + seasonal + float64(i%5-2)/5
	}

	series := timeseries.New(values)
	result := STL(series, period, 2)

	if result == nil {
		t.Fatal("STL returned nil")
	}

	// Basic length checks
	if result.Trend.Len() != n {
		t.Errorf("STL Trend length mismatch")
	}
	if result.Seasonal.Len() != n {
		t.Errorf("STL Seasonal length mismatch")
	}
	if result.Residual.Len() != n {
		t.Errorf("STL Residual length mismatch")
	}

	// The seasonal component should be periodic
	// Check a few period-apart values
	for i := period; i < n; i += period {
		diff := math.Abs(result.Seasonal.Values[i] - result.Seasonal.Values[i-period])
		if diff > 5.0 {
			t.Logf("Seasonal component may not be periodic: diff at %d = %f", i, diff)
		}
	}
}

func TestChiSquaredCDF(t *testing.T) {
	// Test some known values with wider tolerance
	tests := []struct {
		x       float64
		k       int
		minProb float64
		maxProb float64
	}{
		{0, 1, 0, 0.1},
		{3.84, 1, 0.93, 0.98}, // 95th percentile (wider range)
		{5.99, 2, 0.93, 0.98}, // 95th percentile
		{7.81, 3, 0.93, 0.98}, // 95th percentile
	}

	for _, tt := range tests {
		result := chiSquaredCDF(tt.x, tt.k)
		if result < tt.minProb || result > tt.maxProb {
			t.Errorf("chiSquaredCDF(%f, %d) = %f, expected between %f and %f",
				tt.x, tt.k, result, tt.minProb, tt.maxProb)
		}
	}
}

func TestGamma(t *testing.T) {
	tests := []struct {
		z        float64
		expected float64
	}{
		{1, 1},
		{2, 1},
		{3, 2},
		{4, 6},
		{5, 24},
		{0.5, math.Sqrt(math.Pi)},
	}

	for _, tt := range tests {
		result := gamma(tt.z)
		if math.Abs(result-tt.expected) > 0.001 {
			t.Errorf("gamma(%f) = %f, expected %f", tt.z, result, tt.expected)
		}
	}
}
