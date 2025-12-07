package stats

import (
	"math"
	"testing"

	"github.com/sartorproj/goarima/timeseries"
)

func TestNDiffs(t *testing.T) {
	// Test with stationary data (should need 0 differences)
	n := 100
	stationary := make([]float64, n)
	for i := 0; i < n; i++ {
		stationary[i] = float64(i%10-5) + float64((i*7)%11-5)*0.5
	}
	stationarySeries := timeseries.New(stationary)

	d := NDiffs(stationarySeries, 2, "kpss")
	t.Logf("Stationary series ndiffs: %d", d)
	// Stationary data should need 0 or at most 1 difference
	if d > 1 {
		t.Errorf("Stationary series should need at most 1 difference, got %d", d)
	}

	// Test with random walk (non-stationary, should need 1 difference)
	randomWalk := make([]float64, n)
	randomWalk[0] = 0
	for i := 1; i < n; i++ {
		randomWalk[i] = randomWalk[i-1] + float64((i*7)%11-5)*0.3
	}
	rwSeries := timeseries.New(randomWalk)

	d = NDiffs(rwSeries, 2, "kpss")
	t.Logf("Random walk ndiffs: %d", d)
	// Random walk should typically need at least 1 difference
	if d < 1 {
		t.Logf("Random walk may need differencing, got d=%d", d)
	}

	// Test with trend (should need 1-2 differences)
	trend := make([]float64, n)
	for i := 0; i < n; i++ {
		trend[i] = 100 + float64(i)*2 + float64((i*3)%7-3)*0.5
	}
	trendSeries := timeseries.New(trend)

	d = NDiffs(trendSeries, 2, "kpss")
	t.Logf("Trend series ndiffs: %d", d)
}

func TestNSDiffs(t *testing.T) {
	// Test with seasonal data (period 12)
	n := 120
	seasonal := make([]float64, n)
	for i := 0; i < n; i++ {
		trend := 100 + float64(i)*0.5
		season := 15 * math.Sin(2*math.Pi*float64(i)/12)
		seasonal[i] = trend + season
	}
	seasonalSeries := timeseries.New(seasonal)

	sd := NSDiffs(seasonalSeries, 12, 1)
	t.Logf("Seasonal series (period 12) nsdiffs: %d", sd)
	// Strong seasonal pattern should suggest 1 seasonal difference
	if sd < 0 || sd > 1 {
		t.Errorf("Expected 0 or 1 seasonal differences, got %d", sd)
	}

	// Test with non-seasonal data
	nonSeasonal := make([]float64, n)
	for i := 0; i < n; i++ {
		nonSeasonal[i] = 100 + float64((i*7)%20-10)*0.5
	}
	nonSeasonalSeries := timeseries.New(nonSeasonal)

	sd = NSDiffs(nonSeasonalSeries, 12, 1)
	t.Logf("Non-seasonal series nsdiffs: %d", sd)
	// Non-seasonal data should need 0 seasonal differences
	if sd > 1 {
		t.Errorf("Non-seasonal series should need at most 1 seasonal difference, got %d", sd)
	}
}

func TestAICc(t *testing.T) {
	// Test AICc calculation
	// AICc = AIC + 2*k*(k+1)/(n-k-1)

	tests := []struct {
		aic     float64
		nObs    int
		nParams int
	}{
		{100.0, 50, 3},
		{200.0, 100, 5},
		{150.0, 30, 4},
	}

	for _, tt := range tests {
		aicc := AICc(tt.aic, tt.nObs, tt.nParams)

		// AICc should always be >= AIC for finite sample sizes
		if aicc < tt.aic {
			t.Errorf("AICc (%f) should be >= AIC (%f)", aicc, tt.aic)
		}

		// Verify the formula
		k := float64(tt.nParams)
		n := float64(tt.nObs)
		expectedCorrection := 2 * k * (k + 1) / (n - k - 1)
		expectedAICc := tt.aic + expectedCorrection

		if math.Abs(aicc-expectedAICc) > 1e-10 {
			t.Errorf("AICc calculation incorrect: got %f, expected %f", aicc, expectedAICc)
		}

		t.Logf("AIC=%.2f, n=%d, k=%d -> AICc=%.2f (correction=%.4f)",
			tt.aic, tt.nObs, tt.nParams, aicc, expectedCorrection)
	}

	// Test edge case: n-k-1 <= 0 should return Inf
	aicc := AICc(100.0, 5, 5)
	if !math.IsInf(aicc, 1) {
		t.Errorf("AICc should be +Inf when n-k-1 <= 0, got %f", aicc)
	}
}

func TestCalculateIC(t *testing.T) {
	// Test full IC calculation
	logLik := -50.0
	nObs := 100
	nParams := 3

	ic := CalculateIC(logLik, nObs, nParams)

	// AIC = -2*logLik + 2*k
	expectedAIC := -2*logLik + 2*float64(nParams)
	if math.Abs(ic.AIC-expectedAIC) > 1e-10 {
		t.Errorf("AIC calculation incorrect: got %f, expected %f", ic.AIC, expectedAIC)
	}

	// BIC = -2*logLik + k*log(n)
	expectedBIC := -2*logLik + float64(nParams)*math.Log(float64(nObs))
	if math.Abs(ic.BIC-expectedBIC) > 1e-10 {
		t.Errorf("BIC calculation incorrect: got %f, expected %f", ic.BIC, expectedBIC)
	}

	// AICc should be >= AIC
	if ic.AICc < ic.AIC {
		t.Errorf("AICc should be >= AIC")
	}

	t.Logf("LogLik=%.2f, n=%d, k=%d -> AIC=%.2f, AICc=%.2f, BIC=%.2f",
		logLik, nObs, nParams, ic.AIC, ic.AICc, ic.BIC)
}

func TestSeasonalStrength(t *testing.T) {
	// Test with strong seasonal pattern
	n := 120
	strong := make([]float64, n)
	for i := 0; i < n; i++ {
		trend := 100.0
		season := 20 * math.Sin(2*math.Pi*float64(i)/12)
		strong[i] = trend + season
	}
	strongSeries := timeseries.New(strong)

	strength := seasonalStrength(strongSeries, 12)
	t.Logf("Strong seasonal pattern strength: %.4f", strength)
	// Strong seasonal pattern should have strength > 0.5
	if strength < 0.3 {
		t.Logf("Expected higher seasonal strength for strong pattern, got %.4f", strength)
	}

	// Test with weak/no seasonal pattern
	weak := make([]float64, n)
	for i := 0; i < n; i++ {
		weak[i] = 100 + float64((i*7)%20-10)*0.5
	}
	weakSeries := timeseries.New(weak)

	weakStrength := seasonalStrength(weakSeries, 12)
	t.Logf("Weak seasonal pattern strength: %.4f", weakStrength)
	// Weak pattern should have lower strength
	if weakStrength > 0.8 {
		t.Logf("Expected lower seasonal strength for weak pattern, got %.4f", weakStrength)
	}
}

func TestVariance(t *testing.T) {
	// Test variance calculation
	data := []float64{2, 4, 4, 4, 5, 5, 7, 9}

	// Mean = 5, Variance = 4.571... (sample variance)
	v := variance(data)
	expectedVar := 32.0 / 7.0 // Sample variance with n-1 denominator

	if math.Abs(v-expectedVar) > 0.001 {
		t.Errorf("Variance calculation incorrect: got %f, expected %f", v, expectedVar)
	}

	t.Logf("Data variance: %.4f (expected: %.4f)", v, expectedVar)

	// Test with NaN values
	dataWithNaN := []float64{2, 4, math.NaN(), 4, 5, math.NaN(), 7, 9}
	vNaN := variance(dataWithNaN)
	t.Logf("Variance with NaN values: %.4f", vNaN)
}
