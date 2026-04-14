package timeseries

import (
	"math"
	"testing"
)

func TestNew(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5}
	s := New(values)

	if s.Len() != 5 {
		t.Errorf("Expected length 5, got %d", s.Len())
	}

	for i, v := range s.Values {
		if v != values[i] {
			t.Errorf("Expected value %f at index %d, got %f", values[i], i, v)
		}
	}
}

func TestMean(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		expected float64
	}{
		{"simple", []float64{1, 2, 3, 4, 5}, 3.0},
		{"single", []float64{5}, 5.0},
		{"negative", []float64{-1, -2, -3}, -2.0},
		{"mixed", []float64{-1, 0, 1}, 0.0},
		{"empty", []float64{}, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := New(tt.values)
			result := s.Mean()
			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("Expected mean %f, got %f", tt.expected, result)
			}
		})
	}
}

func TestVariance(t *testing.T) {
	s := New([]float64{2, 4, 4, 4, 5, 5, 7, 9})
	expected := 4.571428571428571

	result := s.Variance()
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("Expected variance %f, got %f", expected, result)
	}
}

func TestStd(t *testing.T) {
	s := New([]float64{2, 4, 4, 4, 5, 5, 7, 9})
	expected := math.Sqrt(4.571428571428571)

	result := s.Std()
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("Expected std %f, got %f", expected, result)
	}
}

func TestMinMax(t *testing.T) {
	s := New([]float64{5, 2, 8, 1, 9, 3})

	if s.Min() != 1 {
		t.Errorf("Expected min 1, got %f", s.Min())
	}

	if s.Max() != 9 {
		t.Errorf("Expected max 9, got %f", s.Max())
	}
}

func TestMedian(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		expected float64
	}{
		{"odd", []float64{1, 3, 5}, 3.0},
		{"even", []float64{1, 2, 3, 4}, 2.5},
		{"single", []float64{5}, 5.0},
		{"unsorted", []float64{5, 1, 3}, 3.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := New(tt.values)
			result := s.Median()
			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("Expected median %f, got %f", tt.expected, result)
			}
		})
	}
}

func TestDiff(t *testing.T) {
	s := New([]float64{1, 3, 6, 10, 15})
	diff := s.Diff()

	expected := []float64{2, 3, 4, 5}
	if len(diff.Values) != len(expected) {
		t.Errorf("Expected length %d, got %d", len(expected), len(diff.Values))
	}

	for i, v := range diff.Values {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("Expected %f at index %d, got %f", expected[i], i, v)
		}
	}
}

func TestDiffN(t *testing.T) {
	// DiffN(2) applies Diff() twice (second-order difference)
	s := New([]float64{1, 3, 6, 10, 15, 21})
	diff2 := s.DiffN(2)

	// First diff: [2, 3, 4, 5, 6], second diff: [1, 1, 1, 1]
	expected := []float64{1, 1, 1, 1}
	if len(diff2.Values) != len(expected) {
		t.Errorf("Expected length %d, got %d", len(expected), len(diff2.Values))
	}

	for i, v := range diff2.Values {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("Expected %f at index %d, got %f", expected[i], i, v)
		}
	}
}

func TestDiffLag(t *testing.T) {
	// DiffLag(2) computes y[t] - y[t-2]
	s := New([]float64{1, 3, 6, 10, 15, 21})
	lag2 := s.DiffLag(2)

	expected := []float64{5, 7, 9, 11}
	if len(lag2.Values) != len(expected) {
		t.Errorf("Expected length %d, got %d", len(expected), len(lag2.Values))
	}

	for i, v := range lag2.Values {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("Expected %f at index %d, got %f", expected[i], i, v)
		}
	}
}

func TestSeasonalDiff(t *testing.T) {
	// Monthly data with yearly seasonality
	values := []float64{10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 11, 13, 15, 17}
	s := New(values)

	diff := s.SeasonalDiff(12)

	// Expected: values[12] - values[0], values[13] - values[1], etc.
	expected := []float64{1, 1, 1, 1}
	if len(diff.Values) != len(expected) {
		t.Errorf("Expected length %d, got %d", len(expected), len(diff.Values))
	}

	for i, v := range diff.Values {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("Expected %f at index %d, got %f", expected[i], i, v)
		}
	}
}

func TestLag(t *testing.T) {
	s := New([]float64{1, 2, 3, 4, 5})
	lagged := s.Lag(2)

	expected := []float64{1, 2, 3}
	if len(lagged.Values) != len(expected) {
		t.Errorf("Expected length %d, got %d", len(expected), len(lagged.Values))
	}

	for i, v := range lagged.Values {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("Expected %f at index %d, got %f", expected[i], i, v)
		}
	}
}

func TestSlice(t *testing.T) {
	s := New([]float64{1, 2, 3, 4, 5})
	sliced := s.Slice(1, 4)

	expected := []float64{2, 3, 4}
	if len(sliced.Values) != len(expected) {
		t.Errorf("Expected length %d, got %d", len(expected), len(sliced.Values))
	}

	for i, v := range sliced.Values {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("Expected %f at index %d, got %f", expected[i], i, v)
		}
	}
}

func TestLog(t *testing.T) {
	s := New([]float64{1, math.E, math.E * math.E})
	logged := s.Log()

	expected := []float64{0, 1, 2}
	for i, v := range logged.Values {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("Expected %f at index %d, got %f", expected[i], i, v)
		}
	}
}

func TestMovingAverage(t *testing.T) {
	s := New([]float64{1, 2, 3, 4, 5, 6, 7})
	ma := s.MovingAverage(3)

	expected := []float64{2, 3, 4, 5, 6}
	if len(ma.Values) != len(expected) {
		t.Errorf("Expected length %d, got %d", len(expected), len(ma.Values))
	}

	for i, v := range ma.Values {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("Expected %f at index %d, got %f", expected[i], i, v)
		}
	}
}

func TestNormalize(t *testing.T) {
	s := New([]float64{1, 2, 3, 4, 5})
	normalized := s.Normalize()

	// Mean should be close to 0
	if math.Abs(normalized.Mean()) > 1e-10 {
		t.Errorf("Expected mean close to 0, got %f", normalized.Mean())
	}

	// Std should be close to 1
	if math.Abs(normalized.Std()-1) > 1e-10 {
		t.Errorf("Expected std close to 1, got %f", normalized.Std())
	}
}

func TestCopy(t *testing.T) {
	s := New([]float64{1, 2, 3})
	copied := s.Copy()

	// Modify original
	s.Values[0] = 100

	// Copy should be unchanged
	if copied.Values[0] != 1 {
		t.Errorf("Copy was modified when original changed")
	}
}

func TestBoxCoxLambdaZero(t *testing.T) {
	// lambda=0 should be equivalent to log
	s := New([]float64{1, 2, 3, 4, 5})
	bc := s.BoxCox(0)
	lg := s.Log()

	for i := range bc.Values {
		if math.Abs(bc.Values[i]-lg.Values[i]) > 1e-10 {
			t.Errorf("BoxCox(0) != Log at index %d: %f vs %f", i, bc.Values[i], lg.Values[i])
		}
	}
}

func TestBoxCoxLambdaOne(t *testing.T) {
	// lambda=1: (y^1 - 1)/1 = y - 1
	s := New([]float64{2, 4, 6, 8})
	bc := s.BoxCox(1)

	expected := []float64{1, 3, 5, 7}
	for i, v := range bc.Values {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("BoxCox(1) at %d: got %f want %f", i, v, expected[i])
		}
	}
}

func TestBoxCoxRoundtrip(t *testing.T) {
	s := New([]float64{1.5, 2.3, 5.7, 10.1, 0.3})
	lambdas := []float64{-1, -0.5, 0, 0.25, 0.5, 1, 2}

	for _, lambda := range lambdas {
		transformed := s.BoxCox(lambda)
		recovered := transformed.InverseBoxCox(lambda)

		for i, v := range recovered.Values {
			if math.IsNaN(v) || math.Abs(v-s.Values[i]) > 1e-8 {
				t.Errorf("Roundtrip failed for lambda=%.2f at index %d: got %f want %f",
					lambda, i, v, s.Values[i])
			}
		}
	}
}

func TestBoxCoxNonPositive(t *testing.T) {
	s := New([]float64{1, -2, 3})
	bc := s.BoxCox(0.5)

	if !math.IsNaN(bc.Values[1]) {
		t.Errorf("Expected NaN for non-positive value, got %f", bc.Values[1])
	}
}

func TestBoxCoxLambdaSelection(t *testing.T) {
	// Exponential growth: lambda should be near 0 (log)
	n := 200
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = math.Exp(float64(i) * 0.02)
	}
	s := New(vals)

	lambda, err := BoxCoxLambda(s)
	if err != nil {
		t.Fatalf("BoxCoxLambda failed: %v", err)
	}

	if math.Abs(lambda) > 0.3 {
		t.Errorf("Expected lambda near 0 for exponential data, got %f", lambda)
	}
	t.Logf("Exponential data: lambda = %f", lambda)
}

func TestBoxCoxLambdaLinear(t *testing.T) {
	// Linear data with constant variance: lambda should be near 1
	n := 200
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = 10 + float64(i)*0.1
	}
	s := New(vals)

	lambda, err := BoxCoxLambda(s)
	if err != nil {
		t.Fatalf("BoxCoxLambda failed: %v", err)
	}

	if math.Abs(lambda-1) > 0.5 {
		t.Errorf("Expected lambda near 1 for linear data, got %f", lambda)
	}
	t.Logf("Linear data: lambda = %f", lambda)
}

func TestInverseBoxCoxWithBias(t *testing.T) {
	// Log case: E[exp(X)] = exp(mu + sigma²/2)
	mu := 2.0
	sigma2 := 0.5
	result := InverseBoxCoxWithBias(mu, sigma2, 0)
	expected := math.Exp(mu + sigma2/2)

	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("Bias correction for lambda=0: got %f want %f", result, expected)
	}
}
