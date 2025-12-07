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
	s := New([]float64{1, 3, 6, 10, 15, 21})
	diff2 := s.DiffN(2)

	expected := []float64{5, 7, 9, 11}
	if len(diff2.Values) != len(expected) {
		t.Errorf("Expected length %d, got %d", len(expected), len(diff2.Values))
	}

	for i, v := range diff2.Values {
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
