// Package timeseries provides core time series data structures and operations.
package timeseries

import (
	"errors"
	"math"
	"sort"
	"time"
)

// Series represents a time series with timestamps and values.
type Series struct {
	Timestamps []time.Time
	Values     []float64
	Name       string
}

// New creates a new time series from values.
func New(values []float64) *Series {
	timestamps := make([]time.Time, len(values))
	base := time.Now()
	for i := range timestamps {
		timestamps[i] = base.Add(time.Duration(i) * time.Hour)
	}
	return &Series{
		Timestamps: timestamps,
		Values:     values,
	}
}

// NewWithTimestamps creates a time series with explicit timestamps.
func NewWithTimestamps(timestamps []time.Time, values []float64) (*Series, error) {
	if len(timestamps) != len(values) {
		return nil, errors.New("timestamps and values must have the same length")
	}
	return &Series{
		Timestamps: timestamps,
		Values:     values,
	}, nil
}

// Len returns the length of the series.
func (s *Series) Len() int {
	return len(s.Values)
}

// Mean calculates the arithmetic mean of the series.
func (s *Series) Mean() float64 {
	if len(s.Values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range s.Values {
		sum += v
	}
	return sum / float64(len(s.Values))
}

// Variance calculates the variance of the series.
func (s *Series) Variance() float64 {
	if len(s.Values) < 2 {
		return 0
	}
	mean := s.Mean()
	sumSq := 0.0
	for _, v := range s.Values {
		diff := v - mean
		sumSq += diff * diff
	}
	return sumSq / float64(len(s.Values)-1)
}

// Std calculates the standard deviation of the series.
func (s *Series) Std() float64 {
	return math.Sqrt(s.Variance())
}

// Min returns the minimum value in the series.
func (s *Series) Min() float64 {
	if len(s.Values) == 0 {
		return math.NaN()
	}
	min := s.Values[0]
	for _, v := range s.Values[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

// Max returns the maximum value in the series.
func (s *Series) Max() float64 {
	if len(s.Values) == 0 {
		return math.NaN()
	}
	max := s.Values[0]
	for _, v := range s.Values[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// Median returns the median value of the series.
func (s *Series) Median() float64 {
	if len(s.Values) == 0 {
		return math.NaN()
	}
	sorted := make([]float64, len(s.Values))
	copy(sorted, s.Values)
	sort.Float64s(sorted)

	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

// Diff calculates the first difference of the series (d=1).
func (s *Series) Diff() *Series {
	return s.DiffN(1)
}

// DiffN calculates the n-th order difference of the series.
func (s *Series) DiffN(n int) *Series {
	if n <= 0 || len(s.Values) <= n {
		return &Series{Values: []float64{}}
	}

	result := make([]float64, len(s.Values)-n)
	for i := n; i < len(s.Values); i++ {
		result[i-n] = s.Values[i] - s.Values[i-n]
	}

	timestamps := make([]time.Time, len(result))
	if len(s.Timestamps) > n {
		copy(timestamps, s.Timestamps[n:])
	}

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_diff",
	}
}

// SeasonalDiff calculates the seasonal difference with period m.
func (s *Series) SeasonalDiff(m int) *Series {
	if m <= 0 || len(s.Values) <= m {
		return &Series{Values: []float64{}}
	}

	result := make([]float64, len(s.Values)-m)
	for i := m; i < len(s.Values); i++ {
		result[i-m] = s.Values[i] - s.Values[i-m]
	}

	timestamps := make([]time.Time, len(result))
	if len(s.Timestamps) > m {
		copy(timestamps, s.Timestamps[m:])
	}

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_seasonal_diff",
	}
}

// Lag returns a lagged version of the series.
func (s *Series) Lag(k int) *Series {
	if k <= 0 || k >= len(s.Values) {
		return &Series{Values: []float64{}}
	}

	result := make([]float64, len(s.Values)-k)
	copy(result, s.Values[:len(s.Values)-k])

	timestamps := make([]time.Time, len(result))
	if len(s.Timestamps) > k {
		copy(timestamps, s.Timestamps[k:])
	}

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_lag",
	}
}

// Slice returns a slice of the series from start to end (exclusive).
func (s *Series) Slice(start, end int) *Series {
	if start < 0 {
		start = 0
	}
	if end > len(s.Values) {
		end = len(s.Values)
	}
	if start >= end {
		return &Series{Values: []float64{}}
	}

	values := make([]float64, end-start)
	copy(values, s.Values[start:end])

	timestamps := make([]time.Time, len(values))
	if len(s.Timestamps) >= end {
		copy(timestamps, s.Timestamps[start:end])
	}

	return &Series{
		Timestamps: timestamps,
		Values:     values,
		Name:       s.Name,
	}
}

// Copy creates a deep copy of the series.
func (s *Series) Copy() *Series {
	values := make([]float64, len(s.Values))
	copy(values, s.Values)

	timestamps := make([]time.Time, len(s.Timestamps))
	copy(timestamps, s.Timestamps)

	return &Series{
		Timestamps: timestamps,
		Values:     values,
		Name:       s.Name,
	}
}

// Log applies natural logarithm transformation.
func (s *Series) Log() *Series {
	result := make([]float64, len(s.Values))
	for i, v := range s.Values {
		if v > 0 {
			result[i] = math.Log(v)
		} else {
			result[i] = math.NaN()
		}
	}

	timestamps := make([]time.Time, len(s.Timestamps))
	copy(timestamps, s.Timestamps)

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_log",
	}
}

// MovingAverage calculates a simple moving average with window size.
func (s *Series) MovingAverage(window int) *Series {
	if window <= 0 || window > len(s.Values) {
		return &Series{Values: []float64{}}
	}

	result := make([]float64, len(s.Values)-window+1)
	sum := 0.0

	for i := 0; i < window; i++ {
		sum += s.Values[i]
	}
	result[0] = sum / float64(window)

	for i := window; i < len(s.Values); i++ {
		sum = sum - s.Values[i-window] + s.Values[i]
		result[i-window+1] = sum / float64(window)
	}

	timestamps := make([]time.Time, len(result))
	if len(s.Timestamps) >= window {
		copy(timestamps, s.Timestamps[window-1:])
	}

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_ma",
	}
}

// Normalize standardizes the series (z-score normalization).
func (s *Series) Normalize() *Series {
	mean := s.Mean()
	std := s.Std()

	if std == 0 {
		return s.Copy()
	}

	result := make([]float64, len(s.Values))
	for i, v := range s.Values {
		result[i] = (v - mean) / std
	}

	timestamps := make([]time.Time, len(s.Timestamps))
	copy(timestamps, s.Timestamps)

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_normalized",
	}
}
