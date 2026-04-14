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

// Mean calculates the arithmetic mean of the series, skipping NaN values.
func (s *Series) Mean() float64 {
	if len(s.Values) == 0 {
		return 0
	}
	sum := 0.0
	count := 0
	for _, v := range s.Values {
		if !math.IsNaN(v) {
			sum += v
			count++
		}
	}
	if count == 0 {
		return math.NaN()
	}
	return sum / float64(count)
}

// Variance calculates the variance of the series, skipping NaN values.
func (s *Series) Variance() float64 {
	if len(s.Values) < 2 {
		return 0
	}
	mean := s.Mean()
	if math.IsNaN(mean) {
		return math.NaN()
	}
	sumSq := 0.0
	count := 0
	for _, v := range s.Values {
		if !math.IsNaN(v) {
			diff := v - mean
			sumSq += diff * diff
			count++
		}
	}
	if count < 2 {
		return 0
	}
	return sumSq / float64(count-1)
}

// Std calculates the standard deviation of the series.
func (s *Series) Std() float64 {
	return math.Sqrt(s.Variance())
}

// Min returns the minimum value in the series, skipping NaN values.
func (s *Series) Min() float64 {
	if len(s.Values) == 0 {
		return math.NaN()
	}
	minVal := math.NaN()
	for _, v := range s.Values {
		if math.IsNaN(v) {
			continue
		}
		if math.IsNaN(minVal) || v < minVal {
			minVal = v
		}
	}
	return minVal
}

// Max returns the maximum value in the series, skipping NaN values.
func (s *Series) Max() float64 {
	if len(s.Values) == 0 {
		return math.NaN()
	}
	maxVal := math.NaN()
	for _, v := range s.Values {
		if math.IsNaN(v) {
			continue
		}
		if math.IsNaN(maxVal) || v > maxVal {
			maxVal = v
		}
	}
	return maxVal
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

// Diff calculates the first difference of the series: y[t] - y[t-1].
func (s *Series) Diff() *Series {
	return s.DiffLag(1)
}

// DiffN calculates the n-th order difference by applying Diff() n times.
// For example, DiffN(2) computes (y[t]-y[t-1]) - (y[t-1]-y[t-2]), NOT y[t]-y[t-2].
func (s *Series) DiffN(n int) *Series {
	if n <= 0 {
		return s.Copy()
	}
	result := s
	for i := 0; i < n; i++ {
		result = result.Diff()
		if result.Len() == 0 {
			return result
		}
	}
	return result
}

// DiffLag calculates the lag-k difference: y[t] - y[t-k].
func (s *Series) DiffLag(k int) *Series {
	if k <= 0 || len(s.Values) <= k {
		return &Series{Values: []float64{}}
	}

	result := make([]float64, len(s.Values)-k)
	for i := k; i < len(s.Values); i++ {
		result[i-k] = s.Values[i] - s.Values[i-k]
	}

	timestamps := make([]time.Time, len(result))
	if len(s.Timestamps) > k {
		copy(timestamps, s.Timestamps[k:])
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

// BoxCox applies the Box-Cox power transformation with parameter lambda.
//
//	lambda = 0: log(y)
//	lambda != 0: (y^lambda - 1) / lambda
//
// All values must be positive. Returns NaN for non-positive values.
func (s *Series) BoxCox(lambda float64) *Series {
	result := make([]float64, len(s.Values))
	for i, v := range s.Values {
		if v <= 0 {
			result[i] = math.NaN()
			continue
		}
		if math.Abs(lambda) < 1e-12 {
			result[i] = math.Log(v)
		} else {
			result[i] = (math.Pow(v, lambda) - 1) / lambda
		}
	}

	timestamps := make([]time.Time, len(s.Timestamps))
	copy(timestamps, s.Timestamps)

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_boxcox",
	}
}

// InverseBoxCox reverses the Box-Cox transformation.
//
//	lambda = 0: exp(y)
//	lambda != 0: (lambda*y + 1)^(1/lambda)
func (s *Series) InverseBoxCox(lambda float64) *Series {
	result := make([]float64, len(s.Values))
	for i, v := range s.Values {
		if math.Abs(lambda) < 1e-12 {
			result[i] = math.Exp(v)
		} else {
			inner := lambda*v + 1
			if inner <= 0 {
				result[i] = math.NaN()
			} else {
				result[i] = math.Pow(inner, 1/lambda)
			}
		}
	}

	timestamps := make([]time.Time, len(s.Timestamps))
	copy(timestamps, s.Timestamps)

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_inv_boxcox",
	}
}

// InverseBoxCoxValue applies the inverse Box-Cox transformation to a single scalar value.
func InverseBoxCoxValue(value, lambda float64) float64 {
	if math.Abs(lambda) < 1e-12 {
		return math.Exp(value)
	}
	inner := lambda*value + 1
	if inner <= 0 {
		return math.NaN()
	}
	return math.Pow(inner, 1/lambda)
}

// InverseBoxCoxWithBias applies bias-corrected inverse Box-Cox transformation.
// Standard inverse is biased: E[g⁻¹(X)] ≠ g⁻¹(E[X]) for nonlinear g.
// Correction: ŷ = g⁻¹(μ) · [1 + σ²·(1-λ) / (2·g⁻¹(μ)^(2λ))]
// Use this for back-transforming forecasts with known prediction variance.
func InverseBoxCoxWithBias(value, variance, lambda float64) float64 {
	if math.Abs(lambda) < 1e-12 {
		// Log case: E[exp(X)] = exp(μ + σ²/2)
		return math.Exp(value + variance/2)
	}
	inner := lambda*value + 1
	if inner <= 0 {
		return math.NaN()
	}
	base := math.Pow(inner, 1/lambda)
	correction := 1 + variance*(1-lambda)/(2*inner*inner)
	return base * correction
}

// BoxCoxLambda finds the optimal Box-Cox lambda via profile log-likelihood.
// Searches lambda in [-1, 2]. All series values must be positive.
// Prefers "round" lambdas (0, 0.5, 1, -1) if they're within tolerance of optimal.
func BoxCoxLambda(series *Series) (float64, error) {
	n := len(series.Values)
	if n < 4 {
		return 1, errors.New("series too short for Box-Cox lambda selection")
	}

	// Check all positive
	for _, v := range series.Values {
		if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return 1, errors.New("Box-Cox requires all positive, finite values")
		}
	}

	// Precompute log-sum for the Jacobian term
	logSum := 0.0
	for _, v := range series.Values {
		logSum += math.Log(v)
	}

	// Profile log-likelihood for Box-Cox:
	// L(λ) = -n/2 · log(var(z)) + (λ-1) · Σlog(y_i)
	// where z is the transformed series
	bestLambda := 1.0
	bestLL := math.Inf(-1)

	for li := -100; li <= 200; li++ {
		lambda := float64(li) / 100.0

		// Transform
		var sumZ, sumZ2 float64
		valid := true
		for _, v := range series.Values {
			var z float64
			if math.Abs(lambda) < 1e-12 {
				z = math.Log(v)
			} else {
				z = (math.Pow(v, lambda) - 1) / lambda
			}
			if math.IsNaN(z) || math.IsInf(z, 0) {
				valid = false
				break
			}
			sumZ += z
			sumZ2 += z * z
		}
		if !valid {
			continue
		}

		meanZ := sumZ / float64(n)
		varZ := sumZ2/float64(n) - meanZ*meanZ
		if varZ <= 0 {
			continue
		}

		ll := -float64(n)/2*math.Log(varZ) + (lambda-1)*logSum
		if ll > bestLL {
			bestLL = ll
			bestLambda = lambda
		}
	}

	// Prefer round values if within tolerance of optimal
	roundValues := []float64{-1, -0.5, 0, 1.0 / 3.0, 0.5, 1, 2}
	for _, rv := range roundValues {
		// Evaluate log-likelihood at round value
		var sumZ, sumZ2 float64
		for _, v := range series.Values {
			var z float64
			if math.Abs(rv) < 1e-12 {
				z = math.Log(v)
			} else {
				z = (math.Pow(v, rv) - 1) / rv
			}
			sumZ += z
			sumZ2 += z * z
		}
		meanZ := sumZ / float64(n)
		varZ := sumZ2/float64(n) - meanZ*meanZ
		if varZ <= 0 {
			continue
		}
		ll := -float64(n)/2*math.Log(varZ) + (rv-1)*logSum
		if bestLL-ll < 0.5 { // within 0.5 log-lik units
			bestLambda = rv
			bestLL = ll
		}
	}

	return bestLambda, nil
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

// InverseDiff reverses first-order differencing given the initial value.
// If y was differenced to produce z (z = diff(y)), then InverseDiff(z, y[0]) reconstructs y.
func (s *Series) InverseDiff(initialValue float64) *Series {
	result := make([]float64, len(s.Values)+1)
	result[0] = initialValue
	for i, v := range s.Values {
		result[i+1] = result[i] + v
	}

	timestamps := make([]time.Time, len(result))
	base := time.Now()
	for i := range timestamps {
		timestamps[i] = base.Add(time.Duration(i) * time.Hour)
	}

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_integrated",
	}
}

// InverseSeasonalDiff reverses seasonal differencing given the initial values.
// initialValues must have length m (the seasonal period).
func (s *Series) InverseSeasonalDiff(m int, initialValues []float64) *Series {
	if m <= 0 || len(initialValues) < m {
		return &Series{Values: []float64{}}
	}

	result := make([]float64, len(s.Values)+m)
	copy(result, initialValues[:m])
	for i, v := range s.Values {
		result[i+m] = v + result[i]
	}

	timestamps := make([]time.Time, len(result))
	base := time.Now()
	for i := range timestamps {
		timestamps[i] = base.Add(time.Duration(i) * time.Hour)
	}

	return &Series{
		Timestamps: timestamps,
		Values:     result,
		Name:       s.Name + "_seasonal_integrated",
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
