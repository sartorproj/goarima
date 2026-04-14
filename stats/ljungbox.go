package stats

import (
	"math"

	"github.com/sartorproj/goarima/timeseries"
)

// LjungBoxResult represents the result of a Ljung-Box test.
type LjungBoxResult struct {
	Statistic float64
	PValue    float64
	Lags      int
	DOF       int // Degrees of freedom
}

// LjungBox performs the Ljung-Box test for autocorrelation in residuals.
// The null hypothesis is that there is no autocorrelation up to lag h.
// If p-value < 0.05, we reject the null and conclude there is significant autocorrelation.
// fitdf is the number of parameters estimated in the model (p + q for ARIMA).
func LjungBox(series *timeseries.Series, lags, fitdf int) *LjungBoxResult {
	n := series.Len()
	if n < 10 || lags < 1 {
		return nil
	}

	if lags >= n {
		lags = n - 1
	}

	// Calculate autocorrelations
	acf := ACF(series, lags)
	if acf == nil {
		return nil
	}

	// Ljung-Box Q statistic
	q := 0.0
	for k := 1; k <= lags; k++ {
		q += (acf[k] * acf[k]) / float64(n-k)
	}
	q *= float64(n * (n + 2))

	// Degrees of freedom
	dof := lags - fitdf
	if dof < 1 {
		dof = 1
	}

	// P-value from chi-squared distribution
	pValue := 1 - chiSquaredCDF(q, dof)

	return &LjungBoxResult{
		Statistic: q,
		PValue:    pValue,
		Lags:      lags,
		DOF:       dof,
	}
}

// BoxPierceResult represents the result of a Box-Pierce test.
type BoxPierceResult struct {
	Statistic float64
	PValue    float64
	Lags      int
	DOF       int
}

// BoxPierce performs the Box-Pierce test for autocorrelation.
// Similar to Ljung-Box but with a simpler formula.
func BoxPierce(series *timeseries.Series, lags, fitdf int) *BoxPierceResult {
	n := series.Len()
	if n < 10 || lags < 1 {
		return nil
	}

	if lags >= n {
		lags = n - 1
	}

	acf := ACF(series, lags)
	if acf == nil {
		return nil
	}

	// Box-Pierce Q statistic
	q := 0.0
	for k := 1; k <= lags; k++ {
		q += acf[k] * acf[k]
	}
	q *= float64(n)

	dof := lags - fitdf
	if dof < 1 {
		dof = 1
	}

	pValue := 1 - chiSquaredCDF(q, dof)

	return &BoxPierceResult{
		Statistic: q,
		PValue:    pValue,
		Lags:      lags,
		DOF:       dof,
	}
}

// chiSquaredCDF calculates the CDF of chi-squared distribution.
// Uses regularized incomplete gamma function for numerical stability.
func chiSquaredCDF(x float64, k int) float64 {
	if x < 0 {
		return 0
	}

	a := float64(k) / 2
	xHalf := x / 2

	if xHalf < a+1 {
		// Use series representation directly (regularized form)
		return gammaIncSeries(a, xHalf) / gamma(a)
	}
	// Use continued fraction: P(a,x) = 1 - Q(a,x)
	// Compute Q(a,x) = gammaIncCF(a,x)/gamma(a) directly to avoid catastrophic cancellation
	return 1 - gammaIncCF(a, xHalf)/gamma(a)
}

// gamma wraps math.Gamma from the standard library.
func gamma(z float64) float64 {
	return math.Gamma(z)
}

// lowerIncompleteGamma calculates the lower incomplete gamma function.
func lowerIncompleteGamma(a, x float64) float64 {
	if x < 0 || a <= 0 {
		return 0
	}

	if x < a+1 {
		return gammaIncSeries(a, x)
	}
	// For large x: γ(a,x) = Γ(a) - Γ_upper(a,x)
	// Compute via regularized form to avoid precision loss
	g := gamma(a)
	return g - gammaIncCF(a, x)
}

// gammaIncSeries calculates incomplete gamma using series expansion.
func gammaIncSeries(a, x float64) float64 {
	if x == 0 {
		return 0
	}

	maxIter := 200
	eps := 1e-10

	ap := a
	sum := 1.0 / a
	del := sum

	for n := 1; n < maxIter; n++ {
		ap++
		del *= x / ap
		sum += del
		if math.Abs(del) < math.Abs(sum)*eps {
			break
		}
	}

	return sum * math.Exp(-x+a*math.Log(x)-math.Log(gamma(a)))
}

// gammaIncCF calculates incomplete gamma using continued fraction.
func gammaIncCF(a, x float64) float64 {
	maxIter := 200
	eps := 1e-10
	fpmin := 1e-30

	b := x + 1 - a
	c := 1.0 / fpmin
	d := 1.0 / b
	h := d

	for i := 1; i < maxIter; i++ {
		an := -float64(i) * (float64(i) - a)
		b += 2
		d = an*d + b
		if math.Abs(d) < fpmin {
			d = fpmin
		}
		c = b + an/c
		if math.Abs(c) < fpmin {
			c = fpmin
		}
		d = 1.0 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < eps {
			break
		}
	}

	return math.Exp(-x+a*math.Log(x)-math.Log(gamma(a))) * h
}

// NormalQuantile returns the z-value for a given probability using
// the Abramowitz and Stegun rational approximation.
func NormalQuantile(p float64) float64 {
	if p <= 0 || p >= 1 {
		return 0
	}
	if p < 0.5 {
		return -NormalQuantile(1 - p)
	}

	t := math.Sqrt(-2 * math.Log(1-p))
	c0, c1, c2 := 2.515517, 0.802853, 0.010328
	d1, d2, d3 := 1.432788, 0.189269, 0.001308

	return t - (c0+c1*t+c2*t*t)/(1+d1*t+d2*t*t+d3*t*t*t)
}

// DurbinWatsonResult represents the result of a Durbin-Watson test.
type DurbinWatsonResult struct {
	Statistic float64
	// d ≈ 2: no autocorrelation
	// d < 2: positive autocorrelation
	// d > 2: negative autocorrelation
}

// DurbinWatson calculates the Durbin-Watson statistic for first-order autocorrelation.
func DurbinWatson(residuals []float64) *DurbinWatsonResult {
	n := len(residuals)
	if n < 2 {
		return nil
	}

	numerator := 0.0
	denominator := 0.0

	for i := 1; i < n; i++ {
		diff := residuals[i] - residuals[i-1]
		numerator += diff * diff
	}

	for _, r := range residuals {
		denominator += r * r
	}

	if denominator == 0 {
		return nil
	}

	return &DurbinWatsonResult{
		Statistic: numerator / denominator,
	}
}

// JarqueBeraResult represents the result of a Jarque-Bera normality test.
type JarqueBeraResult struct {
	Statistic float64
	PValue    float64
	Skewness  float64
	Kurtosis  float64
	IsNormal  bool // true if p-value >= 0.05
}

// JarqueBera performs the Jarque-Bera test for normality of residuals.
// H0: residuals are normally distributed. Reject if p-value < 0.05.
func JarqueBera(residuals []float64) *JarqueBeraResult {
	n := len(residuals)
	if n < 8 {
		return nil
	}

	// Calculate mean
	mean := 0.0
	for _, r := range residuals {
		mean += r
	}
	mean /= float64(n)

	// Calculate central moments
	m2, m3, m4 := 0.0, 0.0, 0.0
	for _, r := range residuals {
		d := r - mean
		d2 := d * d
		m2 += d2
		m3 += d2 * d
		m4 += d2 * d2
	}
	m2 /= float64(n)
	m3 /= float64(n)
	m4 /= float64(n)

	if m2 == 0 {
		return &JarqueBeraResult{Statistic: 0, PValue: 1, IsNormal: true}
	}

	// Skewness and excess kurtosis
	skewness := m3 / math.Pow(m2, 1.5)
	kurtosis := m4/(m2*m2) - 3.0 // excess kurtosis

	// JB statistic
	nf := float64(n)
	jb := nf / 6 * (skewness*skewness + kurtosis*kurtosis/4)

	// P-value from chi-squared(2) distribution
	pValue := 1 - chiSquaredCDF(jb, 2)

	return &JarqueBeraResult{
		Statistic: jb,
		PValue:    pValue,
		Skewness:  skewness,
		Kurtosis:  kurtosis,
		IsNormal:  pValue >= 0.05,
	}
}
