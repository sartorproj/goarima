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
func chiSquaredCDF(x float64, k int) float64 {
	if x < 0 {
		return 0
	}

	// Use incomplete gamma function: P(k/2, x/2)
	return lowerIncompleteGamma(float64(k)/2, x/2) / gamma(float64(k)/2)
}

// gamma calculates the gamma function using Lanczos approximation.
func gamma(z float64) float64 {
	if z < 0.5 {
		return math.Pi / (math.Sin(math.Pi*z) * gamma(1-z))
	}

	z--
	g := 7
	c := []float64{
		0.99999999999980993,
		676.5203681218851,
		-1259.1392167224028,
		771.32342877765313,
		-176.61502916214059,
		12.507343278686905,
		-0.13857109526572012,
		9.9843695780195716e-6,
		1.5056327351493116e-7,
	}

	x := c[0]
	for i := 1; i < g+2; i++ {
		x += c[i] / (z + float64(i))
	}

	t := z + float64(g) + 0.5
	return math.Sqrt(2*math.Pi) * math.Pow(t, z+0.5) * math.Exp(-t) * x
}

// lowerIncompleteGamma calculates the lower incomplete gamma function.
func lowerIncompleteGamma(a, x float64) float64 {
	if x < 0 || a <= 0 {
		return 0
	}

	if x < a+1 {
		// Use series representation
		return gammaIncSeries(a, x)
	}
	// Use continued fraction
	return gamma(a) - gammaIncCF(a, x)
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
