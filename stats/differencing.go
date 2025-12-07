package stats

import (
	"math"

	"github.com/sartorproj/goarima/timeseries"
)

// NDiffs determines the number of first differences required for stationarity.
// Uses KPSS test by default. Returns 0, 1, or 2.
// maxD is the maximum number of differences to consider (default 2).
// testType can be "kpss" (default) or "adf".
func NDiffs(series *timeseries.Series, maxD int, testType string) int {
	if maxD <= 0 {
		maxD = 2
	}
	if testType == "" {
		testType = "kpss"
	}

	current := series
	for d := 0; d < maxD; d++ {
		isStationary := false

		if testType == "adf" {
			result := ADF(current, 0)
			if result != nil && result.IsStationary {
				isStationary = true
			}
		} else {
			// KPSS test (default)
			result := KPSS(current, "c", 0)
			if result != nil && result.IsStationary {
				isStationary = true
			}
		}

		if isStationary {
			return d
		}

		// Apply differencing
		current = current.Diff()
		if current.Len() < 10 {
			return d
		}
	}

	return maxD
}

// NSDiffs determines the number of seasonal differences required.
// Uses seasonal strength measure: if F_S >= 0.64, one seasonal difference is suggested.
// period is the seasonal period (e.g., 12 for monthly data with yearly seasonality).
func NSDiffs(series *timeseries.Series, period int, maxD int) int {
	if maxD <= 0 {
		maxD = 1
	}
	if period <= 1 || series.Len() < 2*period {
		return 0
	}

	current := series
	for d := 0; d < maxD; d++ {
		// Calculate seasonal strength
		strength := seasonalStrength(current, period)

		// If seasonal strength < 0.64, no more seasonal differencing needed
		if strength < 0.64 {
			return d
		}

		// Apply seasonal differencing
		current = current.SeasonalDiff(period)
		if current.Len() < 2*period {
			return d
		}
	}

	return maxD
}

// seasonalStrength calculates the strength of seasonality (F_S).
// F_S = max(0, 1 - Var(R) / Var(S+R))
// where S is seasonal component and R is residual.
func seasonalStrength(series *timeseries.Series, period int) float64 {
	if series.Len() < 2*period {
		return 0
	}

	// Simple seasonal decomposition to get seasonal and residual
	decomp := Decompose(series, period, "additive")
	if decomp == nil {
		return 0
	}

	// Calculate variance of residuals
	varR := variance(decomp.Residual.Values)

	// Calculate variance of seasonal + residual
	seasonalPlusResid := make([]float64, len(decomp.Seasonal.Values))
	for i := range seasonalPlusResid {
		if !math.IsNaN(decomp.Seasonal.Values[i]) && !math.IsNaN(decomp.Residual.Values[i]) {
			seasonalPlusResid[i] = decomp.Seasonal.Values[i] + decomp.Residual.Values[i]
		}
	}
	varSR := variance(seasonalPlusResid)

	if varSR == 0 {
		return 0
	}

	strength := 1 - varR/varSR
	if strength < 0 {
		strength = 0
	}

	return strength
}

// variance calculates the variance of a slice, ignoring NaN values.
func variance(data []float64) float64 {
	// Filter out NaN values
	valid := make([]float64, 0, len(data))
	for _, v := range data {
		if !math.IsNaN(v) {
			valid = append(valid, v)
		}
	}

	n := len(valid)
	if n < 2 {
		return 0
	}

	// Calculate mean
	sum := 0.0
	for _, v := range valid {
		sum += v
	}
	mean := sum / float64(n)

	// Calculate variance
	sumSq := 0.0
	for _, v := range valid {
		diff := v - mean
		sumSq += diff * diff
	}

	return sumSq / float64(n-1)
}

// AICc calculates the corrected Akaike Information Criterion.
// AICc = AIC + 2(k)(k+1)/(n-k-1) where k is number of parameters.
// This corrects for small sample sizes.
func AICc(aic float64, nObs int, nParams int) float64 {
	k := float64(nParams)
	n := float64(nObs)

	if n-k-1 <= 0 {
		return math.Inf(1)
	}

	correction := 2 * k * (k + 1) / (n - k - 1)
	return aic + correction
}

// InformationCriteria calculates AIC, AICc, and BIC given model parameters.
type InformationCriteria struct {
	AIC    float64
	AICc   float64
	BIC    float64
	LogLik float64
}

// CalculateIC calculates all information criteria.
// logLik is the log-likelihood, nObs is the number of observations,
// nParams is the number of estimated parameters.
func CalculateIC(logLik float64, nObs int, nParams int) *InformationCriteria {
	k := float64(nParams)
	n := float64(nObs)

	aic := -2*logLik + 2*k
	bic := -2*logLik + k*math.Log(n)

	var aicc float64
	if n-k-1 > 0 {
		aicc = aic + 2*k*(k+1)/(n-k-1)
	} else {
		aicc = math.Inf(1)
	}

	return &InformationCriteria{
		AIC:    aic,
		AICc:   aicc,
		BIC:    bic,
		LogLik: logLik,
	}
}
