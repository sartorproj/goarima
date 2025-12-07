// Package autoarima implements automatic ARIMA model selection.
package autoarima

import (
	"math"

	"github.com/sartorproj/goarima/arima"
	"github.com/sartorproj/goarima/sarima"
	"github.com/sartorproj/goarima/stats"
	"github.com/sartorproj/goarima/timeseries"
)

// Config holds configuration for auto ARIMA search.
type Config struct {
	MaxP        int    // Maximum AR order (default: 5)
	MaxD        int    // Maximum differencing order (default: 2)
	MaxQ        int    // Maximum MA order (default: 5)
	MaxSP       int    // Maximum seasonal AR order (default: 2)
	MaxSD       int    // Maximum seasonal differencing order (default: 1)
	MaxSQ       int    // Maximum seasonal MA order (default: 2)
	Seasonal    bool   // Whether to consider seasonal models
	SeasonalM   int    // Seasonal period (required if Seasonal=true)
	Stepwise    bool   // Use stepwise search instead of exhaustive
	Criterion   string // Information criterion: "aic" or "bic" (default: "aic")
	Trace       bool   // Print progress
	StationTest string // Stationarity test: "adf" or "kpss" (default: "kpss")
}

// DefaultConfig returns the default auto ARIMA configuration.
func DefaultConfig() *Config {
	return &Config{
		MaxP:        5,
		MaxD:        2,
		MaxQ:        5,
		MaxSP:       2,
		MaxSD:       1,
		MaxSQ:       2,
		Seasonal:    false,
		Stepwise:    true,
		Criterion:   "aic",
		StationTest: "kpss",
	}
}

// Result represents the result of auto ARIMA model selection.
type Result struct {
	// Non-seasonal model (if no seasonality)
	Model *arima.Model
	// Seasonal model (if seasonal)
	SeasonalModel *sarima.Model

	// Best parameters found
	P  int
	D  int
	Q  int
	SP int
	SD int
	SQ int
	M  int

	// Model metrics
	AIC       float64
	BIC       float64
	LogLik    float64
	Criterion float64

	// Search information
	ModelsEvaluated int
	IsSeasonal      bool
}

// AutoARIMA automatically selects the best ARIMA or SARIMA model.
func AutoARIMA(series *timeseries.Series, config *Config) (*Result, error) {
	if config == nil {
		config = DefaultConfig()
	}

	// Determine the differencing order
	d := determineDifferencing(series, config.MaxD, config.StationTest)

	// Determine seasonal differencing if applicable
	sd := 0
	if config.Seasonal && config.SeasonalM > 0 {
		sd = determineSeasonalDifferencing(series, config.MaxSD, config.SeasonalM)
	}

	var result *Result

	if config.Seasonal && config.SeasonalM > 0 {
		result = searchSeasonal(series, d, sd, config)
	} else {
		result = searchNonSeasonal(series, d, config)
	}

	return result, nil
}

// determineDifferencing determines the optimal differencing order.
// Uses both KPSS and ADF tests for more robust detection.
func determineDifferencing(series *timeseries.Series, maxD int, testType string) int {
	currentSeries := series

	for d := 0; d < maxD; d++ {
		isStationary := false

		if testType == "adf" {
			// ADF test: H0 = non-stationary, reject if p < 0.05
			result := stats.ADF(currentSeries, 0)
			if result != nil && result.IsStationary {
				isStationary = true
			}
		} else {
			// Use both KPSS and ADF for more robust detection
			// KPSS: H0 = stationary, reject if p < 0.05
			// ADF: H0 = non-stationary, reject if p < 0.05
			kpssResult := stats.KPSS(currentSeries, "c", 0)
			adfResult := stats.ADF(currentSeries, 0)

			kpssStationary := kpssResult != nil && kpssResult.IsStationary
			adfStationary := adfResult != nil && adfResult.IsStationary

			// Consider stationary if both tests agree, or if KPSS strongly suggests it
			// and series variance is stable
			if kpssStationary && adfStationary {
				isStationary = true
			} else if kpssStationary && kpssResult.PValue > 0.1 {
				// KPSS strongly suggests stationarity
				isStationary = true
			}
		}

		if isStationary {
			return d
		}

		// Apply differencing
		currentSeries = currentSeries.Diff()
		if currentSeries.Len() < 10 {
			return d
		}
	}

	return maxD
}

// determineSeasonalDifferencing determines optimal seasonal differencing.
func determineSeasonalDifferencing(series *timeseries.Series, maxSD int, period int) int {
	// Check seasonal autocorrelation
	acf := stats.ACF(series, period*2)
	if acf == nil {
		return 0
	}

	// If strong autocorrelation at seasonal lag, likely need seasonal differencing
	if len(acf) > period && math.Abs(acf[period]) > 0.5 {
		return 1
	}

	return 0
}

// searchNonSeasonal performs ARIMA model search.
func searchNonSeasonal(series *timeseries.Series, d int, config *Config) *Result {
	bestResult := &Result{
		Criterion: math.Inf(1),
	}
	modelsEvaluated := 0

	if config.Stepwise {
		// Stepwise search
		return stepwiseSearchNonSeasonal(series, d, config)
	}

	// Exhaustive grid search
	for p := 0; p <= config.MaxP; p++ {
		for q := 0; q <= config.MaxQ; q++ {
			model := arima.New(p, d, q)
			err := model.Fit(series)
			if err != nil {
				continue
			}

			modelsEvaluated++

			var criterion float64
			if config.Criterion == "bic" {
				criterion = model.BIC
			} else {
				criterion = model.AIC
			}

			if criterion < bestResult.Criterion {
				bestResult = &Result{
					Model:           model,
					P:               p,
					D:               d,
					Q:               q,
					AIC:             model.AIC,
					BIC:             model.BIC,
					LogLik:          model.LogLik,
					Criterion:       criterion,
					ModelsEvaluated: modelsEvaluated,
					IsSeasonal:      false,
				}
			}
		}
	}

	bestResult.ModelsEvaluated = modelsEvaluated
	return bestResult
}

// stepwiseSearchNonSeasonal performs stepwise search for ARIMA.
func stepwiseSearchNonSeasonal(series *timeseries.Series, d int, config *Config) *Result {
	type modelSpec struct {
		p, q int
	}

	getCriterion := func(model *arima.Model) float64 {
		if config.Criterion == "bic" {
			return model.BIC
		}
		return model.AIC
	}

	// Start with simple models
	startModels := []modelSpec{
		{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 2},
	}

	bestSpec := modelSpec{0, 0}
	bestCriterion := math.Inf(1)
	var bestModel *arima.Model
	modelsEvaluated := 0

	// Evaluate starting models
	for _, spec := range startModels {
		if spec.p > config.MaxP || spec.q > config.MaxQ {
			continue
		}

		model := arima.New(spec.p, d, spec.q)
		err := model.Fit(series)
		if err != nil {
			continue
		}

		modelsEvaluated++
		criterion := getCriterion(model)

		if criterion < bestCriterion {
			bestCriterion = criterion
			bestSpec = spec
			bestModel = model
		}
	}

	// Stepwise refinement
	improved := true
	for improved {
		improved = false

		// Try neighboring models
		neighbors := []modelSpec{
			{bestSpec.p + 1, bestSpec.q},
			{bestSpec.p - 1, bestSpec.q},
			{bestSpec.p, bestSpec.q + 1},
			{bestSpec.p, bestSpec.q - 1},
			{bestSpec.p + 1, bestSpec.q + 1},
			{bestSpec.p - 1, bestSpec.q - 1},
		}

		for _, spec := range neighbors {
			if spec.p < 0 || spec.p > config.MaxP || spec.q < 0 || spec.q > config.MaxQ {
				continue
			}

			model := arima.New(spec.p, d, spec.q)
			err := model.Fit(series)
			if err != nil {
				continue
			}

			modelsEvaluated++
			criterion := getCriterion(model)

			if criterion < bestCriterion {
				bestCriterion = criterion
				bestSpec = spec
				bestModel = model
				improved = true
			}
		}
	}

	return &Result{
		Model:           bestModel,
		P:               bestSpec.p,
		D:               d,
		Q:               bestSpec.q,
		AIC:             bestModel.AIC,
		BIC:             bestModel.BIC,
		LogLik:          bestModel.LogLik,
		Criterion:       bestCriterion,
		ModelsEvaluated: modelsEvaluated,
		IsSeasonal:      false,
	}
}

// searchSeasonal performs SARIMA model search.
func searchSeasonal(series *timeseries.Series, d, sd int, config *Config) *Result {
	if config.Stepwise {
		return stepwiseSearchSeasonal(series, d, sd, config)
	}

	bestResult := &Result{
		Criterion:  math.Inf(1),
		IsSeasonal: true,
		M:          config.SeasonalM,
	}
	modelsEvaluated := 0

	// Grid search
	for p := 0; p <= config.MaxP; p++ {
		for q := 0; q <= config.MaxQ; q++ {
			for sp := 0; sp <= config.MaxSP; sp++ {
				for sq := 0; sq <= config.MaxSQ; sq++ {
					model := sarima.New(p, d, q, sp, sd, sq, config.SeasonalM)
					err := model.Fit(series)
					if err != nil {
						continue
					}

					modelsEvaluated++

					var criterion float64
					if config.Criterion == "bic" {
						criterion = model.BIC
					} else {
						criterion = model.AIC
					}

					if criterion < bestResult.Criterion {
						bestResult = &Result{
							SeasonalModel:   model,
							P:               p,
							D:               d,
							Q:               q,
							SP:              sp,
							SD:              sd,
							SQ:              sq,
							M:               config.SeasonalM,
							AIC:             model.AIC,
							BIC:             model.BIC,
							LogLik:          model.LogLik,
							Criterion:       criterion,
							ModelsEvaluated: modelsEvaluated,
							IsSeasonal:      true,
						}
					}
				}
			}
		}
	}

	bestResult.ModelsEvaluated = modelsEvaluated
	return bestResult
}

// stepwiseSearchSeasonal performs stepwise search for SARIMA.
func stepwiseSearchSeasonal(series *timeseries.Series, d, sd int, config *Config) *Result {
	type modelSpec struct {
		p, q, sp, sq int
	}

	getCriterion := func(model *sarima.Model) float64 {
		if config.Criterion == "bic" {
			return model.BIC
		}
		return model.AIC
	}

	// Starting models
	startModels := []modelSpec{
		{0, 0, 0, 0},
		{1, 0, 1, 0},
		{0, 1, 0, 1},
		{1, 1, 1, 1},
		{2, 2, 1, 1},
	}

	bestSpec := modelSpec{0, 0, 0, 0}
	bestCriterion := math.Inf(1)
	var bestModel *sarima.Model
	modelsEvaluated := 0

	for _, spec := range startModels {
		if spec.p > config.MaxP || spec.q > config.MaxQ ||
			spec.sp > config.MaxSP || spec.sq > config.MaxSQ {
			continue
		}

		model := sarima.New(spec.p, d, spec.q, spec.sp, sd, spec.sq, config.SeasonalM)
		err := model.Fit(series)
		if err != nil {
			continue
		}

		modelsEvaluated++
		criterion := getCriterion(model)

		if criterion < bestCriterion {
			bestCriterion = criterion
			bestSpec = spec
			bestModel = model
		}
	}

	// Stepwise refinement
	improved := true
	for improved {
		improved = false

		neighbors := []modelSpec{
			{bestSpec.p + 1, bestSpec.q, bestSpec.sp, bestSpec.sq},
			{bestSpec.p - 1, bestSpec.q, bestSpec.sp, bestSpec.sq},
			{bestSpec.p, bestSpec.q + 1, bestSpec.sp, bestSpec.sq},
			{bestSpec.p, bestSpec.q - 1, bestSpec.sp, bestSpec.sq},
			{bestSpec.p, bestSpec.q, bestSpec.sp + 1, bestSpec.sq},
			{bestSpec.p, bestSpec.q, bestSpec.sp - 1, bestSpec.sq},
			{bestSpec.p, bestSpec.q, bestSpec.sp, bestSpec.sq + 1},
			{bestSpec.p, bestSpec.q, bestSpec.sp, bestSpec.sq - 1},
		}

		for _, spec := range neighbors {
			if spec.p < 0 || spec.p > config.MaxP ||
				spec.q < 0 || spec.q > config.MaxQ ||
				spec.sp < 0 || spec.sp > config.MaxSP ||
				spec.sq < 0 || spec.sq > config.MaxSQ {
				continue
			}

			model := sarima.New(spec.p, d, spec.q, spec.sp, sd, spec.sq, config.SeasonalM)
			err := model.Fit(series)
			if err != nil {
				continue
			}

			modelsEvaluated++
			criterion := getCriterion(model)

			if criterion < bestCriterion {
				bestCriterion = criterion
				bestSpec = spec
				bestModel = model
				improved = true
			}
		}
	}

	if bestModel == nil {
		return &Result{
			Criterion:  math.Inf(1),
			IsSeasonal: true,
			M:          config.SeasonalM,
		}
	}

	return &Result{
		SeasonalModel:   bestModel,
		P:               bestSpec.p,
		D:               d,
		Q:               bestSpec.q,
		SP:              bestSpec.sp,
		SD:              sd,
		SQ:              bestSpec.sq,
		M:               config.SeasonalM,
		AIC:             bestModel.AIC,
		BIC:             bestModel.BIC,
		LogLik:          bestModel.LogLik,
		Criterion:       bestCriterion,
		ModelsEvaluated: modelsEvaluated,
		IsSeasonal:      true,
	}
}

// Predict generates forecasts using the selected model.
func (r *Result) Predict(steps int) ([]float64, error) {
	if r.IsSeasonal && r.SeasonalModel != nil {
		return r.SeasonalModel.Predict(steps)
	}
	if r.Model != nil {
		return r.Model.Predict(steps)
	}
	return nil, nil
}

// Residuals returns the model residuals.
func (r *Result) Residuals() []float64 {
	if r.IsSeasonal && r.SeasonalModel != nil {
		return r.SeasonalModel.Residuals()
	}
	if r.Model != nil {
		return r.Model.Residuals()
	}
	return nil
}
