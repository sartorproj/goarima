// Package autoarima implements automatic ARIMA model selection with
// automatic seasonality detection, model comparison, and cross-validation.
//
// Key features:
//   - Automatic seasonal period detection from ACF analysis
//   - Compares ARIMA vs SARIMA and selects the best model
//   - Cross-validation based model selection (RMSE/MAPE)
//   - Follows R's auto.arima methodology with enhancements
package autoarima

import (
	"math"
	"sort"

	"github.com/sartorproj/goarima/arima"
	"github.com/sartorproj/goarima/sarima"
	"github.com/sartorproj/goarima/stats"
	"github.com/sartorproj/goarima/timeseries"
)

// Common seasonal periods for different data frequencies.
// These are checked during automatic period detection.
var (
	// HourlyPeriods are common periods for hourly data
	HourlyPeriods = []int{6, 12, 24, 48, 168} // 6h, 12h, daily, 2-day, weekly
	// DailyPeriods are common periods for daily data
	DailyPeriods = []int{7, 14, 30, 365} // weekly, biweekly, monthly, yearly
	// DefaultPeriods is the default set of periods to check
	DefaultPeriods = []int{4, 6, 7, 12, 24, 52, 168, 365}
)

// Config holds configuration for auto ARIMA search.
type Config struct {
	// Order constraints
	MaxP  int // Maximum AR order (default: 5)
	MaxD  int // Maximum differencing order (default: 2)
	MaxQ  int // Maximum MA order (default: 5)
	MaxSP int // Maximum seasonal AR order (default: 2)
	MaxSD int // Maximum seasonal differencing order (default: 1)
	MaxSQ int // Maximum seasonal MA order (default: 2)

	// Search settings
	Stepwise    bool   // Use stepwise search instead of exhaustive (default: true)
	Criterion   string // Information criterion: "aic", "aicc", or "bic" (default: "aicc")
	StationTest string // Stationarity test: "adf" or "kpss" (default: "kpss")

	// Seasonality settings
	AutoSeasonal         bool    // Automatically detect seasonality (default: true)
	SeasonalPeriods      []int   // Periods to check (default: DefaultPeriods)
	SeasonalityThreshold float64 // ACF threshold for seasonality detection (default: 0.4)
	MinSeasonalPeriod    int     // Minimum period to consider (default: 4)
	MaxSeasonalPeriod    int     // Maximum period to check (default: 168)

	// Model selection settings
	ModelSelection   string  // "cv", "aicc", "aic", "bic" (default: "cv" for cross-validation)
	CVFolds          int     // Number of CV folds for time series CV (default: 5)
	TestRatio        float64 // Ratio of data for test set in CV (default: 0.2)
	CompareModels    bool    // Compare seasonal vs non-seasonal (default: true)
	PreferSimpler    bool    // Prefer simpler model if CV scores are close (default: true)
	SimplerThreshold float64 // RMSE difference threshold to prefer simpler model (default: 0.05)

	// Trace/debug settings
	Trace bool // Print progress (default: false)
}

// DefaultConfig returns the default auto ARIMA configuration.
// By default, it automatically detects seasonality and uses cross-validation
// for model selection.
func DefaultConfig() *Config {
	return &Config{
		// Order constraints
		MaxP:  5,
		MaxD:  2,
		MaxQ:  5,
		MaxSP: 2,
		MaxSD: 1,
		MaxSQ: 2,

		// Search settings
		Stepwise:    true,
		Criterion:   "aicc",
		StationTest: "kpss",

		// Seasonality settings - AUTO-DETECT BY DEFAULT
		AutoSeasonal:         true,
		SeasonalPeriods:      DefaultPeriods,
		SeasonalityThreshold: 0.4,
		MinSeasonalPeriod:    4,
		MaxSeasonalPeriod:    168,

		// Model selection - CROSS-VALIDATION BY DEFAULT
		ModelSelection:   "cv",
		CVFolds:          5,
		TestRatio:        0.2,
		CompareModels:    true,
		PreferSimpler:    true,
		SimplerThreshold: 0.05,

		Trace: false,
	}
}

// ModelCandidate represents a candidate model with its evaluation metrics.
type ModelCandidate struct {
	Name       string // e.g., "ARIMA(1,1,1)" or "SARIMA(1,1,1)(1,0,1)[24]"
	IsSeasonal bool   // Whether this is a seasonal model
	Period     int    // Seasonal period (0 for non-seasonal)
	P, D, Q    int    // Non-seasonal orders
	SP, SD, SQ int    // Seasonal orders

	// Fitted model
	ARIMAModel  *arima.Model
	SARIMAModel *sarima.Model

	// Information criteria
	AIC    float64
	AICc   float64
	BIC    float64
	LogLik float64

	// Cross-validation metrics
	RMSE     float64 // Root mean squared error
	MAE      float64 // Mean absolute error
	MAPE     float64 // Mean absolute percentage error
	CVScores []float64

	// Selection
	Selected bool
	Rank     int // Rank among candidates (1 = best)
}

// Result represents the result of auto ARIMA model selection.
type Result struct {
	// Selected model (one will be non-nil)
	Model         *arima.Model
	SeasonalModel *sarima.Model

	// Best parameters found
	P  int
	D  int
	Q  int
	SP int
	SD int
	SQ int
	M  int // Seasonal period

	// Model metrics
	AIC    float64
	AICc   float64
	BIC    float64
	LogLik float64

	// Cross-validation metrics
	RMSE float64
	MAE  float64
	MAPE float64

	// Search information
	ModelsEvaluated int
	IsSeasonal      bool

	// Seasonality detection
	DetectedPeriod      int     // Auto-detected period (0 if none)
	SeasonalityStrength float64 // ACF value at detected period
	DetectionMethod     string  // "acf", "spectral", or "none"

	// ACF/PACF suggested orders (for diagnostics)
	SuggestedP  int
	SuggestedQ  int
	SuggestedSP int
	SuggestedSQ int

	// Model comparison (all candidates evaluated)
	Candidates []ModelCandidate
}

// AutoARIMA automatically selects the best ARIMA or SARIMA model.
//
// The function:
//  1. Automatically detects if the series has seasonality
//  2. Fits both ARIMA and SARIMA models (if seasonality detected)
//  3. Selects the best model using cross-validation
//
// Example:
//
//	// Fully automatic - detects seasonality and selects best model
//	result, err := autoarima.AutoARIMA(series, nil)
//
//	// With custom config
//	cfg := autoarima.DefaultConfig()
//	cfg.SeasonalPeriods = []int{24, 168}  // Only check daily/weekly
//	result, err := autoarima.AutoARIMA(series, cfg)
func AutoARIMA(series *timeseries.Series, config *Config) (*Result, error) {
	if config == nil {
		config = DefaultConfig()
	}

	n := series.Len()
	if n < 10 {
		return nil, nil
	}

	// Step 1: Detect seasonality if enabled
	detectedPeriod := 0
	seasonalityStrength := 0.0
	detectionMethod := "none"

	if config.AutoSeasonal {
		detectedPeriod, seasonalityStrength = detectSeasonalPeriod(series, config)
		if detectedPeriod > 0 {
			detectionMethod = "acf"
		}
	}

	// Step 2: Determine differencing orders
	d := determineDifferencing(series, config.MaxD, config.StationTest)

	sd := 0
	if detectedPeriod > 0 && n >= detectedPeriod*2 {
		sd = determineSeasonalDifferencing(series, config.MaxSD, detectedPeriod)
	}

	// Step 3: Fit candidate models
	var candidates []ModelCandidate

	// Always fit non-seasonal ARIMA as baseline
	arimaCandidate := fitBestARIMA(series, d, config)
	if arimaCandidate != nil {
		candidates = append(candidates, *arimaCandidate)
	}

	// Fit seasonal SARIMA if seasonality detected
	if detectedPeriod > 0 && n >= detectedPeriod*2 && config.CompareModels {
		sarimaCandidate := fitBestSARIMA(series, d, sd, detectedPeriod, config)
		if sarimaCandidate != nil {
			candidates = append(candidates, *sarimaCandidate)
		}
	}

	if len(candidates) == 0 {
		return nil, nil
	}

	// Step 4: Evaluate candidates using cross-validation
	if config.ModelSelection == "cv" {
		for i := range candidates {
			evaluateWithCV(series, &candidates[i], config)
		}
	}

	// Step 5: Select best model
	best := selectBestModel(candidates, config)

	// Build result
	result := &Result{
		P:                   best.P,
		D:                   best.D,
		Q:                   best.Q,
		SP:                  best.SP,
		SD:                  best.SD,
		SQ:                  best.SQ,
		M:                   best.Period,
		AIC:                 best.AIC,
		AICc:                best.AICc,
		BIC:                 best.BIC,
		LogLik:              best.LogLik,
		RMSE:                best.RMSE,
		MAE:                 best.MAE,
		MAPE:                best.MAPE,
		ModelsEvaluated:     countModelsEvaluated(candidates),
		IsSeasonal:          best.IsSeasonal,
		DetectedPeriod:      detectedPeriod,
		SeasonalityStrength: seasonalityStrength,
		DetectionMethod:     detectionMethod,
		Candidates:          candidates,
	}

	if best.IsSeasonal {
		result.SeasonalModel = best.SARIMAModel
	} else {
		result.Model = best.ARIMAModel
	}

	return result, nil
}

// detectSeasonalPeriod detects the seasonal period from ACF analysis.
// Returns (period, strength) where strength is the ACF value at that lag.
//
// The algorithm:
//  1. Calculates ACF up to maxSeasonalPeriod
//  2. Checks common periods for significant ACF values
//  3. Also looks for local maxima in ACF
//  4. Returns the period with highest ACF above threshold
func detectSeasonalPeriod(series *timeseries.Series, config *Config) (period int, strength float64) {
	n := series.Len()
	maxLag := min(config.MaxSeasonalPeriod, n/3)
	if maxLag < config.MinSeasonalPeriod {
		return 0, 0
	}

	acf := stats.ACF(series, maxLag)
	if acf == nil {
		return 0, 0
	}

	type periodScore struct {
		period   int
		strength float64
	}
	var candidates []periodScore

	// Check configured periods
	for _, period := range config.SeasonalPeriods {
		if period >= config.MinSeasonalPeriod && period <= maxLag && period < len(acf) {
			if acf[period] > config.SeasonalityThreshold {
				candidates = append(candidates, periodScore{period, acf[period]})
			}
		}
	}

	// Also check for local maxima in ACF (periods not in the list)
	for i := config.MinSeasonalPeriod; i < len(acf)-1 && i <= maxLag; i++ {
		// Local maximum check
		if acf[i] > acf[i-1] && acf[i] > acf[i+1] && acf[i] > config.SeasonalityThreshold {
			// Check if already in candidates
			found := false
			for _, c := range candidates {
				if c.period == i {
					found = true
					break
				}
			}
			if !found {
				candidates = append(candidates, periodScore{i, acf[i]})
			}
		}
	}

	if len(candidates) == 0 {
		return 0, 0
	}

	// Sort by strength (descending)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].strength > candidates[j].strength
	})

	return candidates[0].period, candidates[0].strength
}

// determineDifferencing determines the optimal differencing order using stationarity tests.
func determineDifferencing(series *timeseries.Series, maxD int, testType string) int {
	currentSeries := series

	for d := 0; d < maxD; d++ {
		isStationary := false

		if testType == "adf" {
			result := stats.ADF(currentSeries, 0)
			if result != nil && result.IsStationary {
				isStationary = true
			}
		} else {
			// Use both KPSS and ADF for more robust detection
			kpssResult := stats.KPSS(currentSeries, "c", 0)
			adfResult := stats.ADF(currentSeries, 0)

			kpssStationary := kpssResult != nil && kpssResult.IsStationary
			adfStationary := adfResult != nil && adfResult.IsStationary

			if kpssStationary && adfStationary {
				isStationary = true
			} else if kpssStationary && kpssResult.PValue > 0.1 {
				isStationary = true
			}
		}

		if isStationary {
			return d
		}

		currentSeries = currentSeries.Diff()
		if currentSeries.Len() < 10 {
			return d
		}
	}

	return maxD
}

// determineSeasonalDifferencing determines optimal seasonal differencing.
func determineSeasonalDifferencing(series *timeseries.Series, _, period int) int {
	acf := stats.ACF(series, period*2)
	if acf == nil {
		return 0
	}

	// If strong autocorrelation at seasonal lag, need seasonal differencing
	if len(acf) > period && math.Abs(acf[period]) > 0.5 {
		return 1
	}

	return 0
}

// fitBestARIMA fits the best non-seasonal ARIMA model.
func fitBestARIMA(series *timeseries.Series, d int, config *Config) *ModelCandidate {
	type modelSpec struct {
		p, q int
	}

	getCriterion := func(model *arima.Model) float64 {
		switch config.Criterion {
		case "bic":
			return model.BIC
		case "aicc":
			return model.AICc
		default:
			return model.AIC
		}
	}

	// Get ACF/PACF suggestions
	diffSeries := series
	for i := 0; i < d; i++ {
		diffSeries = diffSeries.Diff()
		if diffSeries.Len() < 10 {
			break
		}
	}
	suggestedP, suggestedQ := suggestOrdersFromACF(diffSeries, config.MaxP, config.MaxQ)

	// Starting models
	startModels := []modelSpec{
		{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 2},
		{2, 0}, {0, 2}, {2, 1}, {1, 2}, {3, 0}, {0, 3},
	}

	if suggestedP > 0 || suggestedQ > 0 {
		startModels = append(startModels,
			modelSpec{suggestedP, suggestedQ},
			modelSpec{suggestedP, 0},
			modelSpec{0, suggestedQ},
			modelSpec{suggestedP + 1, suggestedQ},
			modelSpec{suggestedP, suggestedQ + 1},
		)
	}

	evaluated := make(map[modelSpec]bool)
	bestSpec := modelSpec{0, 0}
	bestCriterion := math.Inf(1)
	var bestModel *arima.Model

	// Evaluate starting models
	for _, spec := range startModels {
		if spec.p < 0 || spec.p > config.MaxP || spec.q < 0 || spec.q > config.MaxQ {
			continue
		}
		if evaluated[spec] {
			continue
		}
		evaluated[spec] = true

		model := arima.New(spec.p, d, spec.q)
		if err := model.Fit(series); err != nil {
			continue
		}

		criterion := getCriterion(model)
		if criterion < bestCriterion {
			bestCriterion = criterion
			bestSpec = spec
			bestModel = model
		}
	}

	// Stepwise refinement
	if config.Stepwise {
		improved := true
		for improved {
			improved = false

			neighbors := []modelSpec{
				{bestSpec.p + 1, bestSpec.q}, {bestSpec.p - 1, bestSpec.q},
				{bestSpec.p, bestSpec.q + 1}, {bestSpec.p, bestSpec.q - 1},
				{bestSpec.p + 1, bestSpec.q + 1}, {bestSpec.p - 1, bestSpec.q - 1},
				{bestSpec.p + 1, bestSpec.q - 1}, {bestSpec.p - 1, bestSpec.q + 1},
			}

			for _, spec := range neighbors {
				if spec.p < 0 || spec.p > config.MaxP || spec.q < 0 || spec.q > config.MaxQ {
					continue
				}
				if evaluated[spec] {
					continue
				}
				evaluated[spec] = true

				model := arima.New(spec.p, d, spec.q)
				if err := model.Fit(series); err != nil {
					continue
				}

				criterion := getCriterion(model)
				if criterion < bestCriterion {
					bestCriterion = criterion
					bestSpec = spec
					bestModel = model
					improved = true
				}
			}
		}
	}

	if bestModel == nil {
		return nil
	}

	return &ModelCandidate{
		Name:       formatARIMAOrder(bestSpec.p, d, bestSpec.q),
		IsSeasonal: false,
		P:          bestSpec.p,
		D:          d,
		Q:          bestSpec.q,
		ARIMAModel: bestModel,
		AIC:        bestModel.AIC,
		AICc:       bestModel.AICc,
		BIC:        bestModel.BIC,
		LogLik:     bestModel.LogLik,
	}
}

// fitBestSARIMA fits the best seasonal SARIMA model.
func fitBestSARIMA(series *timeseries.Series, d, sd, period int, config *Config) *ModelCandidate {
	type modelSpec struct {
		p, q, sp, sq int
	}

	getCriterion := func(model *sarima.Model) float64 {
		switch config.Criterion {
		case "bic":
			return model.BIC
		case "aicc":
			return model.AICc
		default:
			return model.AIC
		}
	}

	// Get ACF/PACF suggestions
	diffSeries := series
	for i := 0; i < d; i++ {
		diffSeries = diffSeries.Diff()
		if diffSeries.Len() < 10 {
			break
		}
	}
	for i := 0; i < sd && period > 0; i++ {
		diffSeries = diffSeries.SeasonalDiff(period)
		if diffSeries.Len() < 10 {
			break
		}
	}

	suggestedP, suggestedQ := suggestOrdersFromACF(diffSeries, config.MaxP, config.MaxQ)
	suggestedSP, suggestedSQ := suggestSeasonalOrdersFromACF(diffSeries, period, config.MaxSP, config.MaxSQ)

	// Starting models - reduced for seasonal to limit computation
	startModels := []modelSpec{
		{0, 0, 0, 0}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 1, 1, 1},
		{2, 0, 1, 0}, {0, 2, 0, 1}, {1, 0, 0, 1}, {0, 1, 1, 0},
	}

	if suggestedP > 0 || suggestedQ > 0 || suggestedSP > 0 || suggestedSQ > 0 {
		startModels = append(startModels,
			modelSpec{suggestedP, suggestedQ, suggestedSP, suggestedSQ},
			modelSpec{suggestedP, suggestedQ, 0, 0},
			modelSpec{0, 0, suggestedSP, suggestedSQ},
		)
	}

	evaluated := make(map[modelSpec]bool)
	bestSpec := modelSpec{}
	bestCriterion := math.Inf(1)
	var bestModel *sarima.Model

	for _, spec := range startModels {
		if spec.p < 0 || spec.p > config.MaxP || spec.q < 0 || spec.q > config.MaxQ ||
			spec.sp < 0 || spec.sp > config.MaxSP || spec.sq < 0 || spec.sq > config.MaxSQ {
			continue
		}
		if evaluated[spec] {
			continue
		}
		evaluated[spec] = true

		model := sarima.New(spec.p, d, spec.q, spec.sp, sd, spec.sq, period)
		if err := model.Fit(series); err != nil {
			continue
		}

		criterion := getCriterion(model)
		if criterion < bestCriterion {
			bestCriterion = criterion
			bestSpec = spec
			bestModel = model
		}
	}

	// Stepwise refinement
	if config.Stepwise {
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
				if spec.p < 0 || spec.p > config.MaxP || spec.q < 0 || spec.q > config.MaxQ ||
					spec.sp < 0 || spec.sp > config.MaxSP || spec.sq < 0 || spec.sq > config.MaxSQ {
					continue
				}
				if evaluated[spec] {
					continue
				}
				evaluated[spec] = true

				model := sarima.New(spec.p, d, spec.q, spec.sp, sd, spec.sq, period)
				if err := model.Fit(series); err != nil {
					continue
				}

				criterion := getCriterion(model)
				if criterion < bestCriterion {
					bestCriterion = criterion
					bestSpec = spec
					bestModel = model
					improved = true
				}
			}
		}
	}

	if bestModel == nil {
		return nil
	}

	return &ModelCandidate{
		Name:        formatSARIMAOrder(bestSpec.p, d, bestSpec.q, bestSpec.sp, sd, bestSpec.sq, period),
		IsSeasonal:  true,
		Period:      period,
		P:           bestSpec.p,
		D:           d,
		Q:           bestSpec.q,
		SP:          bestSpec.sp,
		SD:          sd,
		SQ:          bestSpec.sq,
		SARIMAModel: bestModel,
		AIC:         bestModel.AIC,
		AICc:        bestModel.AICc,
		BIC:         bestModel.BIC,
		LogLik:      bestModel.LogLik,
	}
}

// evaluateWithCV evaluates a candidate using time series cross-validation.
func evaluateWithCV(series *timeseries.Series, candidate *ModelCandidate, config *Config) {
	n := series.Len()
	testSize := int(float64(n) * config.TestRatio)
	if testSize < 5 {
		testSize = 5
	}
	if testSize > n/2 {
		testSize = n / 2
	}

	trainSize := n - testSize
	train := series.Slice(0, trainSize)
	test := series.Slice(trainSize, n)

	// Refit model on training data
	var forecasts []float64
	var err error

	if candidate.IsSeasonal {
		model := sarima.New(candidate.P, candidate.D, candidate.Q, candidate.SP, candidate.SD, candidate.SQ, candidate.Period)
		if err = model.Fit(train); err == nil {
			forecasts, err = model.Predict(testSize)
		}
	} else {
		model := arima.New(candidate.P, candidate.D, candidate.Q)
		if err = model.Fit(train); err == nil {
			forecasts, err = model.Predict(testSize)
		}
	}

	if err != nil || forecasts == nil {
		candidate.RMSE = math.Inf(1)
		candidate.MAE = math.Inf(1)
		candidate.MAPE = math.Inf(1)
		return
	}

	// Calculate metrics
	candidate.RMSE, candidate.MAE, candidate.MAPE = calculateMetrics(test.Values, forecasts)
}

// calculateMetrics calculates RMSE, MAE, and MAPE.
func calculateMetrics(actual, predicted []float64) (rmse, mae, mape float64) {
	n := min(len(actual), len(predicted))
	if n == 0 {
		return math.Inf(1), math.Inf(1), math.Inf(1)
	}

	for i := 0; i < n; i++ {
		d := actual[i] - predicted[i]
		rmse += d * d
		mae += math.Abs(d)
		if actual[i] != 0 {
			mape += math.Abs(d) / math.Abs(actual[i]) * 100
		}
	}

	return math.Sqrt(rmse / float64(n)), mae / float64(n), mape / float64(n)
}

// selectBestModel selects the best model from candidates.
func selectBestModel(candidates []ModelCandidate, config *Config) *ModelCandidate {
	if len(candidates) == 0 {
		return nil
	}

	// Score and rank candidates
	for i := range candidates {
		candidates[i].Selected = false
	}

	// Sort based on selection criteria
	switch config.ModelSelection {
	case "cv":
		// Primary: RMSE, Secondary: MAPE, Tertiary: model complexity
		sort.Slice(candidates, func(i, j int) bool {
			// If RMSE is very close, prefer simpler model
			rmseDiff := math.Abs(candidates[i].RMSE-candidates[j].RMSE) / (candidates[i].RMSE + 1e-10)
			if config.PreferSimpler && rmseDiff < config.SimplerThreshold {
				// Prefer non-seasonal over seasonal if scores are close
				if candidates[i].IsSeasonal != candidates[j].IsSeasonal {
					return !candidates[i].IsSeasonal
				}
				// Prefer lower order
				complexityI := candidates[i].P + candidates[i].Q + candidates[i].SP + candidates[i].SQ
				complexityJ := candidates[j].P + candidates[j].Q + candidates[j].SP + candidates[j].SQ
				return complexityI < complexityJ
			}
			return candidates[i].RMSE < candidates[j].RMSE
		})
	case "aicc":
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].AICc < candidates[j].AICc
		})
	case "aic":
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].AIC < candidates[j].AIC
		})
	case "bic":
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].BIC < candidates[j].BIC
		})
	}

	// Assign ranks and select best
	for i := range candidates {
		candidates[i].Rank = i + 1
	}
	candidates[0].Selected = true

	return &candidates[0]
}

// Helper functions

func suggestOrdersFromACF(series *timeseries.Series, maxP, maxQ int) (p, q int) {
	n := series.Len()
	confBound := 1.96 / math.Sqrt(float64(n))

	maxLag := max(maxP, maxQ)
	if maxLag > n/4 {
		maxLag = n / 4
	}
	if maxLag < 1 {
		return 0, 0
	}

	pacf := stats.PACF(series, maxLag)
	acf := stats.ACF(series, maxLag)

	if pacf == nil || acf == nil {
		return 0, 0
	}

	suggestedP := 0
	for i := 1; i < len(pacf) && i <= maxP; i++ {
		if math.Abs(pacf[i]) > confBound {
			suggestedP = i
		}
	}

	suggestedQ := 0
	for i := 1; i < len(acf) && i <= maxQ; i++ {
		if math.Abs(acf[i]) > confBound {
			suggestedQ = i
		}
	}

	// Limit suggestions if many significant lags
	pacfSig := countSignificantLags(pacf, confBound, maxP)
	acfSig := countSignificantLags(acf, confBound, maxQ)

	if pacfSig > 3 && suggestedP > 2 {
		suggestedP = min(2, suggestedP)
	}
	if acfSig > 3 && suggestedQ > 2 {
		suggestedQ = min(2, suggestedQ)
	}

	return suggestedP, suggestedQ
}

func suggestSeasonalOrdersFromACF(series *timeseries.Series, period, maxSP, maxSQ int) (sp, sq int) {
	n := series.Len()
	confBound := 1.96 / math.Sqrt(float64(n))

	maxLag := period * max(maxSP, maxSQ)
	if maxLag > n/2 {
		maxLag = n / 2
	}
	if maxLag < period {
		return 0, 0
	}

	pacf := stats.PACF(series, maxLag)
	acf := stats.ACF(series, maxLag)

	if pacf == nil || acf == nil {
		return 0, 0
	}

	suggestedSP := 0
	for k := 1; k <= maxSP; k++ {
		lag := k * period
		if lag < len(pacf) && math.Abs(pacf[lag]) > confBound {
			suggestedSP = k
		}
	}

	suggestedSQ := 0
	for k := 1; k <= maxSQ; k++ {
		lag := k * period
		if lag < len(acf) && math.Abs(acf[lag]) > confBound {
			suggestedSQ = k
		}
	}

	return suggestedSP, suggestedSQ
}

func countSignificantLags(values []float64, confBound float64, maxLag int) int {
	count := 0
	for i := 1; i < len(values) && i <= maxLag; i++ {
		if math.Abs(values[i]) > confBound {
			count++
		}
	}
	return count
}

func countModelsEvaluated(candidates []ModelCandidate) int {
	// This is approximate since we don't track all models in stepwise
	return len(candidates)
}

func formatARIMAOrder(p, d, q int) string {
	return "ARIMA(" + itoa(p) + "," + itoa(d) + "," + itoa(q) + ")"
}

func formatSARIMAOrder(p, d, q, sp, sd, sq, m int) string {
	return "SARIMA(" + itoa(p) + "," + itoa(d) + "," + itoa(q) + ")(" +
		itoa(sp) + "," + itoa(sd) + "," + itoa(sq) + ")[" + itoa(m) + "]"
}

func itoa(i int) string {
	if i < 0 {
		return "-" + itoa(-i)
	}
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
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

// PredictWithInterval generates forecasts with confidence intervals at a single level.
// Returns point forecasts, lower bounds, and upper bounds.
// Common confidence levels: 0.80 (80%), 0.90 (90%), 0.95 (95%), 0.99 (99%)
func (r *Result) PredictWithInterval(steps int, confidence float64) (forecast, lower, upper []float64, err error) {
	if r.IsSeasonal && r.SeasonalModel != nil {
		return r.SeasonalModel.PredictWithInterval(steps, confidence)
	}
	if r.Model != nil {
		return r.Model.PredictWithInterval(steps, confidence)
	}
	return nil, nil, nil, nil
}

// ForecastResult contains forecasts with multiple confidence interval levels.
// Inspired by R's forecast package which returns 80% and 95% intervals by default.
type ForecastResult struct {
	Forecasts []float64             // Point forecasts
	Lower     map[float64][]float64 // Lower bounds by confidence level (e.g., 0.80, 0.95)
	Upper     map[float64][]float64 // Upper bounds by confidence level
	Levels    []float64             // Confidence levels used (e.g., [0.80, 0.95])
}

// DefaultForecastLevels are the confidence levels used by default (matches R's forecast)
var DefaultForecastLevels = []float64{0.80, 0.95}

// PredictWithLevels generates forecasts with multiple confidence interval levels.
// Similar to R's forecast() function which returns 80% and 95% intervals by default.
//
// Example:
//
//	result := autoarima.AutoARIMA(series, nil)
//	fc, _ := result.PredictWithLevels(24, nil)  // Uses 80% and 95% levels
//	fc.Lower[0.95]  // 95% lower bound
//	fc.Upper[0.80]  // 80% upper bound
//
//	// Custom levels
//	fc, _ := result.PredictWithLevels(24, []float64{0.90, 0.99})
func (r *Result) PredictWithLevels(steps int, levels []float64) (*ForecastResult, error) {
	if len(levels) == 0 {
		levels = DefaultForecastLevels
	}

	// Get base forecast (without intervals)
	forecasts, err := r.Predict(steps)
	if err != nil {
		return nil, err
	}

	result := &ForecastResult{
		Forecasts: forecasts,
		Lower:     make(map[float64][]float64),
		Upper:     make(map[float64][]float64),
		Levels:    levels,
	}

	// Compute intervals for each level
	for _, level := range levels {
		_, lower, upper, err := r.PredictWithInterval(steps, level)
		if err != nil {
			continue
		}
		result.Lower[level] = lower
		result.Upper[level] = upper
	}

	return result, nil
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

// Order returns a string representation of the selected model order.
func (r *Result) Order() string {
	if r.IsSeasonal {
		return formatSARIMAOrder(r.P, r.D, r.Q, r.SP, r.SD, r.SQ, r.M)
	}
	return formatARIMAOrder(r.P, r.D, r.Q)
}
