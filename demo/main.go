// Package main demonstrates ARIMA, SARIMA, and Auto-ARIMA with real data.
// Based on: Forecasting: Principles and Practice (https://otexts.com/fpppy)
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/sartorproj/goarima/arima"
	"github.com/sartorproj/goarima/autoarima"
	"github.com/sartorproj/goarima/sarima"
	"github.com/sartorproj/goarima/stats"
	"github.com/sartorproj/goarima/timeseries"
)

// Dataset defines a time series dataset to analyze
type Dataset struct {
	Name        string  // Display name
	Description string  // Brief description
	File        string  // CSV filename
	Column      string  // Value column name
	FilterCol   string  // Column to filter on (optional)
	FilterVal   string  // Value to filter for (optional)
	Period      int     // Seasonal period (0 = non-seasonal)
	Scale       float64 // Scale factor for values (e.g., 1e-6 for millions)
	SkipFirst   int     // Number of initial observations to skip
	MaxObs      int     // Max observations to use (0 = all, from end)
}

// ForecastResult holds model results for JSON export
type ForecastResult struct {
	ModelName       string    `json:"model_name"`
	Order           string    `json:"order"`
	AIC             float64   `json:"aic"`
	AICc            float64   `json:"aicc"`
	BIC             float64   `json:"bic"`
	RMSE            float64   `json:"rmse"`
	MAE             float64   `json:"mae"`
	MAPE            float64   `json:"mape"`
	Forecasts       []float64 `json:"forecasts"`
	ModelsEvaluated int       `json:"models_evaluated,omitempty"`
	SuggestedOrder  string    `json:"suggested_order,omitempty"` // ACF/PACF suggested order
}

// DatasetResult holds analysis results for a dataset
type DatasetResult struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	NObs         int                    `json:"n_obs"`
	TrainData    []float64              `json:"train_data"`
	TestData     []float64              `json:"test_data"`
	TrainIndex   []int                  `json:"train_index"`
	TestIndex    []int                  `json:"test_index"`
	Models       []ForecastResult       `json:"models"`
	Stationarity map[string]interface{} `json:"stationarity"`
	ACF          []float64              `json:"acf"`
	PACF         []float64              `json:"pacf"`
}

// OutputData holds all results for visualization
type OutputData struct {
	Datasets []DatasetResult `json:"datasets"`
}

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("GoARIMA Demonstration - ARIMA/SARIMA/Auto-ARIMA")
	fmt.Println("Reference: https://otexts.com/fpppy/nbs/09-arima.html")
	fmt.Println(strings.Repeat("=", 80))

	dataDir := findDataDir()
	fmt.Printf("\nData directory: %s\n", dataDir)

	// Define datasets - all configuration in one place
	datasets := []Dataset{
		{Name: "Australian Population", File: "aus_economy.csv", FilterCol: "unique_id", FilterVal: "Australia", Column: "y", Scale: 1e-6, Description: "Annual population (millions)"},
		{Name: "Australian Cement", File: "aus_production.csv", Column: "Cement", Period: 4, SkipFirst: 128, Description: "Quarterly cement production"},
		{Name: "Australian Beer", File: "aus_production.csv", Column: "Beer", Period: 4, SkipFirst: 80, Description: "Quarterly beer production"},
		{Name: "Australian Electricity", File: "aus_production.csv", Column: "Electricity", Period: 4, SkipFirst: 80, Description: "Quarterly electricity production"},
		{Name: "Australian Gas", File: "aus_production.csv", Column: "Gas", Period: 4, SkipFirst: 80, Description: "Quarterly gas production"},
		{Name: "US Eggs", File: "eggs.csv", Column: "y", Description: "Annual eggs per capita"},
		{Name: "US House Sales", File: "hsales.csv", Column: "y", Period: 12, Description: "Monthly new house sales"},
		{Name: "US Strikes", File: "strikes.csv", Column: "y", Description: "Annual strikes count"},
		{Name: "US Employment", File: "us_employment.csv", FilterCol: "unique_id", FilterVal: "Total Private", Column: "y", Period: 12, MaxObs: 240, Description: "Monthly private employment (000s)"},
		{Name: "Google Stock", File: "gafa_stock.csv", FilterCol: "unique_id", FilterVal: "GOOG_Close", Column: "y", MaxObs: 500, Description: "Daily closing price"},
	}

	output := OutputData{Datasets: []DatasetResult{}}

	for i, ds := range datasets {
		fmt.Printf("\n%s\n[%d/%d] %s\n%s\n", strings.Repeat("=", 80), i+1, len(datasets), ds.Name, strings.Repeat("=", 80))

		result := analyze(dataDir, ds)
		if result != nil {
			output.Datasets = append(output.Datasets, *result)
		}
	}

	// Export results
	fmt.Printf("\n%s\nEXPORTING RESULTS\n%s\n", strings.Repeat("=", 80), strings.Repeat("=", 80))

	if data, err := json.MarshalIndent(output, "", "  "); err == nil {
		os.WriteFile("forecast_results.json", data, 0644)
		fmt.Printf("Exported %d datasets to forecast_results.json\n", len(output.Datasets))
	}

	fmt.Println("\nTo visualize: python visualize.py")
	fmt.Println(strings.Repeat("=", 80))
}

// findDataDir locates the data directory
func findDataDir() string {
	for _, p := range []string{"data", "./data", "../data"} {
		if _, err := os.Stat(filepath.Join(p, "aus_economy.csv")); err == nil {
			return p
		}
	}
	return "data"
}

// analyze performs complete analysis on a dataset
func analyze(dataDir string, ds Dataset) *DatasetResult {
	// Load data
	series, err := loadData(dataDir, ds)
	if err != nil {
		fmt.Printf("   Error loading: %v\n", err)
		return nil
	}

	n := series.Len()
	fmt.Printf("   Loaded %d observations (%.2f to %.2f)\n", n, series.Min(), series.Max())

	// Train/test split
	testSize := calculateTestSize(n, ds.Period)
	trainSize := n - testSize
	train := series.Slice(0, trainSize)
	test := series.Slice(trainSize, n)
	fmt.Printf("   Train: %d, Test: %d\n", trainSize, testSize)

	// Create result
	result := &DatasetResult{
		Name:         ds.Name,
		Description:  ds.Description,
		NObs:         n,
		TrainData:    train.Values,
		TestData:     test.Values,
		TrainIndex:   makeRange(1, trainSize),
		TestIndex:    makeRange(trainSize+1, n),
		Models:       []ForecastResult{},
		Stationarity: make(map[string]interface{}),
	}

	// Stationarity tests
	if adf := stats.ADF(train, 0); adf != nil {
		result.Stationarity["adf_pvalue"] = adf.PValue
		result.Stationarity["adf_stationary"] = adf.IsStationary
	}
	if kpss := stats.KPSS(train, "c", 0); kpss != nil {
		result.Stationarity["kpss_pvalue"] = kpss.PValue
		result.Stationarity["kpss_stationary"] = kpss.IsStationary
	}
	result.Stationarity["ndiffs"] = stats.NDiffs(train, 2, "kpss")

	// ACF/PACF
	maxLag := min(24, trainSize/2)
	if acf := stats.ACF(train, maxLag); acf != nil {
		result.ACF = acf
	}
	if pacf := stats.PACF(train, maxLag); pacf != nil {
		result.PACF = pacf
	}

	// Fit models
	if ds.Period > 0 {
		fitSeasonalModels(result, train, test, ds.Period, testSize)
	} else {
		fitNonSeasonalModels(result, train, test, testSize)
	}

	return result
}

// loadData loads a dataset based on configuration
func loadData(dataDir string, ds Dataset) (*timeseries.Series, error) {
	path := filepath.Join(dataDir, ds.File)

	var series *timeseries.Series
	var err error

	if ds.FilterCol != "" {
		series, err = timeseries.LoadCSVFiltered(path, ds.FilterCol, ds.FilterVal, ds.Column)
	} else {
		series, err = timeseries.LoadCSVColumn(path, ds.Column)
	}
	if err != nil {
		return nil, err
	}

	// Apply transformations
	if ds.SkipFirst > 0 && series.Len() > ds.SkipFirst {
		series = series.Slice(ds.SkipFirst, series.Len())
	}
	if ds.MaxObs > 0 && series.Len() > ds.MaxObs {
		series = series.Slice(series.Len()-ds.MaxObs, series.Len())
	}
	if ds.Scale != 0 {
		for i := range series.Values {
			series.Values[i] *= ds.Scale
		}
	}

	return series, nil
}

// calculateTestSize determines appropriate test set size
func calculateTestSize(n, period int) int {
	testSize := n / 5
	if period > 0 {
		testSize = max(testSize, period)
	}
	return max(min(testSize, 30), 3)
}

// fitNonSeasonalModels fits ARIMA models for non-seasonal data
func fitNonSeasonalModels(result *DatasetResult, train, test *timeseries.Series, testSize int) {
	fmt.Println("   Fitting ARIMA models...")

	// Standard models to try
	models := []struct{ p, d, q int }{
		{0, 1, 0}, // Random walk
		{1, 1, 0}, // AR(1) with diff
		{1, 1, 1}, // ARIMA(1,1,1)
	}

	for _, m := range models {
		model := arima.New(m.p, m.d, m.q)
		if err := model.Fit(train); err == nil {
			forecasts, _ := model.Predict(testSize)
			rmse, mae, mape := metrics(test.Values, forecasts)
			order := fmt.Sprintf("(%d,%d,%d)", m.p, m.d, m.q)
			fmt.Printf("   ARIMA%s: RMSE=%.4f\n", order, rmse)
			result.Models = append(result.Models, ForecastResult{
				ModelName: "ARIMA", Order: order, AIC: model.AIC, AICc: model.AICc,
				BIC: model.BIC, RMSE: rmse, MAE: mae, MAPE: mape, Forecasts: forecasts,
			})
		}
	}

	// Auto-ARIMA with AICc criterion (non-seasonal)
	cfg := autoarima.DefaultConfig()
	cfg.MaxP, cfg.MaxQ = 3, 3
	cfg.Criterion = "aicc"
	cfg.AutoSeasonal = false // Disable auto-seasonality for non-seasonal data
	cfg.CompareModels = false
	if auto, err := autoarima.AutoARIMA(train, cfg); err == nil && auto.Model != nil {
		forecasts, _ := auto.Predict(testSize)
		rmse, mae, mape := metrics(test.Values, forecasts)
		order := fmt.Sprintf("(%d,%d,%d)", auto.P, auto.D, auto.Q)
		suggestedOrder := fmt.Sprintf("(%d,%d,%d)", auto.SuggestedP, auto.D, auto.SuggestedQ)
		fmt.Printf("   Auto-ARIMA%s: RMSE=%.4f (%d models, ACF/PACF suggested: %s)\n",
			order, rmse, auto.ModelsEvaluated, suggestedOrder)
		result.Models = append(result.Models, ForecastResult{
			ModelName: "Auto-ARIMA", Order: order, AIC: auto.AIC, AICc: auto.AICc,
			BIC: auto.BIC, RMSE: rmse, MAE: mae, MAPE: mape, Forecasts: forecasts,
			ModelsEvaluated: auto.ModelsEvaluated, SuggestedOrder: suggestedOrder,
		})
	}
}

// fitSeasonalModels fits SARIMA models for seasonal data
func fitSeasonalModels(result *DatasetResult, train, test *timeseries.Series, period, testSize int) {
	fmt.Printf("   Fitting SARIMA models (period=%d)...\n", period)

	// Standard seasonal models
	models := []struct{ p, d, q, sp, sd, sq int }{
		{1, 0, 0, 1, 1, 0}, // SAR with seasonal diff
		{0, 1, 1, 0, 1, 1}, // Airline model
		{1, 1, 1, 1, 1, 1}, // Full model
	}

	for _, m := range models {
		model := sarima.New(m.p, m.d, m.q, m.sp, m.sd, m.sq, period)
		if err := model.Fit(train); err == nil {
			forecasts, _ := model.Predict(testSize)
			rmse, mae, mape := metrics(test.Values, forecasts)
			order := fmt.Sprintf("(%d,%d,%d)(%d,%d,%d)[%d]", m.p, m.d, m.q, m.sp, m.sd, m.sq, period)
			fmt.Printf("   SARIMA%s: RMSE=%.4f\n", order, rmse)
			result.Models = append(result.Models, ForecastResult{
				ModelName: "SARIMA", Order: order, AIC: model.AIC, AICc: model.AICc,
				BIC: model.BIC, RMSE: rmse, MAE: mae, MAPE: mape, Forecasts: forecasts,
			})
		}
	}

	// Auto-SARIMA with AICc criterion - force the known period
	cfg := autoarima.DefaultConfig()
	cfg.AutoSeasonal = true
	cfg.SeasonalPeriods = []int{period} // Force the known period
	cfg.SeasonalityThreshold = 0.1      // Low threshold to ensure detection
	cfg.MaxP, cfg.MaxQ, cfg.MaxSP, cfg.MaxSQ = 2, 2, 2, 2
	cfg.Criterion = "aicc"
	cfg.CompareModels = false // Only fit seasonal since we know it's seasonal
	if auto, err := autoarima.AutoARIMA(train, cfg); err == nil && auto.IsSeasonal && auto.SeasonalModel != nil {
		forecasts, _ := auto.Predict(testSize)
		rmse, mae, mape := metrics(test.Values, forecasts)
		order := fmt.Sprintf("(%d,%d,%d)(%d,%d,%d)[%d]", auto.P, auto.D, auto.Q, auto.SP, auto.SD, auto.SQ, auto.M)
		suggestedOrder := fmt.Sprintf("(%d,%d,%d)(%d,%d,%d)[%d]",
			auto.SuggestedP, auto.D, auto.SuggestedQ, auto.SuggestedSP, auto.SD, auto.SuggestedSQ, auto.M)
		fmt.Printf("   Auto-SARIMA%s: RMSE=%.4f (%d models, ACF/PACF suggested: %s)\n",
			order, rmse, auto.ModelsEvaluated, suggestedOrder)
		result.Models = append(result.Models, ForecastResult{
			ModelName: "Auto-SARIMA", Order: order, AIC: auto.AIC, AICc: auto.AICc,
			BIC: auto.BIC, RMSE: rmse, MAE: mae, MAPE: mape, Forecasts: forecasts,
			ModelsEvaluated: auto.ModelsEvaluated, SuggestedOrder: suggestedOrder,
		})
	}
}

// metrics calculates forecast accuracy metrics
func metrics(actual, predicted []float64) (rmse, mae, mape float64) {
	n := min(len(actual), len(predicted))
	if n == 0 {
		return
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

func makeRange(start, end int) []int {
	r := make([]int, end-start+1)
	for i := range r {
		r[i] = start + i
	}
	return r
}
