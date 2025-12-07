// Package timeseries provides time series data structures and utilities.
//
// This package includes the Series type for representing time series data,
// along with functions for data loading, transformation, and analysis.
//
// # Creating a Series
//
// Create a time series from a slice:
//
//	values := []float64{100, 102, 105, 103, 108, 110}
//	series := timeseries.New(values)
//
// # Loading from CSV
//
// Load time series data from CSV files:
//
//	// Load a specific column
//	series, err := timeseries.LoadCSVColumn("data.csv", "value")
//
//	// Load with filtering
//	series, err := timeseries.LoadCSVFiltered(
//	    "data.csv",
//	    "country", "Australia",  // filter column and value
//	    "population",            // value column
//	)
//
// # Basic Statistics
//
// Calculate summary statistics:
//
//	mean := series.Mean()
//	std := series.Std()
//	min := series.Min()
//	max := series.Max()
//	median := series.Median()
//
// # Transformations
//
// Transform the time series:
//
//	// Differencing
//	diff := series.Diff()           // First difference
//	diff2 := series.DiffN(2)        // Second difference
//	sdiff := series.SeasonalDiff(12) // Seasonal difference
//
//	// Other transformations
//	logged := series.Log()          // Natural log
//	normalized := series.Normalize() // Z-score normalization
//	ma := series.MovingAverage(7)   // Moving average
//
// # Slicing and Manipulation
//
// Work with subsets of the data:
//
//	// Get a slice
//	subset := series.Slice(10, 50)
//
//	// Create lagged version
//	lagged := series.Lag(1)
//
//	// Copy the series
//	copy := series.Copy()
//
// # CSV Options
//
// Customize CSV loading:
//
//	opts := &timeseries.CSVOptions{
//	    DateColumn:  "date",
//	    ValueColumn: "value",
//	    DateFormat:  "2006-01-02",
//	    HasHeader:   true,
//	}
//	series, err := timeseries.LoadCSVFromReader(reader, opts)
package timeseries
