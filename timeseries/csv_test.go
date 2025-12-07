package timeseries

import (
	"strings"
	"testing"
)

func TestLoadCSVFromReader(t *testing.T) {
	// Test basic CSV loading
	csvData := `ds,y
2020-01-01,100
2020-01-02,101
2020-01-03,102
2020-01-04,103
2020-01-05,104`

	reader := strings.NewReader(csvData)
	opts := DefaultCSVOptions()

	series, err := LoadCSVFromReader(reader, opts)
	if err != nil {
		t.Fatalf("Failed to load CSV: %v", err)
	}

	if series.Len() != 5 {
		t.Errorf("Expected 5 observations, got %d", series.Len())
	}

	// Check values
	expected := []float64{100, 101, 102, 103, 104}
	for i, v := range expected {
		if series.Values[i] != v {
			t.Errorf("Value at index %d: expected %f, got %f", i, v, series.Values[i])
		}
	}

	t.Logf("Loaded %d values: %v", series.Len(), series.Values)
}

func TestLoadCSVWithFilter(t *testing.T) {
	// Test filtered CSV loading
	csvData := `unique_id,ds,y
A,2020-01-01,100
B,2020-01-01,200
A,2020-01-02,101
B,2020-01-02,201
A,2020-01-03,102`

	reader := strings.NewReader(csvData)
	opts := DefaultCSVOptions()
	opts.IDColumn = "unique_id"
	opts.IDFilter = "A"

	series, err := LoadCSVFromReader(reader, opts)
	if err != nil {
		t.Fatalf("Failed to load CSV: %v", err)
	}

	if series.Len() != 3 {
		t.Errorf("Expected 3 observations for 'A', got %d", series.Len())
	}

	// Check values (should only have A's values)
	expected := []float64{100, 101, 102}
	for i, v := range expected {
		if series.Values[i] != v {
			t.Errorf("Value at index %d: expected %f, got %f", i, v, series.Values[i])
		}
	}

	t.Logf("Filtered series: %v", series.Values)
}

func TestLoadCSVWithNAValues(t *testing.T) {
	// Test handling of NA values
	csvData := `ds,y
2020-01-01,100
2020-01-02,NA
2020-01-03,102
2020-01-04,NaN
2020-01-05,104`

	reader := strings.NewReader(csvData)
	opts := DefaultCSVOptions()

	series, err := LoadCSVFromReader(reader, opts)
	if err != nil {
		t.Fatalf("Failed to load CSV: %v", err)
	}

	// NA and NaN values should be skipped
	if series.Len() != 3 {
		t.Errorf("Expected 3 observations (NA values skipped), got %d", series.Len())
	}

	expected := []float64{100, 102, 104}
	for i, v := range expected {
		if series.Values[i] != v {
			t.Errorf("Value at index %d: expected %f, got %f", i, v, series.Values[i])
		}
	}

	t.Logf("Series with NA skipped: %v", series.Values)
}

func TestLoadCSVMultipleColumns(t *testing.T) {
	// Test loading specific column
	csvData := `ds,Beer,Cement,Gas
2020-01-01,100,200,50
2020-01-02,110,210,55
2020-01-03,120,220,60`

	reader := strings.NewReader(csvData)
	opts := DefaultCSVOptions()
	opts.ValueColumn = "Cement"

	series, err := LoadCSVFromReader(reader, opts)
	if err != nil {
		t.Fatalf("Failed to load CSV: %v", err)
	}

	expected := []float64{200, 210, 220}
	for i, v := range expected {
		if series.Values[i] != v {
			t.Errorf("Value at index %d: expected %f, got %f", i, v, series.Values[i])
		}
	}

	t.Logf("Cement column: %v", series.Values)
}

func TestLoadCSVQuotedFields(t *testing.T) {
	// Test handling of quoted fields
	csvData := `"unique_id","ds","y"
"Australia","2020-01-01","1000000"
"Australia","2020-01-02","1000100"
"Australia","2020-01-03","1000200"`

	reader := strings.NewReader(csvData)
	opts := DefaultCSVOptions()

	series, err := LoadCSVFromReader(reader, opts)
	if err != nil {
		t.Fatalf("Failed to load CSV: %v", err)
	}

	if series.Len() != 3 {
		t.Errorf("Expected 3 observations, got %d", series.Len())
	}

	t.Logf("Quoted fields loaded: %v", series.Values)
}

func TestLoadCSVDateFormats(t *testing.T) {
	// Test various date formats
	testCases := []struct {
		name    string
		csvData string
	}{
		{
			"ISO format",
			`ds,y
2020-01-01,100
2020-01-02,101`,
		},
		{
			"Year only",
			`ds,y
2020,100
2021,101`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			reader := strings.NewReader(tc.csvData)
			opts := DefaultCSVOptions()

			series, err := LoadCSVFromReader(reader, opts)
			if err != nil {
				t.Fatalf("Failed to load CSV: %v", err)
			}

			if series.Len() != 2 {
				t.Errorf("Expected 2 observations, got %d", series.Len())
			}
		})
	}
}

func TestDefaultCSVOptions(t *testing.T) {
	opts := DefaultCSVOptions()

	if opts.ValueColumn != "y" {
		t.Errorf("Expected default value column 'y', got '%s'", opts.ValueColumn)
	}

	if opts.DateFormat != "2006-01-02" {
		t.Errorf("Expected default date format '2006-01-02', got '%s'", opts.DateFormat)
	}

	if !opts.HasHeader {
		t.Error("Expected HasHeader to be true by default")
	}

	if opts.Delimiter != ',' {
		t.Errorf("Expected default delimiter ',', got '%c'", opts.Delimiter)
	}
}
