package timeseries

import (
	"bufio"
	"encoding/csv"
	"errors"
	"io"
	"os"
	"strconv"
	"strings"
	"time"
)

// CSVOptions holds options for CSV loading.
type CSVOptions struct {
	DateColumn  string // Column name for dates (optional)
	ValueColumn string // Column name for values (default: "y")
	IDColumn    string // Column name for series ID (optional, for filtering)
	IDFilter    string // Value to filter by ID column
	DateFormat  string // Date format (default: "2006-01-02")
	HasHeader   bool   // Whether CSV has header row (default: true)
	Delimiter   rune   // Field delimiter (default: ',')
	SkipRows    int    // Number of rows to skip at start
}

// DefaultCSVOptions returns default options for CSV loading.
func DefaultCSVOptions() *CSVOptions {
	return &CSVOptions{
		ValueColumn: "y",
		DateFormat:  "2006-01-02",
		HasHeader:   true,
		Delimiter:   ',',
	}
}

// LoadCSV loads a time series from a CSV file.
func LoadCSV(filename string, opts *CSVOptions) (*Series, error) {
	if opts == nil {
		opts = DefaultCSVOptions()
	}

	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return LoadCSVFromReader(file, opts)
}

// LoadCSVFromReader loads a time series from an io.Reader.
func LoadCSVFromReader(r io.Reader, opts *CSVOptions) (*Series, error) {
	if opts == nil {
		opts = DefaultCSVOptions()
	}

	reader := csv.NewReader(r)
	reader.Comma = opts.Delimiter
	reader.TrimLeadingSpace = true

	// Skip rows if needed
	for i := 0; i < opts.SkipRows; i++ {
		_, err := reader.Read()
		if err != nil {
			return nil, err
		}
	}

	// Read header
	var headers []string
	var valueIdx, dateIdx, idIdx int = -1, -1, -1

	if opts.HasHeader {
		header, err := reader.Read()
		if err != nil {
			return nil, err
		}
		headers = header

		// Find column indices
		for i, h := range headers {
			h = strings.TrimSpace(strings.Trim(h, "\""))
			switch {
			case h == opts.ValueColumn || (opts.ValueColumn == "" && (h == "y" || h == "value" || h == "Value")):
				valueIdx = i
			case opts.DateColumn != "" && h == opts.DateColumn:
				dateIdx = i
			case h == "ds" || h == "date" || h == "Date" || h == "Month" || h == "Year":
				if dateIdx == -1 {
					dateIdx = i
				}
			case opts.IDColumn != "" && h == opts.IDColumn:
				idIdx = i
			case h == "unique_id" || h == "id" || h == "ID":
				if idIdx == -1 && opts.IDColumn == "" {
					idIdx = i
				}
			}
		}

		// If value column not found, try to find numeric column
		if valueIdx == -1 {
			// Default to last column if not specified
			valueIdx = len(headers) - 1
		}
	} else {
		// No header - use column indices
		valueIdx = 1 // Assume second column is value
		dateIdx = 0  // Assume first column is date
	}

	var values []float64
	var timestamps []time.Time

	// Read data rows
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		// Filter by ID if specified
		if opts.IDFilter != "" && idIdx >= 0 && idIdx < len(record) {
			id := strings.TrimSpace(strings.Trim(record[idIdx], "\""))
			if id != opts.IDFilter {
				continue
			}
		}

		// Parse value
		if valueIdx >= 0 && valueIdx < len(record) {
			valStr := strings.TrimSpace(strings.Trim(record[valueIdx], "\""))
			if valStr == "" || valStr == "NA" || valStr == "NaN" || valStr == "null" {
				continue
			}
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil {
				continue // Skip invalid values
			}
			values = append(values, val)

			// Parse date if available
			if dateIdx >= 0 && dateIdx < len(record) {
				dateStr := strings.TrimSpace(strings.Trim(record[dateIdx], "\""))
				// Try multiple date formats
				formats := []string{
					opts.DateFormat,
					"2006-01-02",
					"2006-01-02T15:04:05",
					"2006/01/02",
					"01/02/2006",
					"02-Jan-2006",
					"2006",
				}
				var ts time.Time
				for _, fmt := range formats {
					ts, err = time.Parse(fmt, dateStr)
					if err == nil {
						break
					}
				}
				if err == nil {
					timestamps = append(timestamps, ts)
				}
			}
		}
	}

	if len(values) == 0 {
		return nil, errors.New("no valid data found in CSV")
	}

	// Create series
	if len(timestamps) == len(values) {
		return &Series{
			Timestamps: timestamps,
			Values:     values,
		}, nil
	}

	return New(values), nil
}

// LoadCSVColumn loads a specific column from a CSV file as a series.
func LoadCSVColumn(filename string, column string) (*Series, error) {
	opts := DefaultCSVOptions()
	opts.ValueColumn = column
	return LoadCSV(filename, opts)
}

// LoadCSVFiltered loads a filtered series from a CSV file.
func LoadCSVFiltered(filename string, idColumn, idValue, valueColumn string) (*Series, error) {
	opts := DefaultCSVOptions()
	opts.IDColumn = idColumn
	opts.IDFilter = idValue
	if valueColumn != "" {
		opts.ValueColumn = valueColumn
	}
	return LoadCSV(filename, opts)
}

// SaveCSV saves a time series to a CSV file.
func SaveCSV(series *Series, filename string, includeIndex bool) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Write header
	if includeIndex && len(series.Timestamps) == len(series.Values) {
		writer.WriteString("ds,y\n")
	} else if includeIndex {
		writer.WriteString("index,y\n")
	} else {
		writer.WriteString("y\n")
	}

	// Write data
	for i, v := range series.Values {
		if includeIndex {
			if len(series.Timestamps) == len(series.Values) {
				writer.WriteString(series.Timestamps[i].Format("2006-01-02"))
			} else {
				writer.WriteString(strconv.Itoa(i + 1))
			}
			writer.WriteString(",")
		}
		writer.WriteString(strconv.FormatFloat(v, 'f', -1, 64))
		writer.WriteString("\n")
	}

	return nil
}
