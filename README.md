## Data Analytics Version Control

A professional-grade data cleaning and preprocessing pipeline supporting multiple file formats with comprehensive validation, logging, and quality reporting capabilities.

## Overview

This project provides an enterprise-level data cleaning framework designed for data analytics workflows. It handles CSV, TSV, Excel, JSON, Parquet, and Feather formats with robust error handling and detailed audit trails.

## Features

* **Multi-format Support**: CSV, TSV, Excel (.xlsx, .xls), JSON, Parquet, Feather
* **Comprehensive Data Profiling**: Statistical analysis and quality metrics
* **Smart Missing Value Handling**: Configurable strategies (mean, median, mode, interpolation)
* **Outlier Detection & Treatment**: IQR and Z-score methods with clip/remove/transform options
* **Automated Type Detection**: Intelligent conversion of numeric and datetime columns
* **Duplicate Removal**: Flexible duplicate identification and removal
* **Column Standardization**: Automatic naming convention enforcement (snake_case, camelCase)
* **Audit Logging**: Complete operation tracking with timestamps
* **Export Reports**: JSON reports documenting all cleaning operations

## Requirements

* Python 3.8+
* pandas >= 2.0.0
* numpy >= 1.24.0
* openpyxl >= 3.1.0 (for Excel support)
* pyarrow >= 12.0.0 (for Parquet support)

## Installation

```bash
git clone https://github.com/jaideepkathiresan/data-analytics-version-control.git
cd data-analytics-version-control

pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from data_cleaning import DataCleaningPipeline

# Initialize pipeline
pipeline = DataCleaningPipeline(log_level='INFO')

# Load data
pipeline.load_data('input_data.csv')

# Clean data
pipeline.standardize_columns(naming_convention='snake_case')
pipeline.handle_missing_values(strategy='smart', threshold=0.7)
pipeline.remove_duplicates()
pipeline.handle_outliers(method='clip', detection_method='iqr')
pipeline.validate_data_types(auto_detect=True)

# Export results
pipeline.export_data('cleaned_data.csv')
pipeline.generate_report('cleaning_report.json')
```

### Advanced Configuration

**Custom missing value handling**

```python
pipeline.handle_missing_values(
    strategy='fill',
    threshold=0.5,
    numeric_method='median',
    categorical_method='mode'
)
```

**Outlier detection with custom parameters**

```python
outliers = pipeline.detect_outliers(
    method='zscore',
    threshold=3.0,
    columns=['price', 'quantity']
)
```

**Type validation with explicit mapping**

```python
pipeline.validate_data_types(
    type_map={'date': 'datetime64', 'price': 'float64'},
    auto_detect=True
)
```

## Supported File Formats

| Format  | Extension   | Read | Write |
| ------- | ----------- | ---- | ----- |
| CSV     | .csv        | ✅    | ✅     |
| TSV     | .tsv        | ✅    | ✅     |
| Excel   | .xlsx, .xls | ✅    | ✅     |
| JSON    | .json       | ✅    | ✅     |
| Parquet | .parquet    | ✅    | ✅     |
| Feather | .feather    | ✅    | ✅     |

## 🔧 API Reference

### DataCleaningPipeline Class

**Methods:**

* `load_data(filepath, **kwargs)` - Load data from file
* `generate_profile()` - Generate statistical profile
* `handle_missing_values(strategy, threshold, numeric_method, categorical_method)` - Handle missing data
* `remove_duplicates(subset, keep)` - Remove duplicate rows
* `detect_outliers(method, threshold, columns)` - Detect outliers
* `handle_outliers(method, detection_method, threshold, columns)` - Handle outliers
* `standardize_columns(naming_convention, strip_whitespace)` - Standardize column names
* `validate_data_types(type_map, auto_detect)` - Validate and convert data types
* `export_data(filepath, format, **kwargs)` - Export cleaned data
* `generate_report(output_path)` - Generate cleaning report

## Project Structure

```
data-analytics-version-control/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── data_cleaning.py          # Main pipeline module
```

*This project was developed as part of the Foundations of Data Analytics coursework, Semester 7.*
