"""
Data Cleaning Pipeline
==================================
Data preprocessing and validation framework
supporting multiple file formats with comprehensive error handling,
logging, and data quality reporting.

Author: Jaideep Kathiresan
License: MIT
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCleaningPipeline:
    """
   Data cleaning and validation pipeline.
    
    Supports: CSV, TSV, Excel (.xlsx, .xls), JSON, Parquet, Feather
    Features: Missing value handling, outlier detection, duplicate removal,
              data type validation, statistical profiling, and audit logging.
    """
    
    SUPPORTED_FORMATS = {
        '.csv': pd.read_csv,
        '.tsv': lambda path: pd.read_csv(path, sep='\t'),
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.json': pd.read_json,
        '.parquet': pd.read_parquet,
        '.feather': pd.read_feather
    }
    
    def __init__(self, log_level: str = 'INFO'):
        """
        Initialize the data cleaning pipeline.
        
        Args:
            log_level: Logging verbosity level (DEBUG, INFO, WARNING, ERROR)
        """
        self._setup_logging(log_level)
        self.data: Optional[pd.DataFrame] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}
        self.cleaning_report: Dict = {}
        
    def _setup_logging(self, log_level: str) -> None:
        """Configure logging with timestamp and formatting."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from supported file formats with automatic format detection.
        
        Args:
            filepath: Path to data file
            **kwargs: Additional arguments passed to pandas read functions
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        file_ext = filepath.suffix.lower()
        
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_ext}. "
                f"Supported: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        
        self.logger.info(f"Loading data from {filepath}")
        
        try:
            read_func = self.SUPPORTED_FORMATS[file_ext]
            self.data = read_func(filepath, **kwargs)
            self.original_data = self.data.copy()
            
            self.metadata = {
                'source_file': str(filepath),
                'load_timestamp': datetime.now().isoformat(),
                'original_shape': self.data.shape,
                'original_columns': list(self.data.columns),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
            }
            
            self.logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_profile(self) -> Dict:
        """
        Generate comprehensive statistical profile of the dataset.
        
        Returns:
            Dictionary containing detailed statistics and quality metrics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info("Generating data profile...")
        
        profile = {
            'shape': self.data.shape,
            'columns': len(self.data.columns),
            'rows': len(self.data),
            'memory_usage_mb': round(self.data.memory_usage(deep=True).sum() / 1024**2, 2),
            'duplicates': self.data.duplicated().sum(),
            'column_info': {}
        }
        
        for col in self.data.columns:
            col_data = self.data[col]
            col_info = {
                'dtype': str(col_data.dtype),
                'missing_count': int(col_data.isna().sum()),
                'missing_percentage': round(col_data.isna().mean() * 100, 2),
                'unique_count': int(col_data.nunique()),
                'unique_percentage': round(col_data.nunique() / len(col_data) * 100, 2)
            }
            
            # Numeric column statistics
            if pd.api.types.is_numeric_dtype(col_data):
                col_info.update({
                    'mean': float(col_data.mean()) if not col_data.isna().all() else None,
                    'median': float(col_data.median()) if not col_data.isna().all() else None,
                    'std': float(col_data.std()) if not col_data.isna().all() else None,
                    'min': float(col_data.min()) if not col_data.isna().all() else None,
                    'max': float(col_data.max()) if not col_data.isna().all() else None,
                    'zeros_count': int((col_data == 0).sum())
                })
            
            # Categorical column statistics
            elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                col_info['top_values'] = col_data.value_counts().head(5).to_dict()
            
            profile['column_info'][col] = col_info
        
        return profile
    
    def handle_missing_values(
        self,
        strategy: str = 'smart',
        threshold: float = 0.5,
        numeric_method: str = 'median',
        categorical_method: str = 'mode'
    ) -> pd.DataFrame:
        """
        Handle missing values using configurable strategies.
        
        Args:
            strategy: 'smart', 'drop', 'fill', or 'interpolate'
            threshold: Drop columns if missing % exceeds this (0-1)
            numeric_method: 'mean', 'median', 'mode', or custom value
            categorical_method: 'mode', 'constant', or custom value
            
        Returns:
            DataFrame with missing values handled
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        self.logger.info(f"Handling missing values with strategy: {strategy}")
        initial_missing = self.data.isna().sum().sum()
        
        # Drop columns exceeding threshold
        missing_pct = self.data.isna().mean()
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            self.logger.warning(f"Dropping {len(cols_to_drop)} columns exceeding {threshold*100}% missing threshold")
            self.data = self.data.drop(columns=cols_to_drop)
        
        if strategy == 'drop':
            self.data = self.data.dropna()
            
        elif strategy == 'fill' or strategy == 'smart':
            for col in self.data.columns:
                if self.data[col].isna().any():
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        if numeric_method == 'mean':
                            fill_value = self.data[col].mean()
                        elif numeric_method == 'median':
                            fill_value = self.data[col].median()
                        elif numeric_method == 'mode':
                            fill_value = self.data[col].mode()[0] if not self.data[col].mode().empty else 0
                        else:
                            fill_value = numeric_method
                        self.data[col].fillna(fill_value, inplace=True)
                    else:
                        if categorical_method == 'mode':
                            fill_value = self.data[col].mode()[0] if not self.data[col].mode().empty else 'Unknown'
                        elif categorical_method == 'constant':
                            fill_value = 'Unknown'
                        else:
                            fill_value = categorical_method
                        self.data[col].fillna(fill_value, inplace=True)
        
        elif strategy == 'interpolate':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        final_missing = self.data.isna().sum().sum()
        self.cleaning_report['missing_values'] = {
            'initial_missing': int(initial_missing),
            'final_missing': int(final_missing),
            'removed': int(initial_missing - final_missing),
            'columns_dropped': cols_to_drop
        }
        
        self.logger.info(f"Missing values handled. Removed: {initial_missing - final_missing}")
        return self.data
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            subset: Column labels to consider for duplicate identification
            keep: 'first', 'last', or False to drop all duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset, keep=keep)
        duplicates_removed = initial_rows - len(self.data)
        
        self.cleaning_report['duplicates'] = {
            'initial_count': int(initial_rows),
            'duplicates_removed': int(duplicates_removed),
            'final_count': len(self.data)
        }
        
        self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        return self.data
    
    def detect_outliers(
        self,
        method: str = 'iqr',
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.Series]:
        """
        Detect outliers using IQR or Z-score methods.
        
        Args:
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier (default 1.5) or Z-score threshold (default 3.0)
            columns: Specific columns to check (None for all numeric)
            
        Returns:
            Dictionary mapping column names to boolean Series indicating outliers
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers[col] = z_scores > threshold
        
        total_outliers = sum([outliers[col].sum() for col in outliers])
        self.logger.info(f"Detected {total_outliers} outliers using {method} method")
        
        return outliers
    
    def handle_outliers(
        self,
        method: str = 'clip',
        detection_method: str = 'iqr',
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle outliers by clipping, removing, or transforming.
        
        Args:
            method: 'clip', 'remove', or 'log_transform'
            detection_method: 'iqr' or 'zscore'
            threshold: Detection threshold
            columns: Columns to process
            
        Returns:
            DataFrame with outliers handled
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        outliers = self.detect_outliers(detection_method, threshold, columns)
        
        if method == 'remove':
            mask = pd.Series([False] * len(self.data))
            for col in outliers:
                mask |= outliers[col]
            initial_rows = len(self.data)
            self.data = self.data[~mask]
            self.logger.info(f"Removed {initial_rows - len(self.data)} rows containing outliers")
            
        elif method == 'clip':
            for col in outliers:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
            self.logger.info(f"Clipped outliers in {len(outliers)} columns")
            
        elif method == 'log_transform':
            for col in outliers:
                if (self.data[col] > 0).all():
                    self.data[col] = np.log1p(self.data[col])
                    self.logger.info(f"Log-transformed {col}")
        
        return self.data
    
    def standardize_columns(
        self,
        naming_convention: str = 'snake_case',
        strip_whitespace: bool = True
    ) -> pd.DataFrame:
        """
        Standardize column names to consistent format.
        
        Args:
            naming_convention: 'snake_case', 'camelCase', or 'lowercase'
            strip_whitespace: Remove leading/trailing spaces
            
        Returns:
            DataFrame with standardized column names
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        new_columns = []
        
        for col in self.data.columns:
            new_col = str(col)
            
            if strip_whitespace:
                new_col = new_col.strip()
            
            if naming_convention == 'snake_case':
                new_col = new_col.lower().replace(' ', '_').replace('-', '_')
                new_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in new_col)
            elif naming_convention == 'camelCase':
                words = new_col.replace('_', ' ').replace('-', ' ').split()
                new_col = words[0].lower() + ''.join(w.capitalize() for w in words[1:])
            elif naming_convention == 'lowercase':
                new_col = new_col.lower()
            
            new_columns.append(new_col)
        
        self.data.columns = new_columns
        self.logger.info(f"Standardized column names to {naming_convention}")
        return self.data
    
    def validate_data_types(
        self,
        type_map: Optional[Dict[str, str]] = None,
        auto_detect: bool = True
    ) -> pd.DataFrame:
        """
        Validate and convert data types.
        
        Args:
            type_map: Dictionary mapping column names to desired types
            auto_detect: Automatically detect and convert appropriate types
            
        Returns:
            DataFrame with corrected data types
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if type_map:
            for col, dtype in type_map.items():
                if col in self.data.columns:
                    try:
                        self.data[col] = self.data[col].astype(dtype)
                        self.logger.info(f"Converted {col} to {dtype}")
                    except Exception as e:
                        self.logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
        
        if auto_detect:
            for col in self.data.columns:
                # Try datetime conversion
                if self.data[col].dtype == 'object':
                    try:
                        self.data[col] = pd.to_datetime(self.data[col], errors='raise')
                        self.logger.info(f"Auto-detected datetime format for {col}")
                        continue
                    except:
                        pass
                    
                    # Try numeric conversion
                    try:
                        self.data[col] = pd.to_numeric(self.data[col], errors='raise')
                        self.logger.info(f"Auto-converted {col} to numeric")
                    except:
                        pass
        
        return self.data
    
    def export_data(
        self,
        filepath: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Export cleaned data to file.
        
        Args:
            filepath: Output file path
            format: Output format (auto-detected from extension if None)
            **kwargs: Additional arguments for export function
        """
        if self.data is None:
            raise ValueError("No data to export.")
        
        filepath = Path(filepath)
        file_ext = format or filepath.suffix.lower()
        
        export_funcs = {
            '.csv': lambda: self.data.to_csv(filepath, index=False, **kwargs),
            '.xlsx': lambda: self.data.to_excel(filepath, index=False, **kwargs),
            '.json': lambda: self.data.to_json(filepath, **kwargs),
            '.parquet': lambda: self.data.to_parquet(filepath, **kwargs),
            '.feather': lambda: self.data.to_feather(filepath, **kwargs)
        }
        
        if file_ext not in export_funcs:
            raise ValueError(f"Unsupported export format: {file_ext}")
        
        export_funcs[file_ext]()
        self.logger.info(f"Data exported to {filepath}")
    
    def generate_report(self, output_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Generate comprehensive cleaning report.
        
        Args:
            output_path: Optional path to save report as JSON
            
        Returns:
            Dictionary containing complete cleaning report
        """
        report = {
            'metadata': self.metadata,
            'cleaning_operations': self.cleaning_report,
            'final_profile': self.generate_profile() if self.data is not None else {},
            'timestamp': datetime.now().isoformat()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Report saved to {output_path}")
        
        return report


def main():
    """
    Example usage demonstrating the data cleaning pipeline.
    """
    # Initialize pipeline
    pipeline = DataCleaningPipeline(log_level='INFO')
    
    # Load data
    try:
        data = pipeline.load_data('input_data.csv')
        
        # Generate initial profile
        print("\nINITIAL DATA PROFILE")
        profile = pipeline.generate_profile()
        print(json.dumps(profile, indent=2, default=str))
        
        # Clean data
        pipeline.standardize_columns(naming_convention='snake_case')
        pipeline.handle_missing_values(strategy='smart', threshold=0.7)
        pipeline.remove_duplicates()
        pipeline.handle_outliers(method='clip', detection_method='iqr')
        pipeline.validate_data_types(auto_detect=True)
        
        # Export cleaned data
        pipeline.export_data('cleaned_data.csv')
        
        # Generate and save report
        report = pipeline.generate_report('cleaning_report.json')
        print("\nCLEANING COMPLETE")
        print(f"Original shape: {report['metadata']['original_shape']}")
        print(f"Final shape: {report['final_profile']['shape']}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
