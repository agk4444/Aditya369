"""
Data Cleaner for EQM Preprocessing

This module provides data cleaning and quality improvement functionality
for physiological sensor data used in emotion detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.signal import medfilt
import warnings

logger = logging.getLogger(__name__)


class CleaningStrategy(Enum):
    """Data cleaning strategies"""
    INTERPOLATION = "interpolation"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    STATISTICAL = "statistical"
    MODEL_BASED = "model_based"


@dataclass
class CleaningConfig:
    """Configuration for data cleaning"""
    strategy: CleaningStrategy = CleaningStrategy.INTERPOLATION
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 1.5
    max_missing_ratio: float = 0.1
    smoothing_window: int = 5
    enable_detrending: bool = True
    enable_normalization: bool = True


@dataclass
class CleaningReport:
    """Report of cleaning operations performed"""
    original_rows: int
    cleaned_rows: int
    missing_values_filled: int
    outliers_removed: int
    data_points_smoothed: int
    operations_performed: List[str]
    quality_score: float


class DataCleaner:
    """Comprehensive data cleaning for physiological signals"""

    def __init__(self, config: CleaningConfig):
        self.config = config
        self.sensor_ranges = {
            'heart_rate': (30, 220),
            'heart_rate_variability': (0, 200),
            'temperature': (20, 45),
            'galvanic_skin_response': (0.5, 20),
            'blood_oxygen': (70, 100),
            'accelerometer_x': (-78, 78),
            'accelerometer_y': (-78, 78),
            'accelerometer_z': (-78, 78)
        }

    def clean_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Clean physiological sensor data

        Args:
            data: DataFrame with sensor data

        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        original_rows = len(data)
        operations_performed = []

        # Make a copy to avoid modifying original
        cleaned_data = data.copy()

        # 1. Handle missing values
        cleaned_data, missing_filled = self._handle_missing_values(cleaned_data)
        if missing_filled > 0:
            operations_performed.append(f"Filled {missing_filled} missing values")

        # 2. Remove outliers
        cleaned_data, outliers_removed = self._remove_outliers(cleaned_data)
        if outliers_removed > 0:
            operations_performed.append(f"Removed {outliers_removed} outliers")

        # 3. Apply smoothing if configured
        if self.config.smoothing_window > 1:
            cleaned_data = self._apply_smoothing(cleaned_data)
            operations_performed.append(f"Applied smoothing with window {self.config.smoothing_window}")

        # 4. Detrend data if configured
        if self.config.enable_detrending:
            cleaned_data = self._detrend_data(cleaned_data)
            operations_performed.append("Applied detrending")

        # 5. Normalize data if configured
        if self.config.enable_normalization:
            cleaned_data = self._normalize_data(cleaned_data)
            operations_performed.append("Applied normalization")

        # Calculate quality score
        quality_score = self._calculate_quality_score(cleaned_data, original_rows)

        # Create cleaning report
        report = CleaningReport(
            original_rows=original_rows,
            cleaned_rows=len(cleaned_data),
            missing_values_filled=missing_filled,
            outliers_removed=outliers_removed,
            data_points_smoothed=len(cleaned_data) if self.config.smoothing_window > 1 else 0,
            operations_performed=operations_performed,
            quality_score=quality_score
        )

        return cleaned_data, report

    def _handle_missing_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Handle missing values in the data"""
        missing_before = data.isnull().sum().sum()

        if self.config.strategy == CleaningStrategy.INTERPOLATION:
            data = self._interpolate_missing(data)
        elif self.config.strategy == CleaningStrategy.FORWARD_FILL:
            data = data.fillna(method='ffill')
        elif self.config.strategy == CleaningStrategy.BACKWARD_FILL:
            data = data.fillna(method='bfill')
        elif self.config.strategy == CleaningStrategy.STATISTICAL:
            data = self._statistical_fill(data)
        elif self.config.strategy == CleaningStrategy.MODEL_BASED:
            data = self._model_based_fill(data)

        missing_after = data.isnull().sum().sum()
        filled_count = missing_before - missing_after

        return data, filled_count

    def _interpolate_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values"""
        # Linear interpolation for physiological data
        for column in data.columns:
            if data[column].isnull().any():
                # Check if we have enough data for interpolation
                non_null_ratio = data[column].notnull().sum() / len(data)
                if non_null_ratio >= 0.5:  # At least 50% non-null
                    data[column] = data[column].interpolate(method='linear', limit_direction='both')
                else:
                    # Fall back to forward fill for sparse data
                    data[column] = data[column].fillna(method='ffill').fillna(method='bfill')

        return data

    def _statistical_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with statistical measures"""
        for column in data.columns:
            if data[column].isnull().any():
                # Use median for robustness (less sensitive to outliers)
                median_value = data[column].median()
                data[column] = data[column].fillna(median_value)

        return data

    def _model_based_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using simple model-based approach"""
        # For now, fall back to interpolation (could be extended with ML models)
        return self._interpolate_missing(data)

    def _remove_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove or handle outliers in the data"""
        rows_before = len(data)

        if self.config.outlier_method == "iqr":
            data = self._remove_outliers_iqr(data)
        elif self.config.outlier_method == "zscore":
            data = self._remove_outliers_zscore(data)
        elif self.config.outlier_method == "isolation_forest":
            data = self._remove_outliers_isolation_forest(data)

        rows_after = len(data)
        outliers_removed = rows_before - rows_after

        return data, outliers_removed

    def _remove_outliers_iqr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        for column in data.columns:
            if column in self.sensor_ranges:
                # Calculate IQR
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1

                # Define bounds
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR

                # Also respect physiological ranges
                min_val, max_val = self.sensor_ranges[column]
                lower_bound = max(lower_bound, min_val)
                upper_bound = min(upper_bound, max_val)

                # Clip values to bounds
                data[column] = data[column].clip(lower_bound, upper_bound)

        return data

    def _remove_outliers_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        for column in data.columns:
            if column in self.sensor_ranges:
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(data[column], nan_policy='omit'))

                # Define threshold
                threshold = self.config.outlier_threshold

                # Create mask for valid values
                valid_mask = z_scores < threshold

                # Replace outliers with median
                median_value = data[column].median()
                data.loc[~valid_mask, column] = median_value

        return data

    def _remove_outliers_isolation_forest(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Isolation Forest (simplified version)"""
        try:
            from sklearn.ensemble import IsolationForest

            # Prepare data for Isolation Forest
            numeric_data = data.select_dtypes(include=[np.number])

            if len(numeric_data.columns) > 0:
                # Fit isolation forest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_predictions = iso_forest.fit_predict(numeric_data)

                # Remove outliers (where prediction is -1)
                data = data[outlier_predictions == 1]

        except ImportError:
            logger.warning("sklearn not available, falling back to IQR method")
            data = self._remove_outliers_iqr(data)

        return data

    def _apply_smoothing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to reduce noise"""
        for column in data.columns:
            if data[column].dtype in [np.float64, np.int64]:
                # Use median filter for robust smoothing
                data[column] = medfilt(data[column], kernel_size=self.config.smoothing_window)

        return data

    def _detrend_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove trends from the data"""
        for column in data.columns:
            if data[column].dtype in [np.float64, np.int64]:
                # Simple detrending by subtracting rolling mean
                rolling_mean = data[column].rolling(window=60, center=True).mean()
                data[column] = data[column] - rolling_mean

                # Fill NaN values created by rolling operation
                data[column] = data[column].fillna(method='ffill').fillna(method='bfill')

        return data

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data to standard scale"""
        for column in data.columns:
            if data[column].dtype in [np.float64, np.int64]:
                # Z-score normalization
                mean_val = data[column].mean()
                std_val = data[column].std()

                if std_val > 0:
                    data[column] = (data[column] - mean_val) / std_val

        return data

    def _calculate_quality_score(self, data: pd.DataFrame, original_rows: int) -> float:
        """Calculate a quality score for the cleaned data"""
        score = 100.0

        # Penalty for data loss
        data_loss_ratio = (original_rows - len(data)) / original_rows
        score -= data_loss_ratio * 50

        # Penalty for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_ratio * 30

        # Bonus for data consistency
        for column in data.columns:
            if column in self.sensor_ranges:
                min_val, max_val = self.sensor_ranges[column]
                in_range_ratio = ((data[column] >= min_val) & (data[column] <= max_val)).mean()
                score += in_range_ratio * 10

        return max(0, min(100, score))

    def detect_data_quality_issues(self, data: pd.DataFrame) -> List[str]:
        """Detect potential data quality issues"""
        issues = []

        # Check for high missing ratio
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > self.config.max_missing_ratio:
            issues.append(f"High missing data ratio: {missing_ratio:.2%}")

        # Check for data outside physiological ranges
        for column in data.columns:
            if column in self.sensor_ranges:
                min_val, max_val = self.sensor_ranges[column]
                out_of_range = ((data[column] < min_val) | (data[column] > max_val)).sum()
                if out_of_range > 0:
                    issues.append(f"{out_of_range} values in {column} outside physiological range")

        # Check for constant values (potential sensor failure)
        for column in data.columns:
            if data[column].dtype in [np.float64, np.int64]:
                unique_ratio = data[column].nunique() / len(data)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    issues.append(f"Low variability in {column}: {unique_ratio:.2%} unique values")

        # Check for unrealistic changes
        for column in data.columns:
            if data[column].dtype in [np.float64, np.int64]:
                if column in self.sensor_ranges:
                    diff = data[column].diff().abs()
                    max_expected_change = (self.sensor_ranges[column][1] - self.sensor_ranges[column][0]) * 0.1
                    sudden_changes = (diff > max_expected_change).sum()
                    if sudden_changes > len(data) * 0.01:  # More than 1% sudden changes
                        issues.append(f"Unrealistic changes detected in {column}: {sudden_changes} instances")

        return issues


class BatchDataCleaner:
    """Data cleaner for batch processing of multiple files"""

    def __init__(self, config: CleaningConfig):
        self.config = config
        self.cleaner = DataCleaner(config)

    def clean_batch(self, file_paths: List[str]) -> Dict[str, CleaningReport]:
        """Clean multiple data files"""
        reports = {}

        for file_path in file_paths:
            try:
                # Load data
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    data = pd.read_json(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue

                # Clean data
                cleaned_data, report = self.cleaner.clean_data(data)
                reports[file_path] = report

                # Save cleaned data
                output_path = file_path.replace('.csv', '_cleaned.csv').replace('.json', '_cleaned.json')
                if output_path.endswith('_cleaned.csv'):
                    cleaned_data.to_csv(output_path, index=False)
                else:
                    cleaned_data.to_json(output_path, orient='records')

                logger.info(f"Cleaned {file_path} -> {output_path}")

            except Exception as e:
                logger.error(f"Error cleaning {file_path}: {e}")

        return reports

    def generate_batch_report(self, reports: Dict[str, CleaningReport]) -> Dict[str, Any]:
        """Generate summary report for batch cleaning"""
        if not reports:
            return {}

        total_files = len(reports)
        total_original = sum(report.original_rows for report in reports.values())
        total_cleaned = sum(report.cleaned_rows for report in reports.values())
        total_missing_filled = sum(report.missing_values_filled for report in reports.values())
        total_outliers_removed = sum(report.outliers_removed for report in reports.values())

        avg_quality = np.mean([report.quality_score for report in reports.values()])

        return {
            'total_files_processed': total_files,
            'total_original_rows': total_original,
            'total_cleaned_rows': total_cleaned,
            'total_missing_values_filled': total_missing_filled,
            'total_outliers_removed': total_outliers_removed,
            'average_quality_score': round(avg_quality, 2),
            'data_retention_rate': round(total_cleaned / total_original * 100, 2) if total_original > 0 else 0
        }


# Example usage and demonstration
def demo_data_cleaning():
    """Demonstrate data cleaning functionality"""

    # Create sample physiological data with issues
    np.random.seed(42)
    n_samples = 1000

    # Generate base signals
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='S')

    # Heart rate with some outliers and missing values
    heart_rate = 70 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 60)  # 60-second cycle
    heart_rate += np.random.normal(0, 2, n_samples)  # Add noise

    # Add outliers
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    heart_rate[outlier_indices] = np.random.choice([250, 30], size=20)  # Extreme values

    # Add missing values
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    heart_rate[missing_indices] = np.nan

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rate,
        'heart_rate_variability': 40 + np.random.normal(0, 5, n_samples),
        'temperature': 36.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 3600),  # Daily cycle
        'galvanic_skin_response': 2.0 + np.random.exponential(1, n_samples)
    })

    print("Data Cleaning Demo")
    print("=" * 50)
    print(f"Original data shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(".2f")
    print(".2f")

    # Create cleaning configuration
    config = CleaningConfig(
        strategy=CleaningStrategy.INTERPOLATION,
        outlier_method="iqr",
        outlier_threshold=1.5,
        smoothing_window=5,
        enable_detrending=True,
        enable_normalization=True
    )

    # Create cleaner and clean data
    cleaner = DataCleaner(config)
    cleaned_data, report = cleaner.clean_data(data)

    print("
Cleaning Results:")
    print(f"  Original rows: {report.original_rows}")
    print(f"  Cleaned rows: {report.cleaned_rows}")
    print(f"  Missing values filled: {report.missing_values_filled}")
    print(f"  Outliers removed: {report.outliers_removed}")
    print(f"  Quality score: {report.quality_score:.1f}")
    print("
Operations performed:"
    for operation in report.operations_performed:
        print(f"  - {operation}")

    # Check for data quality issues
    issues = cleaner.detect_data_quality_issues(cleaned_data)
    if issues:
        print("
Data quality issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo data quality issues detected!")

    # Show some statistics
    print("
Data Statistics:")
    print(cleaned_data.describe())

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    demo_data_cleaning()