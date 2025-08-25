"""
Preprocessing Pipeline for EQM

This module orchestrates the complete data preprocessing workflow
including cleaning, feature extraction, and feature selection.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from .data_cleaner import DataCleaner, CleaningConfig, CleaningStrategy
from .feature_extractor import FeatureExtractor, FeatureConfig, ExtractedFeatures

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline"""
    # Data cleaning config
    cleaning_strategy: CleaningStrategy = CleaningStrategy.INTERPOLATION
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    enable_smoothing: bool = True
    smoothing_window: int = 5

    # Feature extraction config
    window_size: int = 60
    sampling_rate: int = 10
    enable_fft: bool = True
    enable_wavelet: bool = True
    enable_temporal: bool = True

    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_method: str = "variance"  # variance, correlation, mutual_info
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    max_features: Optional[int] = None

    # Output configuration
    save_intermediate: bool = True
    output_format: str = "parquet"  # csv, parquet, json
    compression: str = "snappy"


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline"""
    original_data_shape: Tuple[int, int]
    cleaned_data_shape: Tuple[int, int]
    extracted_features_shape: Tuple[int, int]
    selected_features_shape: Optional[Tuple[int, int]]
    processing_time: float
    quality_score: float
    features_extracted: int
    features_selected: Optional[int]
    preprocessing_steps: List[str]


class PreprocessingPipeline:
    """Complete preprocessing pipeline for EQM data"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.cleaner = None
        self.extractor = None
        self.feature_selector = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize pipeline components"""
        # Initialize data cleaner
        cleaning_config = CleaningConfig(
            strategy=self.config.cleaning_strategy,
            outlier_method=self.config.outlier_method,
            outlier_threshold=self.config.outlier_threshold,
            smoothing_window=self.config.smoothing_window if self.config.enable_smoothing else 1,
            enable_detrending=True,
            enable_normalization=True
        )
        self.cleaner = DataCleaner(cleaning_config)

        # Initialize feature extractor
        feature_config = FeatureConfig(
            window_size=self.config.window_size,
            sampling_rate=self.config.sampling_rate,
            enable_fft=self.config.enable_fft,
            enable_wavelet=self.config.enable_wavelet,
            enable_statistical=True,  # Always enabled
            enable_temporal=self.config.enable_temporal
        )
        self.extractor = FeatureExtractor(feature_config)

    def preprocess_data(self, data: pd.DataFrame, sensor_columns: List[str]) -> Tuple[pd.DataFrame, PreprocessingResult]:
        """
        Run complete preprocessing pipeline on data

        Args:
            data: Raw sensor data DataFrame
            sensor_columns: List of sensor column names to process

        Returns:
            Tuple of (processed_features, preprocessing_result)
        """
        start_time = datetime.utcnow()
        preprocessing_steps = []

        # Store original shape
        original_shape = data.shape

        # Step 1: Data Cleaning
        logger.info("Starting data cleaning...")
        cleaned_data, cleaning_report = self.cleaner.clean_data(data)
        preprocessing_steps.append("Data cleaning completed")

        if self.config.save_intermediate:
            self._save_intermediate_data(cleaned_data, "cleaned_data")

        # Step 2: Feature Extraction
        logger.info("Starting feature extraction...")
        extracted_features = self.extractor.extract_features(cleaned_data, sensor_columns)

        # Convert features to DataFrame
        feature_data = self._features_to_dataframe(extracted_features, len(cleaned_data))
        preprocessing_steps.append("Feature extraction completed")

        if self.config.save_intermediate:
            self._save_intermediate_data(feature_data, "extracted_features")

        # Step 3: Feature Selection (if enabled)
        selected_features = feature_data
        if self.config.enable_feature_selection:
            logger.info("Starting feature selection...")
            selected_features = self._select_features(feature_data)
            preprocessing_steps.append("Feature selection completed")

        if self.config.save_intermediate:
            self._save_intermediate_data(selected_features, "selected_features")

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Create result object
        result = PreprocessingResult(
            original_data_shape=original_shape,
            cleaned_data_shape=cleaned_data.shape,
            extracted_features_shape=feature_data.shape,
            selected_features_shape=selected_features.shape if self.config.enable_feature_selection else None,
            processing_time=processing_time,
            quality_score=cleaning_report.quality_score,
            features_extracted=feature_data.shape[1],
            features_selected=selected_features.shape[1] if self.config.enable_feature_selection else None,
            preprocessing_steps=preprocessing_steps
        )

        logger.info(f"Preprocessing completed in {processing_time:.2f} seconds")
        return selected_features, result

    def _features_to_dataframe(self, features: ExtractedFeatures, num_samples: int) -> pd.DataFrame:
        """Convert extracted features to DataFrame format"""
        # For this implementation, we'll create a single row of features
        # In a real scenario, you might want to extract windowed features
        feature_dict = {}

        # Add all feature types
        feature_dict.update(features.statistical_features)
        feature_dict.update(features.frequency_features)
        feature_dict.update(features.wavelet_features)
        feature_dict.update(features.temporal_features)

        # Create DataFrame with one row (for single sample analysis)
        # For time series, you would create multiple rows for different windows
        df = pd.DataFrame([feature_dict])

        return df

    def _select_features(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """Select most relevant features"""
        if self.config.feature_selection_method == "variance":
            return self._variance_threshold_selection(feature_data)
        elif self.config.feature_selection_method == "correlation":
            return self._correlation_based_selection(feature_data)
        elif self.config.feature_selection_method == "mutual_info":
            return self._mutual_info_selection(feature_data)
        else:
            logger.warning(f"Unknown feature selection method: {self.config.feature_selection_method}")
            return feature_data

    def _variance_threshold_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance"""
        try:
            from sklearn.feature_selection import VarianceThreshold

            selector = VarianceThreshold(threshold=self.config.variance_threshold)
            selected_data = selector.fit_transform(data)

            # Get selected feature names
            selected_features = data.columns[selector.get_support()].tolist()
            logger.info(f"Variance threshold selected {len(selected_features)} features")

            return pd.DataFrame(selected_data, columns=selected_features)

        except ImportError:
            logger.warning("sklearn not available for variance threshold selection")
            return data

    def _correlation_based_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        corr_matrix = data.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation above threshold
        to_drop = []
        for column in upper.columns:
            correlated_features = upper.index[upper[column] > self.config.correlation_threshold].tolist()
            if correlated_features:
                # Keep the first feature, drop the others
                to_drop.extend(correlated_features[1:])

        # Remove duplicates
        to_drop = list(set(to_drop))

        selected_data = data.drop(to_drop, axis=1)
        logger.info(f"Correlation-based selection removed {len(to_drop)} features")

        return selected_data

    def _mutual_info_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select features based on mutual information"""
        try:
            from sklearn.feature_selection import SelectKBest, mutual_info_regression

            # For now, assume we want to predict a target (you would need actual labels)
            # This is a placeholder implementation
            if self.config.max_features:
                selector = SelectKBest(score_func=mutual_info_regression, k=self.config.max_features)
                # Note: This would need actual target values for real implementation
                selected_data = selector.fit_transform(data, np.zeros(len(data)))

                return pd.DataFrame(selected_data, columns=data.columns[selector.get_support()].tolist())
            else:
                return data

        except ImportError:
            logger.warning("sklearn not available for mutual info selection")
            return data

    def _save_intermediate_data(self, data: pd.DataFrame, name: str):
        """Save intermediate preprocessing results"""
        try:
            output_dir = Path("./data/intermediate")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}"

            if self.config.output_format == "parquet":
                data.to_parquet(output_dir / f"{filename}.parquet", compression=self.config.compression)
            elif self.config.output_format == "csv":
                data.to_csv(output_dir / f"{filename}.csv", index=False)
            elif self.config.output_format == "json":
                data.to_json(output_dir / f"{filename}.json", orient="records")

            logger.debug(f"Saved intermediate data: {name}")

        except Exception as e:
            logger.error(f"Error saving intermediate data {name}: {e}")

    def get_feature_importance(self, feature_data: pd.DataFrame, target_data: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate feature importance scores"""
        importance_scores = {}

        try:
            # Method 1: Variance-based importance
            variances = feature_data.var()
            for feature, variance in variances.items():
                importance_scores[f"{feature}_variance"] = variance

            # Method 2: Correlation with target (if available)
            if target_data is not None:
                correlations = feature_data.corrwith(target_data)
                for feature, corr in correlations.items():
                    importance_scores[f"{feature}_correlation"] = abs(corr)

            # Method 3: Mutual information (if sklearn available)
            if target_data is not None:
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    mi_scores = mutual_info_regression(feature_data, target_data)
                    for i, score in enumerate(mi_scores):
                        feature_name = feature_data.columns[i]
                        importance_scores[f"{feature_name}_mutual_info"] = score
                except ImportError:
                    pass

        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")

        return importance_scores


class BatchPreprocessingPipeline:
    """Batch processing version of the preprocessing pipeline"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.pipeline = PreprocessingPipeline(config)

    async def preprocess_batch(self, file_paths: List[str], sensor_columns: List[str]) -> Dict[str, PreprocessingResult]:
        """Preprocess multiple data files"""
        results = {}

        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")

                # Load data
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith('.parquet'):
                    data = pd.read_parquet(file_path)
                elif file_path.endswith('.json'):
                    data = pd.read_json(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue

                # Preprocess data
                processed_data, result = self.pipeline.preprocess_data(data, sensor_columns)
                results[file_path] = result

                # Save processed data
                output_path = self._get_output_path(file_path)
                self._save_processed_data(processed_data, output_path)

                logger.info(f"Completed processing {file_path}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        return results

    def _get_output_path(self, input_path: str) -> str:
        """Generate output path for processed data"""
        input_path = Path(input_path)
        output_dir = Path("./data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_processed_{timestamp}{input_path.suffix}"

        return str(output_dir / output_filename)

    def _save_processed_data(self, data: pd.DataFrame, output_path: str):
        """Save processed data to file"""
        try:
            output_path = Path(output_path)

            if output_path.suffix == '.parquet':
                data.to_parquet(output_path, compression=self.config.compression)
            elif output_path.suffix == '.csv':
                data.to_csv(output_path, index=False)
            elif output_path.suffix == '.json':
                data.to_json(output_path, orient="records")

            logger.info(f"Saved processed data to {output_path}")

        except Exception as e:
            logger.error(f"Error saving processed data: {e}")

    def generate_batch_report(self, results: Dict[str, PreprocessingResult]) -> Dict[str, Any]:
        """Generate summary report for batch processing"""
        if not results:
            return {}

        total_files = len(results)
        total_original_rows = sum(result.original_data_shape[0] for result in results.values())
        total_cleaned_rows = sum(result.cleaned_data_shape[0] for result in results.values())
        total_features_extracted = sum(result.features_extracted for result in results.values())

        avg_quality_score = np.mean([result.quality_score for result in results.values()])
        avg_processing_time = np.mean([result.processing_time for result in results.values()])

        return {
            'total_files_processed': total_files,
            'total_original_rows': total_original_rows,
            'total_cleaned_rows': total_cleaned_rows,
            'total_features_extracted': total_features_extracted,
            'average_quality_score': round(avg_quality_score, 2),
            'average_processing_time': round(avg_processing_time, 2),
            'data_retention_rate': round(total_cleaned_rows / total_original_rows * 100, 2) if total_original_rows > 0 else 0
        }


# Example usage and demonstration
def demo_preprocessing_pipeline():
    """Demonstrate the preprocessing pipeline functionality"""

    # Create sample physiological data with noise and artifacts
    np.random.seed(42)
    n_samples = 2000

    # Generate realistic physiological signals
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='500ms')

    # Heart rate with realistic patterns
    t_seconds = np.arange(n_samples) * 0.5
    heart_rate = 70 + 8 * np.sin(2 * np.pi * t_seconds / 3600)  # Daily rhythm
    heart_rate += 3 * np.sin(2 * np.pi * t_seconds / 4)  # Breathing influence
    heart_rate += np.random.normal(0, 1.5, n_samples)  # Natural variation

    # Add some outliers and missing values
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    heart_rate[outlier_indices] = np.random.choice([250, 30], size=20)

    missing_indices = np.random.choice(n_samples, size=100, replace=False)
    heart_rate[missing_indices] = np.nan

    # Heart rate variability
    hrv = 40 + 15 * np.sin(2 * np.pi * t_seconds / 60) + np.random.normal(0, 5, n_samples)

    # Skin conductance with emotional responses
    gsr = 2.0 + np.random.exponential(1, n_samples)
    emotion_spikes = np.random.choice(n_samples, size=15, replace=False)
    gsr[emotion_spikes] += np.random.uniform(2, 5, 15)

    # Temperature
    temperature = 36.5 + 0.3 * np.sin(2 * np.pi * t_seconds / 3600)
    temperature += np.random.normal(0, 0.1, n_samples)

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rate,
        'heart_rate_variability': hrv,
        'galvanic_skin_response': gsr,
        'temperature': temperature,
        'blood_oxygen': 98 + np.random.normal(0, 1, n_samples)
    })

    print("Preprocessing Pipeline Demo")
    print("=" * 50)
    print(f"Original data shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Data duration: {n_samples * 0.5:.1f} seconds")

    # Create preprocessing configuration
    config = PreprocessingConfig(
        cleaning_strategy=CleaningStrategy.INTERPOLATION,
        outlier_method="iqr",
        outlier_threshold=1.5,
        enable_smoothing=True,
        smoothing_window=5,
        window_size=60,
        sampling_rate=2,  # 2 Hz for 500ms intervals
        enable_fft=True,
        enable_wavelet=True,
        enable_temporal=True,
        enable_feature_selection=True,
        feature_selection_method="variance",
        variance_threshold=0.01,
        save_intermediate=True,
        output_format="csv"
    )

    # Create and run preprocessing pipeline
    pipeline = PreprocessingPipeline(config)
    sensor_columns = ['heart_rate', 'heart_rate_variability', 'galvanic_skin_response', 'temperature', 'blood_oxygen']

    print("\nRunning preprocessing pipeline...")
    processed_features, result = pipeline.preprocess_data(data, sensor_columns)

    # Display results
    print("
Preprocessing Results:")
    print(f"  Original shape: {result.original_data_shape}")
    print(f"  Cleaned shape: {result.cleaned_data_shape}")
    print(f"  Features extracted: {result.features_extracted}")
    print(f"  Features selected: {result.features_selected}")
    print(f"  Processing time: {result.processing_time:.2f} seconds")
    print(f"  Quality score: {result.quality_score:.1f}")

    print("
Processing steps completed:")
    for step in result.preprocessing_steps:
        print(f"  - {step}")

    print(f"\nFinal feature set shape: {processed_features.shape}")

    # Show some extracted features
    print("
Sample of extracted features:")
    feature_cols = processed_features.columns[:10].tolist()
    print(f"  Features: {', '.join(feature_cols)}")

    if len(processed_features.columns) > 10:
        print(f"  ... and {len(processed_features.columns) - 10} more features")

    # Show feature values
    print("
Feature values:")
    for col in feature_cols:
        value = processed_features[col].iloc[0]
        print(f"  {col}: {value:.4f}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    demo_preprocessing_pipeline()