"""
Feature Extractor for EQM Preprocessing

This module provides comprehensive feature extraction from physiological
sensor data for emotion detection and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from scipy.signal import welch, find_peaks
from scipy.fft import fft
import pywt
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    window_size: int = 60  # seconds
    overlap: float = 0.5  # 50% overlap
    sampling_rate: int = 10  # Hz
    enable_fft: bool = True
    enable_wavelet: bool = True
    enable_statistical: bool = True
    enable_temporal: bool = True


@dataclass
class ExtractedFeatures:
    """Container for extracted features"""
    statistical_features: Dict[str, float]
    frequency_features: Dict[str, float]
    wavelet_features: Dict[str, float]
    temporal_features: Dict[str, float]
    metadata: Dict[str, Any]


class FeatureExtractor:
    """Comprehensive feature extraction from physiological signals"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_names = []

    def extract_features(self, data: pd.DataFrame, sensor_columns: List[str]) -> ExtractedFeatures:
        """
        Extract features from physiological sensor data

        Args:
            data: DataFrame containing sensor data
            sensor_columns: List of sensor column names to process

        Returns:
            ExtractedFeatures object with all feature types
        """
        all_statistical = {}
        all_frequency = {}
        all_wavelet = {}
        all_temporal = {}

        # Process each sensor column
        for sensor_col in sensor_columns:
            if sensor_col not in data.columns:
                logger.warning(f"Sensor column {sensor_col} not found in data")
                continue

            sensor_data = data[sensor_col].values

            # Remove NaN values for processing
            sensor_data = sensor_data[~np.isnan(sensor_data)]

            if len(sensor_data) == 0:
                logger.warning(f"No valid data for sensor {sensor_col}")
                continue

            # Extract features for this sensor
            if self.config.enable_statistical:
                statistical = self._extract_statistical_features(sensor_data, sensor_col)
                all_statistical.update(statistical)

            if self.config.enable_fft:
                frequency = self._extract_frequency_features(sensor_data, sensor_col)
                all_frequency.update(frequency)

            if self.config.enable_wavelet:
                wavelet = self._extract_wavelet_features(sensor_data, sensor_col)
                all_wavelet.update(wavelet)

            if self.config.enable_temporal:
                temporal = self._extract_temporal_features(sensor_data, sensor_col)
                all_temporal.update(temporal)

        # Create metadata
        metadata = {
            'extraction_timestamp': datetime.utcnow().isoformat(),
            'window_size': self.config.window_size,
            'sampling_rate': self.config.sampling_rate,
            'sensors_processed': sensor_columns,
            'total_features': len(all_statistical) + len(all_frequency) + len(all_wavelet) + len(all_temporal)
        }

        return ExtractedFeatures(
            statistical_features=all_statistical,
            frequency_features=all_frequency,
            wavelet_features=all_wavelet,
            temporal_features=all_temporal,
            metadata=metadata
        )

    def _extract_statistical_features(self, data: np.ndarray, sensor_name: str) -> Dict[str, float]:
        """Extract statistical features from sensor data"""
        features = {}
        prefix = f"{sensor_name}_"

        try:
            # Basic statistics
            features[prefix + 'mean'] = np.mean(data)
            features[prefix + 'std'] = np.std(data)
            features[prefix + 'var'] = np.var(data)
            features[prefix + 'min'] = np.min(data)
            features[prefix + 'max'] = np.max(data)
            features[prefix + 'range'] = features[prefix + 'max'] - features[prefix + 'min']
            features[prefix + 'median'] = np.median(data)

            # Percentiles
            features[prefix + 'p25'] = np.percentile(data, 25)
            features[prefix + 'p75'] = np.percentile(data, 75)
            features[prefix + 'p90'] = np.percentile(data, 90)
            features[prefix + 'p95'] = np.percentile(data, 95)
            features[prefix + 'p99'] = np.percentile(data, 99)

            # Higher-order moments
            features[prefix + 'skewness'] = stats.skew(data)
            features[prefix + 'kurtosis'] = stats.kurtosis(data)

            # Robust statistics
            features[prefix + 'iqr'] = stats.iqr(data)
            features[prefix + 'mad'] = stats.median_abs_deviation(data)

            # Signal energy
            features[prefix + 'energy'] = np.sum(data ** 2)
            features[prefix + 'rms'] = np.sqrt(np.mean(data ** 2))

            # Rate of change features
            if len(data) > 1:
                diff = np.diff(data)
                features[prefix + 'mean_diff'] = np.mean(diff)
                features[prefix + 'std_diff'] = np.std(diff)
                features[prefix + 'max_diff'] = np.max(np.abs(diff))

        except Exception as e:
            logger.error(f"Error extracting statistical features for {sensor_name}: {e}")

        return features

    def _extract_frequency_features(self, data: np.ndarray, sensor_name: str) -> Dict[str, float]:
        """Extract frequency domain features using FFT and power spectral density"""
        features = {}
        prefix = f"{sensor_name}_"

        try:
            # Compute power spectral density
            frequencies, psd = welch(data, fs=self.config.sampling_rate, nperseg=256)

            # Frequency bands (in Hz)
            bands = {
                'vlf': (0.0033, 0.04),    # Very Low Frequency
                'lf': (0.04, 0.15),       # Low Frequency
                'hf': (0.15, 0.4),        # High Frequency
                'vlf_hf': (0.0033, 0.4),  # Very Low to High Frequency
                'lf_hf': (0.04, 0.4)      # Low to High Frequency
            }

            # Extract power in each frequency band
            for band_name, (low_freq, high_freq) in bands.items():
                mask = (frequencies >= low_freq) & (frequencies < high_freq)
                if np.any(mask):
                    band_power = np.sum(psd[mask])
                    features[prefix + f'{band_name}_power'] = band_power

                    # Peak frequency in band
                    if band_power > 0:
                        peak_idx = np.argmax(psd[mask])
                        peak_freq = frequencies[mask][peak_idx]
                        features[prefix + f'{band_name}_peak_freq'] = peak_freq

            # Spectral centroid and spread
            if np.sum(psd) > 0:
                spectral_centroid = np.sum(frequencies * psd) / np.sum(psd)
                features[prefix + 'spectral_centroid'] = spectral_centroid

                spectral_spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * psd) / np.sum(psd))
                features[prefix + 'spectral_spread'] = spectral_spread

            # Spectral rolloff (95% of energy)
            cumulative_psd = np.cumsum(psd)
            rolloff_threshold = 0.95 * cumulative_psd[-1]
            rolloff_idx = np.where(cumulative_psd >= rolloff_threshold)[0][0]
            features[prefix + 'spectral_rolloff'] = frequencies[rolloff_idx]

            # Heart rate specific features (if applicable)
            if sensor_name in ['heart_rate', 'heart_rate_variability']:
                # LF/HF ratio (autonomic balance indicator)
                if features.get(prefix + 'hf_power', 0) > 0:
                    lf_hf_ratio = features[prefix + 'lf_power'] / features[prefix + 'hf_power']
                    features[prefix + 'lf_hf_ratio'] = lf_hf_ratio

        except Exception as e:
            logger.error(f"Error extracting frequency features for {sensor_name}: {e}")

        return features

    def _extract_wavelet_features(self, data: np.ndarray, sensor_name: str) -> Dict[str, float]:
        """Extract wavelet-based features"""
        features = {}
        prefix = f"{sensor_name}_wavelet_"

        try:
            # Wavelet decomposition
            wavelet = 'db4'  # Daubechies 4 wavelet
            levels = 5

            # Perform wavelet decomposition
            coefficients = pywt.wavedec(data, wavelet, level=levels)

            # Extract features from each level
            for i, coeff in enumerate(coefficients):
                level_prefix = f"{prefix}level_{i}_"

                # Energy
                features[level_prefix + 'energy'] = np.sum(coeff ** 2)

                # Entropy
                if np.sum(coeff ** 2) > 0:
                    normalized_coeff = coeff ** 2 / np.sum(coeff ** 2)
                    features[level_prefix + 'entropy'] = -np.sum(normalized_coeff * np.log(normalized_coeff + 1e-10))

                # Statistical features of coefficients
                features[level_prefix + 'mean'] = np.mean(coeff)
                features[level_prefix + 'std'] = np.std(coeff)
                features[level_prefix + 'max'] = np.max(coeff)
                features[level_prefix + 'min'] = np.min(coeff)

                # Number of zero crossings
                zero_crossings = np.sum(np.abs(np.diff(np.sign(coeff)))) / 2
                features[level_prefix + 'zero_crossings'] = zero_crossings

            # Overall wavelet energy distribution
            total_energy = sum(features[f"{prefix}level_{i}_energy"] for i in range(levels + 1))
            for i in range(levels + 1):
                energy_ratio = features[f"{prefix}level_{i}_energy"] / total_energy if total_energy > 0 else 0
                features[f"{prefix}level_{i}_energy_ratio"] = energy_ratio

        except ImportError:
            logger.warning("PyWavelets not available, skipping wavelet features")
        except Exception as e:
            logger.error(f"Error extracting wavelet features for {sensor_name}: {e}")

        return features

    def _extract_temporal_features(self, data: np.ndarray, sensor_name: str) -> Dict[str, float]:
        """Extract temporal pattern features"""
        features = {}
        prefix = f"{sensor_name}_temporal_"

        try:
            # Trend analysis
            if len(data) > 10:
                # Linear trend
                x = np.arange(len(data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
                features[prefix + 'trend_slope'] = slope
                features[prefix + 'trend_r_squared'] = r_value ** 2

                # Detrended fluctuation
                detrended = data - (slope * x + intercept)
                features[prefix + 'detrended_std'] = np.std(detrended)

            # Autocorrelation features
            if len(data) > 5:
                # Autocorrelation at lag 1
                autocorr_1 = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
                autocorr_1 = autocorr_1[autocorr_1.size // 2:]
                autocorr_1 = autocorr_1 / autocorr_1[0]  # Normalize

                if len(autocorr_1) > 1:
                    features[prefix + 'autocorr_lag1'] = autocorr_1[1]

                # Find autocorrelation decay
                if len(autocorr_1) > 10:
                    autocorr_decay = np.where(autocorr_1 < 0.5)[0]
                    if len(autocorr_decay) > 0:
                        features[prefix + 'autocorr_decay_time'] = autocorr_decay[0]

            # Peak analysis
            if len(data) > 10:
                peaks, _ = find_peaks(data)
                if len(peaks) > 0:
                    features[prefix + 'num_peaks'] = len(peaks)
                    features[prefix + 'peak_density'] = len(peaks) / len(data)

                    # Peak intervals
                    if len(peaks) > 1:
                        peak_intervals = np.diff(peaks)
                        features[prefix + 'mean_peak_interval'] = np.mean(peak_intervals)
                        features[prefix + 'std_peak_interval'] = np.std(peak_intervals)

            # Signal complexity measures
            if len(data) > 10:
                # Sample entropy (simplified)
                # This is a basic approximation - a full implementation would be more complex
                features[prefix + 'complexity'] = self._approximate_sample_entropy(data)

            # Stationarity test
            if len(data) > 20:
                # Augmented Dickey-Fuller test for stationarity
                from statsmodels.tsa.stattools import adfuller
                try:
                    adf_result = adfuller(data)
                    features[prefix + 'adf_statistic'] = adf_result[0]
                    features[prefix + 'adf_p_value'] = adf_result[1]
                    features[prefix + 'is_stationary'] = adf_result[1] < 0.05  # 5% significance
                except ImportError:
                    logger.debug("statsmodels not available for stationarity test")

        except Exception as e:
            logger.error(f"Error extracting temporal features for {sensor_name}: {e}")

        return features

    def _approximate_sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Approximate sample entropy calculation"""
        try:
            # Normalize data
            data = (data - np.mean(data)) / (np.std(data) + 1e-10)

            def _phi(m):
                """Calculate phi for sample entropy"""
                N = len(data)
                B = 0

                for i in range(N - m):
                    for j in range(i + 1, N - m):
                        if np.max(np.abs(data[i:i+m] - data[j:j+m])) <= r:
                            B += 1

                return B / ((N - m) * (N - m - 1) / 2)

            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)

            if phi_m > 0:
                return -np.log(phi_m1 / phi_m)
            else:
                return 0

        except Exception:
            return 0

    def extract_sliding_window_features(
        self,
        data: pd.DataFrame,
        sensor_columns: List[str],
        window_sizes: List[int] = [30, 60, 120]
    ) -> pd.DataFrame:
        """Extract features using sliding windows"""
        all_features = []

        for window_size in window_sizes:
            window_features = {}

            for sensor_col in sensor_columns:
                if sensor_col not in data.columns:
                    continue

                sensor_data = data[sensor_col].values

                # Rolling statistics
                rolling_mean = pd.Series(sensor_data).rolling(window=window_size).mean()
                rolling_std = pd.Series(sensor_data).rolling(window=window_size).std()
                rolling_min = pd.Series(sensor_data).rolling(window=window_size).min()
                rolling_max = pd.Series(sensor_data).rolling(window=window_size).max()

                window_features[f'{sensor_col}_rolling_mean_{window_size}s'] = rolling_mean
                window_features[f'{sensor_col}_rolling_std_{window_size}s'] = rolling_std
                window_features[f'{sensor_col}_rolling_min_{window_size}s'] = rolling_min
                window_features[f'{sensor_col}_rolling_max_{window_size}s'] = rolling_max

                # Rate of change
                if len(sensor_data) > window_size:
                    rolling_diff = pd.Series(sensor_data).diff(window_size) / window_size
                    window_features[f'{sensor_col}_rate_of_change_{window_size}s'] = rolling_diff

            # Convert to DataFrame
            window_df = pd.DataFrame(window_features)
            all_features.append(window_df)

        # Combine all window features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            return combined_features

        return pd.DataFrame()

    def extract_emotion_specific_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract features specifically relevant for emotion detection"""
        features = {}

        try:
            # Physiological stress indicators
            if 'heart_rate' in data.columns and 'heart_rate_variability' in data.columns:
                hr_data = data['heart_rate'].values
                hrv_data = data['heart_rate_variability'].values

                # Heart rate variability features
                features['hr_hrv_correlation'] = np.corrcoef(hr_data, hrv_data)[0, 1]

                # Stress indicators
                features['hr_variability'] = np.std(hr_data)
                features['hrv_mean'] = np.mean(hrv_data)
                features['hrv_std'] = np.std(hrv_data)

            # Skin conductance features
            if 'galvanic_skin_response' in data.columns:
                gsr_data = data['galvanic_skin_response'].values

                # Phasic and tonic components (simplified)
                features['gsr_phasic'] = np.mean(gsr_data[gsr_data > np.percentile(gsr_data, 75)])
                features['gsr_tonic'] = np.mean(gsr_data[gsr_data < np.percentile(gsr_data, 25)])
                features['gsr_variability'] = np.std(gsr_data)

            # Temperature features
            if 'temperature' in data.columns:
                temp_data = data['temperature'].values
                features['temp_variability'] = np.std(temp_data)

                # Temperature gradient
                if len(temp_data) > 1:
                    features['temp_gradient'] = np.mean(np.gradient(temp_data))

            # Movement features
            accel_cols = [col for col in data.columns if 'accelerometer' in col]
            if accel_cols:
                accel_data = data[accel_cols].values

                # Activity magnitude
                magnitude = np.sqrt(np.sum(accel_data ** 2, axis=1))
                features['activity_intensity'] = np.mean(magnitude)
                features['activity_std'] = np.std(magnitude)

                # Dominant movement direction
                features['vertical_movement'] = np.std(accel_data[:, 0])  # Assuming first column is vertical
                features['horizontal_movement'] = np.sqrt(np.std(accel_data[:, 1])**2 + np.std(accel_data[:, 2])**2)

            # Blood oxygen features
            if 'blood_oxygen' in data.columns:
                spo2_data = data['blood_oxygen'].values
                features['spo2_mean'] = np.mean(spo2_data)
                features['spo2_variability'] = np.std(spo2_data)

                # Oxygen stress indicator
                low_oxygen = (spo2_data < 95).sum()
                features['low_oxygen_ratio'] = low_oxygen / len(spo2_data)

        except Exception as e:
            logger.error(f"Error extracting emotion-specific features: {e}")

        return features


# Batch feature extraction
class BatchFeatureExtractor:
    """Feature extractor for batch processing of multiple files"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.extractor = FeatureExtractor(config)

    def extract_batch_features(self, file_paths: List[str], sensor_columns: List[str]) -> Dict[str, ExtractedFeatures]:
        """Extract features from multiple data files"""
        results = {}

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

                # Extract features
                features = self.extractor.extract_features(data, sensor_columns)
                results[file_path] = features

                logger.info(f"Extracted features from {file_path}: {features.metadata['total_features']} features")

            except Exception as e:
                logger.error(f"Error extracting features from {file_path}: {e}")

        return results

    def save_features_to_csv(self, features_dict: Dict[str, ExtractedFeatures], output_path: str):
        """Save extracted features to CSV format"""
        all_features = []

        for file_path, features in features_dict.items():
            feature_row = {
                'file_path': file_path,
                'extraction_timestamp': features.metadata['extraction_timestamp']
            }

            # Add all feature types
            feature_row.update(features.statistical_features)
            feature_row.update(features.frequency_features)
            feature_row.update(features.wavelet_features)
            feature_row.update(features.temporal_features)

            all_features.append(feature_row)

        # Create DataFrame and save
        features_df = pd.DataFrame(all_features)
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(all_features)} feature sets to {output_path}")


# Example usage and demonstration
def demo_feature_extraction():
    """Demonstrate feature extraction functionality"""

    # Create sample physiological data
    np.random.seed(42)
    n_samples = 1000
    sampling_rate = 10  # Hz

    # Generate heart rate signal with physiological patterns
    t = np.arange(n_samples) / sampling_rate

    # Base heart rate with circadian rhythm
    heart_rate = 70 + 5 * np.sin(2 * np.pi * t / 3600)  # Daily cycle

    # Add respiratory sinus arrhythmia
    respiratory_component = 2 * np.sin(2 * np.pi * t / 4)  # 4-second breathing cycle
    heart_rate += respiratory_component

    # Add noise
    heart_rate += np.random.normal(0, 1, n_samples)

    # Heart rate variability
    hrv = 40 + 10 * np.sin(2 * np.pi * t / 60) + np.random.normal(0, 2, n_samples)

    # Skin conductance with emotional responses
    gsr = 2.0 + np.random.exponential(1, n_samples)
    # Add emotional peaks
    emotion_peaks = np.random.choice(n_samples, size=10, replace=False)
    gsr[emotion_peaks] += np.random.uniform(1, 3, 10)

    # Temperature with small variations
    temperature = 36.5 + 0.2 * np.sin(2 * np.pi * t / 3600) + np.random.normal(0, 0.1, n_samples)

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='100ms'),
        'heart_rate': heart_rate,
        'heart_rate_variability': hrv,
        'galvanic_skin_response': gsr,
        'temperature': temperature
    })

    print("Feature Extraction Demo")
    print("=" * 50)
    print(f"Data shape: {data.shape}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Duration: {n_samples/sampling_rate:.1f} seconds")

    # Create feature extraction configuration
    config = FeatureConfig(
        window_size=60,
        overlap=0.5,
        sampling_rate=sampling_rate,
        enable_fft=True,
        enable_wavelet=True,
        enable_statistical=True,
        enable_temporal=True
    )

    # Create feature extractor
    extractor = FeatureExtractor(config)
    sensor_columns = ['heart_rate', 'heart_rate_variability', 'galvanic_skin_response', 'temperature']

    # Extract features
    features = extractor.extract_features(data, sensor_columns)

    print("
Feature Extraction Results:")
    print(f"  Statistical features: {len(features.statistical_features)}")
    print(f"  Frequency features: {len(features.frequency_features)}")
    print(f"  Wavelet features: {len(features.wavelet_features)}")
    print(f"  Temporal features: {len(features.temporal_features)}")
    print(f"  Total features: {features.metadata['total_features']}")

    # Show some key features
    print("
Key Features Extracted:")
    key_features = [
        'heart_rate_mean', 'heart_rate_std', 'heart_rate_variability_mean',
        'galvanic_skin_response_mean', 'temperature_variability',
        'heart_rate_lf_hf_ratio', 'heart_rate_vlf_power', 'heart_rate_hf_power'
    ]

    for feature_name in key_features:
        if feature_name in features.statistical_features:
            print(f"  {feature_name}: {features.statistical_features[feature_name]:.3f}")
        elif feature_name in features.frequency_features:
            print(f"  {feature_name}: {features.frequency_features[feature_name]:.3f}")
        elif feature_name in features.temporal_features:
            print(f"  {feature_name}: {features.temporal_features[feature_name]:.3f}")

    # Extract emotion-specific features
    emotion_features = extractor.extract_emotion_specific_features(data)
    print("
Emotion-Specific Features:")
    for feature_name, value in emotion_features.items():
        print(f"  {feature_name}: {value:.3f}")

    # Extract sliding window features
    print("
Sliding Window Features:")
    window_features = extractor.extract_sliding_window_features(data, sensor_columns[:2], [30, 60])
    print(f"  Window features shape: {window_features.shape}")
    print(f"  Window feature columns: {len(window_features.columns)}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    demo_feature_extraction()