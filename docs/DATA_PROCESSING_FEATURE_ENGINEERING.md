# EQM Data Processing and Feature Engineering

## Overview
This document outlines the comprehensive data processing pipeline and feature engineering strategies for the EQM (Aditya369) emotional intelligence system.

## Data Processing Architecture

### Pipeline Stages
```
Raw Data → Cleaning → Normalization → Feature Extraction → Feature Selection → ML Ready Data
```

#### 1. Data Cleaning Stage

##### Missing Data Handling
```python
def handle_missing_data(data):
    strategies = {
        'interpolation': lambda x: x.interpolate(method='linear'),
        'forward_fill': lambda x: x.fillna(method='ffill'),
        'backward_fill': lambda x: x.fillna(method='bfill'),
        'statistical': lambda x: x.fillna(x.mean())
    }

    for sensor_type, strategy in strategies.items():
        if sensor_type in data.columns:
            data[sensor_type] = strategy(data[sensor_type])

    return data
```

##### Outlier Detection and Removal
```python
def remove_outliers(data, method='iqr', threshold=1.5):
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return data.clip(lower_bound, upper_bound, axis=1)
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        return data[(z_scores < threshold).all(axis=1)]
```

#### 2. Signal Processing

##### Noise Filtering
```python
def apply_filters(data, sampling_rate=10):
    # Butterworth low-pass filter for noise reduction
    nyquist = sampling_rate / 2
    cutoff = 5  # 5 Hz cutoff for physiological signals

    b, a = butter(4, cutoff/nyquist, btype='low')
    filtered_data = filtfilt(b, a, data)

    return filtered_data
```

##### Artifact Removal
```python
def remove_motion_artifacts(data, accelerometer_data):
    # Use accelerometer data to identify and remove motion artifacts
    motion_threshold = 2.0  # m/s²

    motion_mask = np.abs(accelerometer_data).max(axis=1) > motion_threshold
    data_cleaned = data.copy()
    data_cleaned[motion_mask] = np.nan

    # Interpolate over motion artifacts
    data_cleaned = data_cleaned.interpolate(method='linear')

    return data_cleaned
```

### Feature Engineering

#### Time-Domain Features

##### Statistical Features
```python
def extract_statistical_features(data, window_size=60):
    features = {}

    # Basic statistics
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['var'] = np.var(data)
    features['min'] = np.min(data)
    features['max'] = np.max(data)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(data)

    # Percentiles
    features['p25'] = np.percentile(data, 25)
    features['p75'] = np.percentile(data, 75)
    features['p95'] = np.percentile(data, 95)

    # Higher-order moments
    features['skewness'] = stats.skew(data)
    features['kurtosis'] = stats.kurtosis(data)

    return features
```

##### Heart Rate Variability Features
```python
def extract_hrv_features(rr_intervals):
    features = {}

    # Time-domain measures
    features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    features['sdnn'] = np.std(rr_intervals)
    features['pnn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)

    # Frequency-domain measures (using FFT)
    fft_result = np.fft.fft(rr_intervals)
    frequencies = np.fft.fftfreq(len(rr_intervals))

    # VLF, LF, HF power bands
    vlf_mask = (frequencies >= 0.0033) & (frequencies < 0.04)
    lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
    hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)

    features['vlf_power'] = np.sum(np.abs(fft_result[vlf_mask])**2)
    features['lf_power'] = np.sum(np.abs(fft_result[lf_mask])**2)
    features['hf_power'] = np.sum(np.abs(fft_result[hf_mask])**2)
    features['lf_hf_ratio'] = features['lf_power'] / features['hf_power']

    return features
```

#### Frequency-Domain Features

##### Power Spectral Density
```python
def extract_psd_features(data, sampling_rate=10):
    frequencies, psd = welch(data, fs=sampling_rate, nperseg=256)

    features = {}

    # Power in different frequency bands
    bands = {
        'very_low': (0, 0.04),
        'low': (0.04, 0.15),
        'high': (0.15, 0.4)
    }

    for band_name, (low_freq, high_freq) in bands.items():
        mask = (frequencies >= low_freq) & (frequencies < high_freq)
        features[f'{band_name}_power'] = np.sum(psd[mask])
        features[f'{band_name}_peak'] = frequencies[mask][np.argmax(psd[mask])]

    return features
```

##### Wavelet Transform Features
```python
def extract_wavelet_features(data, wavelet='db4', levels=5):
    coefficients = pywt.wavedec(data, wavelet, level=levels)

    features = {}
    for i, coeff in enumerate(coefficients):
        features[f'wavelet_level_{i}_energy'] = np.sum(coeff**2)
        features[f'wavelet_level_{i}_entropy'] = -np.sum(coeff**2 * np.log(coeff**2 + 1e-10))

    return features
```

#### Physiological Feature Combinations

##### Emotional Stress Indicators
```python
def extract_emotional_stress_features(hr_data, hrv_data, gsr_data, temp_data):
    features = {}

    # Heart rate acceleration/deceleration patterns
    hr_gradient = np.gradient(hr_data)
    features['hr_acceleration'] = np.mean(hr_gradient)
    features['hr_acceleration_std'] = np.std(hr_gradient)

    # HRV stress correlation
    features['hr_hrv_correlation'] = np.corrcoef(hr_data, hrv_data)[0,1]

    # GSR phasic/tonic components
    features['gsr_phasic'] = np.mean(gsr_data[gsr_data > np.percentile(gsr_data, 75)])
    features['gsr_tonic'] = np.mean(gsr_data[gsr_data < np.percentile(gsr_data, 25)])

    # Temperature variability
    features['temp_variability'] = np.std(temp_data)
    features['temp_gradient'] = np.mean(np.gradient(temp_data))

    return features
```

##### Activity and Context Features
```python
def extract_activity_features(accel_data, window_size=60):
    features = {}

    # Activity intensity
    magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
    features['activity_intensity'] = np.mean(magnitude)
    features['activity_std'] = np.std(magnitude)

    # Movement patterns
    features['dominant_frequency'] = scipy.signal.find_peaks(magnitude)[0].mean()

    # Posture indicators (simplified)
    features['vertical_movement'] = np.std(accel_data[:, 2])  # Z-axis
    features['horizontal_movement'] = np.sqrt(np.std(accel_data[:, 0])**2 + np.std(accel_data[:, 1])**2)

    return features
```

### Feature Selection and Dimensionality Reduction

#### Correlation Analysis
```python
def remove_correlated_features(features_df, threshold=0.95):
    corr_matrix = features_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return features_df.drop(to_drop, axis=1)
```

#### Feature Importance Ranking
```python
def rank_features_by_importance(X, y, method='mutual_info'):
    if method == 'mutual_info':
        importance_scores = mutual_info_classif(X, y)
    elif method == 'f_test':
        f_scores, _ = f_classif(X, y)
        importance_scores = f_scores
    elif method == 'chi2':
        chi2_scores, _ = chi2(X, y)
        importance_scores = chi2_scores

    feature_ranks = pd.Series(importance_scores, index=X.columns).sort_values(ascending=False)
    return feature_ranks
```

#### Principal Component Analysis
```python
def apply_pca(features_df, variance_threshold=0.95):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    pca = PCA(n_components=variance_threshold)
    principal_components = pca.fit_transform(features_scaled)

    # Create DataFrame with principal components
    pc_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC_{i+1}' for i in range(principal_components.shape[1])]
    )

    return pc_df, pca.explained_variance_ratio_
```

### Temporal Feature Engineering

#### Sliding Window Features
```python
def create_sliding_window_features(data, window_sizes=[30, 60, 120]):
    window_features = {}

    for window_size in window_sizes:
        # Rolling statistics
        window_features[f'mean_{window_size}s'] = data.rolling(window=window_size).mean()
        window_features[f'std_{window_size}s'] = data.rolling(window=window_size).std()
        window_features[f'min_{window_size}s'] = data.rolling(window=window_size).min()
        window_features[f'max_{window_size}s'] = data.rolling(window=window_size).max()

        # Rate of change
        window_features[f'rate_of_change_{window_size}s'] = data.diff(window_size) / window_size

    return pd.DataFrame(window_features)
```

#### Sequence Modeling Features
```python
def create_sequence_features(data, sequence_length=100):
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])

    return np.array(sequences), np.array(targets)
```

### Feature Validation and Quality Assurance

#### Feature Distribution Analysis
```python
def analyze_feature_distributions(features_df):
    analysis = {}

    for column in features_df.columns:
        data = features_df[column]

        analysis[column] = {
            'mean': data.mean(),
            'std': data.std(),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'missing_pct': data.isnull().sum() / len(data) * 100,
            'unique_values': data.nunique(),
            'outlier_pct': detect_outliers(data)
        }

    return analysis
```

#### Feature Stability Testing
```python
def test_feature_stability(train_features, test_features, threshold=0.1):
    stability_issues = []

    for feature in train_features.columns:
        if feature in test_features.columns:
            train_stats = train_features[feature].describe()
            test_stats = test_features[feature].describe()

            # Check for significant distribution shifts
            mean_diff = abs(train_stats['mean'] - test_stats['mean']) / train_stats['std']
            std_diff = abs(train_stats['std'] - test_stats['std']) / train_stats['std']

            if mean_diff > threshold or std_diff > threshold:
                stability_issues.append({
                    'feature': feature,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff
                })

    return stability_issues
```

### Implementation Pipeline

#### Real-time Feature Processing
```python
class RealTimeFeatureProcessor:
    def __init__(self, feature_config):
        self.config = feature_config
        self.buffers = {}
        self.feature_extractors = self._initialize_extractors()

    def process_streaming_data(self, sensor_data):
        # Update buffers with new data
        for sensor_type, values in sensor_data.items():
            if sensor_type not in self.buffers:
                self.buffers[sensor_type] = deque(maxlen=self.config['buffer_size'])
            self.buffers[sensor_type].extend(values)

        # Extract features when sufficient data available
        if self._has_sufficient_data():
            features = {}
            for extractor_name, extractor in self.feature_extractors.items():
                features.update(extractor(self.buffers))

            return features
        return None
```

#### Batch Feature Processing
```python
class BatchFeatureProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def process_dataset(self, raw_data_path, output_path):
        # Load raw data
        raw_data = pd.read_csv(raw_data_path)

        # Apply processing pipeline
        processed_data = self._apply_data_cleaning(raw_data)
        features = self._extract_all_features(processed_data)
        selected_features = self._select_features(features)

        # Save processed features
        selected_features.to_csv(output_path, index=False)
        return selected_features
```

This comprehensive feature engineering approach ensures that the EQM system can effectively extract meaningful emotional indicators from physiological sensor data.