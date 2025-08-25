"""
Emotion Predictor for EQM Real-Time Inference

This module provides the core emotion prediction functionality
for real-time analysis of physiological sensor data.
"""

import numpy as np
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class EmotionPrediction:
    """Result of emotion prediction"""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: datetime
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class PredictionConfig:
    """Configuration for emotion prediction"""
    model_path: str
    preprocessor_path: Optional[str] = None
    sequence_length: int = 300
    confidence_threshold: float = 0.5
    smoothing_window: int = 5
    enable_smoothing: bool = True
    prediction_interval: float = 1.0  # seconds


class EmotionPredictor:
    """Core emotion prediction engine"""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = None
        self.preprocessor = None
        self.emotion_labels = [
            'neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust'
        ]
        self.prediction_history = deque(maxlen=config.smoothing_window)
        self.is_loaded = False

    def load_model(self) -> bool:
        """Load the trained emotion detection model"""
        try:
            # Load TensorFlow/Keras model
            if self.config.model_path.endswith('.h5'):
                self.model = tf.keras.models.load_model(self.config.model_path)
            elif self.config.model_path.endswith('.pb'):  # TensorFlow SavedModel
                self.model = tf.saved_model.load(self.config.model_path)
            else:
                logger.error(f"Unsupported model format: {self.config.model_path}")
                return False

            # Load preprocessor if available
            if self.config.preprocessor_path:
                self._load_preprocessor()

            self.is_loaded = True
            logger.info(f"Model loaded successfully from {self.config.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _load_preprocessor(self):
        """Load data preprocessor"""
        try:
            with open(self.config.preprocessor_path, 'r') as f:
                preprocessor_config = json.load(f)

            # Recreate preprocessor from config
            self.preprocessor = StandardScaler()
            if 'scaler_mean' in preprocessor_config and 'scaler_scale' in preprocessor_config:
                self.preprocessor.mean_ = np.array(preprocessor_config['scaler_mean'])
                self.preprocessor.scale_ = np.array(preprocessor_config['scaler_scale'])

            logger.info(f"Preprocessor loaded from {self.config.preprocessor_path}")

        except Exception as e:
            logger.warning(f"Could not load preprocessor, using default: {e}")
            self.preprocessor = StandardScaler()

    def predict_emotion(self, sensor_data: np.ndarray) -> EmotionPrediction:
        """
        Predict emotion from physiological sensor data

        Args:
            sensor_data: Raw sensor data array of shape (sequence_length, n_features)

        Returns:
            EmotionPrediction object with results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = datetime.utcnow()

        try:
            # Preprocess data
            processed_data = self._preprocess_data(sensor_data)

            # Make prediction
            predictions = self.model.predict(processed_data, verbose=0)

            # Apply smoothing if enabled
            if self.config.enable_smoothing and len(self.prediction_history) > 0:
                predictions = self._smooth_predictions(predictions)

            # Store prediction for future smoothing
            self.prediction_history.append(predictions.flatten())

            # Convert to emotion prediction
            prediction_result = self._convert_to_emotion_prediction(predictions, start_time)

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            prediction_result.processing_time = processing_time

            return prediction_result

        except Exception as e:
            logger.error(f"Error during emotion prediction: {e}")
            raise

    def _preprocess_data(self, sensor_data: np.ndarray) -> np.ndarray:
        """Preprocess sensor data for model input"""
        # Ensure correct shape
        if sensor_data.ndim == 1:
            # Single sample, reshape to (1, sequence_length, features)
            sensor_data = sensor_data.reshape(1, -1)

        if sensor_data.ndim == 2:
            # Add sequence dimension if needed
            if sensor_data.shape[0] == self.config.sequence_length:
                # Shape is (sequence_length, features), add batch dimension
                sensor_data = sensor_data.reshape(1, sensor_data.shape[0], sensor_data.shape[1])
            else:
                # Shape is (batch, features), add sequence dimension
                sensor_data = sensor_data.reshape(sensor_data.shape[0], 1, sensor_data.shape[1])

        # Apply preprocessing if available
        if self.preprocessor is not None:
            # Flatten, scale, reshape back
            original_shape = sensor_data.shape
            sensor_data_flat = sensor_data.reshape(-1, sensor_data.shape[-1])
            sensor_data_scaled = self.preprocessor.transform(sensor_data_flat)
            sensor_data = sensor_data_scaled.reshape(original_shape)

        return sensor_data

    def _smooth_predictions(self, current_prediction: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to predictions"""
        # Get recent predictions
        recent_predictions = list(self.prediction_history)
        recent_predictions.append(current_prediction.flatten())

        # Calculate weighted average (more weight to recent predictions)
        weights = np.linspace(0.1, 1.0, len(recent_predictions))
        weights = weights / np.sum(weights)

        smoothed = np.average(recent_predictions, axis=0, weights=weights)
        return smoothed.reshape(1, -1)

    def _convert_to_emotion_prediction(self, predictions: np.ndarray, timestamp: datetime) -> EmotionPrediction:
        """Convert model output to EmotionPrediction object"""
        # Get prediction probabilities
        probabilities = predictions.flatten()

        # Get most likely emotion
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]

        # Create probability dictionary
        prob_dict = {
            emotion: float(prob)
            for emotion, prob in zip(self.emotion_labels, probabilities)
        }

        # Check confidence threshold
        if confidence < self.config.confidence_threshold:
            predicted_emotion = 'uncertain'
        else:
            predicted_emotion = self.emotion_labels[predicted_idx]

        return EmotionPrediction(
            emotion=predicted_emotion,
            confidence=float(confidence),
            probabilities=prob_dict,
            timestamp=timestamp,
            processing_time=0.0,  # Will be set by caller
            metadata={
                'model_path': self.config.model_path,
                'prediction_method': 'neural_network',
                'confidence_threshold': self.config.confidence_threshold,
                'smoothing_enabled': self.config.enable_smoothing
            }
        )

    def predict_batch(self, sensor_data_batch: List[np.ndarray]) -> List[EmotionPrediction]:
        """Predict emotions for a batch of sensor data"""
        results = []

        for sensor_data in sensor_data_batch:
            try:
                prediction = self.predict_emotion(sensor_data)
                results.append(prediction)
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                # Return uncertain prediction for failed cases
                uncertain_prediction = EmotionPrediction(
                    emotion='error',
                    confidence=0.0,
                    probabilities={emotion: 0.0 for emotion in self.emotion_labels},
                    timestamp=datetime.utcnow(),
                    processing_time=0.0,
                    metadata={'error': str(e)}
                )
                results.append(uncertain_prediction)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {'status': 'not_loaded'}

        try:
            return {
                'status': 'loaded',
                'model_path': self.config.model_path,
                'emotion_labels': self.emotion_labels,
                'sequence_length': self.config.sequence_length,
                'confidence_threshold': self.config.confidence_threshold,
                'smoothing_enabled': self.config.enable_smoothing,
                'smoothing_window': self.config.smoothing_window,
                'model_summary': self.model.summary() if hasattr(self.model, 'summary') else 'N/A'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


class BatchEmotionPredictor:
    """Batch processing version for multiple samples"""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.predictor = EmotionPredictor(config)

    async def initialize(self) -> bool:
        """Initialize the predictor"""
        return self.predictor.load_model()

    def predict_emotions_from_dataframe(self, data: 'pd.DataFrame', feature_columns: List[str]) -> List[EmotionPrediction]:
        """Predict emotions from DataFrame data"""
        predictions = []

        for idx, row in data.iterrows():
            try:
                # Extract features for this sample
                features = row[feature_columns].values

                # Reshape if needed for sequence processing
                if len(features.shape) == 1:
                    # Single timepoint, reshape for single-step prediction
                    features = features.reshape(1, -1)

                # Make prediction
                prediction = self.predictor.predict_emotion(features)
                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Error predicting for sample {idx}: {e}")
                continue

        return predictions


class StreamingEmotionPredictor:
    """Streaming predictor for real-time emotion detection"""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.predictor = EmotionPredictor(config)
        self.data_buffer = deque(maxlen=config.sequence_length)
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the streaming predictor"""
        self.is_initialized = self.predictor.load_model()
        return self.is_initialized

    def add_sensor_data(self, sensor_data: np.ndarray) -> Optional[EmotionPrediction]:
        """Add new sensor data and predict emotion if buffer is full"""
        # Add data to buffer
        if sensor_data.ndim == 1:
            self.data_buffer.append(sensor_data)
        else:
            # If multi-dimensional, take the latest reading
            self.data_buffer.append(sensor_data[-1])

        # Check if we have enough data for prediction
        if len(self.data_buffer) >= self.config.sequence_length:
            # Convert buffer to array for prediction
            buffer_array = np.array(list(self.data_buffer))

            try:
                return self.predictor.predict_emotion(buffer_array)
            except Exception as e:
                logger.error(f"Error in streaming prediction: {e}")
                return None

        return None

    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        return len(self.data_buffer)

    def clear_buffer(self):
        """Clear the data buffer"""
        self.data_buffer.clear()


# Example usage and demonstration
def demo_emotion_prediction():
    """Demonstrate emotion prediction functionality"""

    print("Emotion Prediction Demo")
    print("=" * 50)

    # Create sample physiological data
    np.random.seed(42)
    sequence_length = 300
    n_features = 5

    # Simulate different emotional states
    emotions_to_test = ['happy', 'sad', 'angry', 'neutral']

    for emotion in emotions_to_test:
        print(f"\nTesting emotion: {emotion.upper()}")

        # Create emotion-specific patterns
        if emotion == 'happy':
            hr_pattern = 75 + 8 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
            hrv_pattern = 45 + np.random.normal(0, 5, sequence_length)
            gsr_pattern = 2.5 + np.random.exponential(1.5, sequence_length)
        elif emotion == 'sad':
            hr_pattern = 65 + 3 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
            hrv_pattern = 35 + np.random.normal(0, 3, sequence_length)
            gsr_pattern = 1.2 + np.random.exponential(0.8, sequence_length)
        elif emotion == 'angry':
            hr_pattern = 85 + 12 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
            hrv_pattern = 25 + np.random.normal(0, 8, sequence_length)
            gsr_pattern = 3.0 + np.random.exponential(2.0, sequence_length)
        else:  # neutral
            hr_pattern = 70 + 5 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
            hrv_pattern = 40 + np.random.normal(0, 4, sequence_length)
            gsr_pattern = 1.8 + np.random.exponential(1.0, sequence_length)

        # Add noise
        hr_pattern += np.random.normal(0, 2, sequence_length)
        hrv_pattern += np.random.normal(0, 2, sequence_length)

        # Create feature array
        features = np.column_stack([
            hr_pattern,                    # Heart rate
            hrv_pattern,                   # Heart rate variability
            gsr_pattern,                   # Skin conductance
            36.5 + np.random.normal(0, 0.1, sequence_length),  # Temperature
            98 + np.random.normal(0, 1, sequence_length)       # Blood oxygen
        ])

        print(f"  Generated data shape: {features.shape}")
        print(f"  Heart rate range: {hr_pattern.min():.1f} - {hr_pattern.max():.1f} BPM")
        print(f"  GSR range: {gsr_pattern.min():.2f} - {gsr_pattern.max():.2f} μS")

        # Note: In a real scenario, you would load an actual trained model
        print("  Note: This demo shows data generation. Real prediction requires a trained model.")

    print("\nDemo completed successfully!")
    print("\nTo use real emotion prediction:")
    print("1. Train a model using the model_training module")
    print("2. Create a PredictionConfig with the model path")
    print("3. Load the model and call predict_emotion()")


if __name__ == "__main__":
    demo_emotion_prediction()