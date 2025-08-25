"""
Real-Time Emotion Processor for EQM

This module provides real-time emotion processing capabilities
for streaming physiological sensor data.
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import threading
import time

from .emotion_predictor import EmotionPredictor, PredictionConfig, EmotionPrediction

logger = logging.getLogger(__name__)


@dataclass
class RealTimeConfig:
    """Configuration for real-time emotion processing"""
    prediction_interval: float = 1.0  # seconds
    buffer_size_seconds: int = 300  # 5 minutes of data
    sampling_rate: int = 10  # Hz
    enable_adaptive_prediction: bool = True
    min_confidence_threshold: float = 0.3
    max_confidence_threshold: float = 0.8
    alert_on_emotion_change: bool = True
    emotion_stability_threshold: int = 3  # consecutive predictions


@dataclass
class EmotionState:
    """Current emotional state"""
    current_emotion: str
    confidence: float
    stability_count: int
    last_changed: datetime
    emotion_history: List[str]
    confidence_history: List[float]


@dataclass
class ProcessingStats:
    """Real-time processing statistics"""
    predictions_per_second: float
    average_confidence: float
    emotion_changes: int
    processing_delays: List[float]
    buffer_utilization: float


class RealTimeEmotionProcessor:
    """Real-time emotion processing for streaming data"""

    def __init__(self, predictor: EmotionPredictor, config: RealTimeConfig):
        self.predictor = predictor
        self.config = config

        # Data buffers
        self.data_buffer = deque(maxlen=config.buffer_size_seconds * config.sampling_rate)
        self.timestamp_buffer = deque(maxlen=config.buffer_size_seconds * config.sampling_rate)

        # Emotion state
        self.emotion_state = EmotionState(
            current_emotion='neutral',
            confidence=0.0,
            stability_count=0,
            last_changed=datetime.utcnow(),
            emotion_history=[],
            confidence_history=[]
        )

        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.last_prediction_time = datetime.utcnow()

        # Callbacks
        self.emotion_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []

        # Statistics
        self.stats = ProcessingStats(
            predictions_per_second=0.0,
            average_confidence=0.0,
            emotion_changes=0,
            processing_delays=[],
            buffer_utilization=0.0
        )

        # Adaptive thresholds
        self.confidence_threshold = (config.min_confidence_threshold + config.max_confidence_threshold) / 2

        logger.info("Real-time emotion processor initialized")

    def add_emotion_callback(self, callback: Callable):
        """Add callback for emotion updates"""
        self.emotion_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable):
        """Add callback for emotion alerts"""
        self.alert_callbacks.append(callback)

    def add_sensor_data(self, sensor_data: np.ndarray, timestamp: Optional[datetime] = None):
        """Add new sensor data to the buffer"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Add data to buffers
        self.data_buffer.append(sensor_data)
        self.timestamp_buffer.append(timestamp)

        # Update buffer utilization
        self.stats.buffer_utilization = len(self.data_buffer) / self.data_buffer.maxlen

    async def start_processing(self):
        """Start real-time emotion processing"""
        if self.is_processing:
            logger.warning("Processing is already running")
            return

        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        logger.info("Real-time emotion processing started")

    def stop_processing(self):
        """Stop real-time emotion processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        logger.info("Real-time emotion processing stopped")

    def _processing_loop(self):
        """Main processing loop"""
        while self.is_processing:
            try:
                current_time = datetime.utcnow()

                # Check if enough time has passed for next prediction
                time_since_last_prediction = (current_time - self.last_prediction_time).total_seconds()

                if time_since_last_prediction >= self.config.prediction_interval:
                    self._process_prediction()
                    self.last_prediction_time = current_time

                # Small sleep to prevent busy waiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1.0)

    def _process_prediction(self):
        """Process emotion prediction from current buffer"""
        try:
            # Check if we have enough data
            min_samples = int(self.config.prediction_interval * self.config.sampling_rate)
            if len(self.data_buffer) < min_samples:
                logger.debug("Not enough data for prediction")
                return

            # Get recent data for prediction
            recent_data = np.array(list(self.data_buffer)[-min_samples:])

            # Make prediction
            prediction_start = time.time()
            prediction = self.predictor.predict_emotion(recent_data)
            prediction_time = time.time() - prediction_start

            # Update processing statistics
            self.stats.processing_delays.append(prediction_time)
            if len(self.stats.processing_delays) > 100:  # Keep last 100 delays
                self.stats.processing_delays = self.stats.processing_delays[-100:]

            # Update emotion state
            self._update_emotion_state(prediction)

            # Update statistics
            self._update_processing_stats()

            # Notify callbacks
            self._notify_emotion_callbacks(prediction)

        except Exception as e:
            logger.error(f"Error processing prediction: {e}")

    def _update_emotion_state(self, prediction: EmotionPrediction):
        """Update the current emotion state"""
        # Check if emotion changed
        emotion_changed = prediction.emotion != self.emotion_state.current_emotion

        if emotion_changed:
            # Check confidence threshold for emotion changes
            if prediction.confidence >= self.confidence_threshold:
                self.emotion_state.current_emotion = prediction.emotion
                self.emotion_state.confidence = prediction.confidence
                self.emotion_state.stability_count = 1
                self.emotion_state.last_changed = prediction.timestamp
                self.stats.emotion_changes += 1

                # Trigger alert if enabled
                if self.config.alert_on_emotion_change:
                    self._notify_alert_callbacks(prediction)

                logger.info(f"Emotion changed to: {prediction.emotion} (confidence: {prediction.confidence:.3f})")
            else:
                logger.debug(f"Ignoring emotion change due to low confidence: {prediction.confidence:.3f}")
        else:
            # Increase stability count for consistent predictions
            if prediction.confidence >= self.confidence_threshold:
                self.emotion_state.stability_count += 1
                self.emotion_state.confidence = prediction.confidence

        # Update history
        self.emotion_state.emotion_history.append(prediction.emotion)
        self.emotion_state.confidence_history.append(prediction.confidence)

        # Keep history size manageable
        max_history = 100
        if len(self.emotion_state.emotion_history) > max_history:
            self.emotion_state.emotion_history = self.emotion_state.emotion_history[-max_history:]
            self.emotion_state.confidence_history = self.emotion_state.confidence_history[-max_history:]

        # Adaptive confidence threshold
        if self.config.enable_adaptive_prediction:
            self._update_adaptive_threshold(prediction)

    def _update_adaptive_threshold(self, prediction: EmotionPrediction):
        """Update confidence threshold based on prediction stability"""
        if len(self.emotion_state.confidence_history) >= 10:
            recent_confidence = np.mean(self.emotion_state.confidence_history[-10:])

            # Adjust threshold based on recent confidence stability
            if np.std(self.emotion_state.confidence_history[-10:]) < 0.1:
                # High stability, can be more selective
                self.confidence_threshold = min(
                    self.config.max_confidence_threshold,
                    self.confidence_threshold + 0.05
                )
            else:
                # Low stability, be more accepting
                self.confidence_threshold = max(
                    self.config.min_confidence_threshold,
                    self.confidence_threshold - 0.02
                )

    def _update_processing_stats(self):
        """Update processing statistics"""
        if len(self.stats.processing_delays) > 0:
            self.stats.predictions_per_second = 1.0 / np.mean(self.stats.processing_delays)

        if len(self.emotion_state.confidence_history) > 0:
            self.stats.average_confidence = np.mean(self.emotion_state.confidence_history[-50:])  # Last 50 predictions

    def _notify_emotion_callbacks(self, prediction: EmotionPrediction):
        """Notify emotion update callbacks"""
        for callback in self.emotion_callbacks:
            try:
                asyncio.run(callback(prediction, self.emotion_state))
            except Exception as e:
                logger.error(f"Error in emotion callback: {e}")

    def _notify_alert_callbacks(self, prediction: EmotionPrediction):
        """Notify emotion alert callbacks"""
        alert_data = {
            'previous_emotion': self.emotion_state.current_emotion,
            'new_emotion': prediction.emotion,
            'confidence': prediction.confidence,
            'timestamp': prediction.timestamp,
            'stability_count': self.emotion_state.stability_count
        }

        for callback in self.alert_callbacks:
            try:
                asyncio.run(callback(alert_data))
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def get_current_state(self) -> EmotionState:
        """Get current emotion state"""
        return self.emotion_state

    def get_processing_stats(self) -> ProcessingStats:
        """Get processing statistics"""
        return self.stats

    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer information"""
        return {
            'buffer_size': len(self.data_buffer),
            'max_buffer_size': self.data_buffer.maxlen,
            'buffer_utilization': self.stats.buffer_utilization,
            'oldest_timestamp': self.timestamp_buffer[0] if self.timestamp_buffer else None,
            'newest_timestamp': self.timestamp_buffer[-1] if self.timestamp_buffer else None
        }

    def clear_buffers(self):
        """Clear all data buffers"""
        self.data_buffer.clear()
        self.timestamp_buffer.clear()
        logger.info("Data buffers cleared")

    def reset_emotion_state(self):
        """Reset emotion state to neutral"""
        self.emotion_state = EmotionState(
            current_emotion='neutral',
            confidence=0.0,
            stability_count=0,
            last_changed=datetime.utcnow(),
            emotion_history=[],
            confidence_history=[]
        )
        logger.info("Emotion state reset to neutral")


# Streaming Data Simulator for Testing
class SensorDataSimulator:
    """Simulate streaming sensor data for testing"""

    def __init__(self, sampling_rate: int = 10):
        self.sampling_rate = sampling_rate
        self.current_emotion = 'neutral'
        self.emotion_patterns = {
            'neutral': {'hr': 70, 'hrv': 40, 'gsr': 1.8},
            'happy': {'hr': 75, 'hrv': 45, 'gsr': 2.5},
            'sad': {'hr': 65, 'hrv': 35, 'gsr': 1.2},
            'angry': {'hr': 85, 'hrv': 25, 'gsr': 3.0},
            'fear': {'hr': 90, 'hrv': 20, 'gsr': 3.5},
            'surprise': {'hr': 80, 'hrv': 30, 'gsr': 2.8},
            'disgust': {'hr': 68, 'hrv': 38, 'gsr': 1.5}
        }

    def generate_sample(self, emotion: Optional[str] = None) -> np.ndarray:
        """Generate a single sensor data sample"""
        if emotion:
            self.current_emotion = emotion

        pattern = self.emotion_patterns[self.current_emotion]

        # Generate physiological data with some variation
        hr = pattern['hr'] + np.random.normal(0, 2)
        hrv = pattern['hrv'] + np.random.normal(0, 3)
        gsr = max(0.5, pattern['gsr'] + np.random.exponential(0.5))
        temp = 36.5 + np.random.normal(0, 0.1)
        spo2 = 98 + np.random.normal(0, 1)

        return np.array([hr, hrv, gsr, temp, spo2])

    def change_emotion(self, emotion: str):
        """Change the current emotion being simulated"""
        if emotion in self.emotion_patterns:
            self.current_emotion = emotion
            logger.info(f"Simulator emotion changed to: {emotion}")
        else:
            logger.warning(f"Unknown emotion: {emotion}")


# Example usage and demonstration
async def demo_real_time_processing():
    """Demonstrate real-time emotion processing"""

    print("Real-Time Emotion Processing Demo")
    print("=" * 50)

    # Note: This demo shows the structure but requires a trained model
    print("Note: This demo requires a trained emotion detection model.")
    print("In a real scenario, you would:")
    print("1. Load a trained model using EmotionPredictor")
    print("2. Create RealTimeEmotionProcessor with the predictor")
    print("3. Start processing and feed sensor data")
    print()

    # Create simulator for demonstration
    simulator = SensorDataSimulator(sampling_rate=10)

    print("Simulating different emotional states:")
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fear']

    for emotion in emotions:
        simulator.change_emotion(emotion)

        # Generate multiple samples for this emotion
        samples = []
        for _ in range(10):
            sample = simulator.generate_sample()
            samples.append(sample)

        samples = np.array(samples)

        print(f"\n{emotion.upper()} emotion samples:")
        print(f"  Heart Rate: {samples[:, 0].mean():.1f} ± {samples[:, 0].std():.1f} BPM")
        print(f"  HRV: {samples[:, 1].mean():.1f} ± {samples[:, 1].std():.1f} ms")
        print(f"  GSR: {samples[:, 2].mean():.2f} ± {samples[:, 2].std():.2f} μS")

    print("\nDemo completed successfully!")
    print("\nTo implement real-time processing:")
    print("1. Initialize RealTimeEmotionProcessor with a trained model")
    print("2. Call add_sensor_data() with new sensor readings")
    print("3. Start processing with start_processing()")
    print("4. Register callbacks for emotion updates")


if __name__ == "__main__":
    asyncio.run(demo_real_time_processing())