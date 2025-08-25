"""
Inference Pipeline for EQM Emotion Detection

This module provides a complete pipeline for emotion detection
from physiological sensor data, integrating all EQM components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .emotion_predictor import EmotionPredictor, PredictionConfig
from .real_time_processor import RealTimeEmotionProcessor, RealTimeConfig
from ..data_ingestion import IngestionPipeline, IngestionConfig
from ..data_preprocessing import PreprocessingPipeline, PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class EQMConfig:
    """Complete configuration for EQM system"""
    # Model configuration
    model_path: str
    preprocessor_path: Optional[str] = None

    # Real-time processing configuration
    prediction_interval: float = 1.0
    buffer_size_seconds: int = 300
    sampling_rate: int = 10

    # Data ingestion configuration
    enable_data_ingestion: bool = True
    storage_path: str = "./data/emotion_data"

    # Preprocessing configuration
    enable_preprocessing: bool = True
    sequence_length: int = 300

    # System configuration
    enable_real_time: bool = True
    enable_alerts: bool = True
    log_level: str = "INFO"


@dataclass
class EQMStatus:
    """Status of the EQM system"""
    is_running: bool = False
    model_loaded: bool = False
    data_ingestion_active: bool = False
    preprocessing_active: bool = False
    real_time_processing_active: bool = False
    connected_devices: List[str] = None
    current_emotion: Optional[str] = None
    confidence: float = 0.0
    start_time: Optional[datetime] = None

    def __post_init__(self):
        if self.connected_devices is None:
            self.connected_devices = []


class EQMInferencePipeline:
    """Complete EQM emotion detection pipeline"""

    def __init__(self, config: EQMConfig):
        self.config = config
        self.status = EQMStatus()

        # Initialize components
        self.predictor = None
        self.real_time_processor = None
        self.ingestion_pipeline = None
        self.preprocessing_pipeline = None

        # Callbacks
        self.emotion_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []

        logger.info("EQM Inference Pipeline initialized")

    async def initialize(self) -> bool:
        """Initialize all pipeline components"""
        try:
            # Initialize emotion predictor
            prediction_config = PredictionConfig(
                model_path=self.config.model_path,
                preprocessor_path=self.config.preprocessor_path,
                sequence_length=self.config.sequence_length,
                confidence_threshold=0.5,
                smoothing_window=5,
                enable_smoothing=True
            )

            self.predictor = EmotionPredictor(prediction_config)
            if not self.predictor.load_model():
                logger.error("Failed to load emotion detection model")
                return False

            self.status.model_loaded = True

            # Initialize real-time processor if enabled
            if self.config.enable_real_time:
                real_time_config = RealTimeConfig(
                    prediction_interval=self.config.prediction_interval,
                    buffer_size_seconds=self.config.buffer_size_seconds,
                    sampling_rate=self.config.sampling_rate,
                    enable_adaptive_prediction=True,
                    alert_on_emotion_change=self.config.enable_alerts
                )

                self.real_time_processor = RealTimeEmotionProcessor(self.predictor, real_time_config)

                # Register callbacks
                self.real_time_processor.add_emotion_callback(self._on_emotion_update)
                self.real_time_processor.add_alert_callback(self._on_emotion_alert)

            # Initialize data ingestion if enabled
            if self.config.enable_data_ingestion:
                from ..data_ingestion import DeviceConfig, DeviceConnectorFactory

                ingestion_config = IngestionConfig(
                    storage_path=self.config.storage_path,
                    enable_validation=True
                )

                self.ingestion_pipeline = IngestionPipeline(ingestion_config)

                # Add sample device (in real usage, this would be configured)
                # device_config = DeviceConfig(
                #     device_type="apple_watch",
                #     base_url="ws://localhost:8080"
                # )
                # self.ingestion_pipeline.add_device(device_config)

            # Initialize preprocessing if enabled
            if self.config.enable_preprocessing:
                preprocessing_config = PreprocessingConfig(
                    enable_feature_selection=True,
                    feature_selection_method="variance"
                )

                self.preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)

            logger.info("EQM pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize EQM pipeline: {e}")
            return False

    async def start(self) -> bool:
        """Start the EQM system"""
        if self.status.is_running:
            logger.warning("EQM system is already running")
            return True

        try:
            self.status.start_time = datetime.utcnow()
            self.status.is_running = True

            # Start real-time processing
            if self.real_time_processor:
                await self.real_time_processor.start_processing()
                self.status.real_time_processing_active = True

            # Start data ingestion
            if self.ingestion_pipeline:
                await self.ingestion_pipeline.start()
                self.status.data_ingestion_active = True

            logger.info("EQM system started successfully")
            self._notify_status_callbacks()

            return True

        except Exception as e:
            logger.error(f"Failed to start EQM system: {e}")
            self.status.is_running = False
            return False

    async def stop(self) -> bool:
        """Stop the EQM system"""
        if not self.status.is_running:
            return True

        try:
            # Stop real-time processing
            if self.real_time_processor:
                self.real_time_processor.stop_processing()
                self.status.real_time_processing_active = False

            # Stop data ingestion
            if self.ingestion_pipeline:
                await self.ingestion_pipeline.stop()
                self.status.data_ingestion_active = False

            self.status.is_running = False
            logger.info("EQM system stopped successfully")
            self._notify_status_callbacks()

            return True

        except Exception as e:
            logger.error(f"Error stopping EQM system: {e}")
            return False

    def add_emotion_callback(self, callback: Callable):
        """Add callback for emotion updates"""
        self.emotion_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable):
        """Add callback for emotion alerts"""
        self.alert_callbacks.append(callback)

    def add_status_callback(self, callback: Callable):
        """Add callback for system status updates"""
        self.status_callbacks.append(callback)

    async def process_sensor_data(self, sensor_data: 'np.ndarray') -> Optional[Dict[str, Any]]:
        """Process sensor data through the pipeline"""
        if not self.status.is_running:
            logger.warning("EQM system is not running")
            return None

        try:
            if self.real_time_processor:
                # Add to real-time processor
                self.real_time_processor.add_sensor_data(sensor_data)

                # Get current state
                current_state = self.real_time_processor.get_current_state()
                processing_stats = self.real_time_processor.get_processing_stats()

                return {
                    'emotion': current_state.current_emotion,
                    'confidence': current_state.confidence,
                    'stability_count': current_state.stability_count,
                    'processing_stats': processing_stats,
                    'timestamp': datetime.utcnow()
                }
            else:
                # Direct prediction
                if self.predictor:
                    prediction = self.predictor.predict_emotion(sensor_data)
                    return {
                        'emotion': prediction.emotion,
                        'confidence': prediction.confidence,
                        'probabilities': prediction.probabilities,
                        'timestamp': prediction.timestamp
                    }

        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            return None

    def get_system_status(self) -> EQMStatus:
        """Get current system status"""
        return self.status

    def get_processing_stats(self) -> Optional[Dict[str, Any]]:
        """Get processing statistics"""
        if self.real_time_processor:
            stats = self.real_time_processor.get_processing_stats()
            buffer_info = self.real_time_processor.get_buffer_info()

            return {
                'predictions_per_second': stats.predictions_per_second,
                'average_confidence': stats.average_confidence,
                'emotion_changes': stats.emotion_changes,
                'average_processing_delay': np.mean(stats.processing_delays) if stats.processing_delays else 0,
                'buffer_info': buffer_info
            }

        return None

    def _on_emotion_update(self, prediction, emotion_state):
        """Handle emotion updates from real-time processor"""
        # Update system status
        self.status.current_emotion = prediction.emotion
        self.status.confidence = prediction.confidence

        # Notify callbacks
        for callback in self.emotion_callbacks:
            try:
                asyncio.create_task(callback(prediction, emotion_state))
            except Exception as e:
                logger.error(f"Error in emotion callback: {e}")

    def _on_emotion_alert(self, alert_data):
        """Handle emotion alerts from real-time processor"""
        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                asyncio.create_task(callback(alert_data))
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _notify_status_callbacks(self):
        """Notify status update callbacks"""
        for callback in self.status_callbacks:
            try:
                asyncio.create_task(callback(self.status))
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    async def simulate_emotion_data(self, duration_seconds: int = 60):
        """Simulate emotion data for testing"""
        if not self.status.is_running:
            logger.warning("System must be running to simulate data")
            return

        logger.info(f"Starting emotion data simulation for {duration_seconds} seconds")

        # Import simulator
        from .real_time_processor import SensorDataSimulator

        simulator = SensorDataSimulator(sampling_rate=self.config.sampling_rate)
        emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']

        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).seconds < duration_seconds:
            # Randomly change emotion occasionally
            if np.random.random() < 0.02:  # 2% chance per sample
                new_emotion = np.random.choice(emotions)
                simulator.change_emotion(new_emotion)
                logger.info(f"Simulation: Emotion changed to {new_emotion}")

            # Generate and process sensor data
            sensor_data = simulator.generate_sample()
            await self.process_sensor_data(sensor_data)

            # Wait for next sample
            await asyncio.sleep(1.0 / self.config.sampling_rate)

        logger.info("Emotion data simulation completed")


# WebSocket-based real-time interface
class EQMWebSocketServer:
    """WebSocket server for real-time EQM communication"""

    def __init__(self, eqm_pipeline: EQMInferencePipeline, host: str = "localhost", port: int = 8080):
        self.eqm_pipeline = eqm_pipeline
        self.host = host
        self.port = port
        self.connected_clients = set()

    async def start_server(self):
        """Start WebSocket server"""
        try:
            import websockets

            async def handler(websocket, path):
                await self._handle_client(websocket, path)

            server = await websockets.serve(handler, self.host, self.port)
            logger.info(f"EQM WebSocket server started on ws://{self.host}:{self.port}")

            # Keep server running
            await server.wait_closed()

        except ImportError:
            logger.error("websockets package not installed")
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")

    async def _handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.connected_clients.add(websocket)

        try:
            # Send initial status
            status = self.eqm_pipeline.get_system_status()
            await websocket.send(json.dumps({
                'type': 'status',
                'data': {
                    'is_running': status.is_running,
                    'current_emotion': status.current_emotion,
                    'confidence': status.confidence
                }
            }))

            # Register callbacks for this client
            def emotion_callback(prediction, emotion_state):
                asyncio.create_task(self._send_emotion_update(websocket, prediction, emotion_state))

            def alert_callback(alert_data):
                asyncio.create_task(self._send_alert(websocket, alert_data))

            self.eqm_pipeline.add_emotion_callback(emotion_callback)
            self.eqm_pipeline.add_alert_callback(alert_callback)

            # Listen for client messages
            async for message in websocket:
                await self._handle_client_message(websocket, message)

        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            self.connected_clients.remove(websocket)

    async def _send_emotion_update(self, websocket, prediction, emotion_state):
        """Send emotion update to client"""
        try:
            await websocket.send(json.dumps({
                'type': 'emotion_update',
                'data': {
                    'emotion': prediction.emotion,
                    'confidence': float(prediction.confidence),
                    'probabilities': {k: float(v) for k, v in prediction.probabilities.items()},
                    'stability_count': emotion_state.stability_count,
                    'timestamp': prediction.timestamp.isoformat()
                }
            }))
        except Exception as e:
            logger.error(f"Error sending emotion update: {e}")

    async def _send_alert(self, websocket, alert_data):
        """Send alert to client"""
        try:
            await websocket.send(json.dumps({
                'type': 'alert',
                'data': alert_data
            }))
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

    async def _handle_client_message(self, websocket, message):
        """Handle message from client"""
        try:
            data = json.loads(message)

            if data.get('type') == 'start_simulation':
                duration = data.get('duration', 60)
                asyncio.create_task(self.eqm_pipeline.simulate_emotion_data(duration))

            elif data.get('type') == 'get_status':
                status = self.eqm_pipeline.get_system_status()
                await websocket.send(json.dumps({
                    'type': 'status_response',
                    'data': {
                        'is_running': status.is_running,
                        'current_emotion': status.current_emotion,
                        'confidence': status.confidence,
                        'connected_devices': status.connected_devices
                    }
                }))

        except Exception as e:
            logger.error(f"Error handling client message: {e}")


# Example usage and demonstration
async def demo_eqm_pipeline():
    """Demonstrate the complete EQM inference pipeline"""

    print("EQM Inference Pipeline Demo")
    print("=" * 50)

    # Note: This demo shows the structure but requires a trained model
    print("Note: This demo requires a trained emotion detection model.")
    print("The EQM pipeline integrates:")
    print("1. Emotion prediction from trained models")
    print("2. Real-time processing of sensor data")
    print("3. Data ingestion from multiple devices")
    print("4. Preprocessing and feature extraction")
    print("5. WebSocket interface for real-time communication")
    print()

    # Create EQM configuration
    config = EQMConfig(
        model_path="./models/emotion_model.h5",  # Would need actual model
        enable_real_time=True,
        enable_data_ingestion=True,
        enable_preprocessing=True,
        prediction_interval=1.0,
        sampling_rate=10,
        buffer_size_seconds=300
    )

    print("EQM Configuration:")
    print(f"  Model path: {config.model_path}")
    print(f"  Real-time processing: {'Enabled' if config.enable_real_time else 'Disabled'}")
    print(f"  Data ingestion: {'Enabled' if config.enable_data_ingestion else 'Disabled'}")
    print(f"  Preprocessing: {'Enabled' if config.enable_preprocessing else 'Disabled'}")
    print(f"  Prediction interval: {config.prediction_interval}s")
    print(f"  Sampling rate: {config.sampling_rate}Hz")
    print()

    print("To use the complete EQM system:")
    print("1. Train an emotion detection model using model_training module")
    print("2. Create EQMInferencePipeline with trained model")
    print("3. Initialize and start the pipeline")
    print("4. Feed sensor data for real-time emotion detection")
    print("5. Optionally start WebSocket server for remote access")
    print()

    print("Example code structure:")
    print("""
    # Initialize EQM pipeline
    eqm_config = EQMConfig(model_path="./models/trained_model.h5")
    eqm_pipeline = EQMInferencePipeline(eqm_config)

    # Initialize and start
    await eqm_pipeline.initialize()
    await eqm_pipeline.start()

    # Process sensor data
    sensor_data = np.array([heart_rate, hrv, gsr, temperature, spo2])
    result = await eqm_pipeline.process_sensor_data(sensor_data)

    # Start WebSocket server for real-time communication
    ws_server = EQMWebSocketServer(eqm_pipeline)
    await ws_server.start_server()
    """)

    print("\nDemo completed successfully!")
    print("The EQM system is ready for emotion detection from physiological data.")


if __name__ == "__main__":
    asyncio.run(demo_eqm_pipeline())