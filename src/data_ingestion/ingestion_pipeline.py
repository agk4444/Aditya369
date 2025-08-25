"""
Ingestion Pipeline for EQM Data Collection

This module orchestrates the complete data ingestion process from multiple
devices, including validation, processing, and storage.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import aiofiles
from pathlib import Path

from .device_connectors import (
    DeviceConnector,
    DeviceConnectorFactory,
    DeviceConfig,
    SensorData
)
from .data_validator import DataValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline"""
    batch_size: int = 100
    flush_interval: int = 60  # seconds
    max_queue_size: int = 10000
    storage_path: str = "./data/raw"
    enable_validation: bool = True
    enable_compression: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class IngestionStats:
    """Statistics for ingestion pipeline performance"""
    total_readings: int = 0
    valid_readings: int = 0
    invalid_readings: int = 0
    batches_processed: int = 0
    errors_encountered: int = 0
    devices_connected: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def get_processing_rate(self) -> float:
        """Calculate readings processed per second"""
        if not self.start_time or not self.end_time:
            return 0.0
        duration = (self.end_time - self.start_time).total_seconds()
        return self.total_readings / duration if duration > 0 else 0.0


class IngestionPipeline:
    """Main ingestion pipeline for collecting and processing sensor data"""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.devices: Dict[str, DeviceConnector] = {}
        self.validator = DataValidator()
        self.stats = IngestionStats()
        self.is_running = False
        self.data_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Callbacks
        self.data_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []

        logger.info(f"Initialized ingestion pipeline with config: {config}")

    def add_device(self, device_config: DeviceConfig) -> bool:
        """Add a device to the ingestion pipeline"""
        try:
            device_id = device_config.device_type
            if device_id in self.devices:
                logger.warning(f"Device {device_id} already exists")
                return False

            connector = DeviceConnectorFactory.create_connector(
                device_config.device_type,
                device_config
            )
            self.devices[device_id] = connector
            logger.info(f"Added device: {device_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add device {device_config.device_type}: {e}")
            return False

    def remove_device(self, device_id: str) -> bool:
        """Remove a device from the ingestion pipeline"""
        if device_id not in self.devices:
            logger.warning(f"Device {device_id} not found")
            return False

        try:
            connector = self.devices[device_id]
            asyncio.create_task(connector.disconnect())
            del self.devices[device_id]
            logger.info(f"Removed device: {device_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing device {device_id}: {e}")
            return False

    def add_data_callback(self, callback: Callable):
        """Add callback for processed data"""
        self.data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable):
        """Add callback for errors"""
        self.error_callbacks.append(callback)

    async def start(self) -> bool:
        """Start the ingestion pipeline"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return False

        try:
            self.is_running = True
            self.stats.start_time = datetime.utcnow()

            # Connect to all devices
            connection_tasks = []
            for device_id, connector in self.devices.items():
                connection_tasks.append(self._connect_device(device_id, connector))

            await asyncio.gather(*connection_tasks, return_exceptions=True)

            # Start processing tasks
            processing_task = asyncio.create_task(self._process_data_queue())
            storage_task = asyncio.create_task(self._periodic_storage_flush())

            # Start data collection from devices
            collection_tasks = []
            for device_id, connector in self.devices.items():
                collection_tasks.append(self._collect_from_device(device_id, connector))

            # Wait for all tasks to complete or be cancelled
            await asyncio.gather(
                processing_task,
                storage_task,
                *collection_tasks,
                return_exceptions=True
            )

            return True

        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self.is_running = False
            return False

    async def stop(self) -> bool:
        """Stop the ingestion pipeline"""
        if not self.is_running:
            return True

        try:
            self.is_running = False
            self.stats.end_time = datetime.utcnow()

            # Disconnect from all devices
            disconnect_tasks = []
            for device_id, connector in self.devices.items():
                disconnect_tasks.append(connector.disconnect())

            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

            logger.info("Pipeline stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
            return False

    async def _connect_device(self, device_id: str, connector: DeviceConnector) -> bool:
        """Connect to a specific device"""
        try:
            connected = await connector.connect()
            if connected:
                self.stats.devices_connected += 1
                logger.info(f"Connected to device: {device_id}")
                return True
            else:
                logger.error(f"Failed to connect to device: {device_id}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to device {device_id}: {e}")
            return False

    async def _collect_from_device(self, device_id: str, connector: DeviceConnector):
        """Collect data from a specific device"""
        sensor_types = [
            "heart_rate", "heart_rate_variability", "temperature",
            "galvanic_skin_response", "blood_oxygen",
            "accelerometer_x", "accelerometer_y", "accelerometer_z"
        ]

        while self.is_running:
            try:
                # Get sensor data from device
                data = await connector.get_sensor_data(sensor_types)

                if data:
                    # Add device ID to each reading
                    for reading in data:
                        reading.device_id = device_id

                    # Add to processing queue
                    for reading in data:
                        await self.data_queue.put(reading)

                    logger.debug(f"Collected {len(data)} readings from {device_id}")

                # Wait before next collection
                await asyncio.sleep(1)  # 1 second intervals

            except asyncio.QueueFull:
                logger.warning(f"Data queue full for device {device_id}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error collecting data from {device_id}: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def _process_data_queue(self):
        """Process data from the queue"""
        batch = []
        last_flush = datetime.utcnow()

        while self.is_running:
            try:
                # Get data from queue with timeout
                try:
                    data = await asyncio.wait_for(self.data_queue.get(), timeout=1.0)
                    batch.append(data)
                    self.stats.total_readings += 1
                except asyncio.TimeoutError:
                    # Process current batch even if not full
                    if batch:
                        await self._process_batch(batch)
                        batch = []
                    continue

                # Process batch if it's full or time to flush
                current_time = datetime.utcnow()
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (current_time - last_flush).total_seconds() >= self.config.flush_interval
                )

                if should_flush and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_flush = current_time

            except Exception as e:
                logger.error(f"Error processing data queue: {e}")
                self.stats.errors_encountered += 1
                await self._notify_error_callbacks(e)

    async def _process_batch(self, batch: List[SensorData]):
        """Process a batch of sensor data"""
        try:
            # Validate data if enabled
            if self.config.enable_validation:
                validated_data = await self._validate_batch(batch)
            else:
                validated_data = [(data, []) for data in batch]

            # Separate valid and invalid data
            valid_data = []
            invalid_data = []

            for data, validation_results in validated_data:
                has_errors = any(not result.is_valid for result in validation_results)
                if has_errors:
                    invalid_data.append((data, validation_results))
                    self.stats.invalid_readings += 1
                else:
                    valid_data.append(data)
                    self.stats.valid_readings += 1

            # Process valid data
            if valid_data:
                await self._store_data(valid_data)
                await self._notify_data_callbacks(valid_data)

            # Log invalid data
            if invalid_data:
                await self._handle_invalid_data(invalid_data)

            self.stats.batches_processed += 1

            logger.debug(f"Processed batch: {len(valid_data)} valid, {len(invalid_data)} invalid")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.stats.errors_encountered += 1
            await self._notify_error_callbacks(e)

    async def _validate_batch(self, batch: List[SensorData]) -> List[tuple]:
        """Validate a batch of sensor data"""
        validated = []

        for data in batch:
            validation_results = self.validator.validate_sensor_data(data)
            validated.append((data, validation_results))

        return validated

    async def _store_data(self, data: List[SensorData]):
        """Store valid sensor data"""
        try:
            # Group data by timestamp for file organization
            timestamp = datetime.utcnow()
            filename = f"sensor_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.jsonl"

            filepath = self.storage_path / filename

            # Write data in JSON Lines format
            async with aiofiles.open(filepath, 'a', encoding='utf-8') as f:
                for reading in data:
                    json_data = {
                        'device_id': reading.device_id,
                        'user_id': reading.user_id,
                        'timestamp': reading.timestamp.isoformat() if isinstance(reading.timestamp, datetime) else reading.timestamp,
                        'sensor_type': reading.sensor_type,
                        'value': reading.value,
                        'unit': reading.unit,
                        'metadata': reading.metadata
                    }
                    await f.write(json.dumps(json_data) + '\n')

            logger.debug(f"Stored {len(data)} readings to {filepath}")

        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise

    async def _handle_invalid_data(self, invalid_data: List[tuple]):
        """Handle invalid sensor data"""
        try:
            # Log invalid data for analysis
            for data, validation_results in invalid_data:
                logger.warning(f"Invalid data: {data.device_id} - {data.sensor_type}: {data.value}")
                for result in validation_results:
                    if not result.is_valid:
                        logger.warning(f"  Validation: {result.message}")

        except Exception as e:
            logger.error(f"Error handling invalid data: {e}")

    async def _periodic_storage_flush(self):
        """Periodically flush any remaining data"""
        while self.is_running:
            await asyncio.sleep(self.config.flush_interval)
            # Force flush of any remaining data in queue
            if not self.data_queue.empty():
                logger.info("Flushing remaining data in queue")

    async def _notify_data_callbacks(self, data: List[SensorData]):
        """Notify all registered data callbacks"""
        for callback in self.data_callbacks:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Error in data callback {callback}: {e}")

    async def _notify_error_callbacks(self, error: Exception):
        """Notify all registered error callbacks"""
        for callback in self.error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"Error in error callback {callback}: {e}")

    def get_stats(self) -> IngestionStats:
        """Get current pipeline statistics"""
        return self.stats

    def get_device_status(self) -> Dict[str, bool]:
        """Get status of all devices"""
        return {
            device_id: connector.is_connected
            for device_id, connector in self.devices.items()
        }


class StreamingIngestionPipeline(IngestionPipeline):
    """Streaming version of the ingestion pipeline with real-time processing"""

    def __init__(self, config: IngestionConfig):
        super().__init__(config)
        self.streaming_callbacks: List[Callable] = []

    def add_streaming_callback(self, callback: Callable):
        """Add callback for real-time streaming data"""
        self.streaming_callbacks.append(callback)

    async def _process_data_queue(self):
        """Override to provide streaming processing"""
        while self.is_running:
            try:
                # Get data from queue
                data = await self.data_queue.get()

                # Process single reading for streaming
                await self._process_single_reading(data)

                self.stats.total_readings += 1

            except Exception as e:
                logger.error(f"Error in streaming processing: {e}")
                self.stats.errors_encountered += 1

    async def _process_single_reading(self, data: SensorData):
        """Process a single sensor reading for streaming"""
        try:
            # Validate data
            if self.config.enable_validation:
                validation_results = self.validator.validate_sensor_data(data)
                has_errors = any(not result.is_valid for result in validation_results)

                if has_errors:
                    self.stats.invalid_readings += 1
                    await self._handle_invalid_data([(data, validation_results)])
                    return

            self.stats.valid_readings += 1

            # Notify streaming callbacks immediately
            await self._notify_streaming_callbacks(data)

            # Store data
            await self._store_data([data])

        except Exception as e:
            logger.error(f"Error processing single reading: {e}")

    async def _notify_streaming_callbacks(self, data: SensorData):
        """Notify streaming callbacks with individual readings"""
        for callback in self.streaming_callbacks:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Error in streaming callback {callback}: {e}")


# Example usage and demonstration
async def demo_ingestion_pipeline():
    """Demonstrate the ingestion pipeline functionality"""

    # Create pipeline configuration
    config = IngestionConfig(
        batch_size=10,
        flush_interval=30,
        storage_path="./demo_data",
        enable_validation=True
    )

    # Create pipeline
    pipeline = IngestionPipeline(config)

    # Add devices
    devices = [
        DeviceConfig(
            device_type="apple_watch",
            base_url="ws://localhost:8080/apple_watch"
        ),
        DeviceConfig(
            device_type="oura_ring",
            base_url="https://api.ouraring.com",
            api_key="demo_key",
            api_secret="demo_secret"
        )
    ]

    for device_config in devices:
        pipeline.add_device(device_config)

    # Add callbacks
    async def data_handler(data: List[SensorData]):
        print(f"Received {len(data)} sensor readings:")
        for reading in data:
            print(f"  {reading.device_id}: {reading.sensor_type} = {reading.value} {reading.unit}")

    async def error_handler(error: Exception):
        print(f"Pipeline error: {error}")

    pipeline.add_data_callback(data_handler)
    pipeline.add_error_callback(error_handler)

    print("Starting ingestion pipeline...")
    print("Note: This demo will run for 10 seconds with simulated data")
    print("In a real scenario, actual devices would be connected")

    # Start pipeline
    start_task = asyncio.create_task(pipeline.start())

    # Let it run for a demo period
    await asyncio.sleep(10)

    # Stop pipeline
    await pipeline.stop()

    # Print statistics
    stats = pipeline.get_stats()
    print("\nPipeline Statistics:")
    print(f"  Total readings: {stats.total_readings}")
    print(f"  Valid readings: {stats.valid_readings}")
    print(f"  Invalid readings: {stats.invalid_readings}")
    print(f"  Batches processed: {stats.batches_processed}")
    print(f"  Errors encountered: {stats.errors_encountered}")
    print(f"  Devices connected: {stats.devices_connected}")
    print(".2f")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_ingestion_pipeline())