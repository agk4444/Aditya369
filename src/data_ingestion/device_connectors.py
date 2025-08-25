"""
Device Connectors for EQM Data Ingestion

This module provides connectors for various smart devices and wearables
to collect physiological data for emotion analysis.
"""

import asyncio
import json
import websockets
import aiohttp
import requests
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Data class for sensor readings"""
    device_id: str
    user_id: str
    timestamp: datetime
    sensor_type: str
    value: float
    unit: str
    metadata: Dict[str, Any]


@dataclass
class DeviceConfig:
    """Configuration for device connections"""
    device_type: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    webhook_url: Optional[str] = None
    polling_interval: int = 60  # seconds


class DeviceConnector(ABC):
    """Abstract base class for device connectors"""

    def __init__(self, config: DeviceConfig):
        self.config = config
        self.is_connected = False
        self.data_callbacks: List[Callable] = []

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to device"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from device"""
        pass

    @abstractmethod
    async def get_sensor_data(self, sensor_types: List[str]) -> List[SensorData]:
        """Retrieve sensor data from device"""
        pass

    def add_data_callback(self, callback: Callable):
        """Add callback function for data updates"""
        self.data_callbacks.append(callback)

    def remove_data_callback(self, callback: Callable):
        """Remove callback function"""
        self.data_callbacks.remove(callback)

    async def _notify_callbacks(self, data: List[SensorData]):
        """Notify all registered callbacks with new data"""
        for callback in self.data_callbacks:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Error in callback {callback}: {e}")


class AppleWatchConnector(DeviceConnector):
    """Connector for Apple Watch devices"""

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.health_store = None
        self.session_token = None

    async def connect(self) -> bool:
        """Connect to Apple Watch via HealthKit"""
        try:
            # In a real implementation, this would use HealthKit framework
            # For simulation, we'll use REST API approach
            self.is_connected = True
            logger.info("Connected to Apple Watch")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Apple Watch: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Apple Watch"""
        self.is_connected = False
        logger.info("Disconnected from Apple Watch")
        return True

    async def get_sensor_data(self, sensor_types: List[str]) -> List[SensorData]:
        """Get sensor data from Apple Watch"""
        if not self.is_connected:
            return []

        data = []
        base_timestamp = datetime.utcnow()

        for sensor_type in sensor_types:
            try:
                # Simulate Apple Watch HealthKit data retrieval
                if sensor_type == "heart_rate":
                    value = 72 + (5 * (0.5 - 0.5))  # Simulate 65-85 BPM
                    unit = "BPM"
                elif sensor_type == "heart_rate_variability":
                    value = 35 + (10 * (0.5 - 0.5))  # Simulate 25-45 ms
                    unit = "ms"
                elif sensor_type == "blood_oxygen":
                    value = 96 + (2 * (0.5 - 0.5))  # Simulate 94-98%
                    unit = "%"
                else:
                    continue

                sensor_data = SensorData(
                    device_id=self.config.device_type,
                    user_id="user_001",
                    timestamp=base_timestamp,
                    sensor_type=sensor_type,
                    value=round(value, 2),
                    unit=unit,
                    metadata={"device_model": "Apple Watch Series 7"}
                )
                data.append(sensor_data)

            except Exception as e:
                logger.error(f"Error getting {sensor_type} data: {e}")

        return data


class OuraRingConnector(DeviceConnector):
    """Connector for Oura Ring devices"""

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.session = aiohttp.ClientSession()
        self.access_token = None

    async def connect(self) -> bool:
        """Connect to Oura Ring API"""
        try:
            # OAuth2 authentication flow would go here
            self.access_token = await self._authenticate()
            self.is_connected = True
            logger.info("Connected to Oura Ring")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Oura Ring: {e}")
            return False

    async def _authenticate(self) -> str:
        """Authenticate with Oura API"""
        # Simulate authentication
        return "simulated_oura_token"

    async def disconnect(self) -> bool:
        """Disconnect from Oura Ring"""
        if self.session:
            await self.session.close()
        self.is_connected = False
        logger.info("Disconnected from Oura Ring")
        return True

    async def get_sensor_data(self, sensor_types: List[str]) -> List[SensorData]:
        """Get sensor data from Oura Ring"""
        if not self.is_connected:
            return []

        data = []
        base_timestamp = datetime.utcnow()

        for sensor_type in sensor_types:
            try:
                # Simulate Oura API data retrieval
                if sensor_type == "heart_rate":
                    value = 65 + (10 * 0.5)  # Simulate resting HR
                    unit = "BPM"
                elif sensor_type == "temperature":
                    value = 36.5 + (0.5 * 0.5)  # Simulate body temperature
                    unit = "°C"
                elif sensor_type == "hrv":
                    value = 40 + (15 * 0.5)  # Simulate HRV
                    unit = "ms"
                else:
                    continue

                sensor_data = SensorData(
                    device_id=self.config.device_type,
                    user_id="user_001",
                    timestamp=base_timestamp,
                    sensor_type=sensor_type,
                    value=round(value, 2),
                    unit=unit,
                    metadata={"device_model": "Oura Ring Gen 3"}
                )
                data.append(sensor_data)

            except Exception as e:
                logger.error(f"Error getting {sensor_type} data: {e}")

        return data


class FitbitConnector(DeviceConnector):
    """Connector for Fitbit devices"""

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.client_id = config.api_key
        self.client_secret = config.api_secret
        self.access_token = None

    async def connect(self) -> bool:
        """Connect to Fitbit API"""
        try:
            self.access_token = await self._get_access_token()
            self.is_connected = True
            logger.info("Connected to Fitbit")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Fitbit: {e}")
            return False

    async def _get_access_token(self) -> str:
        """Get Fitbit API access token"""
        # OAuth2 flow would go here
        return "simulated_fitbit_token"

    async def disconnect(self) -> bool:
        """Disconnect from Fitbit"""
        self.is_connected = False
        logger.info("Disconnected from Fitbit")
        return True

    async def get_sensor_data(self, sensor_types: List[str]) -> List[SensorData]:
        """Get sensor data from Fitbit"""
        if not self.is_connected:
            return []

        data = []
        base_timestamp = datetime.utcnow()

        for sensor_type in sensor_types:
            try:
                # Simulate Fitbit API data
                if sensor_type == "heart_rate":
                    value = 70 + (8 * 0.5)  # Simulate HR
                    unit = "BPM"
                elif sensor_type == "steps":
                    value = 8500 + (2000 * 0.5)  # Simulate daily steps
                    unit = "steps"
                elif sensor_type == "sleep":
                    value = 8.5  # Simulate sleep hours
                    unit = "hours"
                else:
                    continue

                sensor_data = SensorData(
                    device_id=self.config.device_type,
                    user_id="user_001",
                    timestamp=base_timestamp,
                    sensor_type=sensor_type,
                    value=round(value, 2),
                    unit=unit,
                    metadata={"device_model": "Fitbit Versa 3"}
                )
                data.append(sensor_data)

            except Exception as e:
                logger.error(f"Error getting {sensor_type} data: {e}")

        return data


class WebSocketDataConnector(DeviceConnector):
    """Generic WebSocket connector for real-time data streaming"""

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.websocket = None
        self.streaming_task = None

    async def connect(self) -> bool:
        """Connect to WebSocket endpoint"""
        try:
            self.websocket = await websockets.connect(self.config.base_url)
            self.is_connected = True
            self.streaming_task = asyncio.create_task(self._stream_data())
            logger.info(f"Connected to WebSocket: {self.config.base_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from WebSocket"""
        if self.streaming_task:
            self.streaming_task.cancel()
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info("Disconnected from WebSocket")
        return True

    async def _stream_data(self):
        """Stream data from WebSocket"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                sensor_data = self._parse_websocket_message(data)
                if sensor_data:
                    await self._notify_callbacks(sensor_data)
        except Exception as e:
            logger.error(f"Error in WebSocket streaming: {e}")

    def _parse_websocket_message(self, message: Dict) -> List[SensorData]:
        """Parse WebSocket message into SensorData objects"""
        data = []
        timestamp = datetime.fromisoformat(message.get('timestamp', datetime.utcnow().isoformat()))

        for sensor_reading in message.get('readings', []):
            sensor_data = SensorData(
                device_id=message.get('device_id', 'unknown'),
                user_id=message.get('user_id', 'unknown'),
                timestamp=timestamp,
                sensor_type=sensor_reading['type'],
                value=sensor_reading['value'],
                unit=sensor_reading.get('unit', ''),
                metadata=sensor_reading.get('metadata', {})
            )
            data.append(sensor_data)

        return data

    async def get_sensor_data(self, sensor_types: List[str]) -> List[SensorData]:
        """WebSocket connector doesn't support polling"""
        logger.warning("WebSocket connector doesn't support polling")
        return []


class DeviceConnectorFactory:
    """Factory for creating device connectors"""

    @staticmethod
    def create_connector(device_type: str, config: DeviceConfig) -> DeviceConnector:
        """Create appropriate connector based on device type"""
        connectors = {
            'apple_watch': AppleWatchConnector,
            'oura_ring': OuraRingConnector,
            'fitbit': FitbitConnector,
            'websocket': WebSocketDataConnector
        }

        connector_class = connectors.get(device_type.lower())
        if not connector_class:
            raise ValueError(f"Unsupported device type: {device_type}")

        return connector_class(config)


# Example usage and testing
async def demo_device_connectors():
    """Demonstrate device connector usage"""

    # Configure devices
    devices = [
        DeviceConfig(
            device_type="apple_watch",
            api_key="apple_health_kit_key",
            base_url="https://healthkit.apple.com/api"
        ),
        DeviceConfig(
            device_type="oura_ring",
            api_key="oura_client_id",
            api_secret="oura_client_secret",
            base_url="https://api.ouraring.com"
        ),
        DeviceConfig(
            device_type="fitbit",
            api_key="fitbit_client_id",
            api_secret="fitbit_client_secret",
            base_url="https://api.fitbit.com"
        )
    ]

    # Data callback function
    async def process_sensor_data(data: List[SensorData]):
        for sensor_data in data:
            print(f"Received: {sensor_data.device_id} - {sensor_data.sensor_type}: "
                  f"{sensor_data.value} {sensor_data.unit}")

    # Connect and collect data
    connectors = []
    for device_config in devices:
        connector = DeviceConnectorFactory.create_connector(
            device_config.device_type,
            device_config
        )
        connector.add_data_callback(process_sensor_data)
        connectors.append(connector)

    # Connect to devices
    for connector in connectors:
        await connector.connect()

    # Collect sensor data
    sensor_types = ["heart_rate", "heart_rate_variability", "temperature"]

    for connector in connectors:
        data = await connector.get_sensor_data(sensor_types)
        for sensor_data in data:
            print(f"Polled: {sensor_data.device_id} - {sensor_data.sensor_type}: "
                  f"{sensor_data.value} {sensor_data.unit}")

    # Disconnect
    for connector in connectors:
        await connector.disconnect()


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_device_connectors())