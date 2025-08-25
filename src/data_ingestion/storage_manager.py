"""
Storage Manager for EQM Data Ingestion

This module handles data persistence, retrieval, and management for the
EQM emotional intelligence system.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
import aiosqlite
import aioredis
from collections import defaultdict
import gzip
import shutil
from enum import Enum

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Supported storage types"""
    FILE = "file"
    SQLITE = "sqlite"
    REDIS = "redis"
    POSTGRESQL = "postgresql"


@dataclass
class StorageConfig:
    """Configuration for storage manager"""
    storage_type: StorageType = StorageType.FILE
    base_path: str = "./data"
    database_url: Optional[str] = None
    redis_url: str = "redis://localhost:6379"
    retention_days: int = 30
    compression_enabled: bool = True
    batch_size: int = 1000
    enable_caching: bool = True
    max_cache_size: int = 10000


@dataclass
class SensorData:
    """Sensor data structure for storage"""
    device_id: str
    user_id: str
    timestamp: datetime
    sensor_type: str
    value: float
    unit: str
    metadata: Dict[str, Any]


class StorageManager:
    """Unified storage manager for sensor data"""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.storage_type = config.storage_type
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize storage backend
        self._backend = None
        self._cache = {}
        self._cache_size = 0

        logger.info(f"Initialized storage manager with type: {self.storage_type}")

    async def initialize(self):
        """Initialize the storage backend"""
        if self.storage_type == StorageType.FILE:
            await self._init_file_storage()
        elif self.storage_type == StorageType.SQLITE:
            await self._init_sqlite_storage()
        elif self.storage_type == StorageType.REDIS:
            await self._init_redis_storage()
        elif self.storage_type == StorageType.POSTGRESQL:
            await self._init_postgresql_storage()

    async def _init_file_storage(self):
        """Initialize file-based storage"""
        self.raw_data_path = self.base_path / "raw"
        self.processed_data_path = self.base_path / "processed"
        self.archive_path = self.base_path / "archive"

        for path in [self.raw_data_path, self.processed_data_path, self.archive_path]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"File storage initialized at: {self.base_path}")

    async def _init_sqlite_storage(self):
        """Initialize SQLite storage"""
        db_path = self.base_path / "eqm_data.db"
        self._backend = await aiosqlite.connect(str(db_path))

        # Create tables
        await self._create_tables()
        logger.info(f"SQLite storage initialized at: {db_path}")

    async def _init_redis_storage(self):
        """Initialize Redis storage"""
        self._backend = await aioredis.create_redis_pool(self.config.redis_url)
        logger.info(f"Redis storage initialized at: {self.config.redis_url}")

    async def _init_postgresql_storage(self):
        """Initialize PostgreSQL storage"""
        # Placeholder for PostgreSQL implementation
        logger.warning("PostgreSQL storage not yet implemented, falling back to file storage")
        await self._init_file_storage()

    async def _create_tables(self):
        """Create database tables for SQLite storage"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            sensor_type TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_device_user_timestamp
        ON sensor_readings(device_id, user_id, timestamp);

        CREATE INDEX IF NOT EXISTS idx_sensor_type_timestamp
        ON sensor_readings(sensor_type, timestamp);
        """

        await self._backend.execute(create_table_sql)
        await self._backend.commit()

    async def store_sensor_data(self, data: List[SensorData]) -> bool:
        """Store sensor data"""
        try:
            if self.storage_type == StorageType.FILE:
                return await self._store_file(data)
            elif self.storage_type == StorageType.SQLITE:
                return await self._store_sqlite(data)
            elif self.storage_type == StorageType.REDIS:
                return await self._store_redis(data)
            else:
                return await self._store_file(data)  # Fallback

        except Exception as e:
            logger.error(f"Error storing sensor data: {e}")
            return False

    async def _store_file(self, data: List[SensorData]) -> bool:
        """Store data in files"""
        try:
            # Group data by date for file organization
            data_by_date = defaultdict(list)

            for reading in data:
                date_key = reading.timestamp.strftime("%Y%m%d") if isinstance(reading.timestamp, datetime) else datetime.utcnow().strftime("%Y%m%d")
                data_by_date[date_key].append(reading)

            # Write to files
            for date_key, date_data in data_by_date.items():
                filename = f"sensor_data_{date_key}.jsonl"
                if self.config.compression_enabled:
                    filename += ".gz"

                filepath = self.raw_data_path / filename

                # Convert data to JSON
                json_lines = []
                for reading in date_data:
                    json_data = {
                        'device_id': reading.device_id,
                        'user_id': reading.user_id,
                        'timestamp': reading.timestamp.isoformat() if isinstance(reading.timestamp, datetime) else reading.timestamp,
                        'sensor_type': reading.sensor_type,
                        'value': reading.value,
                        'unit': reading.unit,
                        'metadata': reading.metadata
                    }
                    json_lines.append(json.dumps(json_data))

                # Write to file
                mode = 'a' if filepath.exists() else 'w'
                if self.config.compression_enabled:
                    with gzip.open(filepath, mode + 't', encoding='utf-8') as f:
                        for line in json_lines:
                            f.write(line + '\n')
                else:
                    async with aiofiles.open(filepath, mode, encoding='utf-8') as f:
                        for line in json_lines:
                            await f.write(line + '\n')

                logger.debug(f"Stored {len(date_data)} readings to {filepath}")

            return True

        except Exception as e:
            logger.error(f"Error in file storage: {e}")
            return False

    async def _store_sqlite(self, data: List[SensorData]) -> bool:
        """Store data in SQLite database"""
        try:
            insert_sql = """
            INSERT INTO sensor_readings
            (device_id, user_id, timestamp, sensor_type, value, unit, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            # Prepare data for batch insert
            values = []
            for reading in data:
                timestamp_str = reading.timestamp.isoformat() if isinstance(reading.timestamp, datetime) else reading.timestamp
                values.append((
                    reading.device_id,
                    reading.user_id,
                    timestamp_str,
                    reading.sensor_type,
                    reading.value,
                    reading.unit,
                    json.dumps(reading.metadata)
                ))

            # Batch insert
            await self._backend.executemany(insert_sql, values)
            await self._backend.commit()

            logger.debug(f"Stored {len(data)} readings in SQLite")
            return True

        except Exception as e:
            logger.error(f"Error in SQLite storage: {e}")
            return False

    async def _store_redis(self, data: List[SensorData]) -> bool:
        """Store data in Redis"""
        try:
            # Store in Redis streams for time-series data
            for reading in data:
                key = f"sensor:{reading.device_id}:{reading.user_id}:{reading.sensor_type}"

                # Prepare data
                timestamp_str = reading.timestamp.isoformat() if isinstance(reading.timestamp, datetime) else reading.timestamp
                data_dict = {
                    'timestamp': timestamp_str,
                    'value': str(reading.value),
                    'unit': reading.unit,
                    'metadata': json.dumps(reading.metadata)
                }

                # Add to Redis stream
                await self._backend.xadd(key, data_dict)

                # Set expiration (TTL)
                await self._backend.expire(key, self.config.retention_days * 24 * 60 * 60)

            logger.debug(f"Stored {len(data)} readings in Redis")
            return True

        except Exception as e:
            logger.error(f"Error in Redis storage: {e}")
            return False

    async def retrieve_sensor_data(
        self,
        device_id: Optional[str] = None,
        user_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[SensorData]:
        """Retrieve sensor data with filtering"""
        try:
            if self.storage_type == StorageType.FILE:
                return await self._retrieve_file(device_id, user_id, sensor_type, start_time, end_time, limit)
            elif self.storage_type == StorageType.SQLITE:
                return await self._retrieve_sqlite(device_id, user_id, sensor_type, start_time, end_time, limit)
            elif self.storage_type == StorageType.REDIS:
                return await self._retrieve_redis(device_id, user_id, sensor_type, start_time, end_time, limit)
            else:
                return await self._retrieve_file(device_id, user_id, sensor_type, start_time, end_time, limit)

        except Exception as e:
            logger.error(f"Error retrieving sensor data: {e}")
            return []

    async def _retrieve_file(
        self,
        device_id: Optional[str],
        user_id: Optional[str],
        sensor_type: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int
    ) -> List[SensorData]:
        """Retrieve data from files"""
        results = []

        # Get all data files
        file_pattern = "*.jsonl*"
        data_files = list(self.raw_data_path.glob(file_pattern))

        for filepath in sorted(data_files, reverse=True):  # Most recent first
            try:
                # Open file (handle compression)
                if filepath.suffix == '.gz':
                    import gzip
                    f = gzip.open(filepath, 'rt', encoding='utf-8')
                else:
                    f = open(filepath, 'r', encoding='utf-8')

                with f:
                    for line in f:
                        try:
                            json_data = json.loads(line.strip())

                            # Apply filters
                            if device_id and json_data.get('device_id') != device_id:
                                continue
                            if user_id and json_data.get('user_id') != user_id:
                                continue
                            if sensor_type and json_data.get('sensor_type') != sensor_type:
                                continue

                            # Parse timestamp and apply time filters
                            timestamp = datetime.fromisoformat(json_data['timestamp'])
                            if start_time and timestamp < start_time:
                                continue
                            if end_time and timestamp > end_time:
                                continue

                            # Create SensorData object
                            sensor_data = SensorData(
                                device_id=json_data['device_id'],
                                user_id=json_data['user_id'],
                                timestamp=timestamp,
                                sensor_type=json_data['sensor_type'],
                                value=json_data['value'],
                                unit=json_data['unit'],
                                metadata=json_data.get('metadata', {})
                            )

                            results.append(sensor_data)

                            # Check limit
                            if len(results) >= limit:
                                break

                        except json.JSONDecodeError:
                            continue

                if len(results) >= limit:
                    break

            except Exception as e:
                logger.error(f"Error reading file {filepath}: {e}")
                continue

        return results[:limit]

    async def _retrieve_sqlite(
        self,
        device_id: Optional[str],
        user_id: Optional[str],
        sensor_type: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int
    ) -> List[SensorData]:
        """Retrieve data from SQLite"""
        try:
            # Build query
            conditions = []
            params = []

            if device_id:
                conditions.append("device_id = ?")
                params.append(device_id)
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            if sensor_type:
                conditions.append("sensor_type = ?")
                params.append(sensor_type)
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query = f"""
            SELECT device_id, user_id, timestamp, sensor_type, value, unit, metadata
            FROM sensor_readings
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
            """
            params.append(limit)

            cursor = await self._backend.execute(query, params)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                sensor_data = SensorData(
                    device_id=row[0],
                    user_id=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    sensor_type=row[3],
                    value=row[4],
                    unit=row[5],
                    metadata=json.loads(row[6]) if row[6] else {}
                )
                results.append(sensor_data)

            return results

        except Exception as e:
            logger.error(f"Error in SQLite retrieval: {e}")
            return []

    async def _retrieve_redis(
        self,
        device_id: Optional[str],
        user_id: Optional[str],
        sensor_type: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int
    ) -> List[SensorData]:
        """Retrieve data from Redis"""
        try:
            results = []

            # For Redis, we need to query specific keys
            if device_id and user_id and sensor_type:
                key = f"sensor:{device_id}:{user_id}:{sensor_type}"

                # Get stream entries
                entries = await self._backend.xrevrange(key, count=limit)

                for entry_id, data in entries:
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue

                    sensor_data = SensorData(
                        device_id=device_id,
                        user_id=user_id,
                        timestamp=timestamp,
                        sensor_type=sensor_type,
                        value=float(data['value']),
                        unit=data['unit'],
                        metadata=json.loads(data['metadata']) if data['metadata'] else {}
                    )
                    results.append(sensor_data)

            return results

        except Exception as e:
            logger.error(f"Error in Redis retrieval: {e}")
            return []

    async def cleanup_old_data(self) -> int:
        """Clean up old data beyond retention period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)

            if self.storage_type == StorageType.FILE:
                return await self._cleanup_files(cutoff_date)
            elif self.storage_type == StorageType.SQLITE:
                return await self._cleanup_sqlite(cutoff_date)
            elif self.storage_type == StorageType.REDIS:
                return await self._cleanup_redis(cutoff_date)
            else:
                return 0

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0

    async def _cleanup_files(self, cutoff_date: datetime) -> int:
        """Clean up old files"""
        total_cleaned = 0

        # Move old files to archive
        for filepath in self.raw_data_path.glob("*.jsonl*"):
            try:
                # Extract date from filename
                filename = filepath.name
                if 'sensor_data_' in filename:
                    date_str = filename.split('sensor_data_')[1].split('.')[0][:8]
                    file_date = datetime.strptime(date_str, '%Y%m%d')

                    if file_date < cutoff_date:
                        # Move to archive
                        archive_path = self.archive_path / filepath.name
                        shutil.move(str(filepath), str(archive_path))
                        total_cleaned += 1

            except Exception as e:
                logger.error(f"Error processing file {filepath}: {e}")

        return total_cleaned

    async def _cleanup_sqlite(self, cutoff_date: datetime) -> int:
        """Clean up old SQLite data"""
        try:
            delete_sql = "DELETE FROM sensor_readings WHERE timestamp < ?"
            cursor = await self._backend.execute(delete_sql, (cutoff_date.isoformat(),))
            deleted_count = cursor.rowcount
            await self._backend.commit()
            return deleted_count or 0
        except Exception as e:
            logger.error(f"Error cleaning up SQLite data: {e}")
            return 0

    async def _cleanup_redis(self, cutoff_date: datetime) -> int:
        """Clean up old Redis data"""
        # Redis data has TTL, so minimal cleanup needed
        return 0

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            if self.storage_type == StorageType.FILE:
                return await self._get_file_stats()
            elif self.storage_type == StorageType.SQLITE:
                return await self._get_sqlite_stats()
            elif self.storage_type == StorageType.REDIS:
                return await self._get_redis_stats()
            else:
                return {}

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}

    async def _get_file_stats(self) -> Dict[str, Any]:
        """Get file storage statistics"""
        total_files = 0
        total_size = 0

        for filepath in self.raw_data_path.rglob("*"):
            if filepath.is_file():
                total_files += 1
                total_size += filepath.stat().st_size

        return {
            'storage_type': 'file',
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'raw_data_path': str(self.raw_data_path),
            'archive_path': str(self.archive_path)
        }

    async def _get_sqlite_stats(self) -> Dict[str, Any]:
        """Get SQLite storage statistics"""
        try:
            cursor = await self._backend.execute("SELECT COUNT(*) FROM sensor_readings")
            count = (await cursor.fetchone())[0]

            cursor = await self._backend.execute("SELECT COUNT(DISTINCT device_id) FROM sensor_readings")
            device_count = (await cursor.fetchone())[0]

            return {
                'storage_type': 'sqlite',
                'total_readings': count,
                'unique_devices': device_count,
                'database_path': str(self.base_path / "eqm_data.db")
            }
        except Exception as e:
            logger.error(f"Error getting SQLite stats: {e}")
            return {}

    async def _get_redis_stats(self) -> Dict[str, Any]:
        """Get Redis storage statistics"""
        try:
            info = await self._backend.info()
            return {
                'storage_type': 'redis',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'total_connections_received': info.get('total_connections_received', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {}

    async def close(self):
        """Close storage connections"""
        try:
            if self._backend:
                if self.storage_type == StorageType.SQLITE:
                    await self._backend.close()
                elif self.storage_type == StorageType.REDIS:
                    self._backend.close()
                    await self._backend.wait_closed()

            logger.info("Storage manager closed")
        except Exception as e:
            logger.error(f"Error closing storage manager: {e}")


# Example usage and demonstration
async def demo_storage_manager():
    """Demonstrate storage manager functionality"""

    # Create storage configuration
    config = StorageConfig(
        storage_type=StorageType.FILE,
        base_path="./demo_storage",
        retention_days=7,
        compression_enabled=True
    )

    # Create storage manager
    storage = StorageManager(config)
    await storage.initialize()

    print("Storage Manager Demo")
    print("=" * 50)

    # Create sample data
    sample_data = []
    base_time = datetime.utcnow()

    for i in range(100):
        sensor_data = SensorData(
            device_id="apple_watch_001",
            user_id="user_001",
            timestamp=base_time + timedelta(seconds=i),
            sensor_type="heart_rate",
            value=65 + (i % 20),  # Varying heart rate
            unit="BPM",
            metadata={"confidence": 0.95, "quality": "good"}
        )
        sample_data.append(sensor_data)

    print(f"Generated {len(sample_data)} sample sensor readings")

    # Store data
    print("\nStoring data...")
    success = await storage.store_sensor_data(sample_data)
    print(f"Storage {'successful' if success else 'failed'}")

    # Retrieve data
    print("\nRetrieving data...")
    retrieved_data = await storage.retrieve_sensor_data(
        device_id="apple_watch_001",
        sensor_type="heart_rate",
        limit=10
    )

    print(f"Retrieved {len(retrieved_data)} readings:")
    for reading in retrieved_data[:5]:  # Show first 5
        print(f"  {reading.timestamp}: {reading.value} {reading.unit}")

    # Get storage statistics
    print("\nStorage Statistics:")
    stats = await storage.get_storage_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Clean up
    await storage.close()
    print("\nDemo completed successfully!")

    # Clean up demo directory
    import shutil
    shutil.rmtree("./demo_storage", ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(demo_storage_manager())