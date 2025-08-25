"""
Data Validator for EQM Data Ingestion

This module provides validation and quality assurance for sensor data
collected from various devices before processing and storage.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Any = None
    suggestion: Optional[str] = None


@dataclass
class SensorData:
    """Simplified sensor data structure for validation"""
    device_id: str
    user_id: str
    timestamp: datetime
    sensor_type: str
    value: float
    unit: str
    metadata: Dict[str, Any]


class DataValidator:
    """Comprehensive data validator for sensor readings"""

    def __init__(self):
        # Define validation rules for each sensor type
        self.validation_rules = {
            'heart_rate': {
                'range': (30, 220),  # BPM
                'unit': 'BPM',
                'max_change_rate': 50,  # BPM per minute
                'required_fields': ['value', 'timestamp', 'device_id']
            },
            'heart_rate_variability': {
                'range': (0, 200),  # ms
                'unit': 'ms',
                'max_change_rate': 100,  # ms per minute
                'required_fields': ['value', 'timestamp', 'device_id']
            },
            'temperature': {
                'range': (20, 45),  # Celsius
                'unit': '°C',
                'max_change_rate': 2,  # °C per minute
                'required_fields': ['value', 'timestamp', 'device_id']
            },
            'galvanic_skin_response': {
                'range': (0.5, 20),  # microsiemens
                'unit': 'μS',
                'max_change_rate': 5,  # μS per minute
                'required_fields': ['value', 'timestamp', 'device_id']
            },
            'blood_oxygen': {
                'range': (70, 100),  # percentage
                'unit': '%',
                'max_change_rate': 10,  # % per minute
                'required_fields': ['value', 'timestamp', 'device_id']
            },
            'accelerometer_x': {
                'range': (-78, 78),  # m/s²
                'unit': 'm/s²',
                'max_change_rate': 100,
                'required_fields': ['value', 'timestamp', 'device_id']
            },
            'accelerometer_y': {
                'range': (-78, 78),
                'unit': 'm/s²',
                'max_change_rate': 100,
                'required_fields': ['value', 'timestamp', 'device_id']
            },
            'accelerometer_z': {
                'range': (-78, 78),
                'unit': 'm/s²',
                'max_change_rate': 100,
                'required_fields': ['value', 'timestamp', 'device_id']
            }
        }

        # Track recent readings for temporal validation
        self.recent_readings = defaultdict(list)
        self.max_history_size = 100

    def validate_sensor_data(self, data: SensorData) -> List[ValidationResult]:
        """
        Validate a single sensor data reading

        Args:
            data: SensorData object to validate

        Returns:
            List of validation results
        """
        results = []

        # Basic structure validation
        results.extend(self._validate_data_structure(data))

        # Sensor-specific validation
        if data.sensor_type in self.validation_rules:
            results.extend(self._validate_sensor_specific(data))
        else:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Unknown sensor type: {data.sensor_type}",
                field="sensor_type",
                value=data.sensor_type,
                suggestion="Add validation rules for this sensor type"
            ))

        # Temporal consistency validation
        results.extend(self._validate_temporal_consistency(data))

        # Store reading for future temporal validation
        self._store_reading(data)

        return results

    def _validate_data_structure(self, data: SensorData) -> List[ValidationResult]:
        """Validate basic data structure and required fields"""
        results = []

        # Check required fields
        required_fields = ['device_id', 'user_id', 'timestamp', 'sensor_type', 'value']
        for field in required_fields:
            if not hasattr(data, field) or getattr(data, field) is None:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.CRITICAL,
                    message=f"Missing required field: {field}",
                    field=field,
                    suggestion="Ensure all required fields are provided"
                ))

        # Validate timestamp
        if hasattr(data, 'timestamp') and data.timestamp:
            if isinstance(data.timestamp, str):
                try:
                    datetime.fromisoformat(data.timestamp.replace('Z', '+00:00'))
                except ValueError:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message="Invalid timestamp format",
                        field="timestamp",
                        value=data.timestamp,
                        suggestion="Use ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)"
                    ))
            elif not isinstance(data.timestamp, datetime):
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Timestamp must be datetime object or ISO string",
                    field="timestamp",
                    value=data.timestamp,
                    suggestion="Convert to datetime object or ISO 8601 string"
                ))

        # Validate numeric fields
        if hasattr(data, 'value') and data.value is not None:
            if not isinstance(data.value, (int, float)):
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Value must be numeric",
                    field="value",
                    value=data.value,
                    suggestion="Convert value to numeric type"
                ))
            elif np.isnan(data.value) or np.isinf(data.value):
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Value must be finite",
                    field="value",
                    value=data.value,
                    suggestion="Check for division by zero or invalid calculations"
                ))

        return results

    def _validate_sensor_specific(self, data: SensorData) -> List[ValidationResult]:
        """Validate sensor-specific rules"""
        results = []
        rules = self.validation_rules[data.sensor_type]

        # Range validation
        if not (rules['range'][0] <= data.value <= rules['range'][1]):
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Value {data.value} outside expected range {rules['range']}",
                field="value",
                value=data.value,
                suggestion=f"Expected range: {rules['range'][0]} - {rules['range'][1]} {rules['unit']}"
            ))

        # Unit validation
        if hasattr(data, 'unit') and data.unit != rules['unit']:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Unit mismatch: expected {rules['unit']}, got {data.unit}",
                field="unit",
                value=data.unit,
                suggestion=f"Use correct unit: {rules['unit']}"
            ))

        return results

    def _validate_temporal_consistency(self, data: SensorData) -> List[ValidationResult]:
        """Validate temporal consistency with recent readings"""
        results = []
        key = f"{data.device_id}_{data.user_id}_{data.sensor_type}"

        if key not in self.recent_readings or len(self.recent_readings[key]) < 2:
            return results

        recent_data = self.recent_readings[key][-5:]  # Last 5 readings
        current_timestamp = data.timestamp if isinstance(data.timestamp, datetime) else datetime.fromisoformat(data.timestamp.replace('Z', '+00:00'))

        # Check for duplicate timestamps
        for reading in recent_data:
            reading_timestamp = reading.timestamp if isinstance(reading.timestamp, datetime) else datetime.fromisoformat(reading.timestamp.replace('Z', '+00:00'))
            if abs((current_timestamp - reading_timestamp).total_seconds()) < 1:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message="Duplicate timestamp detected",
                    field="timestamp",
                    value=data.timestamp,
                    suggestion="Ensure unique timestamps for each reading"
                ))
                break

        # Check for unrealistic change rates
        if len(recent_data) >= 1:
            latest_reading = recent_data[-1]
            latest_timestamp = latest_reading.timestamp if isinstance(latest_reading.timestamp, datetime) else datetime.fromisoformat(latest_reading.timestamp.replace('Z', '+00:00'))

            time_diff_minutes = (current_timestamp - latest_timestamp).total_seconds() / 60
            if time_diff_minutes > 0:
                value_change = abs(data.value - latest_reading.value)
                change_rate = value_change / time_diff_minutes

                rules = self.validation_rules.get(data.sensor_type, {})
                max_change_rate = rules.get('max_change_rate', float('inf'))

                if change_rate > max_change_rate:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"Rapid change detected: {change_rate:.2f} per minute",
                        field="value",
                        value=data.value,
                        suggestion=f"Change rate exceeds {max_change_rate} per minute"
                    ))

        return results

    def _store_reading(self, data: SensorData):
        """Store reading for future temporal validation"""
        key = f"{data.device_id}_{data.user_id}_{data.sensor_type}"
        self.recent_readings[key].append(data)

        # Maintain history size limit
        if len(self.recent_readings[key]) > self.max_history_size:
            self.recent_readings[key] = self.recent_readings[key][-self.max_history_size:]

    def validate_batch(self, data_batch: List[SensorData]) -> Dict[str, List[ValidationResult]]:
        """
        Validate a batch of sensor data

        Args:
            data_batch: List of SensorData objects

        Returns:
            Dictionary mapping data index to validation results
        """
        results = {}

        for i, data in enumerate(data_batch):
            batch_results = self.validate_sensor_data(data)
            if batch_results:
                results[str(i)] = batch_results

        return results

    def get_validation_summary(self, validation_results: Dict[str, List[ValidationResult]]) -> Dict[str, Any]:
        """
        Generate a summary of validation results

        Args:
            validation_results: Results from validate_batch

        Returns:
            Summary statistics
        """
        total_readings = len(validation_results)
        errors_by_level = defaultdict(int)
        errors_by_field = defaultdict(int)
        errors_by_sensor = defaultdict(int)

        for reading_results in validation_results.values():
            for result in reading_results:
                if not result.is_valid:
                    errors_by_level[result.level.value] += 1
                    if result.field:
                        errors_by_field[result.field] += 1

                    # Try to extract sensor type from the data (this is a simplification)
                    errors_by_sensor['unknown'] += 1

        return {
            'total_readings_with_errors': total_readings,
            'errors_by_level': dict(errors_by_level),
            'errors_by_field': dict(errors_by_field),
            'errors_by_sensor': dict(errors_by_sensor),
            'overall_quality_score': max(0, 100 - (total_readings * 2))  # Simple scoring
        }


class DataQualityMonitor:
    """Monitor data quality over time"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.quality_history = []
        self.validator = DataValidator()

    def monitor_batch(self, data_batch: List[SensorData]) -> Dict[str, Any]:
        """Monitor quality of a batch of data"""
        validation_results = self.validator.validate_batch(data_batch)
        summary = self.validator.get_validation_summary(validation_results)

        # Store in history
        self.quality_history.append(summary)
        if len(self.quality_history) > self.window_size:
            self.quality_history = self.quality_history[-self.window_size:]

        return summary

    def get_quality_trends(self) -> Dict[str, List[float]]:
        """Get quality trends over time"""
        trends = defaultdict(list)

        for summary in self.quality_history:
            trends['quality_score'].append(summary['overall_quality_score'])
            trends['error_count'].append(sum(summary['errors_by_level'].values()))

        return dict(trends)

    def detect_quality_issues(self) -> List[str]:
        """Detect potential quality issues"""
        issues = []

        if len(self.quality_history) < 10:
            return issues

        recent_scores = [s['overall_quality_score'] for s in self.quality_history[-10:]]
        avg_score = np.mean(recent_scores)

        if avg_score < 70:
            issues.append(f"Low quality score: {avg_score:.1f}")
        if np.std(recent_scores) > 20:
            issues.append("High quality score variance detected")

        return issues


# Example usage and testing
def demo_data_validation():
    """Demonstrate data validation functionality"""

    # Create validator
    validator = DataValidator()

    # Create test data
    test_data = [
        SensorData(
            device_id="apple_watch_001",
            user_id="user_001",
            timestamp=datetime.utcnow(),
            sensor_type="heart_rate",
            value=72.5,
            unit="BPM",
            metadata={"confidence": 0.95}
        ),
        SensorData(
            device_id="apple_watch_001",
            user_id="user_001",
            timestamp=datetime.utcnow(),
            sensor_type="heart_rate",
            value=350,  # Out of range
            unit="BPM",
            metadata={"confidence": 0.95}
        ),
        SensorData(
            device_id="oura_ring_001",
            user_id="user_001",
            timestamp=datetime.utcnow(),
            sensor_type="temperature",
            value=36.7,
            unit="°C",
            metadata={"confidence": 0.88}
        ),
        SensorData(
            device_id="fitbit_001",
            user_id="user_001",
            timestamp=datetime.utcnow(),
            sensor_type="unknown_sensor",
            value=100,
            unit="unknown",
            metadata={}
        )
    ]

    # Validate each reading
    print("Individual Validation Results:")
    print("-" * 50)

    for i, data in enumerate(test_data):
        results = validator.validate_sensor_data(data)
        print(f"\nReading {i+1}: {data.sensor_type} = {data.value}")
        for result in results:
            print(f"  [{result.level.value.upper()}] {result.message}")
            if result.suggestion:
                print(f"    Suggestion: {result.suggestion}")

    # Batch validation
    print("\nBatch Validation Results:")
    print("-" * 50)
    batch_results = validator.validate_batch(test_data)
    summary = validator.get_validation_summary(batch_results)

    print("Summary:")
    for key, value in summary.items():
        if key != 'errors_by_level' and key != 'errors_by_field':
            print(f"  {key}: {value}")

    print("\nErrors by Level:")
    for level, count in summary['errors_by_level'].items():
        print(f"  {level}: {count}")

    # Quality monitoring
    print("\nQuality Monitoring:")
    print("-" * 50)
    monitor = DataQualityMonitor()

    # Simulate multiple batches
    for batch_num in range(3):
        batch = test_data * 2  # Duplicate for testing
        summary = monitor.monitor_batch(batch)
        print(f"Batch {batch_num + 1} quality score: {summary['overall_quality_score']}")

    trends = monitor.get_quality_trends()
    print(f"\nQuality trends (last {len(trends['quality_score'])} batches):")
    print(f"  Average quality score: {np.mean(trends['quality_score']):.1f}")

    issues = monitor.detect_quality_issues()
    if issues:
        print("\nDetected Issues:")
        for issue in issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    demo_data_validation()