"""
EQM (Aditya369) - Data Ingestion Module

This module handles real-time data collection from various smart devices
and wearables for the Emotional Quotient Model system.
"""

__version__ = "1.0.0"
__author__ = "EQM Development Team"

from .device_connectors import DeviceConnector
from .data_validator import DataValidator
from .ingestion_pipeline import IngestionPipeline
from .storage_manager import StorageManager

__all__ = [
    'DeviceConnector',
    'DataValidator',
    'IngestionPipeline',
    'StorageManager'
]