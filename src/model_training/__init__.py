"""
EQM (Aditya369) - Model Training Module

This module provides comprehensive machine learning model training
functionality for emotion detection from physiological data.
"""

__version__ = "1.0.0"
__author__ = "EQM Development Team"

from .model_builder import ModelBuilder
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .hyperparameter_tuner import HyperparameterTuner

__all__ = [
    'ModelBuilder',
    'ModelTrainer',
    'ModelEvaluator',
    'HyperparameterTuner'
]