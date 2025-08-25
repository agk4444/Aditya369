"""
EQM (Aditya369) - Emotion Inference Module

This module provides real-time emotion detection and inference
capabilities for the Emotional Quotient Model system.

Copyright (c) 2023 AGK FIRE INC. All rights reserved.
"""

__version__ = "1.0.0"
__author__ = "AGK FIRE INC"

from .emotion_predictor import EmotionPredictor
from .real_time_processor import RealTimeEmotionProcessor
from .model_manager import ModelManager
from .inference_pipeline import InferencePipeline

__all__ = [
    'EmotionPredictor',
    'RealTimeEmotionProcessor',
    'ModelManager',
    'InferencePipeline'
]