"""
Model Trainer for EQM Emotion Detection

This module provides comprehensive training functionality for emotion
detection models using physiological sensor data.
"""

import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .model_builder import ModelBuilder, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 15
    learning_rate_patience: int = 7
    save_best_only: bool = True
    enable_data_augmentation: bool = True
    class_weights: Optional[Dict[str, float]] = None
    random_state: int = 42


@dataclass
class TrainingResult:
    """Results from model training"""
    model_name: str
    final_accuracy: float
    final_loss: float
    val_accuracy: float
    val_loss: float
    training_time: float
    epochs_trained: int
    best_epoch: int
    history: Dict[str, List[float]]
    classification_report: Dict[str, Any]
    confusion_matrix: np.ndarray
    model_path: Optional[str] = None


class EmotionDataPreprocessor:
    """Preprocessor for emotion classification data"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.emotion_mapping = {
            0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry',
            4: 'fear', 5: 'surprise', 6: 'disgust'
        }
        self.reverse_emotion_mapping = {v: k for k, v in self.emotion_mapping.items()}

    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess training data"""
        # Handle different input shapes
        if len(X.shape) == 3:  # (samples, sequence_length, features)
            # For sequence data, reshape for scaling
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(original_shape)
        else:  # (samples, features)
            X = self.scaler.fit_transform(X)

        # Encode labels if they're strings
        if y.dtype.kind in ['U', 'S']:  # Unicode or byte strings
            y = self.label_encoder.fit_transform(y)

        return X, y

    def preprocess_single_sample(self, X: np.ndarray) -> np.ndarray:
        """Preprocess a single sample for inference"""
        if len(X.shape) == 2:  # (sequence_length, features)
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
            X = X_scaled.reshape(original_shape)
        else:  # (features,)
            X = self.scaler.transform(X.reshape(1, -1)).flatten()

        return X

    def get_emotion_label(self, prediction_idx: int) -> str:
        """Convert prediction index to emotion label"""
        return self.emotion_mapping.get(prediction_idx, 'unknown')

    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)

        return dict(zip(classes, weights))


class ModelTrainer:
    """Trainer for emotion detection models"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_preprocessor = EmotionDataPreprocessor()
        self.training_results: List[TrainingResult] = []

    def train_model(
        self,
        model_config: ModelConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_name: Optional[str] = None
    ) -> Tuple[tf.keras.Model, TrainingResult]:
        """
        Train a single model

        Args:
            model_config: Configuration for the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            model_name: Name for the model (optional)

        Returns:
            Tuple of (trained_model, training_result)
        """
        start_time = datetime.utcnow()

        # Set model name
        if model_name is None:
            model_name = f"{model_config.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting training for model: {model_name}")

        # Preprocess data
        X_train_processed, y_train_processed = self.data_preprocessor.preprocess_data(X_train, y_train)

        # Prepare validation data
        if X_val is not None and y_val is not None:
            X_val_processed, y_val_processed = self.data_preprocessor.preprocess_data(X_val, y_val)
        else:
            # Split training data for validation
            split_idx = int(len(X_train_processed) * (1 - self.config.validation_split))
            X_val_processed = X_train_processed[split_idx:]
            y_val_processed = y_train_processed[split_idx:]
            X_train_processed = X_train_processed[:split_idx]
            y_train_processed = y_train_processed[:split_idx]

        # Convert labels to categorical
        num_classes = len(np.unique(y_train_processed))
        y_train_categorical = tf.keras.utils.to_categorical(y_train_processed, num_classes)
        y_val_categorical = tf.keras.utils.to_categorical(y_val_processed, num_classes)

        # Build and compile model
        builder = ModelBuilder(model_config)
        model = builder.build_model()
        model = builder.compile_model(model)

        # Get callbacks
        callbacks = builder.get_model_callbacks(model_name)

        # Calculate class weights
        class_weights = self.config.class_weights
        if class_weights is None:
            class_weights = self.data_preprocessor.get_class_weights(y_train_processed)
            class_weights = {i: weight for i, weight in enumerate(class_weights.values())}

        # Data augmentation for training
        if self.config.enable_data_augmentation:
            train_dataset = self._create_augmented_dataset(
                X_train_processed, y_train_categorical, self.config.batch_size
            )
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (X_train_processed, y_train_categorical)
            ).batch(self.config.batch_size).shuffle(1000)

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val_processed, y_val_categorical)
        ).batch(self.config.batch_size)

        # Train model
        logger.info(f"Training model with {len(X_train_processed)} samples")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Calculate training time
        training_time = (datetime.utcnow() - start_time).total_seconds()

        # Evaluate model
        val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)

        # Generate predictions for detailed metrics
        y_pred = model.predict(val_dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val_categorical, axis=1)

        # Calculate final training metrics
        final_accuracy = history.history['accuracy'][-1]
        final_loss = history.history['loss'][-1]

        # Find best epoch
        best_epoch = np.argmax(history.history['val_accuracy']) + 1

        # Generate classification report
        class_report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=[self.data_preprocessor.get_emotion_label(i) for i in range(num_classes)],
            output_dict=True
        )

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

        # Save model
        model_path = self._save_model(model, model_name)

        # Create training result
        result = TrainingResult(
            model_name=model_name,
            final_accuracy=final_accuracy,
            final_loss=final_loss,
            val_accuracy=val_accuracy,
            val_loss=val_loss,
            training_time=training_time,
            epochs_trained=len(history.history['accuracy']),
            best_epoch=best_epoch,
            history=history.history,
            classification_report=class_report,
            confusion_matrix=conf_matrix,
            model_path=model_path
        )

        self.training_results.append(result)
        logger.info(f"Training completed for {model_name}. Accuracy: {val_accuracy:.4f}")

        return model, result

    def train_multiple_models(
        self,
        model_configs: List[ModelConfig],
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> List[Tuple[tf.keras.Model, TrainingResult]]:
        """
        Train multiple models and compare their performance

        Args:
            model_configs: List of model configurations to train
            X: Feature data
            y: Label data
            test_size: Fraction of data to use for testing

        Returns:
            List of (model, result) tuples
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_state, stratify=y
        )

        results = []

        for i, config in enumerate(model_configs):
            logger.info(f"Training model {i+1}/{len(model_configs)}: {config.model_type}")

            try:
                model, result = self.train_model(
                    config, X_train, y_train, X_test, y_test,
                    model_name=f"{config.model_type}_{i}"
                )
                results.append((model, result))

            except Exception as e:
                logger.error(f"Failed to train {config.model_type}: {e}")
                continue

        return results

    def _create_augmented_dataset(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        """Create augmented dataset for training"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        # Add noise augmentation
        def add_noise(x, y):
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1)
            return x + noise, y

        # Add time warping (simple implementation)
        def time_warp(x, y):
            # Randomly shift the signal
            shift = tf.random.uniform([], minval=-10, maxval=10, dtype=tf.int32)
            return tf.roll(x, shift, axis=0), y

        # Apply augmentations
        dataset = dataset.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(time_warp, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _save_model(self, model: tf.keras.Model, model_name: str) -> str:
        """Save trained model"""
        # Create models directory
        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)

        # Save model
        model_path = models_dir / f"{model_name}.h5"
        model.save(model_path)

        # Save model configuration
        config_path = models_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'save_time': datetime.utcnow().isoformat(),
                'model_summary': model.to_json()
            }, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def plot_training_history(self, result: TrainingResult, save_path: Optional[str] = None):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        ax1.plot(result.history['accuracy'], label='Training')
        ax1.plot(result.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Loss
        ax2.plot(result.history['loss'], label='Training')
        ax2.plot(result.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # Precision and Recall
        if 'precision' in result.history:
            ax3.plot(result.history['precision'], label='Precision')
        if 'recall' in result.history:
            ax3.plot(result.history['recall'], label='Recall')
        ax3.set_title('Precision and Recall')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()

        # Confusion Matrix
        sns.heatmap(result.confusion_matrix, annot=True, fmt='d', ax=ax4, cmap='Blues')
        ax4.set_title('Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')

        plt.suptitle(f'Training Results - {result.model_name}', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training plots saved to {save_path}")

        plt.show()

    def compare_models(self, results: List[TrainingResult]) -> pd.DataFrame:
        """Compare multiple trained models"""
        comparison_data = []

        for result in results:
            comparison_data.append({
                'Model': result.model_name,
                'Final Accuracy': result.final_accuracy,
                'Validation Accuracy': result.val_accuracy,
                'Final Loss': result.final_loss,
                'Validation Loss': result.val_loss,
                'Training Time (s)': result.training_time,
                'Epochs Trained': result.epochs_trained,
                'Best Epoch': result.best_epoch,
                'F1-Score': result.classification_report['weighted avg']['f1-score']
            })

        return pd.DataFrame(comparison_data)


# Example usage and demonstration
def demo_model_training():
    """Demonstrate model training functionality"""

    print("Model Training Demo")
    print("=" * 50)

    # Create synthetic physiological data for emotions
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 300
    n_features = 5  # HR, HRV, GSR, Temperature, SpO2

    # Generate synthetic data for different emotions
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']
    emotion_data = {}
    emotion_labels = {}

    for i, emotion in enumerate(emotions):
        # Create emotion-specific patterns
        if emotion == 'neutral':
            hr_pattern = 70 + 5 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
        elif emotion == 'happy':
            hr_pattern = 75 + 8 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
        elif emotion == 'sad':
            hr_pattern = 65 + 3 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
        elif emotion == 'angry':
            hr_pattern = 85 + 12 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
        elif emotion == 'fear':
            hr_pattern = 90 + 15 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
        elif emotion == 'surprise':
            hr_pattern = 80 + 10 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)
        else:  # disgust
            hr_pattern = 68 + 4 * np.sin(2 * np.pi * np.arange(sequence_length) / 60)

        # Add noise and variation
        hr_pattern += np.random.normal(0, 3, sequence_length)

        # Create multi-feature data
        samples_per_emotion = n_samples // len(emotions)
        X_emotion = np.zeros((samples_per_emotion, sequence_length, n_features))

        for j in range(samples_per_emotion):
            # Heart rate
            X_emotion[j, :, 0] = hr_pattern + np.random.normal(0, 2, sequence_length)

            # Heart rate variability
            X_emotion[j, :, 1] = 40 + 10 * np.sin(2 * np.pi * np.arange(sequence_length) / 120)
            X_emotion[j, :, 1] += np.random.normal(0, 5, sequence_length)

            # Skin conductance
            base_gsr = 2.0 if emotion in ['happy', 'surprise'] else 1.5
            X_emotion[j, :, 2] = base_gsr + np.random.exponential(1, sequence_length)

            # Temperature
            X_emotion[j, :, 3] = 36.5 + 0.3 * np.sin(2 * np.pi * np.arange(sequence_length) / 3600)
            X_emotion[j, :, 3] += np.random.normal(0, 0.1, sequence_length)

            # Blood oxygen
            X_emotion[j, :, 4] = 98 + np.random.normal(0, 1, sequence_length)

        emotion_data[emotion] = X_emotion
        emotion_labels[emotion] = np.full(samples_per_emotion, i)

    # Combine all data
    X_data = []
    y_data = []

    for emotion in emotions:
        X_data.append(emotion_data[emotion])
        y_data.append(emotion_labels[emotion])

    X = np.concatenate(X_data, axis=0)
    y = np.concatenate(y_data, axis=0)

    print(f"Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} time steps, {X.shape[2]} features")
    print(f"Emotion distribution: {np.bincount(y)}")

    # Create training configuration
    training_config = TrainingConfig(
        epochs=50,  # Reduced for demo
        batch_size=32,
        validation_split=0.2,
        enable_data_augmentation=True,
        early_stopping_patience=10
    )

    # Create model configurations
    model_configs = [
        ModelConfig(
            model_type='cnn',
            input_shape=(sequence_length, n_features),
            num_classes=len(emotions),
            learning_rate=0.001
        ),
        ModelConfig(
            model_type='lstm',
            input_shape=(sequence_length, n_features),
            num_classes=len(emotions),
            learning_rate=0.001
        )
    ]

    # Train models
    trainer = ModelTrainer(training_config)

    print("\nTraining models...")
    results = trainer.train_multiple_models(model_configs, X, y, test_size=0.2)

    # Compare results
    if results:
        comparison_df = trainer.compare_models([result for _, result in results])
        print("
Model Comparison:")
        print(comparison_df.to_string(index=False))

        # Plot results for best model
        best_result = max(results, key=lambda x: x[1].val_accuracy)[1]
        print(f"\nBest model: {best_result.model_name}")
        print(f"  Validation Accuracy: {best_result.val_accuracy:.4f}")
        print(f"  Training Time: {best_result.training_time:.2f} seconds")

        # Show classification report
        print("
Classification Report:")
        for emotion, metrics in best_result.classification_report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"  {emotion.capitalize()}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

    print("\nDemo completed successfully!")
    print("Note: This demo uses synthetic data. Real emotion detection would require")
    print("actual physiological data collected during emotional states.")


if __name__ == "__main__":
    demo_model_training()