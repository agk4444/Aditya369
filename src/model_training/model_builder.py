"""
Model Builder for EQM Emotion Detection

This module provides various neural network architectures for emotion
detection from physiological sensor data.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout,
    BatchNormalization, Bidirectional, Attention, GlobalAveragePooling1D,
    Input, Concatenate, Reshape, Permute, Multiply
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model building"""
    model_type: str = "cnn"  # cnn, lstm, transformer, ensemble
    input_shape: Tuple[int, ...] = (100, 5)  # (sequence_length, features)
    num_classes: int = 7  # Number of emotion classes
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    batch_size: int = 32

    # Architecture-specific parameters
    cnn_filters: List[int] = None  # [64, 128, 256]
    lstm_units: List[int] = None  # [128, 64]
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_key_dim: int = 64


class ModelBuilder:
    """Builder for various emotion detection model architectures"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.emotion_labels = [
            'neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust'
        ]

        # Set default architecture parameters
        if config.cnn_filters is None:
            config.cnn_filters = [64, 128, 256]
        if config.lstm_units is None:
            config.lstm_units = [128, 64]

    def build_model(self) -> Model:
        """Build model based on configuration"""
        if self.config.model_type == "cnn":
            return self._build_cnn_model()
        elif self.config.model_type == "lstm":
            return self._build_lstm_model()
        elif self.config.model_type == "bidirectional_lstm":
            return self._build_bidirectional_lstm_model()
        elif self.config.model_type == "transformer":
            return self._build_transformer_model()
        elif self.config.model_type == "ensemble":
            return self._build_ensemble_model()
        elif self.config.model_type == "cnn_lstm":
            return self._build_cnn_lstm_model()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _build_cnn_model(self) -> Model:
        """Build 1D CNN model for physiological signals"""
        model = Sequential(name='Emotion_CNN')

        # First convolutional block
        model.add(Conv1D(
            self.config.cnn_filters[0], kernel_size=3,
            activation='relu', input_shape=self.config.input_shape,
            padding='same'
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        # Additional convolutional blocks
        for filters in self.config.cnn_filters[1:]:
            model.add(Conv1D(filters, kernel_size=3, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))

        # Classification head
        model.add(GlobalAveragePooling1D())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.config.dropout_rate))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.config.dropout_rate))
        model.add(Dense(self.config.num_classes, activation='softmax'))

        return model

    def _build_lstm_model(self) -> Model:
        """Build LSTM model for sequential physiological data"""
        model = Sequential(name='Emotion_LSTM')

        # LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            model.add(LSTM(
                units,
                return_sequences=return_sequences,
                input_shape=self.config.input_shape if i == 0 else None
            ))
            model.add(Dropout(self.config.dropout_rate))

        # Classification head
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.config.dropout_rate))
        model.add(Dense(self.config.num_classes, activation='softmax'))

        return model

    def _build_bidirectional_lstm_model(self) -> Model:
        """Build Bidirectional LSTM model with attention"""
        inputs = Input(shape=self.config.input_shape, name='input')

        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(self.config.dropout_rate)(x)

        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Dropout(self.config.dropout_rate)(x)

        # Attention mechanism
        attention = Attention()([x, x])
        x = Multiply()([x, attention])

        # Global pooling
        x = GlobalAveragePooling1D()(x)

        # Classification head
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.config.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Emotion_BiLSTM_Attention')
        return model

    def _build_transformer_model(self) -> Model:
        """Build Transformer-based model for physiological signals"""
        inputs = Input(shape=self.config.input_shape, name='input')

        # Embedding layer
        x = Dense(128, activation='relu')(inputs)

        # Positional encoding
        x = self._add_positional_encoding(x)

        # Transformer encoder blocks
        for i in range(self.config.transformer_layers):
            x = self._transformer_encoder_block(
                x, self.config.transformer_heads, self.config.transformer_key_dim
            )

        # Global pooling
        x = GlobalAveragePooling1D()(x)

        # Classification head
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.config.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Emotion_Transformer')
        return model

    def _transformer_encoder_block(self, inputs, num_heads, key_dim):
        """Create a single transformer encoder block"""
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )(inputs, inputs)

        # Add & normalize
        x = tf.keras.layers.Add()([inputs, attention_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward network
        ffn = tf.keras.Sequential([
            Dense(key_dim * 4, activation='relu'),
            Dense(key_dim)
        ])

        ffn_output = ffn(x)

        # Add & normalize
        x = tf.keras.layers.Add()([x, ffn_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        return x

    def _add_positional_encoding(self, inputs):
        """Add positional encoding to inputs"""
        seq_length = self.config.input_shape[0]
        d_model = inputs.shape[-1]

        # Create positional encoding matrix
        pos_encoding = np.zeros((seq_length, d_model))
        positions = np.arange(seq_length)[:, np.newaxis]
        angles = positions / np.power(10000, (2 * (np.arange(d_model) // 2)) / d_model)

        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])

        pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)

        # Add positional encoding to inputs
        return inputs + pos_encoding

    def _build_cnn_lstm_model(self) -> Model:
        """Build CNN-LSTM hybrid model"""
        inputs = Input(shape=self.config.input_shape, name='input')

        # CNN feature extraction
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        # LSTM sequence processing
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(self.config.dropout_rate)(x)

        x = LSTM(32)(x)
        x = Dropout(self.config.dropout_rate)(x)

        # Classification head
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.config.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Emotion_CNN_LSTM')
        return model

    def _build_ensemble_model(self) -> Model:
        """Build ensemble model combining multiple architectures"""
        inputs = Input(shape=self.config.input_shape, name='input')

        # CNN branch
        cnn_branch = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = GlobalAveragePooling1D()(cnn_branch)

        # LSTM branch
        lstm_branch = LSTM(64, return_sequences=False)(inputs)
        lstm_branch = Dropout(self.config.dropout_rate)(lstm_branch)

        # Combine branches
        combined = Concatenate()([cnn_branch, lstm_branch])

        # Classification head
        x = Dense(128, activation='relu')(combined)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.config.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Emotion_Ensemble')
        return model

    def compile_model(self, model: Model, optimizer_type: str = 'adam') -> Model:
        """Compile model with appropriate optimizer and loss function"""

        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=self.config.learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=self.config.learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = RMSprop(learning_rate=self.config.learning_rate)
        else:
            optimizer = Adam(learning_rate=self.config.learning_rate)

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        return model

    def get_model_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        """Get training callbacks for the model"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        return callbacks

    def create_emotion_data_generator(self, X, y, batch_size: int = 32):
        """Create data generator for emotion classification"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(X))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_model_summary(self, model: Model) -> str:
        """Get model architecture summary"""
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)


# Predefined model configurations
MODEL_CONFIGS = {
    'cnn_simple': ModelConfig(
        model_type='cnn',
        cnn_filters=[64, 128],
        dropout_rate=0.3,
        learning_rate=0.001
    ),
    'cnn_complex': ModelConfig(
        model_type='cnn',
        cnn_filters=[64, 128, 256, 512],
        dropout_rate=0.4,
        learning_rate=0.0005
    ),
    'lstm_simple': ModelConfig(
        model_type='lstm',
        lstm_units=[128, 64],
        dropout_rate=0.3,
        learning_rate=0.001
    ),
    'lstm_bidirectional': ModelConfig(
        model_type='bidirectional_lstm',
        dropout_rate=0.3,
        learning_rate=0.001
    ),
    'transformer_base': ModelConfig(
        model_type='transformer',
        transformer_layers=4,
        transformer_heads=8,
        dropout_rate=0.3,
        learning_rate=0.0005
    ),
    'ensemble': ModelConfig(
        model_type='ensemble',
        dropout_rate=0.4,
        learning_rate=0.001
    ),
    'cnn_lstm_hybrid': ModelConfig(
        model_type='cnn_lstm',
        dropout_rate=0.3,
        learning_rate=0.001
    )
}


# Example usage and demonstration
def demo_model_building():
    """Demonstrate model building functionality"""

    print("Model Building Demo")
    print("=" * 50)

    # Define input shape (sequence_length, features)
    input_shape = (300, 5)  # 300 time steps, 5 physiological features
    num_classes = 7  # 7 emotion classes

    # Create different model configurations
    configs = {
        'CNN': ModelConfig(model_type='cnn', input_shape=input_shape, num_classes=num_classes),
        'LSTM': ModelConfig(model_type='lstm', input_shape=input_shape, num_classes=num_classes),
        'BiLSTM': ModelConfig(model_type='bidirectional_lstm', input_shape=input_shape, num_classes=num_classes),
        'Transformer': ModelConfig(model_type='transformer', input_shape=input_shape, num_classes=num_classes),
        'Ensemble': ModelConfig(model_type='ensemble', input_shape=input_shape, num_classes=num_classes)
    }

    # Build and summarize each model
    for model_name, config in configs.items():
        print(f"\n{model_name} Model:")
        print("-" * 30)

        try:
            builder = ModelBuilder(config)
            model = builder.build_model()
            compiled_model = builder.compile_model(model)

            print(f"Model type: {config.model_type}")
            print(f"Input shape: {config.input_shape}")
            print(f"Number of classes: {config.num_classes}")

            # Count parameters
            total_params = compiled_model.count_params()
            print(f"Total parameters: {total_params:,}")

            # Show model layers
            print("
Model architecture:")
            for i, layer in enumerate(compiled_model.layers[:5]):  # Show first 5 layers
                print(f"  {i+1}. {layer.name} ({layer.__class__.__name__})")

            if len(compiled_model.layers) > 5:
                print(f"  ... and {len(compiled_model.layers) - 5} more layers")

        except Exception as e:
            print(f"Error building {model_name} model: {e}")

    print("
Demo completed successfully!")
    print("\nThese models are ready for training with physiological emotion data.")


if __name__ == "__main__":
    demo_model_building()