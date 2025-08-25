# EQM Machine Learning Model Architecture

## Overview
This document outlines the machine learning architecture for the EQM (Aditya369) emotional intelligence system, designed to analyze physiological data for real-time emotion detection.

## Model Architecture Overview

### Multi-Modal Emotion Classification Pipeline
```
Physiological Data → Feature Extraction → Ensemble Model → Emotion Prediction → Confidence Scoring
```

#### Model Components
1. **Feature-Level Fusion**: Combine physiological features from multiple sensors
2. **Temporal Modeling**: Capture time-series patterns in physiological signals
3. **Emotion Classification**: Multi-class classification with uncertainty estimation
4. **Model Interpretability**: Explainable AI for emotion predictions

## Model Types and Architectures

### 1. Convolutional Neural Networks (CNN)

#### 1D-CNN for Physiological Signals
```python
def create_1d_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling1D(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model
```

#### Application
- **Input**: Time-series physiological data (HR, HRV, GSR, Temperature)
- **Kernel Sizes**: 3-7 time steps for capturing physiological patterns
- **Advantages**: Captures local temporal dependencies, computationally efficient

### 2. Recurrent Neural Networks (RNN)

#### LSTM-Based Emotion Classifier
```python
def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    return model
```

#### Bidirectional LSTM for Context Awareness
```python
def create_bidirectional_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Bidirectional LSTM layers
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_out = Bidirectional(LSTM(32))(lstm_out)

    # Attention mechanism
    attention = Attention()([lstm_out, lstm_out])

    # Classification head
    dense = Dense(64, activation='relu')(attention)
    dropout = Dropout(0.5)(dense)
    outputs = Dense(num_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    return model
```

### 3. Transformer-Based Models

#### Physiological Signal Transformer
```python
class PhysiologicalTransformer(Model):
    def __init__(self, num_layers=4, d_model=128, num_heads=8, num_classes=7):
        super().__init__()

        self.embedding = Dense(d_model)
        self.pos_encoding = self.positional_encoding(1000, d_model)

        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, 256)
            for _ in range(num_layers)
        ]

        self.classification_head = Sequential([
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        # Add positional encoding to embedded inputs
        x = self.embedding(inputs)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]

        # Apply transformer encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Global average pooling and classification
        x = tf.reduce_mean(x, axis=1)
        return self.classification_head(x)
```

### 4. Ensemble and Hybrid Models

#### Multi-Modal Ensemble Architecture
```python
def create_ensemble_model(num_classes):
    # Individual model predictions
    cnn_model = create_1d_cnn_model(input_shape_cnn, num_classes)
    lstm_model = create_lstm_model(input_shape_lstm, num_classes)
    transformer_model = PhysiologicalTransformer(num_classes=num_classes)

    # Ensemble inputs
    inputs = Input(shape=input_shape)

    # Get predictions from each model
    cnn_pred = cnn_model(inputs)
    lstm_pred = lstm_model(inputs)
    transformer_pred = transformer_model(inputs)

    # Weighted ensemble
    ensemble_output = Lambda(lambda x: 0.4*x[0] + 0.3*x[1] + 0.3*x[2])(
        [cnn_pred, lstm_pred, transformer_pred]
    )

    model = Model(inputs=inputs, outputs=ensemble_output)
    return model
```

#### Feature Fusion Network
```python
def create_feature_fusion_model(feature_dims, num_classes):
    inputs = {}
    fusion_layers = []

    # Separate inputs for different feature types
    for feature_name, dim in feature_dims.items():
        inputs[feature_name] = Input(shape=(dim,), name=feature_name)
        fusion_layers.append(Dense(64, activation='relu')(inputs[feature_name]))

    # Concatenate all features
    concatenated = Concatenate()(fusion_layers)

    # Fusion layers
    fusion = Dense(256, activation='relu')(concatenated)
    fusion = Dropout(0.4)(fusion)
    fusion = Dense(128, activation='relu')(fusion)
    fusion = Dropout(0.3)(fusion)

    # Output layer
    output = Dense(num_classes, activation='softmax')(fusion)

    model = Model(inputs=inputs, outputs=output)
    return model
```

## Training Strategy

### Data Preparation

#### Train-Validation-Test Split
```python
def prepare_data_splits(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Time-based split to maintain temporal order
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data
```

#### Data Augmentation
```python
def augment_physiological_data(data, augmentation_factor=2):
    augmented_data = []

    for _ in range(augmentation_factor):
        # Add noise
        noise = np.random.normal(0, 0.1, data.shape)
        augmented_data.append(data + noise)

        # Time warping
        warped = time_warp(data)
        augmented_data.append(warped)

        # Magnitude scaling
        scale_factor = np.random.uniform(0.8, 1.2)
        augmented_data.append(data * scale_factor)

    return np.concatenate([data, augmented_data])
```

### Training Configuration

#### Loss Functions
```python
def custom_emotion_loss(y_true, y_pred):
    # Weighted cross-entropy for emotion classes
    weights = tf.constant([1.0, 1.2, 1.5, 1.0, 1.0, 1.3, 1.1])  # Different weights per emotion

    # Categorical cross-entropy with class weights
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred)

    # Apply class weights
    weight_mask = tf.reduce_sum(y_true * weights, axis=-1)
    weighted_loss = loss * weight_mask

    return tf.reduce_mean(weighted_loss)
```

#### Metrics and Evaluation
```python
def emotion_classification_metrics():
    return [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        F1Score(num_classes=7, name='f1_score')
    ]
```

### Hyperparameter Optimization

#### Bayesian Optimization
```python
def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    def objective(trial):
        # Define hyperparameter search space
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
        lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128, 256])

        # Create and train model
        model = create_lstm_model(
            input_shape=X_train.shape[1:],
            num_classes=y_train.shape[-1],
            lstm_units=lstm_units,
            dropout_rate=dropout_rate
        )

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=emotion_classification_metrics()
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=50,
            callbacks=[EarlyStopping(patience=10)],
            verbose=0
        )

        return max(history.history['val_accuracy'])

    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    return study.best_params
```

## Model Evaluation and Validation

### Cross-Validation Strategy
```python
def time_series_cross_validation(data, n_splits=5):
    # Time series-aware cross-validation
    fold_size = len(data) // n_splits
    scores = []

    for i in range(n_splits):
        start_val = i * fold_size
        end_val = (i + 1) * fold_size

        val_data = data[start_val:end_val]
        train_data = np.concatenate([data[:start_val], data[end_val:]])

        # Train and evaluate
        score = train_and_evaluate(train_data, val_data)
        scores.append(score)

    return np.mean(scores), np.std(scores)
```

### Emotion-Specific Metrics
```python
def calculate_emotion_specific_metrics(y_true, y_pred, emotion_names):
    metrics = {}

    for i, emotion in enumerate(emotion_names):
        # Binary metrics for each emotion
        y_true_binary = y_true[:, i]
        y_pred_binary = (y_pred[:, i] > 0.5).astype(int)

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        metrics[emotion] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    return metrics
```

### Confidence and Uncertainty Estimation

#### Monte Carlo Dropout
```python
def predict_with_uncertainty(model, X, n_samples=50):
    predictions = []

    for _ in range(n_samples):
        pred = model(X, training=True)  # Enable dropout during inference
        predictions.append(pred.numpy())

    predictions = np.array(predictions)

    # Calculate mean prediction and uncertainty
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)

    return mean_pred, uncertainty
```

#### Ensemble Uncertainty
```python
def ensemble_uncertainty(predictions_list):
    # Calculate disagreement among ensemble members
    predictions_array = np.array(predictions_list)
    mean_pred = np.mean(predictions_array, axis=0)
    variance = np.var(predictions_array, axis=0)

    # Entropy-based uncertainty
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)

    return mean_pred, variance, entropy
```

## Deployment and Inference

### Real-Time Inference Pipeline
```python
class RealTimeEmotionClassifier:
    def __init__(self, model_path, feature_processor):
        self.model = tf.keras.models.load_model(model_path)
        self.feature_processor = feature_processor
        self.prediction_buffer = deque(maxlen=30)  # 30-second buffer

    def predict_emotion(self, sensor_data):
        # Extract features from recent sensor data
        features = self.feature_processor.process(sensor_data)

        if features is not None:
            # Make prediction
            prediction = self.model.predict(np.expand_dims(features, axis=0))[0]

            # Add to prediction buffer for smoothing
            self.prediction_buffer.append(prediction)

            # Smooth predictions using moving average
            smoothed_prediction = np.mean(list(self.prediction_buffer), axis=0)

            # Get emotion with highest probability
            emotion_idx = np.argmax(smoothed_prediction)
            confidence = smoothed_prediction[emotion_idx]

            return {
                'emotion': emotion_idx,
                'confidence': confidence,
                'probabilities': smoothed_prediction
            }

        return None
```

### Model Serving Architecture
```python
# FastAPI-based model serving
@app.post("/predict_emotion")
async def predict_emotion(request: SensorDataRequest):
    # Validate and preprocess request data
    sensor_data = validate_sensor_data(request.data)

    # Make prediction
    prediction = emotion_classifier.predict_emotion(sensor_data)

    if prediction:
        return {
            "emotion": emotion_labels[prediction['emotion']],
            "confidence": float(prediction['confidence']),
            "timestamp": datetime.utcnow().isoformat()
        }

    raise HTTPException(status_code=400, detail="Insufficient data for prediction")
```

## Model Interpretability and Explainability

### Feature Importance Analysis
```python
def analyze_feature_importance(model, X_test, feature_names):
    # Permutation importance
    baseline_accuracy = model.evaluate(X_test, y_test)[1]

    importance_scores = {}
    for i, feature_name in enumerate(feature_names):
        # Permute feature values
        X_permuted = X_test.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

        # Calculate drop in performance
        permuted_accuracy = model.evaluate(X_permuted, y_test)[1]
        importance_scores[feature_name] = baseline_accuracy - permuted_accuracy

    return sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
```

### Attention Visualization
```python
def visualize_attention_weights(model, input_sequence, emotion_labels):
    # Extract attention weights from transformer model
    attention_layer = model.get_layer('attention')
    attention_weights = attention_layer.get_weights()[0]

    # Visualize attention patterns
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=emotion_labels,
        yticklabels=[f'Time_{i}' for i in range(len(input_sequence))],
        cmap='viridis'
    )
    plt.title('Attention Weights: Time Steps vs Emotions')
    plt.show()
```

## Performance Optimization

### Model Quantization
```python
def quantize_model_for_edge(model, optimization_level='DEFAULT'):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if optimization_level == 'DEFAULT':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif optimization_level == 'FLOAT16':
        converter.target_spec.supported_types = [tf.float16]

    quantized_model = converter.convert()
    return quantized_model
```

### Model Compression Techniques
```python
def apply_model_pruning(model, pruning_schedule):
    # Apply pruning to reduce model size
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        pruning_schedule=pruning_schedule
    )

    return pruned_model
```

This comprehensive ML architecture provides a robust foundation for real-time emotion detection from physiological data, with considerations for accuracy, interpretability, and deployment efficiency.