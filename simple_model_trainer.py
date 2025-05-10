import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, Bidirectional
from tensorflow.keras.layers import Input, Add, TimeDistributed, SpatialDropout1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable

# Custom Attention Layer
@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                    shape=(input_shape[-1], 1),
                                    initializer='uniform',
                                    trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        e = K.tanh(K.dot(x, self.kernel))
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

# Load data
def load_data(data_path='/kaggle/input/tsl-data2/tsl_data.pkl'):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_data()
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Calculate class weights for imbalance
y_integers = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# Enhanced normalization function with robust scaling
def normalize_data(X):
    X_norm = np.zeros_like(X)
    for i in range(X.shape[0]):
        seq = X[i]
        mask = seq != 0
        if np.any(mask):
            # Use robust scaling with median and IQR
            median = np.median(seq[mask])
            q1 = np.percentile(seq[mask], 25)
            q3 = np.percentile(seq[mask], 75)
            iqr = q3 - q1
            if iqr > 1e-6:
                X_norm[i, mask] = (seq[mask] - median) / (iqr + 1e-6)
            else:
                X_norm[i, mask] = seq[mask] - median
    return X_norm

# Normalize data
X_train_norm = normalize_data(X_train)
X_val_norm = normalize_data(X_val)
X_test_norm = normalize_data(X_test)

def augment_sequence(sequence, noise_factor=0.05, time_warp_factor=0.1):
    """Apply data augmentation to a sequence."""
    augmented = sequence.copy()
    
    # Add small random noise
    noise = np.random.normal(0, noise_factor, sequence.shape)
    augmented += noise
    
    # Time warping (slight speed variations)
    if np.random.random() < 0.5:
        time_steps = sequence.shape[0]
        warp_points = np.random.normal(0, time_warp_factor, time_steps)
        warp_points = np.cumsum(warp_points)
        warp_points = (warp_points - warp_points.min()) / (warp_points.max() - warp_points.min())
        warp_points = (warp_points * (time_steps - 1)).astype(int)
        augmented = augmented[warp_points]
    
    return augmented

# Enhanced model with residual connections and advanced techniques
def build_advanced_model(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial feature extraction with parallel conv layers
    conv1 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(inputs)
    conv2 = Conv1D(128, kernel_size=5, padding='same', activation='relu')(inputs)
    conv3 = Conv1D(128, kernel_size=7, padding='same', activation='relu')(inputs)
    
    # Merge parallel conv layers
    x = Add()([conv1, conv2, conv3])
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.3)(x)  # Increased dropout
    
    # First BiLSTM block with residual
    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = SpatialDropout1D(0.3)(x)  # Increased dropout
    
    # Add residual connection
    x = Add()([x, lstm1])
    
    # Second BiLSTM block with increased units
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.3)(x)  # Increased dropout
    
    # Attention layer
    attention_output = Attention()(x)
    
    # Dense layers with skip connections
    dense1 = Dense(128, activation='relu')(attention_output)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)  # Increased dropout
    
    dense2 = Dense(128, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)  # Increased dropout
    
    # Skip connection
    x = Add()([dense1, dense2])
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with adjusted learning rate and gradient clipping
    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)  # Reduced learning rate
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build model
num_classes = y_train.shape[1]
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_advanced_model(input_shape, num_classes)
model.summary()

# Data augmentation
X_train_aug = np.copy(X_train_norm)
y_train_aug = np.copy(y_train)

# Augment 50% of the training data
aug_indices = np.random.choice(len(X_train_norm), size=len(X_train_norm)//2, replace=False)
for idx in aug_indices:
    X_train_aug[idx] = augment_sequence(X_train_norm[idx])

# Combine original and augmented data
X_train_combined = np.concatenate([X_train_norm, X_train_aug])
y_train_combined = np.concatenate([y_train, y_train_aug])

# Shuffle the combined dataset
shuffle_idx = np.random.permutation(len(X_train_combined))
X_train_combined = X_train_combined[shuffle_idx]
y_train_combined = y_train_combined[shuffle_idx]

# Set up callbacks with adjusted parameters
checkpoint = ModelCheckpoint(
    'simple_tsl_model_v14.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,  # Increased patience
    min_lr=0.000001,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,  # Increased patience
    restore_best_weights=True,
    verbose=1
)

# Train model with augmented data
history = model.fit(
    X_train_combined,
    y_train_combined,
    validation_data=(X_val_norm, y_val),
    epochs=200,
    batch_size=64,
    callbacks=[checkpoint, reduce_lr, early_stopping],
    class_weight=class_weights_dict
)

# Load best model
model.load_weights('simple_tsl_model_v14.keras')

# Evaluate model
train_loss, train_acc = model.evaluate(X_train_norm, y_train)
val_loss, val_acc = model.evaluate(X_val_norm, y_val)
test_loss, test_acc = model.evaluate(X_test_norm, y_test)

print(f"Train accuracy: {train_acc*100:.2f}%")
print(f"Validation accuracy: {val_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")

# Save model
model.save('tsl_simple_model_v14.keras')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

#LATEST BANGER v14