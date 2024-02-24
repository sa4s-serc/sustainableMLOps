import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

def df_to_X_y_multivariate(df_features, df_target, window_size=5):
    df_features_np = df_features.to_numpy()
    df_target_np = df_target.to_numpy()
    X, y = [], []
    for i in range(len(df_features_np) - window_size):
        row = df_features_np[i:i+window_size]  # No need to reshape as we want to keep it multivariate
        X.append(row)
        label = df_target_np[i+window_size]  # Assuming target is to predict the next value right after the window
        y.append(label)
    return np.array(X), np.array(y)


def train_lstm_model(cleaned_data, model_save_path):
    features = ['Calibrated PM2.5', 'Calibrated PM10', 'Calibrated Temperature', 'Calibrated Relative Humidity']
    target = 'AQI'

    df_features = cleaned_data[features]
    df_target = cleaned_data[target]

    WINDOW_SIZE = 5

    X, y = df_to_X_y_multivariate(df_features, df_target, WINDOW_SIZE )

    # Assuming cleaned_data is the entire dataset you want to train on
    train_split = int(len(X) * 0.8)
    val_split = int(len(X) * 0.1)

    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:train_split + val_split], y[train_split:train_split + val_split]
    X_test, y_test = X[train_split + val_split:], y[train_split + val_split:]

    print("Training Data:", X_train.shape, y_train.shape)
    print("Validation Data:", X_val.shape, y_val.shape)
    print("Testing Data:", X_test.shape, y_test.shape)

    model = Sequential([
        InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(16),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
    model.summary()

    best_model = clone_model(model)  # Clone the model structure
    best_val_loss = float('inf')
    for epoch in range(20):  # Number of epochs
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1)  # Train for one epoch at a time

        # Check if the current epoch's validation loss is better (lower) than the best one seen so far
        current_val_loss = history.history['val_loss'][0]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model.set_weights(model.get_weights())  # Update the best model's weights

    # After all epochs, save the best model
    best_model.save('path/to/Models/model_lstm_wo_outlier_5')

    train_predictions = best_model.predict(X_train).flatten()
    # Save the DataFrame to a CSV file
    train_results.to_csv('path/to/Knowledge/train_results_LSTM.csv', index=False)

    print(f"Best model trained and saved at {model_save_path} with validation loss: {best_val_loss}")