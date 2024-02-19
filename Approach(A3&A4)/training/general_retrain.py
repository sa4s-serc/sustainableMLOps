import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib  # For saving the model


def df_to_X_y_for_regression(df_features, df_target, window_size=5):
    df_features_np = df_features.to_numpy()
    df_target_np = df_target.to_numpy()
    X, y = [], []
    for i in range(len(df_features_np) - window_size):
        row = df_features_np[i:i + window_size].flatten()
        X.append(row)
        label = df_target_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)


def train_general_model(cleaned_data, model_save_path):
    features = ['Calibrated PM2.5', 'Calibrated PM10', 'Calibrated Temperature', 'Calibrated Relative Humidity']
    target = 'AQI'

    df_features = cleaned_data[features]
    df_target = cleaned_data[target]

    WINDOW_SIZE = 5
    X, y = df_to_X_y_for_regression(df_features, df_target, WINDOW_SIZE)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Optionally, print model performance on test set
    print("Model training completed. Test set score:", model.score(X_test, y_test))

    # Save the trained model
    joblib.dump(model, 'path/to/Models/regressor_model.pkl')
    print(f"Model saved at {model_save_path}")


