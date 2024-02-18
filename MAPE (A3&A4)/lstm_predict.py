import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pyRAPL

pyRAPL.setup()
# Load the model immediately and keep it ready for predictions
model_path = '/home/shrikara/PycharmProjects/SusMLOps/Models/model_lstm_wo_outlier_5'
model = load_model(model_path)  # Load the model once

def predict_aqi(input_data):
    try:
        global model
        flag = pd.read_csv('/home/shrikara/PycharmProjects/SusMLOps/csv_files/reload_flag_lstm.csv', header=None)
        reload_flag = flag.iloc[0, 0]

        if(reload_flag=='reload'):
            model = load_model(model_path)
            print("LSTM model reloaded")
            pd.DataFrame(['none']).to_csv('/home/shrikara/PycharmProjects/SusMLOps/csv_files/reload_flag_lstm.csv', index=False, header=False)
        meter= pyRAPL.Measurement('predictions')
        meter.begin()
        # Use the pre-loaded `model` for prediction
        prediction = model.predict(input_data)
        meter.end()
        pkg_energy = meter.result.pkg
        dram_energy = meter.result.dram

        # Sum the values in the lists if they are not None, otherwise use 0
        total_pkg_energy = sum(pkg_energy) if pkg_energy is not None else 0
        total_dram_energy = sum(dram_energy) if dram_energy is not None else 0

        # Total energy consumption
        cpu_consumption = total_pkg_energy + total_dram_energy

        return prediction.flatten()[0], cpu_consumption  # Assuming you're predicting a single AQI value
    except Exception as e:
        print(f"Prediction error: {e}")
        return None