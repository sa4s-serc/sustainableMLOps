# general_predict.py
from joblib import load
import numpy as np
import pyRAPL
import pandas as pd
pyRAPL.setup()

# Load the general regression model
try:
    regressor = load('/home/shrikara/PycharmProjects/SusMLOps/Models/regressor_model.pkl')
    print("General model loaded successfully.")
except Exception as e:
    print(f"Error loading general model: {e}")

# Prediction function for the general model
def predict_aqi_general(input_sequence):
    try:

        global regressor
        flag = pd.read_csv('/home/shrikara/PycharmProjects/SusMLOps/csv_files/reload_flag_general.csv', header=None)
        reload_flag = flag.iloc[0, 0]
        if (reload_flag == 'reload'):
            regressor = load('/home/shrikara/PycharmProjects/SusMLOps/Models/regressor_model.pkl')
            print("General model reloaded")
            pd.DataFrame(['none']).to_csv('/home/shrikara/PycharmProjects/SusMLOps/csv_files/reload_flag_general.csv',
                                          index=False, header=False)


        # Assuming input_sequence is already in the correct shape: [1, window_size * number_of_features]
        # Flatten the input sequence if it's not already flat (e.g., if it's [1, window_size, number_of_features])
        if input_sequence.ndim > 2:
            input_data = input_sequence.reshape(1, -1)
        else:
            input_data = input_sequence

        meter= pyRAPL.Measurement('predictions')
        meter.begin()
        # Predict using the loaded general model
        predicted_aqi = regressor.predict(input_data)
        meter.end()
        pkg_energy = meter.result.pkg
        dram_energy = meter.result.dram

        # Sum the values in the lists if they are not None, otherwise use 0
        total_pkg_energy = sum(pkg_energy) if pkg_energy is not None else 0
        total_dram_energy = sum(dram_energy) if dram_energy is not None else 0

        # Total energy consumption
        cpu_consumption = total_pkg_energy + total_dram_energy

        return predicted_aqi[0], cpu_consumption
    except Exception as e:
        print(f"General model prediction error: {e}")
        return None
