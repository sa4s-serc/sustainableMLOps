import pandas as pd
import numpy as np
import time
from lstm_predict import predict_aqi as predict_aqi_lstm
from general_predict import predict_aqi_general
import sys
sys.path.append('../Data_preprocessing/')
from data_cleaning import DataCleaning
import pyRAPL
import warnings
import os
import socket
import json
import threading
pyRAPL.setup()
warnings.filterwarnings('ignore')


HOST = '127.0.0.12'  # Standard loopback interface address (localhost)
PORT = 65428       # Port to listen on (non-privileged ports are > 1023)
dataset_path = 'path/to/Dataset/cleaned_filtered_data_experiment.csv' #path to your dataset on which you want o make the predictions
results_path = 'path/to/Results/prediction_results.csv' #path to the csv where you want to save the prediction results
# Global variable to store the latest input_sequence
latest_input_sequence = None
data_cleaning = DataCleaning(dataset_path)

def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path, header=None)
        print("loaded model")
    except pd.errors.EmptyDataError:
        print(f"Warning: The file {file_path} is empty. Using default model settings.")
        # Return a default configuration or handle this case as needed
        return pd.DataFrame(data=[["lstm"]])

def simulate():
    global latest_input_sequence
    # Check if the CSV file already exists
    csv_file_exists = os.path.isfile(results_path)
    data_min=data_cleaning.clean_and_filter_data()
    data_min = data_min[data_min.index.month == 12]
    # Define the results DataFrame
    results = pd.DataFrame(columns=['Timestamp', 'Actual AQI', 'Predicted AQI', 'Model', 'Time Taken', 'CPU Usage'])

    time_step = 5

    for index in range(len(data_min) - time_step):
        # Read the model type
        model = safe_read_csv('path/to/Model_reload/model.csv') #path to where you read the name of the model which you want to use to make the predictions
        #if you want to get results using linear model write general_model in model.csv and if you want to use lstm for predictions write lstm in model.csv
        model_type = model.iloc[0, 0]
        input_sequence = data_min.iloc[index:index+time_step][['Calibrated PM2.5', 'Calibrated PM10', 'Calibrated Temperature', 'Calibrated Relative Humidity']].to_numpy().reshape(1, time_step, 4)
        latest_input_sequence = input_sequence  # Update the global variable with the latest input_sequence
        actual_aqi = data_min.iloc[index+time_step]['AQI']
        start_time = time.time()

        if model_type == 'lstm':
            predicted_aqi, cpu_consumption = predict_aqi_lstm(input_sequence)
        elif model_type == 'general_model':
            predicted_aqi, cpu_consumption = predict_aqi_general(input_sequence)
        else:
            print("Invalid model type specified.")
            continue

        duration = time.time() - start_time

        # Timestamp for the current prediction is the timestamp of the row immediately after the input sequence
        current_timestamp = data_min.index[index+time_step]
        new_result = {
            'Timestamp': current_timestamp,
            'Actual AQI': actual_aqi,
            'Predicted AQI': predicted_aqi,
            'Model': model_type,
            'Time Taken': duration,
            'CPU Usage': cpu_consumption
        }

        # Append the new result to the results DataFrame
        results = results._append(new_result, ignore_index=True)
        time.sleep(0.035)
        print(f"Timestamp: {current_timestamp} | Actual AQI: {actual_aqi} | Predicted AQI: {predicted_aqi} | Time taken: {duration}s | CPU Usage: {cpu_consumption} | Model: {model_type}")
        # Append the new result as a new row to the CSV file
        pd.DataFrame([new_result]).to_csv(results_path, mode='a', index=False, header=not csv_file_exists)


    print("All predictions saved to CSV successfully.")

def handle_client(conn, addr, latest_input_sequence):
    print(f"Connected by {addr}")
    try:
        while True:
            data = conn.recv(1024)  # Buffer size is 1024 bytes
            if not data:
                break  # Connection closed by the client
            request_message = data.decode('utf-8')
            if request_message == "Requesting input sequence" and latest_input_sequence is not None:
                input_sequence_json = json.dumps(latest_input_sequence.tolist())
                conn.sendall(input_sequence_json.encode('utf-8'))
            else:
                conn.sendall(b"No input sequence available")
    finally:
        conn.close()


def start_server():
    global latest_input_sequence
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print("Server started, waiting for connection...")
        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr, latest_input_sequence))
            client_thread.start()


if __name__ == '__main__':
    # Run the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    # Run the simulation in the main thread
    simulate()

    server_thread.join()