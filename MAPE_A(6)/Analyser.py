# analyser.py
import pandas as pd
from scipy.stats import entropy
import numpy as np
from Planner import Planner
import os
import socket
import json
from lstm_predict import predict_aqi as predict_aqi_lstm


class Analyser:
    def __init__(self, model_csv_path='Model_reload/model.csv'):
        self.model_csv_path = model_csv_path
        self.planner = Planner(model_csv_path=model_csv_path)  # Create an instance of Planner

    def calculate_kl_divergence(self, p, q):
        p = p / np.sum(p)
        q = q / np.sum(q)
        p += 0.0001
        q += 0.0001
        p /= np.sum(p)
        q /= np.sum(q)
        return entropy(p, q)

    def calculate_kl_divergence_for_model(self, baseline_predictions, real_time_predictions, model_type):
        """Calculate the KL divergence for a specific model type."""
        real_time_predictions_filtered = real_time_predictions[real_time_predictions['Model'] == model_type]

        bins = np.linspace(
            min(baseline_predictions['Predicted AQI'].min(), real_time_predictions_filtered['Predicted AQI'].min()),
            max(baseline_predictions['Predicted AQI'].max(), real_time_predictions_filtered['Predicted AQI'].max()),
            num=50)

        baseline_distribution, _ = np.histogram(baseline_predictions['Predicted AQI'], bins=bins, density=True)
        real_time_distribution, _ = np.histogram(real_time_predictions_filtered['Predicted AQI'], bins=bins,
                                                 density=True)

        return self.calculate_kl_divergence(baseline_distribution, real_time_distribution)

    def analyze_drift(self, baseline_predictions_lstm, baseline_predictions_general, real_time_predictions):
        # model = pd.read_csv(self.model_csv_path, header=None)
        # model_in_use = model.iloc[0, 0]

        # Define a minimum number of predictions required for meaningful KL divergence calculation
        MIN_PREDICTIONS_REQUIRED = 100

        # Load the KL divergence thresholds
        KL = pd.read_csv('/Knowledge/KL_general.csv', header=None)
        KL_general_threshold = KL.iloc[0, 0]
        KL = pd.read_csv('/Knowledge/KL_LSTM.csv', header=None)
        KL_LSTM_threshold = KL.iloc[0, 0]

        # Calculate KL divergence for General model
        if len(real_time_predictions[real_time_predictions['Model'] == 'general_model']) >= MIN_PREDICTIONS_REQUIRED:
            kl_div_general = self.calculate_kl_divergence_for_model(baseline_predictions_general, real_time_predictions,
                                                                    'general_model')
            print(f"KL divergence for general model: {kl_div_general}")
            if kl_div_general > KL_general_threshold:
                print(f"KL DIVERGENCE FOR GENERAL MODEL: {kl_div_general}, indicating drift.")
                self.planner.divergence_detected('general_model')
        else:
            print("Insufficient General model real-time predictions for meaningful KL divergence calculation.")

        # Calculate KL divergence for LSTM model
        if len(real_time_predictions[real_time_predictions['Model'] == 'lstm']) >= MIN_PREDICTIONS_REQUIRED:
            kl_div_lstm = self.calculate_kl_divergence_for_model(baseline_predictions_lstm, real_time_predictions,
                                                                 'lstm')
            print(f"KL divergence for LSTM model: {kl_div_lstm}")
            if kl_div_lstm > KL_LSTM_threshold:
                print(f"KL DIVERGENCE FOR LSTM: {kl_div_lstm}, indicating drift.")
                self.planner.divergence_detected('lstm')
        else:
            print("Insufficient LSTM real-time predictions for meaningful KL divergence calculation.")



    def get_input_sequence(self):
        HOST = '127.0.0.12'  # The server's hostname or IP address
        PORT = 65427  # The port used by the server

        request_message = "Requesting input sequence"  # Specific request message

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                s.sendall(request_message.encode('utf-8'))  # Send the specific request message
                data = s.recv(1024)
                input_sequence = json.loads(data.decode('utf-8'))  # Convert JSON string back to list
                return input_sequence
        except ConnectionRefusedError as e:
            print(f"Connection refused: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def analyze_cpu_usage(self, average_cpu_consumption):

        model = pd.read_csv(self.model_csv_path, header=None)
        model_type = model.iloc[0, 0]
        cpu_threshold1 = pd.read_csv('/Knowledge/cpu_threshold.csv', header=None)
        cpu_threshold = cpu_threshold1.iloc[0, 0]
        if model_type == 'lstm' and average_cpu_consumption > cpu_threshold:
            # Exceeds threshold, take action
            print(f"CPU Usage {average_cpu_consumption} exceeds threshold {cpu_threshold}. Taking action.")
            self.planner.take_action_based_on_cpu('general_model')  # Assuming this method exists
        elif model_type == 'general_model':
            input_sequence=self.get_input_sequence()
            prediction,cpu_consumption = predict_aqi_lstm(input_sequence)
            print("cpu consumption lstm: ", cpu_consumption)
            if cpu_consumption < cpu_threshold:
                self.planner.take_action_based_on_cpu('lstm')  # Assuming this method exists
            # print(cpu_consumption)
        else:
            print(f"CPU Usage {average_cpu_consumption} is within threshold {cpu_threshold}. No action required.")