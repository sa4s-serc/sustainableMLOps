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

    def analyze_drift(self):
        model = pd.read_csv(self.model_csv_path, header=None)
        model_in_use = model.iloc[0, 0]

        #comment out the model for which you are not doing periodic retraining
        #for example if you want to periodically retrain lstm you comment out general model planner object

        self.planner.divergence_detected('lstm')
        # self.planner.divergence_detected('general_model')


    def get_input_sequence(self):
        HOST = '127.0.0.12'  # The server's hostname or IP address
        PORT = 65428  # The port used by the server

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
        cpu_threshold1 = pd.read_csv('Knowledge/cpu_threshold.csv', header=None)
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
