import pandas as pd
import numpy as np
import os
import time
from scipy.stats import entropy
import threading
from Analyser import Analyser

analyser_instance = Analyser()  # Create an instance of Analyser


def load_predictions(file_path, skiprows=None):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        predictions = pd.read_csv(file_path, skiprows=skiprows)
        return predictions
    else:
        return None


def calculate_average_non_zero_cpu_consumptions(cpu_consumptions):
    non_zero_consumptions = cpu_consumptions[cpu_consumptions != 0]
    if not non_zero_consumptions.empty:
        return non_zero_consumptions.mean()
    else:
        return None


start_lstm = 0
start_general = 0


def monitor_for_drift():
    global start_lstm
    global start_general
    time.sleep(300)
    while True:
        model = pd.read_csv('path/to/Model_reload/model.csv', header=None)
        model_type = model.iloc[0, 0]
        baseline_file = 'path/to/Knowledge/train_results_LSTM.csv' if model_type == 'lstm' else 'Knowledge/train_results_general_model.csv'
        baseline_file_lstm = 'path/to/Knowledge/train_results_LSTM.csv'
        baseline_file_general = 'path/to/Knowledge/train_results_general_model.csv'
        real_time_file = 'path/to/Results/prediction_results.csv'

        baseline_predictions_lstm = load_predictions(baseline_file_lstm)
        baseline_predictions_general = load_predictions(baseline_file_general)
        baseline_predictions = load_predictions(baseline_file)
        real_time_predictions = load_predictions(real_time_file)
        print('Creating analyser object...')
        analyser_instance.analyze_drift(baseline_predictions_lstm, baseline_predictions_general, real_time_predictions)
        time.sleep(300)  # Adjust frequency as needed


def monitor_emissions_and_switch_model():
    last_processed_row = 0  # Initialize to start from the first row
    while True:
        # Calculate the number of rows to skip, skiprows is 0-based hence the adjustment
        skiprows = list(range(1, last_processed_row + 1)) if last_processed_row > 0 else None
        # print(len(skiprows))
        predictions = load_predictions('path/to/Results/prediction_results.csv',
                                       skiprows=skiprows)
        if predictions is not None and not predictions.empty:
            new_data_count = len(predictions)

            # Assuming the CPU consumption is in the last column
            cpu_consumption_column_name = predictions.columns[-1]
            average_cpu_consumption = calculate_average_non_zero_cpu_consumptions(
                predictions[cpu_consumption_column_name])
            # Pass the average CPU consumption to the analyser, if there's any new non-zero data
            if average_cpu_consumption is not None:
                print("Average CPU", average_cpu_consumption)
                analyser_instance.analyze_cpu_usage(average_cpu_consumption)
            # Update the last processed row count
            last_processed_row += new_data_count
        else:
            print("Waiting for new data...")

        time.sleep(10)  # Sleep and then process any new data


if __name__ == "__main__":
    drift_thread = threading.Thread(target=monitor_for_drift)
    emissions_thread = threading.Thread(target=monitor_emissions_and_switch_model)
    #
    emissions_thread.start()
    drift_thread.start()

    emissions_thread.join()
    drift_thread.join()