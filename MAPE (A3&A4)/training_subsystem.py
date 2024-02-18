import socket
import threading
import csv
import pandas as pd
import sys
from training.lstm_retrain import train_lstm_model
from training.general_retrain import train_general_model
import os
import shutil
import glob
import threading
import pyRAPL
pyRAPL.setup()

sys.path.append('../Data_preprocessing/')
from data_cleaning import DataCleaning

# Global lock object
training_lock = threading.Lock()

# Assuming these are your file paths - adjust them as needed
predictions_csv_path = 'Results/prediction_results.csv'
experiment_csv_path = 'Dataset/cleaned_filtered_data_experiment.csv'
training_csv_path = 'Dataset/cleaned_filtered_data_train.csv'
output_csv_path = 'Dataset/combined_cleaned_data.csv'
model_dir = 'Models/'
versioned_model_dir = 'versioned_models/'

cleaning_instance = DataCleaning(experiment_csv_path)
def update_reload_flag_lstm():
    flag_file_path = "Model_reload/reload_flag_lstm.csv"
    pd.DataFrame(['reload']).to_csv(flag_file_path, index=False, header=False)

def update_reload_flag_general():
    flag_file_path = "Model_reload/reload_flag_general.csv"
    pd.DataFrame(['reload']).to_csv(flag_file_path, index=False, header=False)

def log_training_instance(instance, model, cpu_consumption, log_file_path):
    """Logs training instance details to a CSV file."""
    # Check if the log file exists to determine if we need to write headers
    file_exists = os.path.isfile(log_file_path)

    with open(log_file_path, 'a', newline='') as csvfile:
        fieldnames = ['instance', 'model', 'CPU Consumption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write headers if file does not exist

        writer.writerow({'instance': instance, 'model': model, 'CPU Consumption': cpu_consumption})


def version_model(model_in_use):
    model_filename_map = {
        'lstm': 'model_lstm_wo_outlier_5',
        'general_model': 'regressor_model'
    }
    model_basename = model_filename_map.get(model_in_use)

    if not model_basename:
        print("Model type is unknown, cannot version.")
        return

    source_path = os.path.join(model_dir, f"{model_basename}.pkl") if model_in_use == 'general_model' else os.path.join(
        model_dir, model_basename)
    versioned_model_dir_specific = os.path.join(versioned_model_dir, model_in_use)

    if not os.path.exists(versioned_model_dir_specific):
        os.makedirs(versioned_model_dir_specific)

    pattern = os.path.join(versioned_model_dir_specific, f"{model_basename}_v*")
    existing_versions = glob.glob(pattern)
    version_numbers = [int(os.path.basename(version).split('_v')[-1].split('.')[0]) for version in existing_versions]
    next_version_number = max(version_numbers) + 1 if version_numbers else 1

    destination_path = os.path.join(versioned_model_dir_specific, f"{model_basename}_v{next_version_number}")
    destination_path += ".pkl" if model_in_use == 'general_model' else ""

    if model_in_use == 'lstm':
        # Use shutil.copytree() for LSTM model directory
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)
        else:
            print(f"No existing model directory found at {source_path}.")
    else:
        # Use shutil.copy() for general model file
        if os.path.isfile(source_path ):  # Adjusted to check for the '.pkl' file
            shutil.copy(source_path , destination_path)  # No additional '.pkl' is appended here
        else:
            print(f"No existing model file found at {source_path}.")

    print(f"Versioned model saved to {destination_path}")


def prepare_and_clean_data(predictions_csv_path, experiment_csv_path, training_csv_path, output_csv_path):
    predictions = pd.read_csv(predictions_csv_path, parse_dates=['Timestamp'])
    last_timestamp = predictions['Timestamp'].max()

    experiment = pd.read_csv(experiment_csv_path, parse_dates=['Timestamp'])
    filtered_experiment = experiment[experiment['Timestamp'] <= last_timestamp]

    training = pd.read_csv(training_csv_path, parse_dates=['Timestamp'])
    combined_data = pd.concat([training, filtered_experiment])

    cleaned_data = cleaning_instance.clean_and_filter_dataframe(combined_data)
    print("Combined and cleaned data saved successfully.")
    return cleaned_data

def handle_client_connection(client_socket):
    global training_lock  # Reference the global lock object

    try:
        message = client_socket.recv(1025)
        model_in_use = message.decode('utf-8')
        print(f"Received model in use: {model_in_use}")

        # Attempt to acquire the lock before proceeding with data preprocessing and training
        if training_lock.acquire(blocking=True):
            try:
                # Initialize the instance number (this could be based on a timestamp or a simple increment)
                instance = str(pd.Timestamp.now())  # Using current timestamp as a unique instance identifier

                # Data preprocessing upon drift detection
                cleaned_data = prepare_and_clean_data(predictions_csv_path, experiment_csv_path, training_csv_path, output_csv_path)
                client_socket.sendall("Data preprocessing completed successfully.".encode('utf-8'))


                meter = pyRAPL.Measurement('predictions')
                meter.begin()
                # Version the current model before retraining
                version_model(model_in_use)
                # Proceed with retraining
                if model_in_use == 'lstm':
                    train_lstm_model(cleaned_data, os.path.join(model_dir, 'model_lstm_wo_outlier_5'))
                    update_reload_flag_lstm()
                elif model_in_use == 'general_model':
                    train_general_model(cleaned_data, os.path.join(model_dir, 'regressor_model.pkl'))
                    update_reload_flag_general()

                meter.end()
                pkg_energy = meter.result.pkg
                dram_energy = meter.result.dram

                # Sum the values in the lists if they are not None, otherwise use 0
                total_pkg_energy = sum(pkg_energy) if pkg_energy is not None else 0
                total_dram_energy = sum(dram_energy) if dram_energy is not None else 0

                # Total energy consumption
                cpu_consumption = total_pkg_energy + total_dram_energy
                log_file_path = 'Results/retraining_results.csv'
                log_training_instance(instance, model_in_use, cpu_consumption, log_file_path)

                print(f"{model_in_use} model retraining completed.")
            finally:
                # Ensure the lock is released after training is completed
                training_lock.release()
    finally:
        client_socket.close()



def start_server(host='127.0.0.2', port=65402):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Training subsystem server listening on {host}:{port}")

    try:
        while True:
            client_sock, address = server_socket.accept()
            print(f"Accepted connection from {address[0]}:{address[1]}")
            client_handler = threading.Thread(target=handle_client_connection, args=(client_sock,))
            client_handler.start()
    except KeyboardInterrupt:
        print("Server is shutting down.")
    finally:
        server_socket.close()


if __name__ == "__main__":
    start_server()
