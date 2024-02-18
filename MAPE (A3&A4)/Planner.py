# planner.py
from Executor import Executor
import socket

class Planner:
    def __init__(self, model_csv_path='Model_reload/model.csv', training_subsystem_host='127.0.0.2', training_subsystem_port=65402):
        self.model_csv_path = model_csv_path
        self.training_subsystem_address = (training_subsystem_host, training_subsystem_port)
        self.executor = Executor(model_csv_path=model_csv_path)  # Assuming Executor handles model switching and writing

    def send_to_training_subsystem(self, model_in_use):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(self.training_subsystem_address)
                sock.sendall(model_in_use.encode('utf-8'))
                # Optionally wait for acknowledgment
                response = sock.recv(1024)
                print(f"Acknowledgment from training subsystem: {response.decode('utf-8')}")
        except ConnectionRefusedError:
            print("Connection refused. The training subsystem may not be available.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def divergence_detected(self, model_in_use):
        """Handles actions to be taken when divergence is detected."""
        print("Divergence detected. Considering model evaluation.")
        self.send_to_training_subsystem(model_in_use)

    def switch_model_to_general(self):
        """Switches the current model to the general model."""
        self.executor.write_model_name('general_model')
        print("Switched the model to general_model.")

    def take_action_based_on_cpu(self, model_switch):
        """Takes action based on CPU usage, such as switching models."""
        print(f"Switching the model to {model_switch}.")
        self.executor.write_model_name(model_switch)

# Assuming the rest of the Executor class and functionality is correctly implemented elsewhere.
