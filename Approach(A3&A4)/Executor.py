# executor.py
import pandas as pd

class Executor:
    def __init__(self, model_csv_path='path/to/Model_reload/model.csv'):
        self.model_csv_path = model_csv_path

    def write_model_name(self, model_name):
        print(f"Switching to {model_name} model.")
        pd.DataFrame([model_name]).to_csv(self.model_csv_path, index=False, header=False)
