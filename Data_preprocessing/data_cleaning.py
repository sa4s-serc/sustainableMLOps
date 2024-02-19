import pandas as pd
import os
import numpy as np
class DataCleaning:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def clean_and_filter_dataframe(self, dataframe):
        # Assuming 'data' is a DataFrame passed directly to the method
        dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'], format='%Y-%m-%d %H:%M:%S')
        dataframe.set_index('Timestamp', inplace=True)

        # Resample, identify and mark outliers, then interpolate
        data_min = dataframe.resample('min').mean()
        data_min['AQI'] = np.where(data_min['AQI'] > 500, np.nan, data_min['AQI'])
        data_min.interpolate(inplace=True)

        return data_min

    def clean_and_filter_data(self):
        # Check if the CSV file exists
        if not os.path.isfile(self.csv_file_path):
            print(f"File {self.csv_file_path} does not exist.")
            return None

        # Load and preprocess the data
        data = pd.read_csv(self.csv_file_path)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y-%m-%d %H:%M:%S')
        data.set_index('Timestamp', inplace=True)

        # Resample and interpolate
        data_min = data.resample('T').mean()

        # Identify and mark outliers in the AQI column
        data_min['AQI'] = np.where(data_min['AQI'] > 500, np.nan, data_min['AQI'])  # remove outliers
        data_min.interpolate(inplace=True)

        return data_min

# Usage example
if __name__ == "__main__":
    cleaning_instance = DataCleaning('path/to/Dataset/cleaned_filtered_data_experiment.csv')
    cleaned_data = cleaning_instance.clean_and_filter_data()
    if cleaned_data is not None:
        print(cleaned_data.head())  # Display the first few rows of the cleaned data
    else:
        print("Data cleaning was not successful.")
