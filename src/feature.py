import numpy as np
import pandas as pd
from measurement import Measurement
from database import Database
import os


class FeatureExtracting:
    def __init__(self, audio_folder, weld_folder, database_filepath):
        self.audio_measurements = Measurement(audio_folder)
        self.weld_measurements = Measurement(weld_folder)
        self.database = Database(database_filepath, skip_rows=0, use_cols=None)

    def rename_labels(self):
        if self.database.data is not None:
            # Rename labels in the Dataset column based on the Forced Error Type
            self.database.data['Dataset'] = self.database.data.apply(
                lambda row: f"niO - {row['Forced Error Type']}" if row['Dataset'] == 'niO' else 'iO', axis=1)
        else:
            print("Error: Database data is not loaded correctly.")


    # I randomly picked some features i remembered from the lecture :D
    def compute_stat_features(self, data):
        if len(data) > 0:
            mean = np.mean(data)
            std = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            rms = np.sqrt(np.mean(np.square(data)))
        else:
            mean, std, min_val, max_val, rms = 0, 0, 0, 0, 0
        return mean, std, min_val, max_val, rms

    def compute_features_for_measurement(self, measurement_number):
        audio_file = os.path.join(self.audio_measurements.csv_folder, f"{measurement_number}.csv")
        weld_file = os.path.join(self.weld_measurements.csv_folder, f"{measurement_number}.csv")

        audio_data = pd.read_csv(audio_file) if os.path.exists(audio_file) else None
        weld_data = pd.read_csv(weld_file) if os.path.exists(weld_file) else None

        features = {}

        if audio_data is not None:
            audio_signal = audio_data['M'].dropna().values
            features['audio_mean'], features['audio_std'], features['audio_min'], features['audio_max'], features['audio_rms'] = self.compute_stat_features(audio_signal)
        else:
            features['audio_mean'], features['audio_std'], features['audio_min'], features['audio_max'], features['audio_rms'] = 0, 0, 0, 0, 0

        if weld_data is not None:
            for col in ['Current [A]', 'Voltage [V]', 'Wire [m/min]']:
                data = weld_data[col].dropna().values
                col_name = col.split()[0].lower()  # Get 'current', 'voltage', 'wire' from column names
                features[f'{col_name}_mean'], features[f'{col_name}_std'], features[f'{col_name}_min'], features[f'{col_name}_max'], features[f'{col_name}_rms'] = self.compute_stat_features(data)
        else:
            for col in ['current', 'voltage', 'wire']:
                features[f'{col}_mean'], features[f'{col}_std'], features[f'{col}_min'], features[f'{col}_max'], features[f'{col}_rms'] = 0, 0, 0, 0, 0

        return features

    def cleanse_data(self):
        self.rename_labels()

        if self.database.data is not None:
            # Initialize columns to store features
            feature_columns = ['audio_mean', 'audio_std', 'audio_min', 'audio_max', 'audio_rms',
                               'current_mean', 'current_std', 'current_min', 'current_max', 'current_rms',
                               'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max', 'voltage_rms',
                               'wire_mean', 'wire_std', 'wire_min', 'wire_max', 'wire_rms']
            for feature in feature_columns:
                self.database.data[feature] = 0

            # Process each row separately
            for index, row in self.database.data.iterrows():
                measurement_number = str(row['Number of Measurement'])
                features = self.compute_features_for_measurement(measurement_number)
                for feature, value in features.items():
                    self.database.data.at[index, feature] = value

            return self.database.data
        else:
            print("Error: Database data is not loaded correctly.")
            return None

    def save_to_excel(self, output_filepath):
        if self.database.data is not None:
            # Select only relevant columns
            output_data = self.database.data[['Number of Measurement', 'Dataset'] + [
                'audio_mean', 'audio_std', 'audio_min', 'audio_max', 'audio_rms',
                'current_mean', 'current_std', 'current_min', 'current_max', 'current_rms',
                'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max', 'voltage_rms',
                'wire_mean', 'wire_std', 'wire_min', 'wire_max', 'wire_rms'
            ]]
            output_data.to_excel(output_filepath, index=False)
            print(f"Data saved to {output_filepath}")
        else:
            print("Error: Database data is not loaded correctly, nothing to save.")

# As you can see I stored the Data in a folder ("Data") in the src directory & Added folder for Audio and Weld
audio_folder = r"C:\Users\CT-Laptop-01\PycharmProjects\AiInPE_GroupB\src\Data\01_Audio"
weld_folder = r"C:\Users\CT-Laptop-01\PycharmProjects\AiInPE_GroupB\src\Data\02_Weldqas"
database_filepath = r"C:\Users\CT-Laptop-01\PycharmProjects\AiInPE_GroupB\src\Data\0001_Database.xlsx"

# New Datasheet: Cleansed, only with number, label, and features!
output_filepath = r'C:\Users\CT-Laptop-01\PycharmProjects\AiInPE_GroupB\src\Data\0001_Database_with_features.xlsx'

# Ensure 'openpyxl' is installed before running the code
feature_extractor = FeatureExtracting(audio_folder, weld_folder, database_filepath)

# Print the column names to verify
print(feature_extractor.database.data.columns)

clean_data = feature_extractor.cleanse_data()

if clean_data is not None:
    feature_extractor.save_to_excel(output_filepath)
    print(clean_data[['Number of Measurement', 'Dataset'] + [
        'audio_mean', 'audio_std', 'audio_min', 'audio_max', 'audio_rms',
        'current_mean', 'current_std', 'current_min', 'current_max', 'current_rms',
        'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max', 'voltage_rms',
        'wire_mean', 'wire_std', 'wire_min', 'wire_max', 'wire_rms'
    ]].head())
else:
    print("Data cleansing failed due to errors in loading the database.")
