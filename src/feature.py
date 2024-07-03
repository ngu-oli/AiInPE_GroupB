import numpy as np
import pandas as pd
from src.measurement import Measurement
from src.database import Database
import os
import scipy.stats as stats


class FeatureExtracting:
    def __init__(self, audio_folder, weld_folder, database_filepath):
        self.audio_measurements = Measurement(audio_folder)
        self.weld_measurements = Measurement(weld_folder)
        self.database = Database(database_filepath, skip_rows=10, use_cols="B:O")

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
            median = np.median(data)
            min_val = np.min(data)
            max_val = np.max(data)
            rms = np.sqrt(np.mean(np.square(data)))
            peak_to_peak = max_val - min_val
            crest_factor = max(abs(max_val), abs(min_val)) / rms if rms != 0 else 0
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
        else:
            mean, std, median, min_val, max_val, rms, peak_to_peak, crest_factor, skewness, kurtosis = [0] * 10

        return {
            'mean': mean,
            'std': std,
            'median': median,
            'min': min_val,
            'max': max_val,
            'rms': rms,
            'peak_to_peak': peak_to_peak,
            'crest_factor': crest_factor,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def compute_features_for_measurement(self, measurement_number):
        audio_file = os.path.join(self.audio_measurements.csv_folder, f"{measurement_number}.csv")
        weld_file = os.path.join(self.weld_measurements.csv_folder, f"{measurement_number}.csv")

        audio_data = pd.read_csv(audio_file) if os.path.exists(audio_file) else None
        weld_data = pd.read_csv(weld_file) if os.path.exists(weld_file) else None

        features = {}

        if audio_data is not None:
            audio_signal = audio_data['M'].dropna().values
            audio_features = self.compute_stat_features(audio_signal)
            features.update({f'audio_{k}': v for k, v in audio_features.items()})
        else:
            features.update({f'audio_{k}': 0 for k in
                             ['mean', 'std', 'median', 'min', 'max', 'rms', 'peak_to_peak', 'crest_factor', 'skewness',
                              'kurtosis']})

        if weld_data is not None:
            for col in ['Current [A]', 'Voltage [V]', 'Wire [m/min]']:
                data = weld_data[col].dropna().values
                col_name = col.split()[0].lower()  # Get 'current', 'voltage', 'wire' from column names
                col_features = self.compute_stat_features(data)
                features.update({f'{col_name}_{k}': v for k, v in col_features.items()})
        else:
            for col in ['current', 'voltage', 'wire']:
                features.update({f'{col}_{k}': 0 for k in
                                 ['mean', 'std', 'median', 'min', 'max', 'rms', 'peak_to_peak', 'crest_factor',
                                  'skewness', 'kurtosis']})

        return features

    def cleanse_data(self):
        self.rename_labels()

        if self.database.data is not None:
            # Initialize columns to store features
            feature_columns = [f'{signal}_{feature}' for signal in ['audio', 'current', 'voltage', 'wire']
                               for feature in
                               ['mean', 'std', 'median', 'min', 'max', 'rms', 'peak_to_peak', 'crest_factor',
                                'skewness', 'kurtosis']]
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
            output_columns = ['Number of Measurement', 'Dataset'] + [
                f'{signal}_{feature}' for signal in ['audio', 'current', 'voltage', 'wire']
                for feature in
                ['mean', 'std', 'median', 'min', 'max', 'rms', 'peak_to_peak', 'crest_factor', 'skewness', 'kurtosis']
            ]
            output_data = self.database.data[output_columns]
            output_data.to_excel(output_filepath, index=False)
            print(f"Data saved to {output_filepath}")
        else:
            print("Error: Database data is not loaded correctly, nothing to save.")

