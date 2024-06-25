import os
import pandas as pd
import matplotlib.pyplot as plt

class Measurement:
    def __init__(self, csv_folder):
        self.csv_folder = csv_folder
        self.measurements = self.load_measurements()

    def load_measurements(self):
        # Count files
        files = os.listdir(self.csv_folder)
        file_count = len(files)
        
        print(f'Found {file_count} files in this directory. Data loading starting...')
        
        all_measurements = []
        for file_name in os.listdir(self.csv_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.csv_folder, file_name)
                try:
                    df = pd.read_csv(file_path)
                    df['File'] = file_name  # Add a column to identify which file the data belongs to
                    print(f'Currently reading {file_name}')
                    all_measurements.append(df)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        if all_measurements:
            return pd.concat(all_measurements, ignore_index=True)
        else:
            return None

    # Takes very long time to display all data
    def plot_measurement_over_time_graph_for_audio(self):
        if self.measurements is not None:
            plt.figure(figsize=(12, 8))
            for file_name, group in self.measurements.groupby('File'):
                plt.scatter(group['Time'], group['M'], label=f'Measurement M for {file_name}')

            plt.title('Measurement M over Time')
            plt.xlabel('Time')
            plt.ylabel('Measurement M')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No measurements data to plot.")

