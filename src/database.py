import pandas as pd
import matplotlib.pyplot as plt

class Database:
    def __init__(self, filepath, skip_rows, use_cols):
        self.data = self.read_data_table(filepath, skip_rows=skip_rows, use_cols=use_cols)
        
    def read_data_table(self, filepath, skip_rows, use_cols):
        try:
            if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                # Correctly pass skiprows and usecols arguments
                data = pd.read_excel(filepath, skiprows=skip_rows, usecols=use_cols)
            else:
                raise ValueError("Unsupported file format. Please provide an Excel file.")
            
            return data
        
        except FileNotFoundError:
            print(f"Error: The file at {filepath} was not found.")
        except pd.errors.EmptyDataError:
            print("Error: No data found in the file.")
        except pd.errors.ParserError:
            print("Error: Could not parse the file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
                
    def count_na_in_col(self, col_name):
        num_empty_records =  self.data[col_name].isna().sum()
        print(f'# NA in col {col_name}: {num_empty_records}')
        return num_empty_records

    def plot_box_plot(self, col_name):
        # Check if all non-null values in the column are of type str
        is_str_col_type = self.data[col_name].dropna().apply(type).eq(str).all()
        if is_str_col_type:
            print(f"Column '{col_name}' consists of string values.")
            
        else:
            plt.figure(figsize=(8, 6))
            self.data.boxplot(column=[col_name])
            plt.title(f'Boxplot of column {col_name}')
            plt.ylabel('Values')
            plt.show()