import pandas as pd
import os

def load_data():
    file_path = os.path.join("data", "raw", "hcc-data-complete-balanced.xlsx")
    df = pd.read_excel(file_path)  # uses openpyxl automatically
    return df

if __name__ == "__main__":
    df = load_data()
    print("Dataset loaded successfully!")
    print(df.head())

