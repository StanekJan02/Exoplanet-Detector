import pandas as pd
from config import config

class DataProcessor:
    def __init__(self):
        print("[INFO] Data processor initialized.")

    def load_data(self):
        print(f"[INFO] Loading training data: {config.TRAIN_DATA_PATH}")

        train_df = pd.read_csv(config.TRAIN_DATA_PATH)

        print(f"[INFO] Loading testing data: {config.TEST_DATA_PATH}")

        test_df = pd.read_csv(config.TEST_DATA_PATH)

        return train_df, test_df 

if __name__ == "__main__":
    processor = DataProcessor()
    train_data, test_data = processor.load_data()

    print(f"[DEBUG] Train data shape: {train_data.shape}")
    print(f"[DEBUG] Test data shape: {test_data.shape}")