import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from config import config

from config import config


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def load_data(self):
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        test_df = pd.read_csv(config.TEST_DATA_PATH)
        return train_df, test_df 

    def process_data(self, train_df, test_df):

        y_train = train_df['LABEL'].apply(lambda x: 1 if x == 2 else 0).values
        y_test  = test_df['LABEL'].apply(lambda x: 1 if x == 2 else 0).values

        x_train = train_df.drop('LABEL', axis=1).values
        x_test = test_df.drop('LABEL', axis=1).values

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        planets_before = sum(y_train)
        non_planets_before = len(y_train) - planets_before
        print(f"[DEBUG] Before SMOTE -> Planets: {planets_before}, Non-planets: {non_planets_before}")
        x_train_balanced, y_train_balanced = self.smote.fit_resample(x_train_scaled, y_train)
        
        planets_after = sum(y_train_balanced)
        non_planets_after = len(y_train_balanced) - planets_after
        print(f"[DEBUG] After SMOTE  -> Planets: {planets_after}, Non-planets: {non_planets_after}")

        return x_train_balanced, y_train_balanced, x_test_scaled, y_test




if __name__ == "__main__":
    processor = DataProcessor()
    train_data, test_data = processor.load_data()
    print(f"[DEBUG] Train data shape: {train_data.shape}")
    print(f"[DEBUG] Test data shape: {test_data.shape}")