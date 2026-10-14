import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.signal import savgol_filter
from config import config

class DataProcessor:
    def __init__(self):
        print("[INFO] Initializing Data Processor...")
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def load_data(self):
        print("[INFO] Loading datasets...")
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        test_df = pd.read_csv(config.TEST_DATA_PATH)
        return train_df, test_df

    def process_data(self, train_df, test_df):
        print("[INFO] Processing data (Separation, Smoothing, Scaling, SMOTE)...")

        X_train = train_df.drop('LABEL', axis=1).values
        y_train = train_df['LABEL'].apply(lambda x: 1 if x == 2 else 0).values
        
        X_test = test_df.drop('LABEL', axis=1).values
        y_test = test_df['LABEL'].apply(lambda x: 1 if x == 2 else 0).values

        print("[INFO] Applying Savitzky-Golay filter to reduce high-frequency noise...")
        X_train = savgol_filter(X_train, window_length=15, polyorder=3, axis=1)
        X_test = savgol_filter(X_test, window_length=15, polyorder=3, axis=1)
    
        print("[INFO] Scaling features...")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print("[INFO] Balancing classes with SMOTE...")
        X_train, y_train = self.smote.fit_resample(X_train, y_train)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test