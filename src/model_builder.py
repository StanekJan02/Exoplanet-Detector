from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from config import config

class ExoplanetCNN:
    def __init__(self):
        self.input_shape = (config.SEQ_LENGTH,1)

if __name__ == "__main__":
    builder = ExoplanetCNN()
    print(f"[DEBUG] Model input shape set to: {builder.input_shape}")

    