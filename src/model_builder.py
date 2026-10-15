from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense, Dropout
from config import config

class ExoplanetCNN:
    def __init__(self):
        self.input_shape = (config.SEQ_LENGTH, 1)

    def build_model(self):
        model = Sequential()

        model.add(Conv1D(filters=16, kernel_size=11, activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.2))

        model.add(Conv1D(filters=32, kernel_size=7, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.2))

        model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        return model

if __name__ == "__main__":
    builder = ExoplanetCNN()
    model = builder.build_model()
    model.summary()