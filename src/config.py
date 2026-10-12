import os 
import argparse

class ConfigManager:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.TRAIN_DATA_PATH = os.path.join(self.BASE_DIR, 'data', 'exoTrain.csv')
        self.TEST_DATA_PATH = os.path.join(self.BASE_DIR, 'data', 'exoTest.csv')
        self._parse_arguments()

    def _parse_arguments(self):
        parser = argparse.ArgumentParser(description="==[Exoplanet Detector - models' settings]==")

        parser.add_argument('-s', '--seq-length',type=int, default=3197,
                            help='Length of time series')
        parser.add_argument('-b', '--batch-size',type=int, default=32,
                            help='Batch size')
        parser.add_argument('-e', '--epochs', type=int, default=50,
                            help='No. epochs')
        parser.add_argument('-l','--learning-rate', type=float, default=0.001,
                            help='Learning rate')

        args, _ = parser.parse_known_args()

        self.SEQ_LENGTH = args.seq_length
        self.BATCH_SIZE = args.batch_size
        self.EPOCHS = args.epochs
        self.LEARNING_RATE = args.learning_rate



config = ConfigManager()


if __name__ == "__main__":
    print(f"TEST SEQ_LENGTH = {config.SEQ_LENGTH}")
    print(f"TEST BATCH_SIZE = {config.BATCH_SIZE}")
    print(f"TEST EPOCHS = {config.EPOCHS}")
    print(f"TEST LEARNING_RATE = {config.LEARNING_RATE}")

