from pathlib import Path

# Ścieżki katalogów projektu
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Tworzenie katalogów, jeśli nie istnieją
for directory in (DATA_DIR, MODELS_DIR, ASSETS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Ścieżki do zbioru Kaggle "Exoplanet Hunting in Deep Space"
TRAIN_CSV_PATH = DATA_DIR / "exoTrain.csv"
TEST_CSV_PATH = DATA_DIR / "exoTest.csv"
MODEL_SAVE_PATH = MODELS_DIR / "cnn_exoplanet_detector.keras"

# Parametry sygnału (Kepler Long Cadence: 3197 próbek, krok ~29.4 min = 0.0204 dnia)
SEQUENCE_LENGTH = 3197
KEPLER_CADENCE_DAYS = 0.02043365

# Hiperparametry treningu sieci CNN
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15

# Augmentacja rzadkiej klasy pozytywnej (37 gwiazd z planetą w zbiorze treningowym)
AUGMENT_POSITIVE_CLASS = True
AUGMENTATION_FACTOR = 15  # Ile dodatkowych wariantów wygenerować dla każdej planety

# Parametry klasycznej metody Box Least Squares (BLS z astropy)
BLS_MIN_PERIOD_DAYS = 0.5
BLS_DURATIONS_DAYS = (0.04, 0.08, 0.12, 0.16, 0.20, 0.25)
# Ile próbek ze zbioru testowego ocenić metodą BLS (None = cały zbiór testowy)
BLS_MAX_EVAL_SAMPLES = None