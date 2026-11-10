import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import train_test_split

from src.config import (
    AUGMENT_POSITIVE_CLASS,
    AUGMENTATION_FACTOR,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VALIDATION_SPLIT,
)


def normalize_light_curves(X: np.ndarray, detrend: bool = False) -> np.ndarray:
    """
    Normalizuje każdą krzywą blasku indywidualnie (zero-mean, unit-variance lub względem mediany).
    Opcjonalnie usuwa długookresowy trend za pomocą filtra ruchomego.
    """
    X_proc = np.nan_to_num(X.astype(np.float32), nan=0.0)

    if detrend:
        trend = uniform_filter1d(X_proc, size=151, axis=1, mode="nearest")
        X_proc = X_proc - trend

    medians = np.median(X_proc, axis=1, keepdims=True)
    stds = np.std(X_proc, axis=1, keepdims=True) + 1e-8
    return (X_proc - medians) / stds


def augment_positive_samples(
    X: np.ndarray, y: np.ndarray, factor: int = AUGMENTATION_FACTOR, seed: int = RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """
    Powiększa reprezentację klasy pozytywnej (37 gwiazd z planetą), stosując operacje
    zachowujące okresowość tranzytu:
    - odwrócenie osi czasu (time reversal),
    - cykliczne przesunięcie fazowe (np.roll),
    - delikatny szum gaussowski.
    """
    rng = np.random.default_rng(seed)
    pos_indices = np.where(y == 1)[0]

    if len(pos_indices) == 0 or factor <= 0:
        return X, y

    X_pos = X[pos_indices]
    augmented_curves = []

    for curve in X_pos:
        for i in range(factor):
            aug = curve.copy()
            if i % 2 == 0:
                aug = aug[::-1]
            shift = rng.integers(low=50, high=len(aug) - 50)
            aug = np.roll(aug, shift)
            noise = rng.normal(0.0, 0.02 * np.std(aug), size=aug.shape)
            aug = aug + noise
            augmented_curves.append(aug)

    X_aug = np.vstack([X, np.array(augmented_curves, dtype=np.float32)])
    y_aug = np.concatenate([y, np.ones(len(augmented_curves), dtype=np.int32)])

    perm = rng.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]


def load_kaggle_dataset(
    train_path=TRAIN_CSV_PATH, test_path=TEST_CSV_PATH
) -> dict[str, np.ndarray]:
    """
    Wczytuje zbiór 'Exoplanet Hunting in Deep Space' z Kaggle.
    W oryginalnym zbiorze etykieta 2 oznacza obecność egzoplanety, a 1 jej brak.
    Zwraca zarówno surowe krzywe (dla algorytmu BLS), jak i znormalizowane tensory 3D (dla CNN).
    """
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono plików {train_path} lub {test_path}. "
            "Umieść pliki exoTrain.csv oraz exoTest.csv w katalogu data/."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train_full = train_df.iloc[:, 0].values
    X_train_raw = train_df.iloc[:, 1:].values.astype(np.float32)

    y_test = test_df.iloc[:, 0].values
    X_test_raw = test_df.iloc[:, 1:].values.astype(np.float32)

    # Konwersja etykiet z {1, 2} na {0, 1}
    if set(np.unique(y_train_full)).issubset({1, 2}):
        y_train_full = (y_train_full == 2).astype(np.int32)
        y_test = (y_test == 2).astype(np.int32)
    else:
        y_train_full = y_train_full.astype(np.int32)
        y_test = y_test.astype(np.int32)

    # Podział stratyfikowany na trening i walidację (przed augmentacją, aby uniknąć wycieku danych!)
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw,
        y_train_full,
        test_size=VALIDATION_SPLIT,
        stratify=y_train_full,
        random_state=RANDOM_SEED,
    )

    # Augmentacja wyłącznie na zbiorze treningowym
    if AUGMENT_POSITIVE_CLASS:
        X_tr_raw, y_tr = augment_positive_samples(X_tr_raw, y_tr)

    # Normalizacja pod sieć CNN i dodanie wymiaru kanału (N, 3197, 1)
    X_tr_cnn = normalize_light_curves(X_tr_raw, detrend=True)[..., np.newaxis]
    X_val_cnn = normalize_light_curves(X_val_raw, detrend=True)[..., np.newaxis]
    X_test_cnn = normalize_light_curves(X_test_raw, detrend=True)[..., np.newaxis]

    return {
        "X_train_cnn": X_tr_cnn,
        "y_train": y_tr,
        "X_val_cnn": X_val_cnn,
        "y_val": y_val,
        "X_test_cnn": X_test_cnn,
        "X_test_raw": X_test_raw,
        "y_test": y_test,
    }


def fetch_mast_lightcurve(
    target_name: str = "Kepler-10",
    mission: str = "Kepler",
    target_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pobiera krzywą blasku bezpośrednio z archiwum MAST przy użyciu biblioteki lightkurve
    (Kepler lub TESS) i przygotowuje ją do predykcji zarówno przez BLS, jak i sieć CNN.
    """
    import lightkurve as lk

    search_res = lk.search_lightcurve(target_name, author=mission)
    if len(search_res) == 0:
        raise ValueError(f"Nie znaleziono krzywej blasku w MAST dla obiektu: {target_name}")

    lc = search_res[0].download().remove_nans().flatten(window_length=401)
    flux_values = np.asarray(lc.flux.value, dtype=np.float32)

    # Dopasowanie długości sygnału do wymiaru wejściowego sieci (3197 punktów)
    x_old = np.linspace(0.0, 1.0, len(flux_values))
    x_new = np.linspace(0.0, 1.0, target_length)
    raw_resampled = np.interp(x_new, x_old, flux_values).astype(np.float32)

    cnn_ready = normalize_light_curves(raw_resampled[np.newaxis, ...], detrend=False)[..., np.newaxis]
    return raw_resampled, cnn_ready