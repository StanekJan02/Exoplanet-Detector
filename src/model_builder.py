import numpy as np
import tensorflow as tf
from astropy.timeseries import BoxLeastSquares

from src.config import (
    BLS_DURATIONS_DAYS,
    BLS_MIN_PERIOD_DAYS,
    KEPLER_CADENCE_DAYS,
    LEARNING_RATE,
    SEQUENCE_LENGTH,
)


def build_cnn_model(input_length: int = SEQUENCE_LENGTH) -> tf.keras.Model:
    """
    Buduje jednowymiarową sieć konwolucyjną (1D-CNN) do detekcji tranzytów egzoplanet.
    Kompiluje model z metrykami kluczowymi dla silnie niezbalansowanych danych:
    Recall, Precision oraz PR-AUC (pole pod krzywą Precision-Recall).
    """
    inputs = tf.keras.Input(shape=(input_length, 1), name="lightcurve_input")

    # Blok 1 - szersze filtry wyłapujące ogólny profil tranzytu
    x = tf.keras.layers.Conv1D(32, kernel_size=11, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4, strides=4)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Blok 2 - średnia skala czasowa
    x = tf.keras.layers.Conv1D(64, kernel_size=7, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4, strides=4)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Blok 3 - lokalne cechy kształtu (np. kształt U vs V)
    x = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4, strides=4)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Blok 4 - agregacja globalna i lokalna
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="planet_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Exoplanet_1D_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


class BLSBaselineClassifier:
    """
    Klasyczna metoda Box Least Squares (BLS) z biblioteki astropy jako punkt odniesienia (baseline).
    Oblicza periodogram BLS oraz współczynnik SDE (Signal Detection Efficiency), który
    służy jako miara pewności detekcji do porównania z prawdopodobieństwami sieci CNN.
    """

    def __init__(
        self,
        dt: float = KEPLER_CADENCE_DAYS,
        min_period: float = BLS_MIN_PERIOD_DAYS,
        durations: tuple[float, ...] = BLS_DURATIONS_DAYS,
    ):
        self.dt = dt
        self.min_period = min_period
        self.durations = np.array(durations, dtype=np.float64)

    def analyze_single_curve(self, raw_flux: np.ndarray) -> dict[str, float | np.ndarray]:
        """Wykonuje analizę BLS dla pojedynczej krzywej blasku i zwraca SDE oraz parametry orbity."""
        flux = np.nan_to_num(raw_flux.astype(np.float64), nan=np.nanmedian(raw_flux))

        # Przesunięcie wokół 1.0 i usunięcie trendu wielomianem niskiego stopnia
        time = np.arange(len(flux), dtype=np.float64) * self.dt
        poly_trend = np.polyval(np.polyfit(time, flux, deg=2), time)
        detrended = flux - poly_trend
        std_val = np.std(detrended) + 1e-8
        flux_norm = 1.0 + (detrended / (std_val * 100.0))

        bls = BoxLeastSquares(time, flux_norm)
        max_period = min(20.0, (time[-1] / 2.0))
        results = bls.autopower(
            self.durations,
            minimum_period=self.min_period,
            maximum_period=max_period,
            objective="likelihood",
        )

        power = np.asarray(results.power, dtype=np.float64)
        best_idx = int(np.argmax(power))

        # SDE (Signal Detection Efficiency) - znormalizowana wysokość głównego piku periodogramu
        sde = float((power[best_idx] - np.median(power)) / (np.std(power) + 1e-8))

        best_period = float(results.period[best_idx])
        best_t0 = float(results.transit_time[best_idx])
        best_duration = float(results.duration[best_idx])
        model_curve = bls.model(time, best_period, best_duration, best_t0)

        return {
            "sde": sde,
            "best_period": best_period,
            "best_t0": best_t0,
            "best_duration": best_duration,
            "time": time,
            "flux_norm": flux_norm,
            "bls_model_curve": model_curve,
            "periods": np.asarray(results.period, dtype=np.float64),
            "power": power,
        }

    def score_dataset(self, X_raw: np.ndarray, max_samples: int | None = None) -> np.ndarray:
        """Oblicza wektor wyników SDE dla zbioru krzywych blasku."""
        from tqdm import tqdm

        n_samples = len(X_raw) if max_samples is None else min(len(X_raw), max_samples)
        sde_scores = np.zeros(n_samples, dtype=np.float32)

        for idx in tqdm(range(n_samples), desc="Obliczanie Box Least Squares (BLS)"):
            res = self.analyze_single_curve(X_raw[idx])
            sde_scores[idx] = res["sde"]

        return sde_scores