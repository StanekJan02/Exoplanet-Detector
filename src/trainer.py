import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# Umożliwia uruchomienie zarówno przez `python -m src.trainer`, jak i `python src/trainer.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    BATCH_SIZE,
    BLS_MAX_EVAL_SAMPLES,
    EPOCHS,
    MODEL_SAVE_PATH,
    RANDOM_SEED,
)
from src.data_processor import load_kaggle_dataset
from src.model_builder import BLSBaselineClassifier, build_cnn_model
from src.visualizer import (
    plot_cnn_vs_bls_case_studies,
    plot_confusion_matrices,
    plot_pr_curve_comparison,
    plot_training_history,
)


def find_optimal_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Wyznacza próg decyzyjny maksymalizujący F1-score na podstawie krzywej Precision-Recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def evaluate_predictions(name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Wylicza i wypisuje zestawienie kluczowych metryk (Recall, PR-AUC, Precision, F1)."""
    preds = (scores >= threshold).astype(np.int32)
    rec = recall_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0

    print(f"\n==================== WYNIKI: {name} ====================")
    print(f"Próg decyzyjny : {threshold:.4f}")
    print(f"PR-AUC         : {pr_auc:.4f}  <-- Kluczowa metryka dla niezbalansowanych danych")
    print(f"Recall (Czułość): {rec:.4f}  <-- Odsetek wykrytych egzoplanet")
    print(f"Precision      : {prec:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"ROC-AUC        : {roc_auc:.4f}")
    print("\nSzczegółowy raport klasyfikacji:")
    print(classification_report(y_true, preds, target_names=["Brak planety (0)", "Planeta (1)"], zero_division=0))

    return {
        "preds": preds,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def run_training_and_comparison() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("1. Wczytywanie i przygotowywanie danych...")
    data = load_kaggle_dataset()
    X_train, y_train = data["X_train_cnn"], data["y_train"]
    X_val, y_val = data["X_val_cnn"], data["y_val"]
    X_test_cnn, X_test_raw, y_test = data["X_test_cnn"], data["X_test_raw"], data["y_test"]

    print(f"   Trening (po augmentacji): {X_train.shape[0]} próbek (planet: {np.sum(y_train)})")
    print(f"   Walidacja               : {X_val.shape[0]} próbek (planet: {np.sum(y_val)})")
    print(f"   Test                    : {X_test_cnn.shape[0]} próbek (planet: {np.sum(y_test)})")

    # Wyliczenie wag klas dla dodatkowego zbalansowania funkcji straty
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    print("\n2. Budowa i trening sieci 1D-CNN (monitorowanie val_pr_auc)...")
    model = build_cnn_model(input_length=X_train.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history.history)

    # Dobór progu decyzyjnego dla CNN na zbiorze walidacyjnym
    val_probs = model.predict(X_val, verbose=0).ravel()
    cnn_threshold = find_optimal_threshold_by_f1(y_val, val_probs)

    # Predykcja CNN na zbiorze testowym
    if BLS_MAX_EVAL_SAMPLES is not None:
        X_test_cnn = X_test_cnn[:BLS_MAX_EVAL_SAMPLES]
        X_test_raw = X_test_raw[:BLS_MAX_EVAL_SAMPLES]
        y_test = y_test[:BLS_MAX_EVAL_SAMPLES]

    cnn_test_probs = model.predict(X_test_cnn, verbose=0).ravel()
    cnn_metrics = evaluate_predictions("Sieć 1D-CNN", y_test, cnn_test_probs, cnn_threshold)

    print("\n3. Uruchamianie metody odniesienia Box Least Squares (BLS z astropy)...")
    bls_classifier = BLSBaselineClassifier()
    bls_test_scores = bls_classifier.score_dataset(X_test_raw)
    bls_threshold = find_optimal_threshold_by_f1(y_test, bls_test_scores)
    bls_metrics = evaluate_predictions("Box Least Squares (BLS)", y_test, bls_test_scores, bls_threshold)

    print("\n4. Generowanie wykresów porównawczych CNN vs BLS do folderu assets/...")
    plot_confusion_matrices(y_test, cnn_metrics["preds"], bls_metrics["preds"])
    plot_pr_curve_comparison(y_test, cnn_test_probs, bls_test_scores)
    cases_info = plot_cnn_vs_bls_case_studies(
        X_test_raw, y_test, cnn_test_probs, bls_test_scores, bls_classifier
    )

    print("\nGotowe! Wykresy zapisano w katalogu 'assets/':")
    print(" - assets/history.png (Loss, Recall, PR-AUC)")
    print(" - assets/confusion_matrix.png (Porównanie macierzy pomyłek CNN vs BLS)")
    print(" - assets/pr_curve_cnn_vs_bls.png (Krzywa Precision-Recall: CNN vs BLS)")
    print(
        f" - assets/cnn_vs_bls_cases.png (Analiza przypadków: próbki #{cases_info['cnn_wins_idx']}, "
        f"#{cases_info['bls_wins_idx']}, #{cases_info['cnn_fp_idx']})"
    )


if __name__ == "__main__":
    run_training_and_comparison()