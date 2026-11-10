import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

from src.config import ASSETS_DIR


def plot_training_history(history: dict, save_path=ASSETS_DIR / "history.png") -> None:
    """Generuje wykres przebiegu uczenia sieci: Loss, Recall oraz PR-AUC (zamiast samego Accuracy)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Funkcja straty (Loss)
    axes[0].plot(history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linestyle="--", linewidth=2)
    axes[0].set_title("Funkcja straty (Binary Crossentropy)")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Recall (Czułość)
    axes[1].plot(history["recall"], label="Train Recall", linewidth=2, color="tab:green")
    axes[1].plot(history["val_recall"], label="Val Recall", linestyle="--", linewidth=2, color="tab:olive")
    axes[1].set_title("Recall (Czułość detekcji planet)")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Recall")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. PR-AUC (Precision-Recall AUC)
    axes[2].plot(history["pr_auc"], label="Train PR-AUC", linewidth=2, color="tab:purple")
    axes[2].plot(history["val_pr_auc"], label="Val PR-AUC", linestyle="--", linewidth=2, color="tab:pink")
    axes[2].set_title("PR-AUC (Precision-Recall AUC)")
    axes[2].set_xlabel("Epoka")
    axes[2].set_ylabel("PR-AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(
    y_true: np.ndarray,
    cnn_preds: np.ndarray,
    bls_preds: np.ndarray,
    save_path=ASSETS_DIR / "confusion_matrix.png",
) -> None:
    """Porównuje macierze pomyłek dla sieci 1D-CNN oraz klasycznej metody BLS."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, preds, title in zip(
        axes,
        [cnn_preds, bls_preds],
        ["Sieć 1D-CNN", "Klasyczny Box Least Squares (BLS)"],
    ):
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Brak planety (0)", "Planeta (1)"],
            yticklabels=["Brak planety (0)", "Planeta (1)"],
            ax=ax,
        )
        ax.set_title(f"Macierz pomyłek: {title}")
        ax.set_xlabel("Predykcja modelu")
        ax.set_ylabel("Rzeczywista klasa")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curve_comparison(
    y_true: np.ndarray,
    cnn_probs: np.ndarray,
    bls_scores: np.ndarray,
    save_path=ASSETS_DIR / "pr_curve_cnn_vs_bls.png",
) -> None:
    """
    Rysuje wspólny wykres krzywej Precision-Recall (PR-AUC) dla sieci CNN oraz algorytmu BLS.
    """
    prec_cnn, rec_cnn, _ = precision_recall_curve(y_true, cnn_probs)
    ap_cnn = average_precision_score(y_true, cnn_probs)

    prec_bls, rec_bls, _ = precision_recall_curve(y_true, bls_scores)
    ap_bls = average_precision_score(y_true, bls_scores)

    baseline_rate = float(np.mean(y_true))

    plt.figure(figsize=(8, 6))
    plt.plot(rec_cnn, prec_cnn, label=f"Sieć 1D-CNN (PR-AUC = {ap_cnn:.3f})", linewidth=2.5, color="tab:blue")
    plt.plot(
        rec_bls,
        prec_bls,
        label=f"Box Least Squares - BLS (PR-AUC = {ap_bls:.3f})",
        linewidth=2.5,
        linestyle="--",
        color="tab:orange",
    )
    plt.axhline(
        baseline_rate,
        color="gray",
        linestyle=":",
        label=f"Losowy klasyfikator (udział planet = {baseline_rate:.4f})",
    )
    plt.xlabel("Recall (Czułość - odsetek wykrytych planet)")
    plt.ylabel("Precision (Precyzja)")
    plt.title("Krzywa Precision-Recall: 1D-CNN vs Box Least Squares (BLS)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cnn_vs_bls_case_studies(
    X_raw: np.ndarray,
    y_true: np.ndarray,
    cnn_probs: np.ndarray,
    bls_scores: np.ndarray,
    bls_classifier,
    save_path=ASSETS_DIR / "cnn_vs_bls_cases.png",
) -> dict[str, int]:
    """
    Wizualizuje kluczowy element projektu: konkretne krzywe blasku, gdzie:
    1) CNN wygrywa z BLS (prawdziwa planeta z wysokim prob CNN i niskim SDE w BLS, np. przez zafalowane tło/kształt V),
    2) BLS wygrywa z CNN (prawdziwa planeta lub silny okresowy sygnał, gdzie składanie fazowe BLS daje wysoki wynik,
       lub przypadek, gdzie CNN daje False Positive na pojedynczym artefakcie instrumentalnym).
    """
    # Rangowanie wyników do przedziału [0, 1] w celu łatwego porównania względnej pewności obu metod
    cnn_rank = np.argsort(np.argsort(cnn_probs)) / max(len(cnn_probs) - 1, 1)
    bls_rank = np.argsort(np.argsort(bls_scores)) / max(len(bls_scores) - 1, 1)

    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]

    # Przypadek 1: Prawdziwa planeta, gdzie CNN radzi sobie najlepiej względem BLS
    if len(pos_indices) > 0:
        diff_cnn_adv = cnn_rank[pos_indices] - bls_rank[pos_indices]
        idx_cnn_wins = int(pos_indices[np.argmax(diff_cnn_adv)])
        idx_bls_wins = int(pos_indices[np.argmax(-diff_cnn_adv)])
    else:
        idx_cnn_wins = 0
        idx_bls_wins = min(1, len(X_raw) - 1)

    # Przypadek 3: Fałszywy alarm w CNN na pojedynczej anomalii, poprawnie odrzucony przez BLS
    if len(neg_indices) > 0:
        fp_diff = cnn_rank[neg_indices] - bls_rank[neg_indices]
        idx_cnn_fp = int(neg_indices[np.argmax(fp_diff)])
    else:
        idx_cnn_fp = 0

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))

    cases = [
        (
            idx_cnn_wins,
            "1. CNN wygrywa z BLS (Prawdziwa planeta: nieregularne tło / profil V)",
        ),
        (
            idx_bls_wins,
            "2. BLS wygrywa z CNN (Prawdziwa planeta: płytki, ściśle okresowy sygnał)",
        ),
        (
            idx_cnn_fp,
            "3. BLS odporny na pojedynczy artefakt (Brak planety: False Positive w CNN, niskie SDE w BLS)",
        ),
    ]

    for row_idx, (sample_idx, title) in enumerate(fragment := cases):
        bls_res = bls_classifier.analyze_single_curve(X_raw[sample_idx])
        time = bls_res["time"]
        flux_norm = bls_res["flux_norm"]
        model_curve = bls_res["bls_model_curve"]

        # Lewa kolumna: Krzywa blasku + dopasowane pudełko BLS
        ax_lc = axes[row_idx, 0]
        ax_lc.plot(time, flux_norm, color="black", alpha=0.6, linewidth=0.8, label="Krzywa blasku")
        ax_lc.plot(
            time,
            model_curve,
            color="tab:red",
            linewidth=1.5,
            label=f"Model BLS (P={bls_res['best_period']:.2f} d)",
        )
        ax_lc.set_title(
            f"{title}\n[Próbka #{sample_idx} | P_CNN = {cnn_probs[sample_idx]:.3f} | BLS SDE = {bls_scores[sample_idx]:.2f}]"
        )
        ax_lc.set_xlabel("Czas [dni]")
        ax_lc.set_ylabel("Znormalizowany strumień")
        ax_lc.legend(loc="lower right")
        ax_lc.grid(True, alpha=0.3)

        # Prawa kolumna: Periodogram BLS
        ax_per = axes[row_idx, 1]
        ax_per.plot(bls_res["periods"], bls_res["power"], color="tab:orange", linewidth=1.0)
        ax_per.axvline(
            bls_res["best_period"],
            color="tab:red",
            linestyle="--",
            label=f"Najlepszy okres: {bls_res['best_period']:.2f} d",
        )
        ax_per.set_title(f"Periodogram BLS (SDE = {bls_res['sde']:.2f})")
        ax_per.set_xlabel("Okres próbny [dni]")
        ax_per.set_ylabel("Moc sygnału BLS")
        ax_per.legend(loc="upper right")
        ax_per.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "cnn_wins_idx": idx_cnn_wins,
        "bls_wins_idx": idx_bls_wins,
        "cnn_fp_idx": idx_cnn_fp,
    }