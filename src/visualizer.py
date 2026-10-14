import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class Visualizer:
    def __init__(self):
        print("[INFO] Initializing Visualizer...")

    def plot_history(self, history):
        print("[INFO] Plotting training history...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Wykres funkcji straty (Loss)
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Wykres skuteczności (Accuracy)
        ax2.plot(history.history['accuracy'], label='Train Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred_probs):
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Planet', 'Planet'])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Confusion Matrix')
        plt.show()

    def plot_light_curve(self, curve_data, is_planet):
        title = "Exoplanet Transit Signal" if is_planet == 1 else "Normal Star Noise"
        plt.figure(figsize=(10, 4))
        plt.plot(curve_data, color='black', linewidth=0.5)
        plt.title(title)
        plt.xlabel("Time (Observations)")
        plt.ylabel("Light Flux")
        plt.show()