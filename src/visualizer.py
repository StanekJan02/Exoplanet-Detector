import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class Visualizer:
    def __init__(self):
        print("[INFO] Initializing Visualizer...")
        self.bg_color = '#0d1117'       
        self.panel_color = '#161b22'    
        self.text_color = '#c9d1d9'     
        self.grid_color = '#30363d'    
        self.purple_main = '#a256fa'   
        self.gray_line = '#8b949e'      

    def _apply_theme(self, fig, ax):
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.panel_color)
        ax.tick_params(colors=self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)
        for spine in ax.spines.values():
            spine.set_color(self.grid_color)

    def plot_history(self, history):
        print("[INFO] Plotting training history...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['loss'], label='Train Loss', color=self.purple_main, linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', color=self.gray_line, linewidth=2)
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        self._apply_theme(fig, ax1)
        ax1.grid(True, color=self.grid_color, linestyle='--', alpha=0.7)
        legend = ax1.legend(facecolor=self.panel_color, edgecolor=self.grid_color)
        for text in legend.get_texts(): text.set_color(self.text_color)
        
        # Wykres Accuracy
        ax2.plot(history.history['accuracy'], label='Train Accuracy', color=self.purple_main, linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color=self.gray_line, linewidth=2)
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        self._apply_theme(fig, ax2)
        ax2.grid(True, color=self.grid_color, linestyle='--', alpha=0.7)
        legend = ax2.legend(facecolor=self.panel_color, edgecolor=self.grid_color)
        for text in legend.get_texts(): text.set_color(self.text_color)
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred_probs):
        print("[INFO] Plotting confusion matrix...")
        y_pred = (y_pred_probs > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_purple", [self.panel_color, self.purple_main])
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Planet', 'Planet'])
        disp.plot(cmap=cmap, ax=ax, text_kw={'color': 'white'})
        
        self._apply_theme(fig, ax)
        ax.set_title('Confusion Matrix')
        
        if disp.im_.colorbar:
            disp.im_.colorbar.ax.yaxis.set_tick_params(color=self.text_color, labelcolor=self.text_color)
        
        plt.show()

    def plot_true_positives(self, tp_curves):
        print(f"[INFO] Plotting {len(tp_curves)} detected exoplanets on a single grid...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.patch.set_facecolor(self.bg_color)
        
        axes = axes.flatten()
        
        for i, curve in enumerate(tp_curves):
            if i >= 4: 
                break
            ax = axes[i]
            ax.plot(curve, color=self.purple_main, linewidth=0.8)
            ax.set_title(f"Detected Exoplanet #{i+1}", color=self.text_color)
            ax.set_xlabel("Time (Observations)", color=self.text_color)
            ax.set_ylabel("Normalized Flux", color=self.text_color)
            self._apply_theme(fig, ax)
            ax.grid(True, color=self.grid_color, linestyle='--', alpha=0.5)

        for j in range(len(tp_curves), 4):
            fig.delaxes(axes[j])

        fig.suptitle("Successfully Detected Exoplanet Transits (True Positives)", color=self.text_color, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()