from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from config import config
from data_processor import DataProcessor
from model_builder import ExoplanetCNN
from visualizer import Visualizer
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import numpy as np
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
class ModelTrainer:
    def __init__(self):
        self.processor = DataProcessor()
        self.builder = ExoplanetCNN()

    def run_training_pipeline(self):
        train_df, test_df = self.processor.load_data()
        X_train, y_train, X_test, y_test = self.processor.process_data(train_df, test_df)

        model = self.builder.build_model()

        optimizer = Adam(learning_rate=config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )

        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5,         
            restore_best_weights=True 
        )

        print(f"[INFO] Starting training for max {config.EPOCHS} epochs...")
        history = model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_split=0.2, 
            callbacks=[early_stopping],
            verbose=1 
        )

        print("[INFO] Evaluating on unseen test data...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n[RESULT] Test Accuracy: {test_accuracy * 100:.2f}%")
        visualizer = Visualizer()
        visualizer.plot_history(history)
        y_pred_probs = model.predict(X_test)
        visualizer.plot_confusion_matrix(y_test, y_pred_probs)
        visualizer.plot_light_curve(X_train[0], y_train[0])
        return model, history

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training_pipeline()