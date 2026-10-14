# Exoplanet Detection from Kepler Light Curves

A machine learning pipeline for detecting exoplanet transits from highly imbalanced space telescope data, featuring signal smoothing and a 1D Convolutional Neural Network.

The project addresses the "accuracy paradox" in astronomical data. Since over 99% of stars do not show transits, a naive model can achieve 99% accuracy simply by guessing "no planet". This project focuses on signal processing (Savitzky-Golay filter) and synthetic data generation (SMOTE) to prioritize True Positives (actual planet detections) over raw baseline accuracy.

## How to run

### 1. Install the tools
You need three programs installed on your computer:
- **Python 3** from python.org. On Windows, tick the option "Add Python to PATH" in the installer.
- **Git** from git-scm.com.
- **Visual Studio Code** from code.visualstudio.com. Optionally, install the "Python" extension from the Extensions tab (the square icon on the left side).

### 2. Download the project
1. Choose a place on your computer where the project should be saved, for example the Desktop.
2. Open a terminal there. On Windows, you can right-click an empty space in the folder and choose **Open in Terminal**, or open PowerShell and move to the folder with `cd Desktop`.
3. Type `git clone https://github.com/StanekJan02/exoplanet_detector.git` and press Enter. 

Git creates a new folder named `exoplanet_detector` with all the project files inside.

### 3. Open the project in VS Code
1. Open Visual Studio Code.
2. In the top menu click **File**, then **Open Folder**.
3. Select the `exoplanet_detector` folder and confirm.

The project files (`data`, `src`, `README.md` and others) now appear in the Explorer panel on the left.

### 4. Open the terminal in VS Code
In the top menu click **Terminal**, then **New Terminal**. A terminal panel opens at the bottom of the window. The path shown in it should end with `exoplanet_detector`, which means you are in the project folder and can type commands there.

### 5. Install the dependencies
In the terminal type `pip install -r requirements.txt` and press Enter. This installs `tensorflow`, `scipy`, `scikit-learn`, `imblearn`, `pandas`, and `matplotlib`. Wait until the installation finishes and the cursor appears again.

### 6. Run the simulation
In the same terminal type `python src/trainer.py` and press Enter. If Windows says that python is not recognized, use `py src/trainer.py` instead.

The script will process the data, train the 1D CNN, and apply Early Stopping if the validation loss plateaus. When it finishes, a window with Matplotlib charts (Loss/Accuracy history, Confusion Matrix, and Signal plots) will open.

## How it works

The pipeline transforms raw, noisy light flux measurements into a clean binary classification (Planet / No Planet).

1. **Data preparation:** The `LABEL` column is mapped from its original format (1, 2) to a standard binary format (0, 1). 
2. **Signal Smoothing:** A Savitzky-Golay filter (window length 15, polynomial order 3) is applied to the raw signal. This step is critical because it reduces high-frequency background cosmic noise while preserving the depth and sharp edges of actual transit valleys.
3. **Scaling:** A `StandardScaler` centers the light flux data around zero.
4. **Tenant split / Balancing:** `SMOTE` (Synthetic Minority Over-sampling Technique) is applied **only** to the training data to balance the classes and prevent the network from ignoring the minority class (planets).
5. **Local training (1D CNN):** The network uses two `Conv1D` blocks with `MaxPooling1D` to extract transit features, followed by a `Flatten` and `Dense` layers with `Dropout(0.2)` to prevent overfitting.
6. **Evaluation:** The model is evaluated on a completely unseen, naturally imbalanced test set to expose its real-world sensitivity via a Confusion Matrix.

## Dataset

"Exoplanet Hunting in Deep Space" dataset from Kaggle. It contains light curves from the Kepler Space Observatory. Each record represents continuous light flux measurements for a single star over time (3197 data points per star). The target is whether the star has a confirmed exoplanet transit.
The dataset is split into `exoTrain.csv` (5087 records) and `exoTest.csv` (570 records) and should be placed in the `data/` directory.

## Results

- **Baseline (always predict "no planet"):** ~99.1% overall accuracy.
- Without signal smoothing, the base 1D CNN caught only 1 out of 5 actual planets (True Positives) in the test set, losing the transit signature in the noise.
- Implementing the Savitzky-Golay filter doubled the model's sensitivity, capturing 2 out of 5 planets.
- The trade-off for higher sensitivity is a slight increase in False Positives (predicting a planet where there is none). In astronomical pipelines, this is the preferred behavior, as it is cheaper to manually verify a few false alarms than to permanently miss a new exoplanet.

## Limitations

- The model still misses very faint or rapid transits (3 False Negatives remain).
- The Savitzky-Golay parameters (window=15, poly=3) are static. Dynamic window sizing based on individual stellar properties might yield better noise reduction.
- Results are evaluated on a single dataset split provided by Kaggle.
- A relatively shallow CNN architecture was used to prioritize training speed; deeper architectures with residual connections might capture more complex transit patterns.

## Project structure

```text
├── data/
│   ├── exoTrain.csv       # training dataset
│   └── exoTest.csv        # separate test dataset
├── src/
│   ├── config.py          # hyperparameters and paths
│   ├── data_processor.py  # ETL, SMOTE, and smoothing logic
│   ├── model_builder.py   # 1D CNN architecture
│   ├── trainer.py         # training pipeline and early stopping
│   └── visualizer.py      # matplotlib charts
├── requirements.txt
└── README.md