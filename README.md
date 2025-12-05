# WIVI: Multimodal Activity Recognition using CSI and Images

This project implements a multimodal deep learning system for human activity recognition, combining **Channel State Information (CSI)** from WiFi signals and **Visual Images** from a camera.

## 📂 Project Structure

```
wivi_project/
├── data/                  # Dataset folder
│   └── data_activity/     # Organized by class (Dung, Ngoi, etc.)
├── notebooks/             # Jupyter Notebooks for training and analysis
│   └── multimodal_activity_recognition.ipynb # Main project notebook
├── scripts/               # Utility and maintenance scripts
│   ├── update_nb.py       # Tools to patch/update notebooks
│   └── ...
├── outputs/               # Training outputs (Models, Plots, Logs)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- VS Code (recommended) or Jupyter Lab
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FatCatCute/WIVI.git
    cd WIVI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Model Architecture

The system uses a **Hybrid Fusion Model**:
1.  **CSI Branch**: A **Temporal Convolutional Network (TCN)** processes the time-series WiFi signal data to extract temporal features.
2.  **Image Branch**: A **ResNet50** backbone extracts spatial features from the corresponding video frames.
3.  **Fusion Layer**: Features from both branches are concatenated and passed through a classifier to predict the activity.

## 📊 Dataset

The dataset is expected to be in `data/data_activity/` with the following structure:
- Each activity class has its own folder (e.g., `Dung`, `Ngoi`).
- Inside each class folder:
    - `csi/`: Contains `.csv` files with CSI data.
    - `images/`: Contains corresponding image files.

## 🛠 Usage

### Option 1: Using Jupyter Notebook
1.  Open `notebooks/multimodal_activity_recognition.ipynb` in VS Code.
2.  Select your Python kernel.
3.  Run the cells sequentially.

### Option 2: Using Python Scripts (Recommended for Training)
You can run the training directly from the terminal:

```bash
# Run training with default settings (Fusion mode)
python -m src.train
```

The script will automatically discover classes in `data/data_activity`, train the model, and save the best weights to `outputs/models/`.

## 📈 Results

Training logs, best model weights, and evaluation plots (Accuracy, Confusion Matrix) will be saved in the `outputs/` directory.

## 👥 Contributors

- **Ngoc TTM** - *Initial Work*
