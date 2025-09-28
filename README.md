# Synthetic Wi-Fi Fingerprint Generation with Diffusion Models
This repository contains the implementation for the paper "Synthetic Wi-Fi fingerprint generation using diffusion probabilistic model". The project addresses the resource-intensive challenge of collecting Wi-Fi fingerprint data for indoor localization systems by generating synthetic RSSI measurements using conditional denoising diffusion probabilistic models (cDDPM).
## Installation
```
git clone https://github.com/ImokAAA/diff-rss.git
cd diff-rss
pip install -r requirements.txt
```
## Requirements
Python 3.8+

PyTorch 2.0+

scikit-learn

pandas

numpy

matplotlib

## Usage
### 1. Data Preparation
Place your Wi-Fi fingerprint dataset in CSV format in the data/ directory. The dataset should contain:

RSSI values from multiple access points (columns: AP001, AP002, ...)

Spatial coordinates (columns: X, Y)

### 2. Training the Diffusion Model
```
from model import TableDiffusionModel, train_table_diffusion

# Initialize model
model = TableDiffusionModel(rss_dim=31, cond_dim=2, time_emb_dim=128)

# Train model
trained_model = train_table_diffusion(
    model=model,
    X_cond=X_train_cond_t,
    y_rss=y_train_rss_t,
    diffusion_steps=100,
    batch_size=32,
    n_epochs=200,
    lr=0.0006
)
```
### 3. Generating Synthetic Fingerprints
```
from model import sample_table_diffusion

# Generate synthetic RSSI data
synthetic_rss = sample_table_diffusion(
    model=trained_model,
    X_cond=X_target_locations,
    diffusion_steps=100
)
```
### 4. Evaluating Localization Performance
```
from evaluation import evaluate_localization

# Evaluate KNN performance
results = evaluate_localization(
    real_rss=rss_original,
    synthetic_rss=synthetic_rss,
    locations=X_knn
)

print(f"Original Data Error: {results['original_error']:.4f} m")
print(f"Synthetic Data Error: {results['synthetic_error']:.4f} m")
print(f"Combined Data Error: {results['combined_error']:.4f} m")
```
## Results

Our diffusion model significantly improves localization accuracy, especially in data-scarce scenarios. Key findings from our experiments:

| Real Data % | Real Only Error (m) | Real+Synthetic Error (m) | Improvement |
|-------------|---------------------|--------------------------|-------------|
| 10%         | 2.85                | 2.21                     | 22.5%       |
| 30%         | 1.92                | 1.58                     | 17.7%       |
| 50%         | 1.45                | 1.23                     | 15.2%       |
| 100%        | 0.98                | -                        | -           |
## Dataset
The model was evaluated on two public datasets:

### WiFi RSSI Indoor Localization Dataset (IEEE DataPort)

345 reference points in a 21m × 16m corridor

50 RSSI measurements per point

Robot localization accuracy: 0.07m ± 0.02m

### JUIndoorLoc Dataset (GitHub)

1000+ location points across 3 floors

1m × 1m grid structure

172 access points
## Citation
If you use this work in your research, please cite:

bibtex
@inproceedings{zhumangali2025synthetic,
  title={Synthetic Wi-Fi fingerprint generation using diffusion probabilistic model},
  author={Zhumangali, Imangali and Seytnazarov, Shinnazar},
  year={2025}
}


#################################

1. ADD self attention layers to Table diffursion model
2. Mean VS median for AP
3. Try imputers
4. Optuna
5. Diffusion models for tabular dataset
6. Null values how it was handled

Try to scale to MinMax and set the null values to 0,

Add mode

Read the paper 18

Dataset 2: https://www.kaggle.com/datasets/priyaroycse/juindoorloc-wifi-fingerprint-indoor-localization
