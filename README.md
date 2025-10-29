# 🚲 Bicycle Traffic Flow Prediction
**Using Graph Neural Networks for Predicting Campus Cycling Dynamics**

**Author:** R.J.A. van de Kar  
**Programme:** BSc Civil Engineering, Delft University of Technology  
**Department:** Transport & Planning  
**Supervisors:** Prof. dr. R.M.P. Goverde & Prof. dr. M.P. Hagenzieker  
**Date:** October 2025  

---

## 📖 Overview
This repository contains the full implementation and experimental setup for the BSc thesis *“Bicycle Traffic Flow Prediction — Using Graph Neural Networks for Predicting Campus Cycling Dynamics.”*  
The study investigates how accurately bicycle traffic flow on the TU Delft campus can be predicted and what additional value **Graph Neural Networks (GNNs)** provide over classical approaches.

Three models are compared:
1. **Historical Average Baseline**
2. **Linear Regression Model**
3. **Spatio-Temporal Graph Neural Network (ST-GNN)**

The project combines campus bicycle sensor data with contextual information such as weather, exam schedules, and weekdays to forecast edge-level bicycle flows.

---

## 🧩 Repository Structure
```
📁 Bicycle_Traffic_Flow_GNN/
├── .venv/                          # Local virtual environment (not tracked online)
├── artifacts/
│   └── hp_grid_results.csv         # Hyperparameter search results (if present)
├── data/
│   ├── .gitkeep                    # Placeholder; raw CSVs live here (not committed)
│   ├── davis-TUD-EWI_*.csv         # Weather station exports
│   └── smartcameras_line--*.csv    # Camera edge counts (Jan–Apr 2025)
├── models/
│   └── .gitkeep                    # Trained model checkpoints output here
├── src/
│   ├── __init__.py
│   ├── features.py                 # Exogenous features & imputations
│   ├── preprocessing.py            # Parsing, graph, split, datasets
│   ├── models.py                   # ST-GNN and temporal modules
│   ├── training.py                 # Loops, losses, early stopping, loaders
│   └── visualisation.py            # Plots: network, splits, summaries
├── .gitignore
├── LICENSE
├── main.ipynb                      # End-to-end training & evaluation notebook
├── extras.ipynb                    # Side experiments / exploratory analysis
├── README.md                       # You are here
├── requirements.txt                # Full (GPU-tested) environment
└── requirements-cpu.txt            # Curated CPU variant (untested)
```

---

## 🧠 Method Summary

### Data
- **Source:** TU Delft Campus Real Estate & Facility Management (CREFM)  
- **Period:** January–May 2025  
- **Granularity:** typically 15-minute intervals  
- **Additional features:**  
  - Temperature, rainfall, wind (Davis EWI roof station)  
  - Exam/holiday indicators, weekend flag  
  - Lagged flows (24 h, 1 week)

### Models
| Model | Description |
|--------|-------------|
| **Historical Average** | Time-of-day average per edge. |
| **Linear Regression** | Lagged + contextual features. |
| **ST-GNN** | Graph + temporal convolutions for spatial–temporal dependencies. |

### Training
- Framework: **PyTorch**
- Loss: **Masked MAE** (activity-aware option available)
- Optimiser: **Adam**
- Split: **70–15–15 (train–val–test)**, stratified by day type
- Early stopping; optional Laplacian smoothness regulariser.

---

## ⚙️ Installation

### 1) Clone
```bash
git clone https://github.com/RvdKar/Bicycle_Traffic_Flow_GNN.git
cd Bicycle_Traffic_Flow_GNN
```

### 2) Create & activate env
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3) Install deps
GPU (tested):
```bash
pip install -r requirements.txt
```
CPU only (**untested**):
```bash
pip install -r requirements-cpu.txt
```

---

## 🚀 Usage

### A) End-to-end via notebook
```bash
jupyter notebook main.ipynb
```
The notebook walks through: loading data from `data/`, building the campus graph, constructing exogenous features, training models, and evaluating MAE/RMSE. Checkpoint files (if saved) are written to `models/`; figures to `artifacts/`.

### B) Programmatic example
```python
from src import preprocessing, features, training
from src.training import build_gnn

# 1) Load data + config
cfg = preprocessing.Config()  # set paths/granularity as needed
times, X, M, weather_df, A_hat, A_bin, masks, slices = preprocessing.load_all(cfg)

# 2) Build exogenous features
exog = features.add_exogenous(times=times, E=X.shape[1], config=cfg,
                              weather_df=weather_df, X=X, M=M, A_bin=A_bin)

# 3) Gap-aware loaders
loaders = training.make_loaders_gapaware(times, X, M, exog, cfg, masks)

# 4) Model + train
model = build_gnn(E=X.shape[1], Fin=loaders['train'].dataset.F_in,
                  Hout=cfg.H_out, A_hat=A_hat, cfg=cfg)
model = training.train_model(model, loaders, cfg, A_bin=A_bin)

# 5) Evaluate
metrics = training.evaluate(model, loaders['test'], device=cfg.device)
print(metrics)
```

> **Notes**
> - Paths and file names are configured inside `src/preprocessing.py` (or via `Config`).  
> - For CPU runs, set `cfg.device = "cpu"`.

---

## 📊 Results (thesis summary)
| Model | Mean Absolute Error (MAE) | Improvement vs baseline |
|------|----------------------------|--------------------------|
| Historical Average | baseline | – |
| Linear Regression | ↓ modest | ≈ +12.4% |
| ST-GNN | **↓ substantial** | **≈ +30.1%** |

---

## 🧩 Reproducibility
- Code and hyperparameters are included.  
- Day-level stratified splitting ensures balanced weekends/holidays/exams.  
- Random seeds are fixed where applicable.  
- Notebooks can regenerate results from raw CSVs placed in `data/`.

---

## 📚 Citation
If you use this repository or code, please cite:

```
@bachelorsthesis{vandeKar2025,
  author    = {R.J.A. van de Kar},
  title     = {Bicycle Traffic Flow Prediction: Using Graph Neural Networks for Predicting Campus Cycling Dynamics},
  school    = {Delft University of Technology},
  year      = {2025},
  department= {Transport \& Planning}
}
```

---

## 🧾 License
This project is released under the **MIT License**. See `LICENSE` for details.

---

## 💬 Acknowledgements
- TU Delft **Campus Real Estate & Facility Management (CREFM)** for sensor data  
- **Prof. dr. R.M.P. Goverde** and **Prof. dr. M.P. Hagenzieker** for supervision  
- **Dr.ir. W. Daamen**, **Drs.ing. I.L. Oostlander-Çetin**, and **Mingze Gong** for input and data support
