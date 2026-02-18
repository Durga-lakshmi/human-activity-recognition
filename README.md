
# HOW TO RUN - Human Activity Recognition

This project supports **standard training** and **Optuna-based hyperparameter optimization**.

All experiment variants are controlled via Hydra configuration files.

---

### 0. Environment Setup !!

Create and activate a virtual environment, then install all required dependencies:

```bash
chmod +x [setup.sh]
./setup.sh

```

> **Important:**
> 
> 
> The virtual environment must be activated before running Optuna-based hyperparameter optimization.
> 

---

### 1. Standard Training

For standard training without hyperparameter optimization, run:

To switch between different datasets or experimental settings (e.g. **HAPT** and **RealWorld**), specify the configuration name:
Use config_name = default for HAPT
Use config_name = default_RW for RealWorldHAR

```bash
python3 main.py --config-name=<config_name>   config_name: default / default_RW

python3 main.py --config-name=default
python3 main.py --config-name=default_RW
```

Each configuration defines the dataset, preprocessing pipeline, model architecture, and training parameters.

---

### 2. Hyperparameter Optimization with Optuna

### 2.1 Run Optuna Search

Make sure the virtual environment is activated, then start the Optuna search:

```bash
./venv/bin/python runner_optuna.py
```

This script performs multiple Optuna trials by sampling different hyperparameter combinations and evaluating them on the validation set.

---

### 2.2 Best Hyperparameter Configuration

After the search completes, the best-performing trial is saved as a YAML file, for example:

```
optuna_log/best_optuna_params_16.yaml
```

This file stores the optimal hyperparameters found by Optuna and serves as the selected best configuration.

---

### 2.3 Train with the Best Configuration

To retrain the model using the optimized hyperparameters, run:

```bash
./venv/bin/python run_optuna_best.py
```

This script loads the base configuration and **overrides it with the values from `best_optuna_params_16.yaml`**, then launches a standard training run with the optimized setup.

---

## 3. Test the specified model path

### 3.1 RealWorkd HAR
```bash
python3 eval.py --config-name=default_RW
```
Please find the path files of the best models we trained inside : dl-lab-25w-team03/human_activity/artifacts/models

### 3.1 HAPT
```bash
python3 eval.py --config-name=default 
```
**IMPORTANT:**
HAPT : change some parts in default.yaml
1. check normal model **cnn_tcn**

```yaml
- check_modus: true  
- check_path: <your_check_path_after_training>
```

2. check model **cnn_tcn after using check_optuna**

```yaml
- check_modus: true  
- check_optuna: true  
- check_path: <your_check_path_after_training_HO_with_Optuna>
```



### Notes

- Standard training and hyperparameter optimization are intentionally decoupled.
- Dataset and experiment variants (e.g. HAPT vs. RealWorld) are selected via `-config-name`.
- All dependencies are managed via `requirements.txt`; no manual package installation is required.
- RealWorld HAR :  To view model artifacts and plots such as class distributions, signals, confusion matrix, etc, please navigate to : dl-lab-25w-team03/human_activity/artifacts/plots/HAR

---
## Human Activity Recognition
| Model        | Dataset   | Accuracy (%) | F1-score (%) | Precision (%) | Recall (%) | 
| -------------| --------- | ------------ | ------------ | ------------- | ---------- |
| CNN–TCN      | HAPT      | 96.96        | 91.52        | 92.96         | 90.36      | 
| CNN + Fusion | RealWorld | 93.41        | 93.97        | 94.13         | 93.96      | 

