# Team 03
- Durga Lakshmi Sajith (st192588)
- Lingzhi Wu (st196114)
  
---
# HOW TO RUN - Diabetic Retinopathy 
This project uses **Hydra configurations** to switch between different DR experiment variants.

Different functionalities are activated by specifying `--config-name` in the command line.

All training-related experiments are launched via `main.py`.

### 1. Custom Architecture - CNN
1.1**Default Configurations** (Please use these configurations in default.yaml)

-------------------------------------------
```yaml
  defaults:
    - dataset: idrid
    - model: small_drnet
  task : 2c
```
-------------------------------------------

1.2 Run the command
```python
python3 main.py --config-name=default
```
Note : To view model artifacts and plots such as class distributions, confusion matrix, please navigate to : dl-lab-25w-team03/diabetic_retinopathy/artifacts/images.
       Please find the path files of the best models we trained inside : dl-lab-25w-team03/diabetic_retinopathy/artifacts/models
       
-------------------------
### 2. Transfer Learning for Multiclass Classification
2.1 **Default Configurations** (Please use these configurations in default.yaml)  

-------------------------------------------
```yaml
defaults:
  - dataset: eyepacs
  - model: resnet_multiclass
task : 5c
```
-------------------------------------------
2.2 Run the command
```python
python3 main.py --config-name=default
```
To evaluate the multiclass model on IDRID dataset use this configuration

  -------------------------------------------
  ```yaml
  defaults:
  - dataset: eyepacs
  - model: resnet_multiclass
  task : 5c
  ```
  -------------------------------------------
  
  Run this command
  ```python
  python3 eval.py --config-name=default
  ```

Note : To view model artifacts and plots such as class distributions, confusion matrix, please navigate to : dl-lab-25w-team03/diabetic_retinopathy/artifacts/images.
       Please find the path files of the best models we trained inside : dl-lab-25w-team03/diabetic_retinopathy/artifacts/models  
       
---
### 3. Ensemble learning
**Dataset: IDRiD**  
**Models: DenseNet-121, ConvNeXt, EfficientNet**
#### 3.1 Individual Model

**Two-class classification**

```bash
python3 main.py --config-name=default_2c
```

**Two-class with k-fold cross-validation**

```bash
python3 main.py --config-name=default_2c_kfold
```

**Five-class classification**

```bash
python3 main.py --config-name=default_5c
```

#### 3.2 Feature-level Ensemble 
**IMPORTANT：If want to run feature-level ensemble learning, muss change this part to use the trained models**

```yaml
ensemble_models:
- { cfg: "dense121.yaml", ckpt: <dense121_check_path> }
- { cfg: "convnext.yaml", ckpt: <convnext_check_path> }
- { cfg: "efficientnet.yaml", ckpt: <efficientnet_check_path> }
```

**REMENBER:** In Binary/Multiclasses classification use the corresponding paths   
**Two-class with feature-level ensemble**

```bash
python3 main.py --config-name=default_2c_feature_ensemble
```

**Five-class with feature-level ensemble**

```bash
python3 main.py --config-name=default_5c_feature_ensemble
```

Each configuration controls the training setup (number of classes, cross-validation, and feature-level ensembling) through predefined Hydra config files.

---
#### 3.3 Evaluation: Individual Model + Feature-level Ensemble Model

```bash
python3 test_evaluation.py --config-name=default_2c / default_5c / default_2c_feature_ensemble / default_5c_feature_ensemble
```
**IMPORTANT:change some parts in defaults**

```yaml
- test:  
    check_mode: false -> true  
    check_path: <your_check_path_after_training>
```

  
**If want to do deep visualization (ONLY for 2-class):**


```yaml
- deep_viz:  
   enable: false -> true
```


#### 3.4 Output-level Ensemble (Evaluation Only)

Output-level ensemble is performed **after training**, using a dedicated evaluation script.

**IMPORTANT：If want to run output-level ensemble learning, muss change this part to use the trained models**

```yaml
ensemble_models:
- { cfg: "dense121.yaml", ckpt: <dense121_check_path> }
- { cfg: "convnext.yaml", ckpt: <convnext_check_path> }
- { cfg: "efficientnet.yaml", ckpt: <efficientnet_check_path> }
```

**REMENBER:** In Binary/Multiclasses classification use the corresponding paths     
**Two-class output-level ensemble**

```bash
python3 test_evaluation.py --config-name=default_2c_output_ensemble
```

**Five-class output-level ensemble**

```bash
python3 test_evaluation.py --config-name=default_5c_output_ensemble
```

This step aggregates predictions from multiple trained models at the output level without additional training.

---

### 4. Notes

- All experiment variants are controlled purely by configuration files; no code modification is required.
- Feature-level ensemble is handled during training (`main.py`).
- Output-level ensemble is implemented as a post-hoc evaluation procedure (`test_evaluation.py`).


  
---


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
# RESULTS
## Diabetic Retinopathy
### Custom Architecture and Transfer Learning for Multiclass Classification
| Model           | Training Dataset |   Test Dataset    |Test Accuracy    | Classification Type | Training Strategy |
|-----------------|------------------|-------------------|-----------------|---------------------|--------------------
| small_drnet     | IDRID            |      IDRID        |     79.8%       |       Binary        |Train from Scratch |
| ConvexNet       | IDRID            |      IDRID        |    87. 38%      |       Binary        |Transfer Learning  |            
| Resnet18        | EyePACS          |      IDRID        |    56.21%       |      Multiclass     |Transfer Learning  |           
| Resnet18        | EyePACS          |      EyePACS      |    78.25%       |      Multiclass     |Transfer Learning  | 

---
### Ensemble Learning
Binary | **IDRiD**
| Model / Strategy           | Accuracy (%) | AUC (%) | F1-score (%) | Precision (%) | Recall (%) | 
| -------------------------- | ------------ | ------- | ------------ | ------------- | ---------- |
| DenseNet-121               | 82.52        | 93.15   | 86.96        | 81.08         | 93.75      |
| ConvNeXt                   | **88.35**    | 92.33   | **90.16**    | **94.83**     | 85.94      | 
| EfficientNet               | 83.50        | 88.94   | 86.18        | 89.83         | 82.81      | 
| Feature-level Ensemble     | 84.47        | 91.09   | 86.89        | 91.38         | 82.81      | 
| Output Ensemble (Prob Avg) | 85.44        | 92.91   | 84.74        | 84.37         | 85.28      | 
| Output Ensemble (Others*)  | 84.47        | 92.67   | 83.79        | 83.37         | 84.50      |

Multiclass | **IDRiD**
| Model / Strategy           | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | QWK (%)   |
| -------------------------- | ------------ | ---------------- | ------------ | --------- |
| DenseNet-121               | **58.25**    | **45.42**        | **45.12**    | 69.68     |
| ConvNeXt                   | 54.37        | 40.88            | 40.97        | **70.74** |
| EfficientNet               | 54.37        | 41.97            | 41.47        | 65.68     |
| Feature-level Ensemble     | 57.28        | 42.10            | 41.39        | 67.57     |
| Output Ensemble (Prob Avg) | 52.43        | 39.44            | 38.96        | 66.63     |
| Output Ensemble (Others*)  | 54.37        | 40.62            | 39.98        | 68.54     | 

---
## Human Activity Recognition
| Model        | Dataset   | Accuracy (%) | F1-score (%) | Precision (%) | Recall (%) | 
| -------------| --------- | ------------ | ------------ | ------------- | ---------- |
| CNN–TCN      | HAPT      | 96.96        | 91.52        | 92.96         | 90.36      | 
| CNN + Fusion | RealWorld | 93.41        | 93.97        | 94.13         | 93.96      | 

