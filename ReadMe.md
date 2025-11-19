# README.md

# **Optimizing Machine Learning Classification of Narcolepsy Type 1 Using the Stanford Cataplexy Questionnaire and HLA-DQB1*06:02 Biomarker**

## **Abstract**

Narcolepsy type 1 (NT1) remains critically underdiagnosed due to its rarity, complexity of early symptoms, and limited access to confirmatory testing. Standard approaches such as polysomnography, MSLT, and CSF hypocretin are accurate but costly, invasive, and geographically inaccessible for large population screening.
This work presents a scalable machine-learning framework based on a structured cataplexy questionnaire and minimal genetic information (HLA-DQB1*06:02). Using nested cross-validation, Optuna hyperparameter optimization, and robust thresholding strategies, our models achieve high specificity and clinically meaningful sensitivity across multiple feature sets.
We introduce a biologically grounded **Veto Rule** that corrects non-HLA false positives and markedly improves specificity without degrading sensitivity. We additionally evaluate model behavior across realistic prevalence settings to assess feasibility for population-level screening.
The resulting framework enables practical, interpretable, and scalable pre-screening for NT1 in clinical and epidemiological settings where neurophysiological sleep studies are not available.

---

# **Relevance**

* Provides a low-cost, scalable pre-screening tool for NT1
* Useful for large cohorts such as **UK Biobank**, **digital sleep platforms**, and **public health populations**
* Demonstrates how **clinically structured questionnaires + minimal genotype** can approximate performance of resource-heavy diagnostics
* Introduces reproducible ML methodology, pairing HLA vs non-HLA models with a biologically justified correction rule

---

# **Figures**

> Replace `Figure_1.png`, `Figure_2.png`, `Figure_3.png` with your exact filenames if different.

---

## **Figure 1 — Model Features**

<p align="center">
  <img src="results_git/Figure_1.png" alt="Figure 1" width="30%"/>
</p>

**Caption:**
Figure 1. Venn diagram illustrates the overlap between the full questionnaire feature set (k = 26) and the reduced feature set (k = 10), excluding the HLA-DQB1*06:02 allele. Both feature sets were also evaluated with the addition of the HLA genotype (k=27 and k=11). Resulting in four feature sets to evaluate by each model. The reduced set includes emotional triggers (anger, joking, laughing, quick verbal response), muscle weakness locations (hand, jaw, knees, speech), and the Epworth Sleepiness Scale (ESS) score. 

---

## **Figure 2 — Best Performing Models*

<p align="center">
  <img src="results_git/Figure_2.png" alt="Figure 2" width="80%"/>
</p>

**Caption:**
Figure 2. ROC curves and confusion matrices of the best-performing models across the ESS (K=1), ESS + DQB106:02 (K=2),  full feature set (k = 27), full feature set + DQB106:02 (k = 28), reduced feature set (k = 10), and reduced feature set + DQB106:02 (k = 11) configurations. Confusion matrices depict the classification at the obtained thresholds at training, with non-HLA feature set at the first row, HLA feature sets at the second row, and veto-corrected predictions from non-HLA feature set at the third row. Models incorporating HLA-DQB1*06:02 achieved an average specificity of 98%, while those without HLA resulted with more false positives (average specificity of 96%). Application of the veto rule corrected 81 false positives in the ESS feature set, one in the full feature set (k = 27) set, and none in the Reduced feature set. Random Forest (RF); Linear Discriminant Analysis (LDA); Support Vector Machine (SVM)
---

## **Figure 3 — Feature Importance**

<p align="center">
  <img src="results_git/Figure_3_shap.png" alt="Figure 3" width="80%"/>
</p>

**Caption:**
Figure 3. Normalized SHAP feature importances for the best-performing model in each feature set configuration. Emotional triggers such as laughing and joking were the strongest predictors of NT1, followed by muscle weakness features (head, knees) and ESS score. Incorporation of HLA-DQB106:02 increased its relative contribution, particularly in the Reduced feature sets, underscoring its high discriminative value alongside symptom-based features.
---

# **Code Architecture**

The repository is organized for **clarity, modularity, and reproducibility**, mirroring best practices for clinical ML research.

```
project/
│
├── main_cv.py                    # Main pipeline: nested CV, Optuna, metrics, veto, plots
│
├── config/
│   └── config.py                 # Paths and run configuration
│
├── library/
│   └── ml_questionnaire/
│       ├── training.py           # Nested CV + Optuna search (models, pipelines, folds)
│       ├── scoring.py            # Metrics, CI, model selection, thresholding
│       ├── veto_rule.py          # HLA-negative FP correction + metric recomputation
│       ├── visualization.py      # ROC, CM, HLA-vs-nonHLA, PPV plots
│       └── shap_im_pipeline.py   # SHAP interpretability for tree models
│
└── results_git/                  # Figures used in manuscript and README
```

---

# **Pipeline Overview**

## **Run Analysis**

```python
    main_cv.py
```

* Imports questionnaire categorical variables
* Normalizes continuous variables (ESS, BMI, MSLT where available)
* Removes columns with excessive missingness
* Generates:

  * Full feature set
  * UKBB-compatible reduced set
  * ESS-only baselines
  * Each with ± HLA

Features are kept consistent across all model families.

---

## **Training Pipeline**

The project utilizes the ml_questionnaire library to compute the training pipeline with k-folds cross validation
```python
    library/ml_questionnaire/training.py
```

The pipeline evalute the models:
* Random Forest
* LightGBM
* XGBoost
* Elastic Net
* Logistic Regression
* SVM
* LDA

I perform an inner loop with 
* Optuna TPE sampler (250–300 trials)
* Maximizes AUC or specificity, depending on configuration

* and produces the output files

* `metrics_outer_folds.csv`
* `predictions_outer_folds.csv`
* Full hyperparameter history
* Per-fold validation records

The scoring function for the predictions are defined in 
```python
    library/ml_questionnaire/scoring.py
```
The script implements:

* Youden J
* Maximum specificity (spec_max)
* Probability cutoff 0.5
* Sensitivity/specificity CI
* Best-model selection per feature set (`select_best_model_type_per_config`)

For the `veto_rule.py`

> If model **did not train with HLA**, then:
> **HLA-negative false positives are flipped to negative.**

Outputs:

* `df_predictions_with_veto.parquet`
* `df_metrics_with_veto.parquet`
* `veto_count.csv`

The final visualizations are computed by the `visualization.py` script.

# **Reproducibility**

* Fixed random seed (`42`)
* Deterministic Optuna sampling
* Full logging of hyperparameters
* Exact model configurations stored for manuscript replication
