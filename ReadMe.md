# **NarcCataplexyQuestionnaire**  
A machine learning pipeline for optimizing the **Stanford Cataplexy Questionnaire** scoring to improve **Narcolepsy Type 1 (NT1) detection**. This project integrates **clinical questionnaire data, HLA-DQB1*06:02 biomarker information, and machine learning models** to achieve **high-specificity NT1 classification**, reducing misdiagnoses and unnecessary sleep studies.
![Alternative text](docs/RG_NT1_Poster_Seattle_Big_Updated.png)



## **Project Structure**  

```
NarcCataplexyQuestionnaire
│── config/                     # Configuration files  
│   ├── config.py               # General settings for the project  
│   ├── narc_ukbb_mapper.py      # Maps UKBB questionnaire data to relevant features  
│   ├── SSI_Digital_Questionnaire.py # Handles questionnaire processing  
│  
│── data/                        # Dataset storage  
│   ├── pproc/                   # Preprocessed datasets  
│   │   ├── SSQDX_pp.csv         # Preprocessed Stanford Sleep Questionnaire data  
│   ├── raw/                     # Raw dataset files  
│   │   ├── data_for_paper.xlsx   # Primary dataset for paper  
│   │   ├── ssi_validation_older_added_hla.xlsx # HLA-enhanced dataset  
│  
│── docs/                        # Reference papers and documentation  
│   ├── anniclabatcataplexy.pdf  # Original cataplexy questionnaire study  
│   ├── cataplexyokun.pdf        # Study on narcolepsy-cataplexy across ethnic groups  
│   ├── narcolepsy_ml_ukbb.docx  # Machine learning analysis report  
│   ├── Old Questionnaire - All Pages.pdf # Original questionnaire version  
│  
│── library/                     # Core Python modules  
│   ├── __init__.py              # Package initialization  
│   ├── data_class.py            # Data structures for questionnaire processing  
│   ├── effect_measures_plot.py  # Functions for plotting model evaluation metrics  
│   ├── metrics_functions.py     # Performance metrics for model evaluation  
│   ├── ml_models.py             # Machine learning model implementations  
│   ├── table_one.py             # Generates Table 1 for dataset statistics  
│  
│── results/                     # Stores output and analysis results  
│  
│── src/                         # Script directory  
│   ├── ess_cutoff_model.py      # Model for Epworth Sleepiness Scale (ESS) cutoff analysis  
│   ├── generate_table_one.py    # Script to generate dataset summary statistics  
│   ├── pre_processing.py        # Data preprocessing pipeline  
│  
│── main.py                      # Main entry point for executing the ML pipeline  
│── main_full_and_cross_val.py   # Main entry point for executing the ML pipeline using the full dataset and cv
│── roc_curve_plots.py           # Generata the roc curves to determine the best cut-off for each model
```

## **Setup & Installation**  

### **Requirements**  
- Python 3.8+  
- Dependencies listed in `requirements.txt` (if available)  

### **Installation Steps**  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-repo/NarcCataplexyQuestionnaire.git
   cd NarcCataplexyQuestionnaire
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**  

### **1. Data Preprocessing**  
Run the preprocessing script to clean and format the dataset:  
```bash
python src/pre_processing.py
```

### **2. Train Machine Learning Models**  
To train and evaluate the machine learning models:  
```bash
python main.py
```
This will:  
- Load and preprocess the questionnaire data  
- Train **XGBoost** and **Elastic Net** models  
- Apply the **veto rule** using **HLA-DQB1*06:02**  
- Evaluate the models' specificity and sensitivity  

### **3. Generate Summary Tables**  
To generate **Table 1 (dataset statistics):**  
```bash
python src/generate_table_one.py
```

## **Key Features**  
✔ **Machine learning classification of NT1** using questionnaire data and HLA biomarkers  
✔ **High specificity (>98%)** achieved with XGBoost and Elastic Net  
✔ **Veto rule application** to minimize false positives  
✔ **Preprocessing pipeline** for standardizing questionnaire responses  
✔ **Extensible library of ML models and evaluation functions**  

## **Authors & Acknowledgments**  
This project is based on research conducted at the **Stanford Sleep Clinic**, integrating findings from:  
- **Anic-Labat et al. (1999, Sleep)**
- **Okun et al. (2002, Sleep)**  

## **License**  
[MIT License](LICENSE)  
