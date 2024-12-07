# team6-project
A space for Team 6 to collaborate and create a project 

## Data Source
- [Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

# Disease Prediction with Symptom Minimization Using Machine Learning

## Project Overview

In real-world clinical settings, the range of symptoms assessed is often limited, necessitating streamlined diagnostic approaches. This project aims to optimize the classification of 42 diseases by minimizing the number of symptoms required for accurate prediction while maintaining high model performance. By leveraging machine learning algorithms, we seek to develop a robust framework to assist physicians in diagnostic decision-making, particularly in scenarios where expedited diagnostic processes are unavoidable.

To achieve this, our team has implemented and tested a range of classification models, including logistic regression, random forest, k-nearest neighbors (KNN), and feedforward neural networks, to identify minimal yet effective symptom sets for disease prediction.

---

## Team Members

- **David Vaz** ([davidvaz77](https://github.com/davidvaz77))
- **Niccolo Anjello Alcancia** ([nalcancia](https://github.com/nalcancia))
- **Mariya Kolesnikova** ([MK-DSI](https://github.com/MK-DSI))
- **Markus Amalanathan** ([jeffey97](https://github.com/jeffey97))
- **Olga Mineyeva** ([olga-mineyeva](https://github.com/olga-mineyeva))

---

## Exploratory Data Analysis (EDA)

The dataset, sourced from Kaggle, consists of 132 binary variables (symptoms) across 121 observations (patients), each labelled with one of 42 diseases. Key findings from the EDA include:

Data Integrity: No missing values were detected. An extraneous column, "Unnamed: 133" was removed during cleaning. 

Symptom Variability**: Despite recording 132 symptoms, the highest average symptom count per disease was 16.6 (observed for the common cold). This indicates potential redundancy in the dataset.

Symptom Overlap: The highest symptom overlap (69.2%) occurred between Hepatitis D and Hepatitis E, with 9 shared symptoms. By contrast, the common cold exhibited minimal overlap with other diseases.

These findings suggest that symptom minimization could improve both interpretability and efficiency in disease classification.

---

## Objective

The objective of this project is to reduce the set of symptoms required for accurate disease classification while maintaining high performance. 

This involves:
1) Understanding the dataset, standardization and cleanup 
2) Selection suitable classification algorithms 
3) Creating a pipeline to test various hyperparameter settings for each model
4) Identifying the most important features (symptoms) for each model 
5) Through iterative model training and optimization, evaluate if the number of features (symptoms) can be reduced while preserving minimal impact on precision and recall.

Reduction of symptom may have positive impact on reduction of time and resources during intake and triage.

---

## Models and Evaluation

Our machine learning pipeline involves the implementation of multiple classification algorithms, including:

1. **Logistic Regression**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**
4. **Feedforward Neural Networks**

### Methodology

1. **Baseline Models**: Logistic Regression and Feedforward Neural Network models were trained using all 132 symptoms, yielding near-perfect predictions due to low variability in the dataset.
    
2. **Symptom Minimization**: Recursive Feature Elimination (RFE) was used to iteratively reduce the symptom set to sizes of 80, 90, 100, and 120. Models (Logistic Regression, Random Forestm K-Nearest Neighbours (KNN), and a Feedforward Neural Network) were retrained and evaluated on these reduced sets to assess the impact on performance.
    
3. **Metrics**:
    
For model evaluation and selection, we prioritized accuracy and the F1 score. 
    
Accuracy provides a general measure of overall model performance, while the F1 score balances precision and recall, making it particularly useful when false positives and false negatives have differing levels of importance.

Our dataset is a mix of diseases with varying criticality for classification errors. For chronic conditions, false positives are more critical as they may lead to unnecessary interventions. Conversely, for acute or infectious diseases, which dominate this dataset, false negatives are more critical because they can delay necessary treatment or containment. The F1 score's ability to account for this balance makes it an appropriate choice for evaluating our models across such heterogeneous disease groups.
	
By using both metrics, we aim to identify models that maintain high accuracy while minimizing the impact of misclassifications, ensuring robustness across the diverse conditions represented in the dataset.
	
4. **Cross-Validation**: To ensure robustness, 5-fold cross-validation was conducted during training.
    
### Tools and Libraries

Python: numpy, pandas, scikit-learn, keras
Collaboration: Git/GitHub for version control

---

## Modeling Results Outline

This section will be updated as results are finalized. 
For each classification model, we tested multiple hyperparameters to identify the configuration that provided the best performance. The model with the highest accuracy and F1 score was selected for the report. Below are the results of the selected models evaluated on progressively reduced symptom sets:

**Logistic Regression**

A logistic regression model with _solver=lbfgs_ and _C=0.1_ was selected based on optimal accuracy and F1 scores. The results for this model across reduced symptom sets are as follows:

**Symptom Set (80):** Accuracy = X%, F1 = Y%
**Symptom Set (90):** Accuracy = X%, F1 = Y%
**Symptom Set (100):** Accuracy = X%, F1 = Y%
**Symptom Set (110):** Accuracy = X%, F1 = Y%
**Symptom Set (120):** Accuracy = X%, F1 = Y%

**Random Forest**

A random forest model with _n_estimators=200_ and _max_depth=150 was selected. The results for this model are:

**Symptom Set (80):** Accuracy = X%, F1 = Y%
**Symptom Set (90):** Accuracy = X%, F1 = Y%
**Symptom Set (100):** Accuracy = X%, F1 = Y%
**Symptom Set (110):** Accuracy = X%, F1 = Y%
**Symptom Set (120):** Accuracy = X%, F1 = Y%

**K-Nearest Neighbors (KNN)**

A KNN model with _k=7_ and _weights=distance_ was selected. The results are as follows:

**Symptom Set (80):** Accuracy = X%, F1 = Y%
**Symptom Set (90):** Accuracy = X%, F1 = Y%
**Symptom Set (100):** Accuracy = X%, F1 = Y%
**Symptom Set (110):** Accuracy = X%, F1 = Y%
**Symptom Set (120):** Accuracy = X%, F1 = Y%

**Feedforward Neural Network**

A feedforward neural network with _2 hidden layers of 64 and 32 nodes, dropout=0.3_, and _Adam optimizer_ was selected. The results for this model are:

**Symptom Set (80):** Accuracy = X%, F1 = Y%
**Symptom Set (90):** Accuracy = X%, F1 = Y%
**Symptom Set (100):** Accuracy = X%, F1 = Y%
**Symptom Set (110):** Accuracy = X%, F1 = Y%
**Symptom Set (120):** Accuracy = X%, F1 = Y%

---

## Future Directions

1. **Hyperparameter Optimization**: Refined selection of model parameters to improve performance on reduced symptom sets.
2. **Metrics Optimization**: The dataset is dominated with infections diseases necessitating a particular focus to minimizing false negatives. Therefore, precision and recall can be evaluated to optimize model selection.
3. **Validation:** Reduced symptom set validation can be refined using a larger testing set.  
4. **Interpretability**: Expanding the range of explainability techniques to enhance model transparency for clinical applications.

---

## Environment setup

## Pre-requisites
1. Miniconda: [Miniconda Installation Page](https://docs.conda.io/projects/miniconda/en/latest/index.html).
2. Git: [Git Installation Page](https://git-scm.com/).

---

## Installing project packages
1. To install packages and create env: "conda env create -f environment.yml"
2. To delete the environment: "conda env remove -n team6_project"
3. To activate: "conda activate team6_project"

---

## Loading the dataset
1. Data will automatically be downloaded(if does not exist and not properly formatted) and preprocessed each time load_data, and load_validation_data is called from disease_data_ingredient.py

## Project Structure

```plaintext
/team6_project/
├── README.md                             # Project description and setup instructions
├── environment.yml                       # Conda environment setup file

├── data/                                 # Dataset and database files
│   ├── processed/                        # Preprocessed data files
│   │   ├── README.md                     # Description of processed data
│   │   ├── Testing.csv                   # Processed testing dataset
│   │   └── Training.csv                  # Processed training dataset
│   ├── raw/                              # Raw data files
│   │   ├── Testing.csv                   # Raw testing dataset
│   │   └── Training.csv                  # Raw training dataset
│   └── sql/                              # SQL scripts and databases
│       ├── database_disease_perdiction_using_machine_learning.db
│       └── scripts_disease_perdiction_using_machine_learning.sql

├── diagnose_the_disease/                 # Datasets for diagnosis analysis
│   └── datasets/
│       ├── Testing.csv                   # Duplicated testing dataset
│       └── Training.csv                  # Duplicated training dataset

├── experiments/                          # Jupyter notebooks for experiments and analysis
│   ├── David_EDA_WIP.ipynb               # Exploratory data analysis
│   ├── First Jupyter Notebook.ipynb      # Initial notebook
│   ├── MDK_MLapproach_v3_withSHAP.ipynb  # SHAP analysis and ML approach
│   ├── data_visualization.ipynb          # Data visualization scripts
│   ├── grid_search_analysis.ipynb        # Grid search hyperparameter tuning
│   ├── logistic_regression.ipynb         # Logistic regression experiments
│   └── symptom_overlap_eda.ipynb         # Symptom overlap exploratory analysis

├── jupiter_notebooks/                    # Additional notebooks
│   ├── MDK_MLapproach.ipynb
│   ├── MDK_MLapproach_v2.ipynb
│   ├── MDK_MLapproach_v3.ipynb
│   ├── MDK_MLapproach_v4.ipynb
│   ├── MDK_MLapproach_v5_Olgas_code_pieces.ipynb
│   ├── Testing.csv
│   └── Training.csv

├── logs/                                 # Logging directory
│   ├── *.log                             # Placeholder for log files

├── models/                               # Saved models
│   ├── keras_models/                     # Keras model files
│   │   └── model_CustomNeuralNetMDK_None_241206_03_30_55.pkl
│   ├── model_KNN_None_241205_16_53_01.pkl
│   ├── model_KNN_SelectKBest_241205_17_04_36.pkl
│   ├── model_LogisticRegression_None_241205_16_51_02.pkl
│   ├── model_LogisticRegression_RFE_241205_18_42_43.pkl
│   ├── model_LogisticRegression_SelectKBest_241205_16_53_44.pkl
│   ├── model_RandomForest_None_241205_16_51_08.pkl
│   ├── model_RandomForest_SelectKBest_241205_16_54_11.pkl
│   └── reduced_features/                 # Reduced feature models keras only
│       ├── model_CustomNeuralNetMDK_RFE_241206_04_17_30.pkl
│       └── model_CustomNeuralNetMDK_SelectKBest_241206_03_39_01.pkl

├── reports/                              # Reports and analysis results
│   ├── grid_search_results_CustomNeuralNetMDK_None_241206_03_30_55.csv
│   ├── grid_search_results_CustomNeuralNetMDK_RFE_241206_04_17_30.csv
│   ├── grid_search_results_CustomNeuralNetMDK_SelectKBest_241206_03_39_01.csv
│   ├── grid_search_results_KNN_None_241205_16_53_01.csv
│   ├── grid_search_results_KNN_SelectKBest_241205_17_04_36.csv
│   ├── grid_search_results_LogisticRegression_None_241205_16_51_02.csv
│   ├── grid_search_results_LogisticRegression_RFE_241205_18_42_43.csv
│   ├── grid_search_results_LogisticRegression_SelectKBest_241205_16_53_44.csv
│   ├── grid_search_results_RandomForest_None_241205_16_51_08.csv
│   ├── grid_search_results_RandomForest_SelectKBest_241205_16_54_11.csv

│   ├── confusion_matrices/               # Confusion matrices for models
│   │   ├── confusion_matrix_model_KNN_None_241205_16_53_01.pkl.png
│   │   ├── confusion_matrix_model_KNN_SelectKBest_241205_17_04_36.pkl.png
│   │   ├── confusion_matrix_model_LogisticRegression_None_241205_16_51_02.pkl.png
│   │   ├── confusion_matrix_model_LogisticRegression_RFE_241205_18_42_43.pkl.png
│   │   ├── confusion_matrix_model_LogisticRegression_SelectKBest_241205_16_53_44.pkl.png
│   │   ├── confusion_matrix_model_RandomForest_None_241205_16_51_08.pkl.png
│   │   └── confusion_matrix_model_RandomForest_SelectKBest_241205_16_54_11.pkl.png

│   ├── feature_selection/                # Feature selection results
│   │   ├── CustomNeuralNetMDKClassifier_RFE_241206_04_17_30.csv
│   │   ├── CustomNeuralNetMDKClassifier_SelectKBest_241206_03_39_01.csv
│   │   ├── KNeighborsClassifier_SelectKBest_241205_17_04_36.csv
│   │   ├── LogisticRegression_RFE_241205_18_42_43.csv
│   │   ├── LogisticRegression_SelectKBest_241205_16_53_44.csv
│   │   └── RandomForestClassifier_SelectKBest_241205_16_54_11.csv

│   ├── keras_training/                   # Keras training logs
│   │   ├── history_241206_03_18_54.csv
│   │   ├── history_241206_03_19_59.csv
│   │   ├── history_241206_03_30_55.csv
│   │   ├── history_241206_03_39_01.csv
│   │   └── history_241206_04_17_30.csv

│   └── plots/                            # Model performance plots
│       └── validation_accuracy_model_RandomForest_SelectKBest_241205_16_54_11.pkl.png

└── src/                                  # Source code for the project
    ├── CustomNeuralNetMDKClassifier.py   # Custom neural network implementation
    ├── disease_data_ingredient.py        # Data preparation script
    ├── disease_experiment_classification_report.py  # Classification report generator
    ├── disease_experiment_confusion_matrix.py       # Confusion matrix generator
    ├── disease_experiment_evaluate.py    # Evaluation script
    ├── disease_experiment_tuning.py      # Hyperparameter tuning
    ├── disease_model_ingredient.py       # Model-related utilities
    ├── disease_preproc_ingredient.py     # Data preprocessing script
    ├── logger.py                         # Logging utilities
    └── reports_helper.py                 # Helper for generating reports

    ├── config/                           # Configuration files
        ├── knn_pg.json                   # Config for KNN
        ├── logistic_regression_pg.json   # Config for Logistic Regression
        ├── mdk_v3_pg.json                # Config for MDK V3
        ├── neural_net_pg.json            # Config for Neural Networks
        └── random_forest_pg.json         # Config for Random Forest
```


1. README.md: Contains the project description, setup instructions, and general usage.
2. environment.yml: Conda environment configuration file for dependency setup.
3. data/: Organizes raw, processed datasets, and SQL scripts.
4. experiments/: Notebooks for data exploration, EDA, and modeling experiments.
5. jupiter_notebooks/: Notebooks for data exploration, EDA, and modeling experiments.
6. models/: Stores trained models and serialized files.
7. reports/: Performance evaluation reports, including confusion matrices and feature selection results.
8. src/: Core source code for the project, including data processing, modeling, and utilities.

## Running the Project
### Set Up Models
1. Define preferred models in disease_model_ingredient.py. This script sets up model configurations.

### Set Up Preprocessing
1. Specify preprocessing steps in disease_preproc_ingredient.py, where data transformations are configured.

### Configuration Files
1. Edit the config/ folder to include all run parameters. Configuration files (knn_pg.json, logistic_regression_pg.json, mdk_v3_pg.json, neural_net_pg.json, random_forest_pg.json) hold settings like learning rates, iterations, and regularization methods.

### Hyperparameter Tuning
1. Use disease_experiment_tuning.py for hyperparameter tuning. It performs grid search cross-validation on the selected model, exports the best-performing model, and generates a grid search report. Modify grid search parameters using the cfg method.

### Metrics and Evaluation
1. Use disease_experiment_evaluate.py to evaluate model performance after training. It calculates key metrics such as accuracy, and weighted F1 score.
2. Use disease_experiment_classification_report.py to generate a detailed classification report, providing insights into the model's performance for each class.
3. Use disease_experiment_confusion_matrix.py to generate confusion matrices, helping to visualize model performance by comparing predicted vs. actual values across all classes.