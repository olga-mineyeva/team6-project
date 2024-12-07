# team6-project
A space for Team 6 to collaborate and create a project 

# Data Source
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
- Miniconda: [Miniconda Installation Page](https://docs.conda.io/projects/miniconda/en/latest/index.html).
- Git: [Git Installation Page](https://git-scm.com/).

---

## Installing project packages
- To install packages and create env: "conda env create -f environment.yml"
- To delete the environment: "conda env remove -n team6_project"
- To activate: "conda activate team6_project"

---

## Loading the dataset
- Data will automatically load upon running the experiment.
