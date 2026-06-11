# DataScience_MachineLearning----End-To-End_ModelBuilding----CancerPatient_diagnosis_CaseStudy_project
# Cancer Patient Detection using Machine Learning

An end-to-end machine learning pipeline built to predict and classify breast cancer diagnoses (Malignant vs. Benign). The project follows a rigorous engineering workflow involving data fetching, outlier mitigation via feature clipping, multicollinearity resolution using Variance Inflation Factor (VIF), and predictive modeling with advanced statistical and tree-based classifiers.

## 📌 Project Overview
Early detection of breast cancer plays a critical role in clinical treatments and patient survival rates. This project leverages the **Breast Cancer Dataset** from Kaggle to train machine learning algorithms that identify malignant tumors based on geometric, texture, and structural features extracted from cell nuclei digital images.

## 🛠️ Tech Stack & Dependencies
The pipeline is entirely engineered in Python utilizing the following environments and libraries:
* **Environment:** Jupyter Notebook / Google Colab
* **Data Fetching:** `opendatasets`
* **Core Analytics:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Statistical & ML Modeling:** `scikit-learn`, `statsmodels`, `xgboost`, `catboost`

## 📂 Dataset Details
* **Source:** Kaggle - Breast Cancer Dataset (via `wasiqaliyasir/breast-cancer-dataset`)
* **Size:** 569 instances with 33 initial columns
* **Target Feature (`diagnosis`):** Binary encoded into:
  * `1` - Malignant (212 cases)
  * `0` - Benign (357 cases)

## ⚙️ Data Preprocessing & Feature Engineering
1. **Target Variable Transformation:** Evaluated categorical indicators and applied binary numerical mappings.
2. **Outlier Mitigation:** Handled highly skewed distributions and extreme variations by clipping numeric feature values to the 1st and 99th percentiles.
3. **Feature Cleaning:** Standardized column naming structures (replacing structural spacing with snake_case formatting) and stripped redundant indexes (`id`, `Unnamed:_32`).
4. **Multicolliniarity Resolution (VIF Analysis):** To resolve model instability stemming from highly correlated geometric features (e.g., radius vs. perimeter vs. area), a **Variance Inflation Factor (VIF)** restriction threshold of $\le 80$ was imposed. 
5. **Feature Scaling:** Applied a `StandardScaler` standard normal normalization ($\mu=0, \sigma=1$) across the finalized standard error features.

### Finalized Key Features
Following programmatic statistical elimination, the finalized predictive features focusing on cell standard errors (`_se`) include:
* `texture_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave_points_se`, `symmetry_se`, `fractal_dimension_se`

## 📊 Model Training & Evaluation
The repository includes multiple baseline experiments using statistical modeling. Data is systematically split into an **80/20 train-test ratio** to ensure proper evaluation.

### Statistical Baseline: Logistic Regression (via Statsmodels Logit)
A generalized linear baseline model was established utilizing maximum likelihood estimation (MLE):
* **Train Set Accuracy:** 77% (F1-score: 0.82 Benign, 0.67 Malignant)
* **Test Set Accuracy:** 76% (F1-score: 0.83 Benign, 0.58 Malignant)

### Machine Learning Framework Integration
The framework preps and imports the setup foundations for high-performing modern classification networks, including:
* **Traditional Classifiers:** Decision Trees, Random Forest, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Naive Bayes (Gaussian, Bernoulli, Categorical)
* **Ensemble Boosters:** Bagging, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
* **Meta-Estimators:** Voting Classifier for optimal final decision blending

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/cancer-patient-detection.git](https://github.com/YOUR_USERNAME/cancer-patient-detection.git)
cd cancer-patient-detection
