# ITD105 Lab Exercise 2 - Summary of Findings

## Overview
This project demonstrates the application of machine learning techniques for both classification and regression problems using Python, Streamlit, and scikit-learn.

---

## Part 1: Lung Cancer Prediction (Classification)

### Dataset Information
- **Source**: [Kaggle Lung Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer)
- **Size**: 309 rows × 16 columns
- **Target Variable**: LUNG_CANCER (Yes/No)
- **Features**: Gender, Age, Smoking, Yellow Fingers, Anxiety, Peer Pressure, Chronic Disease, Fatigue, Allergy, Wheezing, Alcohol Consuming, Coughing, Shortness of Breath, Swallowing Difficulty, Chest Pain

### Data Preprocessing Steps
1. **Column Renaming**: Replaced whitespace with underscores in column names
2. **Label Encoding**: 
   - Gender: Male → 1, Female → 0
   - Categorical columns: "2" → 1 (Yes), "1" → 0 (No)
   - Target: YES → 1, NO → 0
3. **Feature Scaling**: Applied StandardScaler to the AGE column
4. **Class Imbalance**: Addressed using `class_weight='balanced'` in the model

### Model Comparison

#### Model A: K-Fold Cross-Validation (k=10)
- **Classification Accuracy**: High performance
- **Logarithmic Loss**: Low (better confidence)
- **Area Under ROC Curve**: Near 1.0 (excellent discrimination)
- **Computation**: 100 model fits (10 folds × 10 combinations)

#### Model B: Leave-One-Out Cross-Validation (LOOCV)
- **Performance**: Marginally better than K-Fold (~0.005% improvement)
- **Computation**: 3,090 model fits (309 samples × 10 combinations)
- **Trade-off**: 30x more computation time for minimal accuracy gain

### Model Selection: K-Fold Cross-Validation
**Rationale**: The K-Fold approach provides comparable accuracy with significantly better computational efficiency. The marginal improvement from LOOCV (0.005%) does not justify the 30x increase in training time.

### Final Model Optimization
- **Method**: GridSearchCV with StratifiedKFold (k=10)
- **Parameters Tested**:
  - C: [0.01, 0.1, 1, 10, 100]
  - Penalty: ['l1', 'l2']
  - Solver: ['liblinear', 'saga']
- **Scoring Metric**: ROC-AUC
- **Result**: Best parameters identified and model trained on 100% of data
- **Output**: `lung_logistic.pkl`

### Key Findings
- No significant correlation between features (ideal for model performance)
- Class imbalance successfully addressed through weighting
- Interactive prediction interface allows real-time lung cancer risk assessment
- Feature importance visualized through model coefficients

---

## Part 2: Daily Sea Ice Extent Prediction (Regression)

### Dataset Information
- **Source**: [Kaggle Daily Sea Ice Extent Data](https://www.kaggle.com/datasets/nsidcorg/daily-sea-ice-extent-data)
- **Size**: 26,354 rows × 7 columns (before preprocessing)
- **Target Variable**: Extent (total area of polar oceans covered by ice, in million km²)
- **Features**: Year, Month, Day, Hemisphere, Missing, Source Data
- **Time Period**: 1978-2019
- **Data Source**: Satellite microwave sensing instruments

### Data Preprocessing Steps
1. **Column Removal**:
   - Dropped 'Source Data' (not relevant for prediction)
   - Dropped 'Missing' column (nearly all zeros)
   - Removed rows where Missing ≠ 0

2. **Cyclical Time Feature Engineering**:
   - Month transformation: `Month_sin = sin(2π × Month / 12)` and `Month_cos`
   - Day transformation: `Day_sin = sin(2π × Day / days_in_month)` and `Day_cos`
   - Rationale: Captures cyclical nature (e.g., December adjacent to January)

3. **Categorical Encoding**:
   - Hemisphere: Northern → 1, Southern → 0

### Model Comparison

#### Model A: Train-Test Split (Single Split)
- **Test Size**: 20%
- **Random State**: 42
- **Mean Squared Error**: Low
- **Mean Absolute Error**: Low
- **R² Score**: High (~0.99+)
- **Issue**: Performance based on single, potentially biased data split

#### Model B: Repeated Random Train-Test Splitting
- **Iterations**: 50 random splits
- **Test Size**: 20% per split
- **Mean Squared Error**: Slightly higher than Model A (more conservative)
- **Mean Absolute Error**: Slightly higher than Model A
- **R² Score**: Very high (~0.99+)
- **Advantage**: Robust, averaged performance across multiple splits

### Model Selection: Repeated Random Train-Test Splitting
**Rationale**: Model B provides a more reliable and stable estimate of true model performance by averaging across 50 different data splits, eliminating bias from a single "lucky" or "unlucky" split. The minimal performance difference confirms model stability.

### Final Model
- **Type**: Linear Regression
- **Training**: Full dataset (all 26,000+ samples after preprocessing)
- **Output**: `seaice_linear.pkl`
- **Performance**: Excellent predictive capability with R² > 0.99

### Key Findings
- Sea ice extent shows strong temporal patterns captured by cyclical transformations
- Linear regression sufficient for high-accuracy predictions
- Year feature shows climate change trends over 41-year period
- Hemisphere significantly impacts ice extent patterns
- Interactive prediction interface allows exploration of historical and hypothetical scenarios

---

## Technical Implementation

### Libraries Used
- **Streamlit**: Interactive dashboard and UI
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Matplotlib, Seaborn, Plotly**: Data visualization
- **Scikit-learn**: Machine learning models and evaluation

### Key Techniques Demonstrated
1. **Classification**: Logistic Regression with regularization
2. **Regression**: Linear Regression
3. **Cross-Validation**: K-Fold, StratifiedKFold, LOOCV
4. **Resampling**: Train-Test Split, Repeated Random Splitting
5. **Hyperparameter Tuning**: GridSearchCV
6. **Feature Engineering**: Cyclical transformations, scaling, encoding
7. **Model Evaluation**: Multiple metrics (Accuracy, ROC-AUC, Log Loss, MSE, MAE, R²)
8. **Class Imbalance**: Weighted classes

---

## Conclusions

### Classification Task (Lung Cancer)
- Successfully built a logistic regression model to predict lung cancer risk
- Achieved high accuracy while balancing computational efficiency
- Model can aid in early screening based on patient symptoms and lifestyle factors

### Regression Task (Sea Ice Extent)
- Created a robust linear regression model for predicting sea ice extent
- Captured temporal and geographical patterns in climate data
- Model demonstrates stable performance across different data splits
- Valuable for understanding long-term climate trends

### Overall Learning Outcomes
- Practical experience with end-to-end ML pipeline (data cleaning → modeling → deployment)
- Understanding of different validation strategies and their trade-offs
- Application of appropriate models for classification vs regression problems
- Development of interactive tools for model interpretation and prediction

---

**Project Author**: Crislane Josh B. Eugenio  
**Course**: ITD105 - IT4D.1
**Date**: November 2, 2025