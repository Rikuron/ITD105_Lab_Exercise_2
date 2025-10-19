# ITD105: Lab Exercise 2 - Classification and Regression Machine Learning Models

A comprehensive Streamlit dashboard showcasing machine learning models for both classification and regression tasks, developed as part of ITD105 coursework.

Live Demo Link: [Streamlit Community Cloud App](https://rikuron-itd105-lab-exercise-2-app-zaxtnk.streamlit.app)

## 🎯 Project Overview

This project demonstrates the implementation and comparison of machine learning models using two distinct datasets:

1. **Part 1: Classification Task** - Lung Cancer Prediction using Logistic Regression
2. **Part 2: Regression Task** - Daily Sea Ice Extent Prediction using Linear Regression

## 🚀 Features

### Part 1: Lung Cancer Prediction App 🫁
- **Data Overview**: Comprehensive analysis of lung cancer survey data
- **Data Preprocessing**: Feature encoding, scaling, and class imbalance handling
- **Model Comparison**: K-Fold vs Leave-One-Out Cross Validation
- **Interactive Prediction**: Real-time lung cancer risk assessment
- **Model Optimization**: Grid Search for hyperparameter tuning

### Part 2: Sea Ice Extent Prediction App 🧊
- **Data Overview**: Analysis of daily sea ice extent data (1978-2019)
- **Data Preprocessing**: Data cleaning and feature engineering
- **Model Comparison**: Different regression techniques
- **Interactive Prediction**: Sea ice extent forecasting
- **Climate Analysis**: Environmental trend visualization

## 📊 Datasets

### Lung Cancer Dataset
- **Source**: [Kaggle - Lung Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer)
- **Size**: 309 rows × 16 columns
- **Features**: Patient demographics and symptoms
- **Target**: Lung cancer diagnosis (Yes/No)

### Sea Ice Extent Dataset
- **Source**: [Kaggle - Daily Sea Ice Extent Data](https://www.kaggle.com/datasets/nsidcorg/daily-sea-ice-extent-data)
- **Size**: 26,354 rows × 7 columns
- **Features**: Date components and hemisphere data
- **Target**: Sea ice extent (million km²)

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Data Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Pickle

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ITD105-Lab-Exercise-2
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📋 Requirements

streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.3.0

## 🎮 Usage

### Running the Application

1. Start the Streamlit app: `streamlit run app.py`
2. Navigate to `http://localhost:8501` in your browser
3. Switch between "Part 1" and "Part 2" tabs

### Part 1: Lung Cancer Prediction
1. **Data Overview**: Explore the dataset structure and visualizations
2. **Data Preprocessing**: Review data cleaning and encoding steps
3. **Model Comparison**: Compare K-Fold vs LOOCV performance
4. **Model Application**: Input patient data for real-time predictions

### Part 2: Sea Ice Extent Prediction
1. **Data Overview**: Analyze sea ice data and climate trends
2. **Data Preprocessing**: Review data cleaning procedures
3. **Model Comparison**: Evaluate different regression models
4. **Model Application**: Predict sea ice extent for given parameters

## 📈 Model Performance

### Classification Model (Lung Cancer)
- **Algorithm**: Logistic Regression with L1 regularization
- **Cross-Validation**: 10-Fold Stratified K-Fold
- **Metrics**: Accuracy, ROC-AUC, Precision, Recall, F1-Score
- **Class Balancing**: Balanced class weights

### Regression Model (Sea Ice Extent)
- **Algorithm**: Linear Regression
- **Cross-Validation**: K-Fold Cross Validation
- **Metrics**: MSE, MAE, R² Score
- **Feature Engineering**: Date-based features

## 📁 Project Structure

ITD105-Lab-Exercise-2/  <br>
├── app.py                              # Main Streamlit application <br>
├── survey lung cancer.csv              # Lung cancer dataset   <br>
├── seaice.csv                          # Sea ice extent dataset    <br>
├── lung_logistic.pkl                   # Trained lung cancer model <br>
├── seaice_linear.pkl                   # Trained sea ice model <br>
├── requirements.txt                    # Python dependencies   <br>
└── README.md                          # Project documentation  <br>

## 🔬 Key Features

### Interactive Dashboards

- Real-time data visualization
- Interactive model predictions
- Comprehensive performance metrics
- Downloadable trained models

### Machine Learning Techniques
- Cross-validation strategies
- Hyperparameter optimization
- Feature importance analysis
- Model comparison frameworks

### Data Science Workflow
- Exploratory data analysis
- Data preprocessing pipelines
- Model training and evaluation
- Results interpretation

## 🎓 Educational Value

This project demonstrates:
- **Classification vs Regression**: Different ML problem types
- **Cross-Validation**: Model evaluation techniques
- **Data Preprocessing**: Real-world data cleaning
- **Model Optimization**: Hyperparameter tuning
- **Interactive ML**: User-friendly model deployment

## 📝 Course Information

- **Course**: ITD105 - Machine Learning
- **Assignment**: Lab Exercise 2
- **Focus**: Classification and Regression Models
- **Technologies**: Python, Streamlit, Scikit-learn

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the author.

## 📄 License

This project is created for educational purposes as part of ITD105 coursework.

---

**Note**: This application is for educational and demonstration purposes. Medical predictions should not be used for actual diagnosis, and climate predictions are simplified models for learning purposes.