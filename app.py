import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import textwrap
import math

# Page Config
st.set_page_config(page_title="ITD105: Lab 2 - Lung Cancer Prediction App", page_icon="🫁", layout="wide")
st.title("ITD105: Lab Exercise 2 - Lung Cancer Prediction App")

# Title of the app
st.markdown('# 🫁 Lung Cancer Prediction App')
st.write("Welcome to the Lung Cancer Prediction App. This project was made as fulfillment for my requirements in my ITD105 Course. This dashboard was made using:")
st.write("1. Streamlit for the dashboard")
st.write("2. Plotly, Matplotlib, and Seaborn for the charts and data visualization")
st.write("3. Scikit-learn for the machine learning models")
st.write("4. Pandas for the data manipulation")
st.write("5. NumPy for the numerical operations")

st.markdown("---")

# Load CSV
df = pd.read_csv('survey lung cancer.csv')

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "⚙️ Data Preprocessing", "🤖 Machine Learning Models"])

# Tab 1: Data Overview
# Initial loading and description of the dataset
with tab1:
    st.header("📊 Data Overview")

    st.subheader('Raw Data')
    st.write(df.head())

    # Source of the data
    st.write("The dataset briefly displayed above is taken from the Lung Cancer dataset from kaggle. For reference, here is the link: https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer")
    st.write("The dataset contains information about the patients and their lung cancer status.")

    # Description of the data
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    buffer.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Data Information')
        st.text(info_text)

    with col2:
        st.markdown(textwrap.dedent("""\
        As displayed above, the dataset contains 309 rows and 16 columns. The columns are as follows:
        
        - GENDER: The gender of the patient
        - AGE: The age of the patient
        - SMOKING: Whether the patient is a smoker (Yes or No)
        - YELLOW_FINGERS: Whether the patient has yellow fingers (Yes or No)
        - ANXIETY: Whether the patient is anxious (Yes or No)
        - PEER_PRESSURE: Whether the patient is under peer pressure (Yes or No)
        - CHRONIC DISEASE: Whether the patient has a chronic disease (Yes or No)
        - FATIGUE: Whether the patient is fatigued (Yes or No)
        - ALLERGY: Whether the patient is allergic (Yes or No)
        - WHEEZING: Whether the patient wheezes (Yes or No)
        - ALCOHOL CONSUMING: Whether the patient consumes alcohol (Yes or No)
        - COUGHING: Whether the patient coughs (Yes or No)
        - SHORTNESS OF BREATH: Whether the patient has shortness of breath (Yes or No)
        - SWALLOWING DIFFICULTY: Whether the patient has difficulty swallowing (Yes or No)
        - CHEST PAIN: Whether the patient has chest pain (Yes or No)

        **Target Variable:**

        - LUNG_CANCER: Whether the patient has lung cancer (Yes or No)
        """))
    
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Missing Values:**")
        st.write(df.isnull().sum())

    with col2:
        st.markdown("**Data Types:**")
        st.write(df.dtypes)

    with col3:
        st.markdown("**Data Statistics:**")
        st.write(df.describe())

    st.markdown("---")

    # Data Visualization
    st.subheader("Data Visualization")
    
    target_column = 'LUNG_CANCER'
    feature_columns = [col for col in df.columns if col != target_column]

    # Helper to decide categorical vs numeric
    def is_categorical(series, max_unique=10):
        return series.dtype == 'object' or series.nunique(dropna=False) <= max_unique

    # 1) All features on one matplotlib figure
    n = len(feature_columns)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(feature_columns):
        ax = axes[i]
        if is_categorical(df[col]):
            sns.countplot(x=df[col].astype(str), ax=ax, order=df[col].astype(str).value_counts().index)
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        else:
            sns.histplot(data=df, x=col, kde=True, ax=ax, bins=20)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        ax.set_title(col)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # 2) Target variable on its own figure
    fig_t, ax_t = plt.subplots(figsize=(6, 4))
    if is_categorical(df[target_column]):
        sns.countplot(x=df[target_column].astype(str), ax=ax_t, order=df[target_column].astype(str).value_counts().index)
        ax_t.set_ylabel('Count')
        ax_t.tick_params(axis='x', rotation=0)
    else:
        sns.histplot(data=df, x=target_column, kde=True, ax=ax_t, bins=20)
        ax_t.set_ylabel('Frequency')
    ax_t.set_xlabel(target_column)
    ax_t.set_title(f'Target: {target_column}')
    plt.tight_layout()
    st.pyplot(fig_t, clear_figure=True)

with tab2:
    st.header("⚙️ Data Preprocessing")

    