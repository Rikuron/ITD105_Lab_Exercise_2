import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pickle
import re
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_predict, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import textwrap
import math

# Page Config
st.set_page_config(page_title="ITD105: Lab Exercise 2 - Classification and Regression Machine Learning Models", page_icon="🤖", layout="wide")
st.title("ITD105: Lab Exercise 2 - Classification and Regression Machine Learning Models")

tab1, tab2 = st.tabs(["Part 1", "Part 2"])

# #################################################################################################
# PART 1: CLASSIFICATION (LUNG CANCER)
# #################################################################################################
with tab1:
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
    df_cancer = pd.read_csv('survey lung cancer.csv')

    part1_tab1, part1_tab2, part1_tab3, part1_tab4 = st.tabs(["📊 Data Overview", "⚙️ Data Preprocessing", "🤖 Model Comparison", "📝 Model Application"])

    # Tab 1: Data Overview
    # Initial loading and description of the dataset
    with part1_tab1:
        st.header("📊 Data Overview")

        st.subheader('Raw Data')
        st.write(df_cancer.head())
        st.write("Source: [Kaggle](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer)")

        # Source of the data
        st.write("The dataset briefly displayed above is taken from the Lung Cancer dataset from kaggle.")
        st.write("The dataset contains information about the patients and their lung cancer status.")

        # Description of the data
        buffer = io.StringIO()
        df_cancer.info(buf=buffer)
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
            st.write(df_cancer.isnull().sum())

        with col2:
            st.markdown("**Data Types:**")
            st.write(df_cancer.dtypes)

        with col3:
            st.markdown("**Data Statistics:**")
            st.write(df_cancer.describe())

        st.markdown("---")

        # Data Visualization
        st.subheader("Data Visualization")
        
        target_column_1 = 'LUNG_CANCER'
        feature_columns_1 = [col for col in df_cancer.columns if col != target_column_1]

        # Helper to decide categorical vs numeric
        def is_categorical(series, max_unique=10):
            return series.dtype == 'object' or series.nunique(dropna=False) <= max_unique

        # 1) All features on one matplotlib figure
        n = len(feature_columns_1)
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = axes.flatten() if n > 1 else [axes]

        for i, col in enumerate(feature_columns_1):
            ax = axes[i]
            if is_categorical(df_cancer[col]):
                sns.countplot(x=df_cancer[col].astype(str), ax=ax, order=df_cancer[col].astype(str).value_counts().index)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
            else:
                sns.histplot(data=df_cancer, x=col, kde=True, ax=ax, bins=20)
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
        if is_categorical(df_cancer[target_column_1]):
            sns.countplot(x=df_cancer[target_column_1].astype(str), ax=ax_t, order=df_cancer[target_column_1].astype(str).value_counts().index)
            ax_t.set_ylabel('Count')
            ax_t.tick_params(axis='x', rotation=0)
        else:
            sns.histplot(data=df_cancer, x=target_column_1, kde=True, ax=ax_t, bins=20)
            ax_t.set_ylabel('Frequency')
        ax_t.set_xlabel(target_column_1)
        ax_t.set_title(f'Target: {target_column_1}')
        plt.tight_layout()
        st.pyplot(fig_t, clear_figure=True)

    # Data Preprocessing
    with part1_tab2:
        st.header("⚙️ Data Preprocessing")
        st.write("In this section, we will be preprocessing the data to make it ready for the machine learning models.")
        st.write('Since there are no missing values in the dataset, there are no steps needed to handle them, which is convenient! However, the feature columns (except Age) are notably not in the correct format, consisting of "2" for "YES" and "1" for "NO". We will need to convert them to "1" and "0" respectively.')

        st.subheader("1. Encoding of Categorical Features + Renaming of Columns ")    

        # Replace columns with whitespace inside to underscore. Ex. CHRONIC DISEASE => CHRONIC_DISEASE
        df_cancer.columns = df_cancer.columns.str.strip().str.replace(' ', '_')

        # Encode categorical features
        le = LabelEncoder()
        columns_to_encode = [col for col in df_cancer.columns if col != 'AGE']
        for col in columns_to_encode:
            df_cancer[col] = le.fit_transform(df_cancer[col])

        # Display the cleaned dataset
        st.subheader("Cleaned Dataset")
        st.write(df_cancer.head())
        st.write('We have replaced the contents of: ')
        st.markdown('''
            `GENDER: "M" => 1 ; "F" => 0. `
            
            `Categorical Columns: "2" => 1 ; "1" => 0.`

            `LUNG_CANCER: "YES" => 1 ; "NO" => 0.`
        ''')
        st.write("With this, the data would be more easily read by the machine learning models.")


        st.markdown("---")


        st.subheader("2. Scaling the AGE feature")
        st.write("The AGE column is the only numeric column that is not scaled. We will need to scale it to a range of 0 to 1.")

        scaler = StandardScaler()
        df_cancer['AGE'] = scaler.fit_transform(df_cancer[['AGE']])

        st.session_state.scaler = scaler

        st.write(df_cancer.head())
        st.write("We have scaled the AGE column to a range of 0 to 1.")


        st.markdown("---")

        st.subheader("3. Resolving the Class Imbalance of the Target Variable")

        col1, col2 = st.columns(2)
        with col1:
            st.write("Counts of the Target Variable")
            st.write(df_cancer['LUNG_CANCER'].value_counts())

            plt.pie(df_cancer['LUNG_CANCER'].value_counts(), labels=['1', '0'], autopct='%1.1f%%')
            plt.title('Target Variable Distribution')
            st.pyplot(plt.gcf())
        with col2:
            st.write("Percentages of the Target Variable")
            st.write(df_cancer['LUNG_CANCER'].value_counts(normalize=True) * 100)

            st.write("As can be seen, the values are heavily skewed towards 1 or YES. This creates a class imbalance for our target variable, which could make our machine learning model biased.")
            st.write("In order to resolve this, we will have to use a technique called 'class weighting' to adjust the class weights of our models later on.")

        
        st.markdown("---")

        st.subheader("4. Correlation Heatmap")
        st.write("The correlation heatmap is a useful tool to visualize the correlation between the features of the dataset.")

        corr = df_cancer.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 7})
        plt.title('Correlation Heatmap')
        st.pyplot(plt.gcf())

        st.write("As can be seen, none of the models are significantly correlated with each other. This is a good sign for our machine learning models.")

        st.markdown("---")

        st.markdown("<center><h3>With this, our dataset is now ready for the machine learning models.</h3></center>", unsafe_allow_html=True)


    # Model Comparison
    with part1_tab3:
        st.header("🤖 Model Comparison")
        st.write("In this section, we will be comparing the performance of Logistic Regression Models on our dataset, with one using the K-Fold Cross Validation technique and one using the Leave-One-Out Cross Validation technique.")
        st.markdown(textwrap.dedent("""\
            We will be using the following metrics to compare the models:
            
            1. **(Classification) Accuracy**: The percentage of total predictions the model got right (both 'YES' and 'NO'). It can be misleading on imbalanced datasets.
            2. **Logarithmic Loss**: Measures the model's confidence. It heavily penalizes confident wrong answers. A lower score is better.
            3. **Confusion Matrix**:
                - **True Negatives (TN):** Correctly predicted 'NO'.
                - **False Positives (FP):** Incorrectly predicted 'YES' (Type I Error).
                - **False Negatives (FN):** Incorrectly predicted 'NO' (Type II Error - very bad for cancer!).
                - **True Positives (TP):** Correctly predicted 'YES'.
            4. **Classification Report:** A breakdown of performance for each class.
                - **Precision:** Of all the times the model predicted 'YES', what percentage was correct? (Minimizes False Positives).
                - **Recall:** Of all the *actual* 'YES' cases, what percentage did the model find? (Minimizes False Negatives).
                - **F1-Score:** The harmonic mean of Precision and Recall. A good all-around metric for imbalanced classes.
            5. **Area Under ROC Curve (ROC-AUC):** Measures how well the model can distinguish between the 'YES' and 'NO' classes. 1.0 is a perfect classifier, 0.5 is as good as random guessing.
        """))

        st.markdown(textwrap.dedent("""\
            Additionally, here are the set parameters for this specific comparison:

            - **Model Type:** Logistic Regression 
            - **Penalty**: l1 (L1 Regularization / Lasso)
            - **C Value**: 1
            - **Solver**: liblinear
            - **Random State**: 42
            - **Class Weight**: 'balanced' (to handle the class imbalance)
        """))

        st.markdown("---")

        # Helper function to plot the confusion matrix
        def plot_confusion_matrix(cm):
            """Plots a Plotly Confusion Matrix"""
            z = [[cm[0][0], cm[0][1]],  # TN, FP
                [cm[1][0], cm[1][1]]]  # FN, TP
            
            z_text = [[str(y) for y in x] for x in z]
            x_labels = ['Predicted 0 (NO)', 'Predicted 1 (YES)']
            y_labels = ['Actual 0 (NO)', 'Actual 1 (YES)']
            
            fig = ff.create_annotated_heatmap(
                z, x=x_labels, y=y_labels, 
                annotation_text=z_text, 
                colorscale='Blues'
            )
            
            fig.update_layout(
                xaxis=dict(side='bottom'),
                yaxis=dict(autorange='reversed')
            )
            return fig



        # 1. Prepare data
        target_column_1 = 'LUNG_CANCER'
        X_df = df_cancer.drop(columns=[target_column_1])
        y = df_cancer[target_column_1]

        # 2. Define Model
        logistic_model = LogisticRegression(
            penalty='l1',
            C=1,
            solver='liblinear',
            random_state=42,
            class_weight='balanced'
        )

        # 3. K-Fold Cross Validation Model
        st.subheader("Model A: Logistic Regression with K-Fold Cross-Validation (k=10)")
        
        with st.spinner("Running 10-Fold Cross-Validation..."):
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            
            # Get predictions and probabilities
            y_pred_kfold = cross_val_predict(logistic_model, X_df, y, cv=kfold, method='predict')
            y_proba_kfold = cross_val_predict(logistic_model, X_df, y, cv=kfold, method='predict_proba')
            
            # Calculate final metrics 
            acc_kfold = accuracy_score(y, y_pred_kfold)
            logloss_kfold = log_loss(y, y_proba_kfold)
            cm_kfold = confusion_matrix(y, y_pred_kfold, labels=[0, 1])
            report_kfold = classification_report(y, y_pred_kfold, target_names=['0 (NO)', '1 (YES)'])
            roc_auc_kfold = roc_auc_score(y, y_proba_kfold[:, 1])
            
            report_kfold_dict = classification_report(y, y_pred_kfold, output_dict=True)
            precision_kfold = report_kfold_dict['macro avg']['precision']
            recall_kfold = report_kfold_dict['macro avg']['recall']
            f1_score_kfold = report_kfold_dict['macro avg']['f1-score']

            # Display metrics
            st.markdown("**Performance Metrics:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg. Classification Accuracy", f"{acc_kfold:.4f}")
            col2.metric("Avg. Area Under ROC Curve", f"{roc_auc_kfold:.4f}")
            col3.metric("Avg. Logarithmic Loss", f"-{logloss_kfold:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Confusion Matrix (Total):**")
                fig_cm_kfold = plot_confusion_matrix(cm_kfold)
                st.plotly_chart(fig_cm_kfold, use_container_width=True)
            with col2:
                st.markdown("**Classification Report:**")
                st.code(report_kfold)

        # --- 4. Model B: Leave-One-Out Cross-Validation ---
        st.markdown("---")
        st.subheader("Model B: Logistic Regression with Leave-One-Out Cross-Validation (LOOCV)") 

        with st.spinner(f"Running Leave-One-Out (this will take a minute for {len(df_cancer)} samples)..."):
            loocv = LeaveOneOut()
            
            # Get predictions and probabilities
            y_pred_loo = cross_val_predict(logistic_model, X_df, y, cv=loocv, method='predict')
            y_proba_loo = cross_val_predict(logistic_model, X_df, y, cv=loocv, method='predict_proba')
            
            # Calculate final metrics 
            acc_loo = accuracy_score(y, y_pred_loo)
            logloss_loo = log_loss(y, y_proba_loo)
            cm_loo = confusion_matrix(y, y_pred_loo, labels=[0, 1])
            report_loo = classification_report(y, y_pred_loo, target_names=['0 (NO)', '1 (YES)'])
            roc_auc_loo = roc_auc_score(y, y_proba_loo[:, 1])

            report_loo_dict = classification_report(y, y_pred_loo, output_dict=True)
            precision_loo = report_loo_dict['macro avg']['precision']
            recall_loo = report_loo_dict['macro avg']['recall']
            f1_score_loo = report_loo_dict['macro avg']['f1-score']
            
            # Display metrics
            st.markdown("**Performance Metrics:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg. Classification Accuracy", f"{acc_loo:.4f}")
            col2.metric("Avg. Area Under ROC Curve", f"{roc_auc_loo:.4f}")
            col3.metric("Avg. Logarithmic Loss", f"-{logloss_loo:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Confusion Matrix (Total):**")
                fig_cm_loo = plot_confusion_matrix(cm_loo)
                st.plotly_chart(fig_cm_loo, use_container_width=True)
            with col2:
                st.markdown("**Classification Report:**")
                st.code(report_loo)

        # --- 5. Model Interpretation and Selection ---
        st.markdown("---")
        st.header("Model Interpretation & Selection")
        
        st.subheader("What does each performance metric indicate?") 
        st.markdown("""
        - **Classification Accuracy:** The percentage of total predictions the model got right (both 'YES' and 'NO'). It can be misleading on imbalanced datasets.
        - **Logarithmic Loss:** Measures the model's confidence. It heavily penalizes confident wrong answers. A lower score is better.
        - **Confusion Matrix:** A table showing the raw counts of correct and incorrect predictions:
            - **True Negatives (TN):** Correctly predicted 'NO'.
            - **False Positives (FP):** Incorrectly predicted 'YES' (Type I Error).
            - **False Negatives (FN):** Incorrectly predicted 'NO' (Type II Error - very bad for cancer!).
            - **True Positives (TP):** Correctly predicted 'YES'.
        - **Classification Report:** A breakdown of performance for each class.
            - **Precision:** Of all the times the model predicted 'YES', what percentage was correct? (Minimizes False Positives).
            - **Recall:** Of all the *actual* 'YES' cases, what percentage did the model find? (Minimizes False Negatives).
            - **F1-Score:** The harmonic mean of Precision and Recall. A good all-around metric for imbalanced classes.
        - **Area Under ROC Curve (ROC-AUC):** Measures how well the model can distinguish between the 'YES' and 'NO' classes. 1.0 is a perfect classifier, 0.5 is as good as random guessing.
        """)

        st.markdown("""
        | Resampling Technique | Classification Accuracy | Logarithmic Loss | Area Under ROC Curve | Precision | Recall | F1-Score |
        |-------|-------------------------|---------------------|----------------|----------------|----------------|----------------|
        | K-Fold Cross Validation (k=10) | {:.4f} | -{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |
        | Leave-One-Out Cross Validation | {:.4f} | -{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |
        """.format(acc_kfold, logloss_kfold, roc_auc_kfold, precision_kfold, recall_kfold, f1_score_kfold, acc_loo, logloss_loo, roc_auc_loo, precision_loo, recall_loo, f1_score_loo))

        st.write("Though very little, the **Leave-One-Out Cross Validation** boasts higher metrics than the K-Fold Cross Validation.")
        st.subheader("HOWEVER...")
        st.write("The K-Fold Cross Validation is much faster and more efficient than the Leave-One-Out Cross Validation, as it can be done in a fraction of the time. The amount of model fits it does is much less than the Leave-One-Out Cross Validation.")
        st.code("""
            K-Fold Cross Validation:
                - Number of model fits: 10 Combinations * 10 Samples = 100 Model Fits
        
            Leave-One-Out Cross Validation:
                - Number of model fits: 10 Combinations * 309 Samples = 3090 Model Fits
        """)

        st.write("Though the LOOCV technique is more accurate, it is only by around 0.005% or less. This measly improvement is not worth the time it takes to fit the model 2990 times more.")
        st.markdown("## Thus, we will be using K-Fold Cross Validation for our model.")

        st.markdown("---")

        # 6. Final Model Optimization
        st.subheader("Model Optimization")
        st.write("Now that we've selected which model performs better on our dataset, it's time to determine which parameters would be best to use to make the most accurate predictions.")

        X_df = df_cancer.drop(columns=[target_column_1])
        y = df_cancer[target_column_1]
        features_names = list(X_df.columns)

        # Parameter Grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        # K Fold
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Grid Search
        grid_search = GridSearchCV(
            estimator = LogisticRegression(random_state=42, class_weight='balanced'),
            param_grid = param_grid,
            cv = cv,
            scoring = 'roc_auc',
            n_jobs = 1
        )
        
        with st.spinner("Running Grid Search..."):
            grid_search.fit(X_df, y)

        st.success("Grid Search completed successfully!")
        st.write(f"**Best Parameters Found:** `{grid_search.best_params_}`")
        st.write(f"**Best Cross-Validation ROC-AUC Score:** `{grid_search.best_score_:.4f}`")

        st.markdown("Now we train one final model on **100% of the data** using the best parameters we just found.")

        # Use the best parameters found by GridSearch
        final_logistic_model = grid_search.best_estimator_
        
        # Store the model in session state
        st.session_state.final_logistic_model = final_logistic_model

        st.success("Final optimized model has been trained!")
        st.write("The model has been trained on 100% of the data using the best parameters found by the Grid Search. Using the button below you can download the model as a pickle file.")
        st.download_button(
            label="Download Model",
            data=pickle.dumps(final_logistic_model),
            file_name="lung_logistic.pkl",
            mime="application/octet-stream"
        )

        st.markdown("### In the next tab, we will be using this exact same model to predict the lung cancer status of a patient based on info YOU will provide.")

    with part1_tab4: 
        st.header("📝 Model Application")
        st.write("In this section, we will be applying the model to the dataset.")
        st.write("We will be using the model that we have chosen in the previous section, which is the Leave-One-Out Cross Validation model.")



        st.subheader("🔮 Interactive Prediction")
        st.write("Use the form below to input patient information and get a lung cancer prediction:")

        with st.form("prediction_form"):
            c1, c2, c3, c4 = st.columns(4)
        
            # We use 20-90 as a reasonable age range
            age_val = c1.slider("AGE", 20, 90, 50)
            
            # Binary toggles
            gender_val = c1.radio("GENDER", ["Female", "Male"], horizontal=True)
            smoking_val = c1.radio("SMOKING", ["No", "Yes"], horizontal=True)
            yellow_val = c1.radio("YELLOW_FINGERS", ["No", "Yes"], horizontal=True)
            anxiety_val = c2.radio("ANXIETY", ["No", "Yes"], horizontal=True)
            peer_val = c2.radio("PEER_PRESSURE", ["No", "Yes"], horizontal=True)
            chronic_val = c2.radio("CHRONIC_DISEASE", ["No", "Yes"], horizontal=True)
            fatigue_val = c2.radio("FATIGUE", ["No", "Yes"], horizontal=True)
            allergy_val = c3.radio("ALLERGY", ["No", "Yes"], horizontal=True)
            wheezing_val = c3.radio("WHEEZING", ["No", "Yes"], horizontal=True)
            alcohol_val = c3.radio("ALCOHOL_CONSUMING", ["No", "Yes"], horizontal=True)
            coughing_val = c3.radio("COUGHING", ["No", "Yes"], horizontal=True)
            shortness_val = c4.radio("SHORTNESS_OF_BREATH", ["No", "Yes"], horizontal=True)
            swallowing_val = c4.radio("SWALLOWING_DIFFICULTY", ["No", "Yes"], horizontal=True)
            chest_val = c4.radio("CHEST_PAIN", ["No", "Yes"], horizontal=True)
            
            submitted = st.form_submit_button("🔍 Get Prediction", type="primary", use_container_width=True)

        if submitted:
            # --- Preprocess User Input ---
            # 1. Map text to 0/1
            input_data = {
                'GENDER': 1 if gender_val == "Male" else 0,
                'AGE': age_val, # Will scale this next
                'SMOKING': 1 if smoking_val == "Yes" else 0,
                'YELLOW_FINGERS': 1 if yellow_val == "Yes" else 0,
                'ANXIETY': 1 if anxiety_val == "Yes" else 0,
                'PEER_PRESSURE': 1 if peer_val == "Yes" else 0,
                'CHRONIC_DISEASE': 1 if chronic_val == "Yes" else 0,
                'FATIGUE': 1 if fatigue_val == "Yes" else 0,
                'ALLERGY': 1 if allergy_val == "Yes" else 0,
                'WHEEZING': 1 if wheezing_val == "Yes" else 0,
                'ALCOHOL_CONSUMING': 1 if alcohol_val == "Yes" else 0,
                'COUGHING': 1 if coughing_val == "Yes" else 0,
                'SHORTNESS_OF_BREATH': 1 if shortness_val == "Yes" else 0,
                'SWALLOWING_DIFFICULTY': 1 if swallowing_val == "Yes" else 0,
                'CHEST_PAIN': 1 if chest_val == "Yes" else 0,
            }
            
            # 2. Create DataFrame in correct order
            input_df = pd.DataFrame([input_data], columns=features_names)
            
            # 3. Scale AGE using the saved scaler from tab2
            if 'scaler' in st.session_state:
                scaler = st.session_state.scaler
                input_df['AGE'] = scaler.transform(input_df[['AGE']])
            else:
                st.error("Scaler not found. Please re-run Tab 2 to save the scaler.")
                st.stop()
            
            # --- Make Prediction ---
            prediction = st.session_state.final_logistic_model.predict(input_df)[0]
            prediction_proba = st.session_state.final_logistic_model.predict_proba(input_df)[0]
            
            # --- Display Prediction ---
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"Prediction: YES (Lung Cancer)", icon="⚠️")
                st.metric("Confidence", f"{prediction_proba[1] * 100:.2f}%")
            else:
                st.success(f"Prediction: NO (Lung Cancer)", icon="✅")
                st.metric("Confidence", f"{prediction_proba[0] * 100:.2f}%")

            st.write("---")
            st.write("### Model Coefficients (Feature Importance)")
            st.write("This shows how much each feature influences the prediction (positive = more likely 'YES', negative = more likely 'NO').")
            coefs = pd.DataFrame(
                st.session_state.final_logistic_model.coef_[0],
                index=features_names,
                columns=['Coefficient']
            ).sort_values(by='Coefficient', ascending=False)
            st.dataframe(coefs.style.format("{:.4f}").bar(align='mid', color=['#d65f5f', '#5fba7d']))
            

with tab2:
    # Title of the app
    st.markdown('# 🧊 Daily Sea Ice Extent Prediction App')
    st.write("Welcome to the Daily Sea Ice Extent Prediction App. This project was made as fulfillment for my requirements in my ITD105 Course. This dashboard was made using:")
    st.write("1. Streamlit for the dashboard")
    st.write("2. Plotly, Matplotlib, and Seaborn for the charts and data visualization")
    st.write("3. Scikit-learn for the machine learning models")
    st.write("4. Pandas for the data manipulation")
    st.write("5. NumPy for the numerical operations")

    st.markdown("---")

    # Load CSV
    df_seaice = pd.read_csv('seaice.csv')

    part2_tab1, part2_tab2, part2_tab3, part2_tab4 = st.tabs(["📊 Data Overview", "⚙️ Data Preprocessing", "🤖 Model Comparison", "📝 Model Application"])

    with part2_tab1:
        st.header("📊 Data Overview")

        st.subheader('Raw Data')
        st.write(df_seaice.head())

        # Source of the data
        st.write("The dataset briefly displayed above is taken from the Daily Sea Ice Extent dataset from kaggle. For reference, here is the link: https://www.kaggle.com/datasets/nsidcorg/daily-sea-ice-extent-data")
        st.write("The dataset provides information about the the total area of the polar oceans covered by ice each day from the year 1978 until 2019, often based on satellite data with a concentration of at least 15% ice. The data is primarily derived from satellite microwave sensing instruments. This dataset is crucial for understanding climate change, as sea ice extent is a key indicator of global temperature trends and environmental shifts in polar regions.")

        # Description of the data
        buffer = io.StringIO()
        df_seaice.info(buf=buffer)
        info_text = buffer.getvalue()
        buffer.close()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Data Information')
            st.text(info_text)

        with col2:
            st.markdown(textwrap.dedent("""\
            As displayed above, the dataset contains 26,354 rows and 7 columns. The columns are as follows:
            
            - Year: The year the data was recorded
            - Month: The month the data was recorded
            - Day: The day the data was recorded
            - Missing: Whether the data is missing
            - Source Data: The website source of the data
            - Hemisphere: The hemisphere of planet earth

            **Target Variable:**

            - Extent: The total area of the polar oceans covered by ice each day
            """))
        
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Missing Values:**")
            st.write(df_seaice.isnull().sum())

        with col2:
            st.markdown("**Data Types:**")
            st.write(df_seaice.dtypes)

        with col3:
            st.markdown("**Data Statistics:**")
            st.write(df_seaice.describe())

        st.markdown("---")

        st.subheader("Data Visualization")

        target_column_2 = 'Extent'
        feature_columns_2 = [col for col in df_seaice.columns if col not in [target_column_2, 'Source Data']]

        n_cols = 3
        n_rows = math.ceil(len(feature_columns_2) / n_cols)
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=feature_columns_2)
        for i, col in enumerate(feature_columns_2):
            row, col_pos = (i // n_cols) + 1, (i % n_cols) + 1
            if col == 'hemisphere':
                counts = df_seaice[col].value_counts()
                fig.add_trace(go.Bar(x=counts.index, y=counts.values, name=col), row=row, col=col_pos)
            else:
                fig.add_trace(go.Histogram(x=df_seaice[col], nbinsx=30, name=col), row=row, col=col_pos)
        fig.update_layout(height=n_rows * 300, showlegend=False, title_text="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)

        fig_target = px.histogram(df_seaice, x=target_column_2, nbins=50, title='Distribution of Target Variable: Sea Ice Extent', marginal="box")
        st.plotly_chart(fig_target, use_container_width=True)

    # Data Preprocessing
    with part2_tab2:
        st.header("⚙️ Data Preprocessing")
        st.write("In this section, we will be preprocessing the data to make it ready for the machine learning models.")
        st.write('Since there are no missing values in the dataset, there are no steps needed to handle them, which is convenient! However, there are some columns that, if we take a closer look, would not be relevant when it comes to predicting Extent.')

        st.subheader("1. Dropping Unnecessary Columns")
        st.write("We will be dropping the 'Source Data' column, as it is not relevant to the prediction of the target variable, 'Extent'.")
        st.write("Additionally, from the graph in the previous tab, almost all of the values of the 'Missing' column are all 0, with a few outliers. Since there is basically only one relevant unique value for it, we can just drop the outlier rows as well as the entire column.")

        st.write("Original Dataset Shape: ", df_seaice.shape)
        
        # Drop rows with Missing != 0
        df_seaice.drop(df_seaice[df_seaice['Missing'] != 0].index, inplace=True)
        st.write("Dataset Shape after dropping rows whose 'Missing' value is not 0: ", df_seaice.shape)

        # Drop Missing and Source Data columns
        df_seaice.drop(columns=['Missing', 'Source Data'], inplace=True)
        st.write("Dataset after dropping the 'Missing' and 'Source Data' columns: ")
        st.write(df_seaice.head())

        st.markdown("---")

        st.subheader("2. Handling of Time-Series Features")
        st.write("To help the model understand the cyclical nature of months and days (e.g., December is next to January), we transform them using sine and cosine functions.")

        # Find number of days in each month
        date_series = pd.to_datetime(df_seaice['Year'].astype(str) + '-' + df_seaice['Month'].astype(str))
        days_in_month = date_series.dt.days_in_month

        # Month Transform
        df_seaice['Month_sin'] = np.sin(2 * np.pi * df_seaice['Month'] / 12)
        df_seaice['Month_cos'] = np.cos(2 * np.pi * df_seaice['Month'] / 12)
        
        # Day Transform
        df_seaice['Day_sin'] = np.sin(2 * np.pi * df_seaice['Day'] / days_in_month)
        df_seaice['Day_cos'] = np.cos(2 * np.pi * df_seaice['Day'] / days_in_month)
        
        # Drop original Month and Day columns
        df_seaice.drop(columns=['Month', 'Day'], inplace=True)
        st.write("Dataset after dropping the original 'Month' and 'Day' columns: ")
        st.write(df_seaice.head())

        st.markdown("---")

        st.subheader("3. Encoding of Categorical Features")

        st.write("The 'hemisphere' column is a categorical feature, which means that the model will not be able to understand it. We will need to encode it to a numerical value.")

        # Encode hemisphere
        df_seaice['hemisphere'] = le.fit_transform(df_seaice['hemisphere'])
        st.code("""
            `hemisphere: "Northern Hemisphere" => 1 ; "Southern Hemisphere" => 0.`
        """)
        st.write("Dataset after encoding the 'hemisphere' column: ")
        st.write(df_seaice.head())

        st.markdown("---")

        st.markdown("<center><h3>With this, our dataset is now ready for the machine learning models.</h3></center>", unsafe_allow_html=True)


    with part2_tab3:
        st.header("🤖 Model Comparison")

        st.write("In this section, we will be comparing the performance of Linear Regression Models on our dataset, with one using the Train-Test Split Sampling Technique and one using the Repeated Random Train Test Splitting technique.")
        st.markdown(textwrap.dedent("""\
            We will be using the following metrics to compare the models:
            
            1. **Mean Squared Error**: Measures the average squared difference between the predicted and actual values. A lower score is better.
            2. **Mean Absolute Error**: Measures the average absolute difference between the predicted and actual values. A lower score is better.
            3. **R-Squared**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher score is better.
        """))
        
        st.markdown("---")

        # 1. Prepare data
        target_column_2 = 'Extent'
        X_df = df_seaice.drop(columns=[target_column_2])
        y = df_seaice[target_column_2]

        # 2. Define Model
        linear_model = LinearRegression()

        # 3. Train-Test Split Sampling Technique
        # Split data once

        st.subheader("Model A: Linear Regression with Train-Test Split Sampling Technique")

        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

        # Fit the model
        linear_model.fit(X_train, y_train)

        # Make predictions
        y_pred_a = linear_model.predict(X_test)

        # Calculate metrics
        mse_a = mean_squared_error(y_test, y_pred_a)
        mae_a = mean_absolute_error(y_test, y_pred_a)
        r2_a = r2_score(y_test, y_pred_a)

        st.write("Metrics for the Train-Test Split Sampling Technique:")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Squared Error", f"{mse_a:.4f}")
        col2.metric("Mean Absolute Error", f"{mae_a:.4f}")
        col3.metric("R-Squared", f"{r2_a:.4f}")
        
        st.markdown("---")

        # 4. Repeated Random Train Test Splitting Technique
        # Split data repeatedly

        st.subheader("Model B: Linear Regression with Repeated Random Train Test Splitting Technique")

        with st.spinner("Running 50 random splits... This could take a moment..."):
            mse_scores, mae_scores, r2_scores = [], [], []

            for i in range(50):
                X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=i)
                linear_model.fit(X_train, y_train)
                y_pred_b = linear_model.predict(X_test)
                mse_scores.append(mean_squared_error(y_test, y_pred_b))
                mae_scores.append(mean_absolute_error(y_test, y_pred_b))
                r2_scores.append(r2_score(y_test, y_pred_b))
            
            # Calculate averages
            mse_b = np.mean(mse_scores)
            mae_b = np.mean(mae_scores)
            r2_b = np.mean(r2_scores)

        st.write("Average Metrics for the Repeated Random Train Test Splitting Technique:")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Squared Error", f"{mse_b:.4f}")
        col2.metric("Mean Absolute Error", f"{mae_b:.4f}")
        col3.metric("R-Squared", f"{r2_b:.4f}")

        st.markdown("---")

        # 5. Model Interpretation and Selection
        st.subheader("Model Interpretation and Selection")

        table_data_s = {
            "Metric": ["MSE (Lower is better)", "MAE (Lower is better)", "R² (Higher is better)"],
            "Model A (Single Split)": [f"{mse_a:.4f}", f"{mae_a:.4f}", f"{r2_a:.4f}"],
            "Model B (Repeated Splits)": [f"{mse_b:.4f}", f"{mae_b:.4f}", f"{r2_b:.4f}"]
        }
        st.dataframe(pd.DataFrame(table_data_s), use_container_width=True)
        
        st.markdown("""
        **Conclusion:**
        - **Model A**'s performance is based on a single, potentially "lucky" or "unlucky" random split of the data.
        - **Model B** provides a much more robust and reliable estimate of the model's true performance by averaging the results over 50 different splits. This smooths out any randomness from a single data split.
        
        Though Model A seems to perform better, it is only based on that certain split and random state of the data. It could have been lower had the random state been different. Model B, on the other hand is much more stable and its performance metrics are really not that far off from Model A.

        For our final application, we will use the approach of training our model on the **full dataset**, as the repeated splits of Model B have given us confidence in the model's general stability and performance.
        """)

        st.markdown("## Thus, we will be using the Repeated Random Train Test Splitting Technique for our model.")

        st.markdown("---")

        st.subheader("Model Optimization")
        st.write("Now that we've selected which model performs better on our dataset, it's time to determine which parameters would be best to use to make the most accurate predictions.")

        X_seaice_df = df_seaice.drop(columns=[target_column_2])
        y_seaice = df_seaice[target_column_2]
        features_names_seaice = list(X_seaice_df.columns)

        # Training the Final Model

        with st.spinner("Training the final model... This could take a moment..."):
            final_linear_model = LinearRegression()
            final_linear_model.fit(X_seaice_df, y_seaice)
            st.session_state.final_linear_model = final_linear_model

        st.success("Final optimized model has been trained!")
        st.write("The model has been trained on the full dataset. Using the button below you can download the model as a pickle file.")
        st.download_button(
            label="Download Model",
            data=pickle.dumps(final_linear_model),
            file_name="seaice_linear.pkl",
            mime="application/octet-stream"
        )
        
        st.markdown("### In the next tab, we will be using this exact same model to predict the extent of the sea ice for a given day.")



    with part2_tab4:
        st.header("📝 Model Application")
        st.write("In this section, we will be applying the model to the dataset.")
        st.write("We will be using the model that we have chosen in the previous section, which is the Repeated Random Train Test Splitting Technique.")



        st.subheader("🔮 Interactive Prediction")
        st.write("Use the form below to input the information and get a sea ice extent prediction:")

        with st.form("prediction_form_seaice"):
            c1, c2, = st.columns(2)
        
            # We use 20-90 as a reasonable age range
            year_val = c1.slider("Year", 1978, 2019, 2000)
            month_val = c1.slider("Month", 1, 12, 6)
            day_val = c2.slider("Day", 1, 31, 15)
            hemisphere_val = c2.radio("Hemisphere", ["North", "South"], horizontal=True, key="seaice_hemisphere")
            
            submitted_seaice = st.form_submit_button("🔍 Get Prediction", type="primary", use_container_width=True)

        if submitted_seaice:
            # --- Preprocess User Input ---
            # 1. Determine number of days
            days_in_month = pd.to_datetime(f"{year_val}-{month_val}").days_in_month
            
            # 2. Create input dictionary with sine and cosine features
            input_data = {
                'Year': year_val,
                'hemisphere': 0 if hemisphere_val == 'North' else 1,
                'Month_sin': np.sin(2 * np.pi * month_val / 12),
                'Month_cos': np.cos(2 * np.pi * month_val / 12),
                'Day_sin': np.sin(2 * np.pi * day_val / days_in_month),
                'Day_cos': np.cos(2 * np.pi * day_val / days_in_month),
            }

            # 3. Create DataFrame in correct order
            input_df = pd.DataFrame([input_data], columns=features_names_seaice)
            
            # --- Make Prediction ---
            prediction = st.session_state.final_linear_model.predict(input_df)[0]
            
            # --- Display Prediction ---
            st.subheader("Prediction Result")
            st.metric("Predicted Sea Ice Extent (in million square km)", f"{prediction:.4f}")

            st.markdown("---")

            st.write("### Model Coefficients (Feature Importance)")
            st.write("This shows how much each feature influences the prediction. A positive coefficient means the feature increases the predicted extent, and a negative one decreases it.")
            
            coefs = pd.DataFrame(
                st.session_state.final_linear_model.coef_[0],
                index=features_names_seaice,
                columns=['Coefficient']
            ).sort_values(by='Coefficient', ascending=False)
            st.dataframe(coefs.style.format("{:.4f}").bar(align='mid', color=['#d65f5f', '#5fba7d']))
