import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pickle
import re
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import textwrap
import math

# Page Config
st.set_page_config(page_title="ITD105: Lab Exercise 2 - Classification and Regression Machine Learning Models", page_icon="🤖", layout="wide")
st.title("ITD105: Lab Exercise 2 - Classification and Regression Machine Learning Models")

tab1, tab2 = st.tabs(["Part 1", "Part 2"])

# Part 1: Classification Task using Logistic Regression Models on Health Datasets

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
    df = pd.read_csv('survey lung cancer.csv')

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "⚙️ Data Preprocessing", "🤖 Model Comparison", "📝 Model Application"])

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

    # Data Preprocessing
    with tab2:
        st.header("⚙️ Data Preprocessing")
        st.write("In this section, we will be preprocessing the data to make it ready for the machine learning models.")
        st.write('Since there are no missing values in the dataset, there are no steps needed to handle them, which is convenient! However, the feature columns (except Age) are notably not in the correct format, consisting of "2" for "YES" and "1" for "NO". We will need to convert them to "1" and "0" respectively.')

        st.subheader("1. Encoding of Categorical Features + Renaming of Columns ")    

        # Replace columns with whitespace inside to underscore. Ex. CHRONIC DISEASE => CHRONIC_DISEASE
        df.columns = df.columns.str.strip().str.replace(' ', '_')

        # Encode categorical features
        le = LabelEncoder()
        columns_to_encode = [col for col in df.columns if col != 'AGE']
        for col in columns_to_encode:
            df[col] = le.fit_transform(df[col])

        # Display the cleaned dataset
        st.subheader("Cleaned Dataset")
        st.write(df.head())
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
        df['AGE'] = scaler.fit_transform(df[['AGE']])

        st.session_state.scaler = scaler

        st.write(df.head())
        st.write("We have scaled the AGE column to a range of 0 to 1.")


        st.markdown("---")

        st.subheader("3. Resolving the Class Imbalance of the Target Variable")

        col1, col2 = st.columns(2)
        with col1:
            st.write("Counts of the Target Variable")
            st.write(df['LUNG_CANCER'].value_counts())

            plt.pie(df['LUNG_CANCER'].value_counts(), labels=['1', '0'], autopct='%1.1f%%')
            plt.title('Target Variable Distribution')
            st.pyplot(plt.gcf())
        with col2:
            st.write("Percentages of the Target Variable")
            st.write(df['LUNG_CANCER'].value_counts(normalize=True) * 100)

            st.write("As can be seen, the values are heavily skewed towards 1 or YES. This creates a class imbalance for our target variable, which could make our machine learning model biased.")
            st.write("In order to resolve this, we will have to use a technique called 'class weighting' to adjust the class weights of our models later on.")

        
        st.markdown("---")

        st.subheader("4. Correlation Heatmap")
        st.write("The correlation heatmap is a useful tool to visualize the correlation between the features of the dataset.")

        corr = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 7})
        plt.title('Correlation Heatmap')
        st.pyplot(plt.gcf())

        st.write("As can be seen, none of the models are significantly correlated with each other. This is a good sign for our machine learning models.")

        st.markdown("---")

        st.markdown("<center><h3>With this, our dataset is now ready for the machine learning models.</h3></center>", unsafe_allow_html=True)


    # Model Comparison
    with tab3:
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
        target_column = 'LUNG_CANCER'
        X_df = df.drop(columns=[target_column])
        y = df[target_column]

        # 2. Define Model
        model = LogisticRegression(
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
            y_pred_kfold = cross_val_predict(model, X_df, y, cv=kfold, method='predict')
            y_proba_kfold = cross_val_predict(model, X_df, y, cv=kfold, method='predict_proba')
            
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

        with st.spinner(f"Running Leave-One-Out (this will take a minute for {len(df)} samples)..."):
            loocv = LeaveOneOut()
            
            # Get predictions and probabilities
            y_pred_loo = cross_val_predict(model, X_df, y, cv=loocv, method='predict')
            y_proba_loo = cross_val_predict(model, X_df, y, cv=loocv, method='predict_proba')
            
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

        X_df = df.drop(columns=[target_column])
        y = df[target_column]
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
        final_model = grid_search.best_estimator_
        
        # Store the model in session state
        st.session_state.final_model = final_model

        st.success("Final optimized model has been trained!")
        st.write("The model has been trained on 100% of the data using the best parameters found by the Grid Search. Using the button below you can download the model as a pickle file.")
        st.download_button(
            label="Download Model",
            data=pickle.dumps(final_model),
            file_name="lung_logistic.pkl",
            mime="application/octet-stream"
        )

        st.markdown("### In the next tab, we will be using this exact same model to predict the lung cancer status of a patient based on info YOU will provide.")

    with tab4: 
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
            prediction = st.session_state.final_model.predict(input_df)[0]
            prediction_proba = st.session_state.final_model.predict_proba(input_df)[0]
            
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
                st.session_state.final_model.coef_[0],
                index=features_names,
                columns=['Coefficient']
            ).sort_values(by='Coefficient', ascending=False)
            st.dataframe(coefs.style.format("{:.4f}").bar(align='mid', color=['#d65f5f', '#5fba7d']))
            