
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Logistic Regression App", layout="wide")
st.title("📊 Logistic Regression Interactive App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Missing values:", df.isnull().sum())

    with st.expander("📊 Summary Statistics"):
        st.write(df.describe(include='all'))

    # Feature selection
    target_column = st.selectbox("Select Target Column", df.columns)
    feature_columns = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_column])

    if st.button("Run Logistic Regression"):
        try:
            X = df[feature_columns]
            y = df[target_column]

            # Handle categorical variables
            X = pd.get_dummies(X, drop_first=True)
            if y.dtype == 'O':
                y = pd.get_dummies(y, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.subheader("✅ Model Evaluation")
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

        except Exception as e:
            st.error(f"An error occurred: {e}")
