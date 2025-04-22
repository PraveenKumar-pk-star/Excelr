
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

st.set_page_config(layout="wide")
st.title("🚢 Titanic Survival Prediction")

# File upload
train_file = st.file_uploader("Upload Titanic Train CSV", type="csv")

if train_file:
    train_df = pd.read_csv(train_file)
    st.subheader("Training Data Preview")
    st.dataframe(train_df.head())

    # Basic Info and Description
    with st.expander("🔍 Dataset Info & Stats"):
        buffer = []
        train_df.info(buf=buffer)
        s = "\n".join(buffer)
        st.text(s)
        st.write(train_df.describe(include='all'))

    # Visualizations
    st.subheader("📊 Data Visualizations")
    numerical_vars = ['Age', 'Fare', 'SibSp', 'Parch']
    categorical_vars = ['Survived', 'Pclass', 'Sex', 'Embarked']

    st.markdown("### Histograms for Numerical Variables")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, var in enumerate(numerical_vars):
        sns.histplot(train_df[var].dropna(), kde=True, bins=30, ax=axes[i//2][i%2])
        axes[i//2][i%2].set_title(f"Histogram of {var}")
    st.pyplot(fig)

    st.markdown("### Box Plots for Numerical Variables")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, var in enumerate(numerical_vars):
        sns.boxplot(x=train_df[var], ax=axes[i//2][i%2])
        axes[i//2][i%2].set_title(f"Box Plot of {var}")
    st.pyplot(fig)

    st.markdown("### Count Plots for Categorical Variables")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, var in enumerate(categorical_vars):
        sns.countplot(x=train_df[var], ax=axes[i//2][i%2])
        axes[i//2][i%2].set_title(f"Count Plot of {var}")
    st.pyplot(fig)

    # Preprocessing
    st.subheader("🧹 Data Preprocessing")
    train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
    train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
    train_df = train_df.drop(columns=['Cabin'])
    train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

    st.write("Null values after processing:", train_df.isnull().sum())

    # Model training
    st.subheader("🧠 Model Training")
    try:
        X = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
        y = train_df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        st.markdown("### 📈 Evaluation Metrics")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")
        st.write(f"**ROC-AUC Score:** {roc_auc:.2f}")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="ROC Curve")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

        st.markdown("### 📌 Feature Importance")
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        })
        st.dataframe(coef_df)

    except Exception as e:
        st.error(f"Error during model training: {e}")
