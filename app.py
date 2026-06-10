import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.markdown("<h1 style='color:#3B82F6;'>💳 Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("This app uses Logistic Regression to predict fraudulent transactions.")

# Upload the CSV
file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head())

    # Data cleaning
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud']  # unnecessary for prediction
    df = df.drop(columns=drop_cols)

    # Encode "type"
    df = pd.get_dummies(df, columns=["type"], drop_first=True)


    # Feature/Target split
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("✅ Logistic Regression model trained successfully!")

    # Classification Report
    st.subheader("📊 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("🧮 Confusion Matrix")
    fig1, ax1 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    # ROC Curve
    st.subheader("📈 ROC Curve")
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='green')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown(f"**AUC Score: {roc_auc:.2f}** – {'Excellent model!' if roc_auc > 0.8 else 'Consider tuning the model.'}")
