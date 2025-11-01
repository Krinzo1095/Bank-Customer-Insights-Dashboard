# front.py

import streamlit as st
import pandas as pd
from back import DataLoader, DataAnalyzer, Visualizer, ModelTrainer
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Bank Customer Insights Dashboard", layout="wide")

st.title("Bank Customer Insights Dashboard")
st.write(
    "This interactive dashboard explores the Bank Customer dataset using data analysis, "
    "visualization, and a simple machine learning model to predict term deposit subscriptions."
)

# Load data
loader = DataLoader("bank.csv")
df = loader.load_data()

if df is not None:
    analyzer = DataAnalyzer(df)
    viz = Visualizer(df)
    trainer = ModelTrainer(df)

    # Section 1: Dataset Overview
    st.header("1. Dataset Overview")

    info = analyzer.basic_info()
    st.markdown(f"**Shape:** {info['shape'][0]} rows × {info['shape'][1]} columns")
    st.markdown(f"**Columns:** {', '.join(info['columns'])}")

    # st.markdown("**Missing Values:**")
    # missing_vals = info["missing_values"]
    # if missing_vals:
    #     for col, val in missing_vals.items():
    #         st.write(f"- {col}: {val}")
    # else:
    #     st.write("No missing values found in the dataset.")

    # Section 2: Summary Statistics
    st.header("1. Summary Statistics")
    st.write("Below is a statistical summary of the numerical columns in the dataset.")
    st.dataframe(analyzer.numeric_summary())

    # Section 3: Key Insights
    st.header("2. Key Insights")

    col1, col2 = st.columns(2)
    with col1:
        conv_rate = analyzer.conversion_rate()
        if conv_rate:
            st.subheader("Deposit Subscription Rate")
            yes_rate = conv_rate.get("yes", 0) * 100 if "yes" in conv_rate else conv_rate.get(1, 0) * 100
            no_rate = conv_rate.get("no", 0) * 100 if "no" in conv_rate else conv_rate.get(0, 0) * 100
            st.metric("Subscribed (%)", f"{yes_rate:.2f}")
            st.metric("Not Subscribed (%)", f"{no_rate:.2f}")

    with col2:
        bal_stats = analyzer.balance_analysis()
        if bal_stats:
            st.subheader("Balance Statistics (NumPy Based)")
            st.metric("Mean Balance", f"{bal_stats['mean_balance']:.2f}")
            st.metric("Standard Deviation", f"{bal_stats['std_balance']:.2f}")

    # Section 4: Visualizations
    st.header("3. Visual Analysis")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Age vs Balance")
        st.write("This plot shows how customer age relates to account balance and deposit subscription.")
        fig1 = viz.plot_age_balance()
        st.pyplot(fig1)

    with col4:
        st.subheader("Job Type Distribution")
        st.write("This chart displays how many customers belong to each job category.")
        fig2 = viz.plot_job_distribution()
        st.pyplot(fig2)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Correlation Heatmap")
        st.write("This heatmap shows the correlation between different numerical features.")
        fig3 = viz.plot_correlation_heatmap()
        st.pyplot(fig3)

    with col6:
        st.subheader("Balance Distribution")
        st.write("This histogram shows how customer account balances are distributed.")
        fig4 = viz.plot_balance_distribution()
        st.pyplot(fig4)

    # Section 5: Logistic Regression Model
    st.header("4. Logistic Regression Model")
    st.write(
        "A logistic regression model is trained to predict whether a customer will subscribe to "
        "a term deposit based on features such as age, balance, job type, and other attributes."
    )

    model_results = trainer.train_model()

    st.subheader("Model Performance Summary")
    st.write(f"**Accuracy:** {model_results['accuracy']:.2f}")

    conf_matrix = pd.DataFrame(
        model_results["confusion_matrix"],
        index=["Actual: No", "Actual: Yes"],
        columns=["Predicted: No", "Predicted: Yes"]
    )
    st.write("**Confusion Matrix:**")
    st.dataframe(conf_matrix)
    st.markdown(
        "The confusion matrix shows how well the model distinguishes between customers who subscribed "
        "and those who didn’t. Values along the diagonal represent correct predictions, while the others "
        "indicate misclassifications."
        )


    report_df = pd.DataFrame(model_results["classification_report"]).transpose()
    st.write("**Classification Report:**")
    st.dataframe(report_df.style.format(precision=2))

    st.markdown(
        "The model achieves good overall accuracy but may show some imbalance in predicting 'Yes' "
        "cases, as seen in precision and recall scores. Further tuning or feature engineering "
        "could improve predictive performance."
    )

else:
    st.error("Unable to load dataset. Please check your file path and try again.")
