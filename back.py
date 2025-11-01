import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Class 1: Data Loading
class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        try:
            # Use semicolon as separator for your CSV
            df = pd.read_csv(self.path, sep=';')
            print("Data loaded successfully.")
            return df
        except Exception as e:
            print("Error loading data:", e)
            return None


# Class 2: Data Analysis using Pandas and NumPy
class DataAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def basic_info(self):
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict()
        }

    def numeric_summary(self):
        return self.df.describe()

    def conversion_rate(self):
        if 'y' in self.df.columns:
            return self.df['y'].value_counts(normalize=True).to_dict()
        return None

    def correlation_matrix(self):
        numeric_df = self.df.select_dtypes(include=np.number)
        return numeric_df.corr()

    def balance_analysis(self):
        if 'balance' in self.df.columns:
            balance_array = self.df['balance'].to_numpy()
            return {
                "mean_balance": np.mean(balance_array),
                "std_balance": np.std(balance_array)
            }
        return None


# Class 3: Visualization using Seaborn
class Visualizer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        sns.set(style="whitegrid")

    def plot_age_balance(self):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=self.df, x='age', y='balance', hue='y', alpha=0.7)
        plt.title("Age vs Balance by Deposit Subscription")
        return plt

    def plot_job_distribution(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x='job', order=self.df['job'].value_counts().index, palette='Set2')
        plt.title("Job Type Distribution")
        plt.xticks(rotation=45)
        return plt

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(6, 4))
        corr = self.df.select_dtypes(include=np.number).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        return plt

    def plot_balance_distribution(self):
        plt.figure(figsize=(6, 4))
        sns.histplot(self.df['balance'], bins=30, kde=True, color='skyblue')
        plt.title("Distribution of Customer Balances")
        return plt


# Class 4: Logistic Regression Model
class ModelTrainer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.model = None
        self.accuracy = None
        self.confusion = None

    def preprocess_data(self):
        df = self.df.copy()

        label_encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        X = df.drop('y', axis=1)
        y = df['y']

        return X, y, label_encoders

    def train_model(self):
        X, y, encoders = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        conf = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.model = model
        self.accuracy = acc
        self.confusion = conf

        return {
            "accuracy": acc,
            "confusion_matrix": conf.tolist(),
            "classification_report": report
        }


# Testing backend functions
if __name__ == "__main__":
    path = "bank.csv"
    loader = DataLoader(path)
    df = loader.load_data()

    if df is not None:
        analyzer = DataAnalyzer(df)
        print("\nBasic Info:", analyzer.basic_info())
        print("\nConversion Rate:", analyzer.conversion_rate())
        print("\nBalance Stats:", analyzer.balance_analysis())

        trainer = ModelTrainer(df)
        model_results = trainer.train_model()
        print("\nLogistic Regression Accuracy:", model_results["accuracy"])

        viz = Visualizer(df)
        viz.plot_age_balance().savefig("plot_age_balance.png")
        viz.plot_job_distribution().savefig("plot_job_distribution.png")
        viz.plot_correlation_heatmap().savefig("plot_heatmap.png")
        viz.plot_balance_distribution().savefig("plot_balance_distribution.png")
        print("\nSample plots saved successfully.")