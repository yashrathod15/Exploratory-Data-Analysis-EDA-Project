import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("iris.csv")

# Display first rows
print(df.head())

# Basic information
print(df.info())

# Statistical summary
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Correlation matrix
print(df.corr(numeric_only=True))

# Visualizations
sns.pairplot(df, hue="species")
plt.savefig("pairplot.png")

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.savefig("heatmap.png")