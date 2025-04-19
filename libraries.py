# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset
    iris = load_iris()
    
    # Convert the dataset to a DataFrame for easier handling
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    
    # Add the 'species' column with target values and map them to species names
    data['species'] = iris.target
    data['species'] = data['species'].replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset loaded successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display the first few rows of the dataset to inspect it
print("First few rows of the dataset:")
print(data.head())

# Check for data types and any missing values in the dataset
print("\nData Types:")
print(data.dtypes)
print("\nMissing Values:")
print(data.isnull().sum())

# Cleaning the dataset (In this case, the Iris dataset has no missing values, so this is just an example)
# data.fillna(value, inplace=True) or data.dropna(inplace=True)

# Task 2: Basic Data Analysis
# Compute basic statistics of numerical columns
print("\nBasic Statistics of the Dataset:")
print(data.describe())

# Group the data by the 'species' column and compute the mean of numerical columns
grouped_data = data.groupby('species').mean()
print("\nMean Values Grouped by Species:")
print(grouped_data)

# Observations
print("\nObservations:")
print("- Setosa generally has smaller petal lengths compared to other species.")
print("- Virginica tends to have the largest values in most columns.")

# Task 3: Data Visualization

# 1. Line Chart (Simulating trends of Sepal and Petal Length over index)
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['sepal length (cm)'], label='Sepal Length', marker='o')
plt.plot(data.index, data['petal length (cm)'], label='Petal Length', marker='x')
plt.title('Simulated Trends of Sepal and Petal Length Over Index', fontsize=16)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Length (cm)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Chart (Average Petal Length per Species)
plt.figure(figsize=(8, 6))
sns.barplot(x=grouped_data.index, y=grouped_data['petal length (cm)'], palette='viridis')
plt.title('Average Petal Length by Species', fontsize=16)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Average Petal Length (cm)', fontsize=12)
plt.show()

# 3. Histogram (Distribution of Sepal Length)
plt.figure(figsize=(8, 6))
sns.histplot(data['sepal length (cm)'], bins=15, kde=True, color='blue')
plt.title('Distribution of Sepal Length', fontsize=16)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.show()

# 4. Scatter Plot (Sepal Length vs Petal Length)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=data,
    x='sepal length (cm)',
    y='petal length (cm)',
    hue='species',
    palette='Dark2',
    s=100
)
plt.title('Relationship Between Sepal Length and Petal Length', fontsize=16)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.legend(title='Species')
plt.grid(True)
plt.show()
