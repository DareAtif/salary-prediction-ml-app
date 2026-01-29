# LOAD & UNDERSTAND DATA
import pandas as pd

# load data from a CSV file
data = pd.read_csv('D:\salary-prediction-ML\data\salary_dataset.csv')
data.head()

data.info()

data.describe()

data.shape

# check for missing values
data.isnull().sum()

print("\nUnique Education:", data["Education"].unique())
print("\nUnique Job Roles:", data["JobRole"].unique())
print("\nUnique Locations:", data["Location"].unique())
print("\nUnique Company Sizes:", data["CompanySize"].unique())


# VISUAL ANALYSIS
import matplotlib.pyplot as plt
import seaborn as sns

# histogram of Salary
plt.figure(figsize=(10, 6))
sns.histplot(data['Salary(LPA)'], bins=30, kde=True) 
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()
