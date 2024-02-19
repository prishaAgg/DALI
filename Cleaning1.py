#Author: Prisha
#Purpose: Data cleaning for "Dartmouth Courses" File

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

DEPT = "AAAS"
FEATURE = "Course Number" #feature against which the GPA is varied

# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv("Dartmouth - Courses.csv")

# Display the first few rows of the DataFrame to understand its structure
print(df.head())

# Data Cleaning

# Remove rows with missing values
df.dropna(inplace=True)

# reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Check for any remaining missing values after cleaning
print(df.isnull().sum())

# Eliminating NaN or missing input numbers
df.fillna(method ='ffill', inplace = True)


# convert columns to appropriate data types
df['Year'] = df['Year'].astype(int)
df['Term Number'] = df['Term Number'].astype(int)


# Save the cleaned data back to a CSV file
df.to_csv("cleaned_data.csv", index=False)
df1 = pd.read_csv("cleaned_data.csv")

#Machine Learning: Linear Regression
df2 = df1[df1['Department']== DEPT]

X = np.array(df2[FEATURE]).reshape(-1, 1)
y = np.array(df2['Median GPA Points']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

regr = LinearRegression()

regr.fit(X_train, y_train)
print("R^2 score:", regr.score(X_test, y_test))  # Evaluating the model on test data

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')
plt.xlabel('Course Number')
plt.ylabel('Median GPA Points')
plt.title('Linear Regression')
plt.show()

