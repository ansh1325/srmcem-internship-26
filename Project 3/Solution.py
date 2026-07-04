import pandas as pd
from sklearn.linear_model import LinearRegression
# Load the data
df = pd.read_csv('canada_per_capita_income.csv')

# Let's take a quick look at the first few rows
print(df.head())

# X must be a 2D array/DataFrame. We select the 'year' column.
X = df[['year']] 

# y is the target we want to predict.
y = df['per capita income (US$)']

# Create the linear regression object
reg = LinearRegression()

# Train the model with our data
reg.fit(X, y)

# Predict the per capita income for the year 2020
prediction_2020 = reg.predict([[2020]])

print("Predicted Income for 2020: ", prediction_2020[0])