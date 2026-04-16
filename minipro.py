# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("data.csv")

# -------------------------------
# Step 2: Convert Categorical Data
# -------------------------------
df['Event_Type'] = df['Event_Type'].map({
    'Wedding': 1,
    'Party': 2,
    'Hostel': 3,
    'Function': 4
})

# -------------------------------
# Step 3: Define Features & Target
# -------------------------------
X = df[['People_Count', 'Food_Prepared', 'Event_Type']]
y = df['Waste']

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# Step 5: Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 6: Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 7: Evaluation
# -------------------------------
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# Step 8: Predict New Data
# -------------------------------
new_data = np.array([[120, 100, 1]])  # Wedding
prediction = model.predict(new_data)
print("Predicted Waste (kg):", round(prediction[0], 2))

# -------------------------------
# Step 9: Graphs
# -------------------------------

# 1. Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Waste")
plt.ylabel("Predicted Waste")
plt.title("Actual vs Predicted Waste")
plt.show()

# 2. Food Prepared vs Waste
plt.figure()
plt.scatter(df['Food_Prepared'], df['Waste'])
plt.xlabel("Food Prepared")
plt.ylabel("Waste")
plt.title("Food Prepared vs Waste")
plt.show()


# 4. Histogram
plt.figure()
plt.hist(df['Waste'], bins=10)
plt.xlabel("Waste")
plt.ylabel("Frequency")
plt.title("Distribution of Food Waste")
plt.show()

# 5. Bar Chart Comparison
sample_actual = y_test.values[:10]
sample_pred = y_pred[:10]

x = np.arange(len(sample_actual))

plt.figure()
plt.bar(x - 0.2, sample_actual, width=0.4)
plt.bar(x + 0.2, sample_pred, width=0.4)
plt.xlabel("Samples")
plt.ylabel("Waste")
plt.title("Actual vs Predicted Comparison")
plt.show()
