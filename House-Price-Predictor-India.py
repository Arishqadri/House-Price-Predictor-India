import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ==================================================
# STEP 1: CREATE INDIAN REAL ESTATE DATA (CSV)
# ==================================================
np.random.seed(42)
n_houses = 300

# Generating features typical for Indian apartments/houses
data = {
    'Area_sqft': np.random.randint(500, 4500, n_houses), # Common flat/villa sizes
    'BHK': np.random.randint(1, 6, n_houses),            # 1 to 5 BHK
    'Age_years': np.random.randint(0, 25, n_houses),     # New to 25 year old buildings
}

# Indian Pricing Logic (Approx. ₹8,000 per sqft + BHK premium - Age depreciation)
prices_lakhs = []
for i in range(n_houses):
    # Base price (₹8k per sqft) + 10 Lakh per BHK - 0.5 Lakh per year of age
    p = (data['Area_sqft'][i] * 0.08) + (data['BHK'][i] * 10) - (data['Age_years'][i] * 0.5)
    # Add a random 'Location Factor' noise (±10 Lakhs)
    p += np.random.randint(-10, 10)
    prices_lakhs.append(round(p, 2))

data['Price_Lakhs'] = prices_lakhs
df = pd.DataFrame(data)
df.to_csv('india_house_data.csv', index=False)
print(" Created 'india_house_data.csv' with 300 records.")

# ==================================================
# STEP 2: TRAIN THE MODEL
# ==================================================
X = df[['Area_sqft', 'BHK', 'Age_years']]
y = df['Price_Lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# ==================================================
# STEP 3: PREDICTION & USER INTERFACE
# ==================================================
print("\n--- 🇮🇳 BHARAT REAL ESTATE PREDICTOR ---")
print("Estimate the value of your property in Lakhs.")

u_sqft = float(input("Enter Total Area in sq. ft. (e.g., 1200): "))
u_bhk = int(input("Enter BHK (e.g., 2 or 3): "))
u_age = int(input("Enter Age of property in years: "))

# Model Prediction
user_input = np.array([[u_sqft, u_bhk, u_age]])
predicted_price = model.predict(user_input)[0]

# Display Result in Indian Format
if predicted_price >= 100:
    crores = predicted_price / 100
    print(f"\n Estimated Value: ₹{crores:.2f} Crores")
else:
    print(f"\n Estimated Value: ₹{predicted_price:.2f} Lakhs")

# Simple Accuracy Check
test_preds = model.predict(X_test)
avg_error = mean_absolute_error(y_test, test_preds)
print(f" Model Confidence: Average error is ±{avg_error:.2f} Lakhs")