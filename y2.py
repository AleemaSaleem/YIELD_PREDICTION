import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# Load the CSV file
df = pd.read_csv("final_merged_weather_yieldd.csv")  # Adjust path if needed

# Split features and target
X = df.drop(columns=['kg_per_acre'])
y = df['kg_per_acre']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
joblib.dump(rf_reg, "yield_model.pkl") 
print("✅ Model saved as yield_model.pkl")

# Evaluate model
# y_pred_rf = rf_reg.predict(X_test)
# print("=== Random Forest Regressor Evaluation ===")
# print("MSE:", mean_squared_error(y_test, y_pred_rf))
# print("R² Score:", r2_score(y_test, y_pred_rf))

# # ---------- Prediction Section ----------

# # New data to predict
# new_data = {
#     "Year": 2025,
#     "tempmax": 27.5,
#     "tempmin": 10.2,
#     "precip": 0.35,
#     "humidity": 36.5,
#     "NDVI": 0.2,   # Change this to test < 0.2 logic
#     "NDMI": 0.21,
#     "MSAVI": 0.31
# }

# # Convert to DataFrame
# new_df = pd.DataFrame([new_data])

# # Yield prediction logic based on NDVI threshold
# if new_df.loc[0, "NDVI"] < 0.2:
#     print("\n=== Prediction Skipped ===")
#     print("Likely residential area detected (NDVI < 0.2). No yield prediction made.")
# else:
#     predicted_kg_per_acre = rf_reg.predict(new_df)
#     print("\n=== Prediction ===")
#     print("Predicted kg per acre:", predicted_kg_per_acre[0])

# # Notes for understanding:
# print("\n📌 Note:")
# print("Prediction depends heavily on NDVI, NDMI, and MSAVI.")
# print("Environmental variables (temperature, humidity, precipitation) play a secondary role.")
# # logic of ndvi less than 0.2 & depends heavily on NDVI, NDRE, MSAVI, and less on temp, humidity, and precipitation.