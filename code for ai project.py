# ---------------------------------------------------
# Random Forest Model - Predict Actual Path + Deviation
# ---------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# 1️⃣ Load dataset
url = "https://raw.githubusercontent.com/cr1825/ai-project-data-sheet/main/EdgeAI_Robot_Motion_Control_Dataset.csv"
data = pd.read_csv(url)

print("✅ Data Loaded Successfully")
print("Shape:", data.shape)
print("Columns:", data.columns.tolist())

# 2️⃣ Identify desired and actual path columns
desired_cols = [col for col in data.columns if "desired" in col.lower()]
actual_cols = [col for col in data.columns if "actual" in col.lower()]

if not desired_cols or not actual_cols:
    raise ValueError("❌ Couldn't find 'desired' or 'actual' columns. Please check dataset column names.")

X = data[desired_cols]
y = data[actual_cols]

print(f"\nInput features (Desired Path): {len(desired_cols)} columns")
print(f"Output features (Actual Path): {len(actual_cols)} columns")

# 3️⃣ Train Random Forest on entire dataset
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

# 4️⃣ Predict actual path for all samples
y_pred = rf.predict(X)

# 5️⃣ Evaluate model
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# 6️⃣ Compute deviation (Euclidean distance)
deviation = np.linalg.norm(y.values - y_pred, axis=1)
avg_dev = np.mean(deviation)
max_dev = np.max(deviation)

# 7️⃣ Print metrics
print("\n📊 Model Evaluation Results")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Average Deviation: {avg_dev:.4f}")
print(f"Maximum Deviation: {max_dev:.4f}")

# 8️⃣ Print all deviation values
print("\n🔹 Deviation values for each sample:")
for i, d in enumerate(deviation):
    print(f"Sample {i+1}: Deviation = {d:.4f}")

# 9️⃣ Plot a comparison for visualization
plt.figure(figsize=(8,5))
plt.plot(y.values[:200,0], label='Actual Path (first dimension)')
plt.plot(y_pred[:200,0], label='Predicted Path (first dimension)', linestyle='dashed')
plt.title("Actual vs Predicted Path (First 200 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Path Value")
plt.legend()
plt.show()

# 🔟 Save model
joblib.dump(rf, "random_forest_path_model.pkl")
print("\n💾 Model saved as 'random_forest_path_model.pkl'")
