import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/cr1825/ai-project-data-sheet/main/EdgeAI_Robot_Motion_Control_Dataset.csv"
data = pd.read_csv(url)

print("✅ Data Loaded Successfully")
print("Shape:", data.shape)
print("Columns:", data.columns.tolist())

# Ensure correct column
if "Joint_ID" not in data.columns:
    raise ValueError("❌ 'Joint_ID' column not found!")

joint_ids = sorted(data["Joint_ID"].unique())
results = []

for jid in joint_ids:
    print(f"\n🤖 Training model for Joint ID: {jid}")
    joint_data = data[data["Joint_ID"] == jid]

    # Input features (desired position + velocity)
    X = joint_data[["Desired_Position", "Desired_Velocity"]]
    # Output (actual position + velocity)
    y = joint_data[["Actual_Position", "Actual_Velocity"]]

    # Train model
    rf = RandomForestRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Predict
    y_pred = rf.predict(X)

    # Ensure 2D shape
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Extract actual and predicted values
    y_true_path = y["Actual_Position"].values.reshape(-1, 1)
    y_true_vel  = y["Actual_Velocity"].values.reshape(-1, 1)
    y_pred_path = y_pred[:, 0].reshape(-1, 1)
    y_pred_vel  = y_pred[:, 1].reshape(-1, 1)

    # Compute metrics
    r2_total = r2_score(y, y_pred)
    rmse_total = np.sqrt(mean_squared_error(y, y_pred))

    # Deviation values
    path_dev = np.abs(y_true_path - y_pred_path)
    vel_dev  = np.abs(y_true_vel - y_pred_vel)

    avg_path_dev = np.mean(path_dev)
    avg_vel_dev  = np.mean(vel_dev)
    max_path_dev = np.max(path_dev)
    max_vel_dev  = np.max(vel_dev)

    # Save results
    results.append({
        "Joint_ID": jid,
        "R2_Total": r2_total,
        "RMSE_Total": rmse_total,
        "Avg_Path_Deviation": avg_path_dev,
        "Max_Path_Deviation": max_path_dev,
        "Avg_Velocity_Deviation": avg_vel_dev,
        "Max_Velocity_Deviation": max_vel_dev
    })

    # Print summary
    print(f"📊 R²: {r2_total:.4f}, RMSE: {rmse_total:.4f}")
    print(f"Path Deviation → Avg: {avg_path_dev:.6f}, Max: {max_path_dev:.6f}")
    print(f"Velocity Deviation → Avg: {avg_vel_dev:.6f}, Max: {max_vel_dev:.6f}")

    # Print first 10 deviation samples
    print("\n🔹 First 10 Path Deviation values:")
    print(path_dev[:10].flatten())
    print("🔹 First 10 Velocity Deviation values:")
    print(vel_dev[:10].flatten())

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(y_true_path[:100], label="Actual Position", color='blue')
    plt.plot(y_pred_path[:100], label="Predicted Position", color='orange', linestyle='dashed')
    plt.title(f"Joint {jid} - Position Prediction (First 100 Samples)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(y_true_vel[:100], label="Actual Velocity", color='green')
    plt.plot(y_pred_vel[:100], label="Predicted Velocity", color='red', linestyle='dashed')
    plt.title(f"Joint {jid} - Velocity Prediction (First 100 Samples)")
    plt.legend()
    plt.show()

# Summary
summary_df = pd.DataFrame(results)
print("\n🏁 Overall Model Summary:")
print(summary_df)
