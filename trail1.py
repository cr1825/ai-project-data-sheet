import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ===========================
# 1️⃣ Load Dataset
# ===========================
url = "https://raw.githubusercontent.com/cr1825/ai-project-data-sheet/main/EdgeAI_Robot_Motion_Control_Dataset.csv"
data = pd.read_csv(url)

print("✅ Data Loaded Successfully")
print("Shape:", data.shape)

if "Joint_ID" not in data.columns:
    raise ValueError("❌ 'Joint_ID' column not found!")

joint_ids = sorted(data["Joint_ID"].unique())

# ===========================
# 2️⃣ Train Model for Each Joint
# ===========================
results = []
models = {}

print("\n🚀 Training Random Forest Models for Each Joint...\n")

for jid in joint_ids:
    joint_data = data[data["Joint_ID"] == jid]

    X = joint_data[["Desired_Position", "Desired_Velocity"]]
    y = joint_data[["Actual_Position", "Actual_Velocity"]]

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=25, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    models[jid] = rf

    # Training performance summary
    y_pred = rf.predict(X)
    r2_total = r2_score(y, y_pred)
    rmse_total = np.sqrt(mean_squared_error(y, y_pred))
    avg_path_dev = np.mean(np.abs(y["Actual_Position"] - y_pred[:, 0]))
    avg_vel_dev = np.mean(np.abs(y["Actual_Velocity"] - y_pred[:, 1]))

    results.append({
        "Joint_ID": jid,
        "R2_Total": r2_total,
        "RMSE_Total": rmse_total,
        "Avg_Path_Deviation": avg_path_dev,
        "Avg_Velocity_Deviation": avg_vel_dev
    })

summary_df = pd.DataFrame(results)
print("\n🏁 Overall Model Summary per Joint:")
print(summary_df)

# ===========================
# 3️⃣ Prediction Function for All 6 Joints
# ===========================
def predict_all_joints(desired_positions, desired_velocities):
    preds = []
    pred_pos = []
    pred_vel = []

    for i, jid in enumerate(range(1, 7)):
        model = models.get(jid)
        if model is None:
            raise ValueError(f"Model for Joint {jid} not found.")
        
        input_data = pd.DataFrame({
            "Desired_Position": [desired_positions[i]],
            "Desired_Velocity": [desired_velocities[i]]
        })
        predicted = model.predict(input_data)
        actual_pos, actual_vel = predicted[0]
        
        pos_dev = abs(desired_positions[i] - actual_pos)
        vel_dev = abs(desired_velocities[i] - actual_vel)
        
        preds.append({
            "Joint_ID": f"J{jid}",
            "Desired_Position": desired_positions[i],
            "Pred_Actual_Position": actual_pos,
            "Position_Deviation": pos_dev,
            "Desired_Velocity": desired_velocities[i],
            "Pred_Actual_Velocity": actual_vel,
            "Velocity_Deviation": vel_dev
        })
        pred_pos.append(actual_pos)
        pred_vel.append(actual_vel)
    
    result_df = pd.DataFrame(preds)

    # ---- Compute overall R² & RMSE for user predictions ----
    r2_pos = r2_score(desired_positions, pred_pos)
    r2_vel = r2_score(desired_velocities, pred_vel)
    rmse_pos = np.sqrt(mean_squared_error(desired_positions, pred_pos))
    rmse_vel = np.sqrt(mean_squared_error(desired_velocities, pred_vel))

    print("\n🔹 Predicted Deviation Summary for All 6 Joints:")
    print(result_df)
    print("\n📈 Overall R² and RMSE for User Inputs:")
    print(f"Position → R²: {r2_pos:.4f}, RMSE: {rmse_pos:.4f}")
    print(f"Velocity → R²: {r2_vel:.4f}, RMSE: {rmse_vel:.4f}")

    # ---- Plot deviations ----
    plt.figure(figsize=(10, 5))
    plt.bar(result_df["Joint_ID"], result_df["Position_Deviation"], alpha=0.7, label="Position Deviation")
    plt.bar(result_df["Joint_ID"], result_df["Velocity_Deviation"], alpha=0.7, label="Velocity Deviation")
    plt.title("Predicted Deviation per Joint (Position & Velocity)")
    plt.xlabel("Joint ID")
    plt.ylabel("Deviation")
    plt.legend()
    plt.show()
    
    return result_df, r2_pos, r2_vel, rmse_pos, rmse_vel

# ===========================
# 4️⃣ User Input Section
# ===========================
print("\n🧠 Enter Desired Positions and Velocities for 6 Joints (J1–J6):")

desired_positions = []
desired_velocities = []

for i in range(1, 7):
    dp = float(input(f"Enter Desired Position for Joint {i}: "))
    dv = float(input(f"Enter Desired Velocity for Joint {i}: "))
    desired_positions.append(dp)
    desired_velocities.append(dv)

# ===========================
# 5️⃣ Predict and Display Results
# ===========================
predicted_results, r2p, r2v, rmsep, rmsev = predict_all_joints(desired_positions, desired_velocities)

# Optionally save results
save_choice = input("\n💾 Do you want to save the results to CSV? (y/n): ").strip().lower()
if save_choice == 'y':
    predicted_results.to_csv("Predicted_Joint_Deviations.csv", index=False)
    print("✅ Results saved as 'Predicted_Joint_Deviations.csv'")
