import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Path
csv_path = "../aaf_features/aaf_41merged_features.csv"
model_save_path = "../aaf_models/FINAL_aaf_41xgb_age_predictor.pkl"
plot_save_path = "../aaf_features/visualization/FINAL_aaf_xgb_age_prediction_41feat.png"

# 1. Data loading
df = pd.read_csv(csv_path)
X = df.drop(columns=["age", "filename"], errors="ignore")
y = df["age"]

# 2. Train / test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model training
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 4. Save model
joblib.dump(model, model_save_path)

# 5. Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("All age model : 41 features]")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# 6. Visualization
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title("XGBoost Prediction vs Actual (All age, 41 features)")
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_save_path)
plt.close()
