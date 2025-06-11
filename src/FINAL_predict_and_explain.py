import cv2
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sys
from FINAL_extract_41feautures import extract_41_features  
from FINAL_explain_SHAP import explain_shap  


# Path and input
image_path = "../test_data/josh.jpg"        
actual_age = int(input("Your Actual Age : "))                       # input actual age

#SELECT MODEL 20~50 / 00~80
model_path = "../aaf_models/FINAL_aaf_41xgb_age_20to50_predictor.pkl"
visual_save_path = "../test_data/SHAP/josh_shap.png"


# 1. extract features
features = extract_41_features(image_path)

if isinstance(features, dict):
    features = pd.DataFrame([features])
elif isinstance(features, pd.Series):
    features = features.to_frame().T

# 2. prediction
model = joblib.load(model_path)
predicted_age = model.predict(features)[0]

# 3. old / young looking
diff = predicted_age - actual_age
status = "older-looking" if diff >= 3 else "younger-looking" if diff <= -3 else "age-appropriate appearance"


print(f"\nActual Age: {actual_age}")
print(f"Predicted Age: {predicted_age:.2f}")
print(f" -->  {status} (Difference: {diff:.2f})")

# 4. SHAP
explainer = shap.Explainer(model)
shap_values = explainer(features)

# print top 5 feature
shap_df = pd.DataFrame({
    "feature": features.columns,
    "value": features.iloc[0],
    "shap": shap_values.values[0]
})
shap_df["abs_shap"] = shap_df["shap"].abs()
shap_df = shap_df.sort_values("abs_shap", ascending=False)



print("\n💡Main feature (Top 5):")
print(shap_df[["feature", "value", "shap"]].head(5))
print("\n")


print(explain_shap(shap_df.head(5)))

# 5. visualization
shap.plots.bar(shap_values[0], show=False)
plt.title(f"SHAP - Predicted Age {predicted_age:.1f}")
plt.tight_layout()
plt.savefig(visual_save_path)
plt.close()
print(f"\n SHAP Visualization Saved: {visual_save_path}")
