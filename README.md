# COSE471_data-science

**by XGBoost**

**1. Download the dataset (AAF)**
- [AAF Dataset](https://github.com/JingchunCheng/All-Age-Faces-Dataset)

**2. Install Requirements**
```bash
pip install opencv-python mediapipe pandas scipy scikit-learn matplotlib joblib xgboost shap absl-py
```

**3. Train the Model**
```bash
python FINAL_aaf_41xgb_train_2050.py
```

or

```bash
python FINAL_aaf_41xgb_train_0080.py
```

- Select age range (20 to 50 / 0 to 80) depending on your actual age
- Trains a XGBoost model on the AAF dataset (based on pre-extracted features in aaf_41merged_features.csv)
- Saves the model in FINAL_aaf_41xgb_age_predictor.pkl (or FINAL_aaf_41xgb_age_20to50_predictor.pkl)

**4. Run Prediction & Analysis (SHAP)**
```bash
python FINAL_predict_and_explain.py
```
- Input your actual age
- Loads the trained model
- Predicts age from an input image
- Computes SHAP values to explain feature contribution (e.g., eyes, forehead, mouth)
- Compares predicted age with true age and interprets appearance (young-looking / old-looking)

**5. Project Structure**
```bash
.
├── aaf_features
    ├── aaf_41merged_features.csv                           # Pre-extracted AAF data features
├── aaf_models
    ├── FINAL_aaf_41xgb_age_20to50_predictor.pkl            # Age 20 to 50 model
    ├── FINAL_aaf_41xgb_age_predictor.pkl                   # All age model
├── src
    ├── extract_20more_feature.py                           # Feature extraction 1
    ├── extract_feature_mediapipe_extra.py                  # Feature extraction 2
    ├── extract_feature_mediapipe_normalized.py             # Feature extraction 3
    ├── extract_feature_opencv_normalized.py                # Feature extraction 4
    ├── FINAL_aaf_41xgb_train_0080.py                       # Training all age model
    ├── FINAL_aaf_41xgb_train_2050.py                       # Training age 20 to 50 model
    ├── FINAL_explain_SHAP.py                               # Natural language explanation of SHAP
    ├── FINAL_extract_41feautures.py                        # Merged feature extraction
    ├── FINAL_predict_and_explain.py                        # Age prediction and SHAP explanation
    ├── moremore_features.py                                # Feature extraction 5
├── test_data                                               # Put your image file here
├── README.md
```

