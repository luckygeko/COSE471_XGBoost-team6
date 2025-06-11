import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from glob import glob


mp_face_mesh = mp.solutions.face_mesh

def extract_mediapipe_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image {image_path} could not be read.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            raise ValueError(f"No face landmarks found in {image_path}")

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = img.shape

        def get_xy(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        left_eye = get_xy(33)
        right_eye = get_xy(263)
        nose_top = get_xy(6)
        nose_tip = get_xy(195)
        mouth_left = get_xy(61)
        mouth_right = get_xy(291)
        mouth_top = get_xy(13)
        mouth_bottom = get_xy(14)

        eye_dist = np.linalg.norm(left_eye - right_eye)
        nose_len = np.linalg.norm(nose_top - nose_tip)
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)

        #noramalize
        features = {
            "nose_len": nose_len / eye_dist,
            "mouth_width": mouth_width / eye_dist,
            "mouth_height": mouth_height / eye_dist,
            "eye_dist": eye_dist 
        }

        return features

def main():
    image_dir = "../aaf_data/aglined_faces/"    ###################################################### data path
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))

    os.makedirs("../aaf_features", exist_ok=True)

    feature_data = []
    filenames = []

    for path in image_paths:
        try:
            features = extract_mediapipe_features(path)
            feature_data.append(features)
            filenames.append(os.path.basename(path))
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            continue

    if len(feature_data) == 0:
        print("No MediaPipe features were extracted.")
        return

    df = pd.DataFrame(feature_data)
    df["filename"] = filenames

    df.to_csv("../aaf_features/aaf_mediapipe_normalized_features.csv", index=False)  ################################ csv path
    print("MediaPipe feature extraction complete.")
    print(df.head())

if __name__ == "__main__":
    main()