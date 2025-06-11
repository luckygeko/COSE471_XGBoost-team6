import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from glob import glob


mp_face_mesh = mp.solutions.face_mesh

def get_xy(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

def extract_additional_features(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise ValueError(f"No face found in {image_path}")
        lm = results.multi_face_landmarks[0].landmark

    eye_ratio = np.linalg.norm(get_xy(lm, 33, w, h) - get_xy(lm, 263, w, h)) / (np.linalg.norm(get_xy(lm, 159, w, h) - get_xy(lm, 145, w, h)) + 1e-5)
    eye_slope = np.arctan2(get_xy(lm, 263, w, h)[1] - get_xy(lm, 33, w, h)[1], get_xy(lm, 263, w, h)[0] - get_xy(lm, 33, w, h)[0]) * 180 / np.pi
    face_width = np.linalg.norm(get_xy(lm, 234, w, h) - get_xy(lm, 454, w, h))
    face_height = np.linalg.norm(get_xy(lm, 10, w, h) - get_xy(lm, 152, w, h))
    face_ratio = face_width / (face_height + 1e-5)
    nose_to_mouth_dist = np.linalg.norm(get_xy(lm, 2, w, h) - ((get_xy(lm, 13, w, h) + get_xy(lm, 14, w, h)) / 2))
    mouth_width = np.linalg.norm(get_xy(lm, 61, w, h) - get_xy(lm, 291, w, h))
    mouth_to_jaw_ratio = mouth_width / (face_width + 1e-5)

    return {
        "eye_ratio": eye_ratio,
        "eye_slope": eye_slope,
        "face_width": face_width,
        "face_height": face_height,
        "face_ratio": face_ratio,
        "nose_to_mouth_dist": nose_to_mouth_dist,
        "mouth_to_jaw_ratio": mouth_to_jaw_ratio
    }

def main():
    image_dir = "../aaf_data/aglined_faces/"  ########################################################## data path
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))

    feature_data = []
    filenames = []

    for i, path in enumerate(image_paths):
        try:
            features = extract_additional_features(path)
            features["filename"] = os.path.basename(path)
            feature_data.append(features)
        except Exception as e:
            print(f"Error with {path}: {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"{i + 1} images processed...")

    df = pd.DataFrame(feature_data)
    os.makedirs("../aaf_features", exist_ok=True)
    df.to_csv("../aaf_features/aaf_additional_mediapipe_features.csv", index=False) ############################# csv path
    print(df.head())

if __name__ == "__main__":
    main()
    
    
    