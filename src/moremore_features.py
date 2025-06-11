import os
from absl import logging
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# Path
IMAGE_DIR = "../aaf_data/aglined_faces/"
OUTPUT_CSV = "../aaf_features/aaf_combined_additional_features.csv"

# Feature extraction
def extract_combined_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OpenCV 
    brightness_std = np.std(gray)
    contrast = float(np.max(gray) - np.min(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    aspect_ratio = float(w / h)

    # Mediapipe
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            raise ValueError(f"Failed to detect face: {image_path}")
        landmarks = results.multi_face_landmarks[0].landmark
        def get_xy(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        left_eye = get_xy(33)
        right_eye = get_xy(263)
        nose_tip = get_xy(1)
        chin = get_xy(152)
        mouth_top = get_xy(13)
        mouth_bottom = get_xy(14)
        left_jaw = get_xy(234)
        right_jaw = get_xy(454)

        face_height = np.linalg.norm(chin - nose_tip)
        face_width = np.linalg.norm(left_jaw - right_jaw)
        face_ratio = face_height / (face_width + 1e-6)

        eye_center = (left_eye + right_eye) / 2
        mouth_center = (mouth_top + mouth_bottom) / 2
        mouth_to_eye_dist = np.linalg.norm(eye_center - mouth_center)
        mouth_to_eye_ratio = mouth_to_eye_dist / (face_height + 1e-6)

        symmetry_pairs = [(33, 263), (234, 454), (61, 291), (13, 14)]
        symmetry_score = np.mean([np.linalg.norm(get_xy(i) - get_xy(j)) for i, j in symmetry_pairs])

    return [
        face_ratio,
        mouth_to_eye_ratio,
        symmetry_score,
        brightness_std,
        contrast,
        laplacian_var,
        aspect_ratio
    ]
    
    
def batch_process_images():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]

    records = []
    for i, filename in enumerate(image_files):
        try:
            features = extract_combined_features(os.path.join(IMAGE_DIR, filename))
            records.append([filename] + features)
        except Exception as e:
            print(f"Failed: {filename} -> {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"✅ {i+1} image processed")

    columns = ["filename", "face_ratio", "mouth_to_eye_ratio", "symmetry_score",
               "brightness_std", "contrast", "laplacian_var", "aspect_ratio"]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Total # of images: {len(df)}")


if __name__ == "__main__":
    batch_process_images()

