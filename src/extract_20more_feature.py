import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd
from scipy.stats import skew
from glob import glob

mp_face_mesh = mp.solutions.face_mesh

def get_xy(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

def calc_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).ravel()
    hist /= hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calc_lbp_uniformity(gray):
    lbp_img = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i,j]
            binary = ''.join(['1' if gray[i+dy,j+dx] > center else '0' 
                              for dx,dy in [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]])
            lbp_img[i,j] = int(binary, 2)
    hist, _ = np.histogram(lbp_img.ravel(), bins=256, range=(0, 256))
    return np.std(hist)

def extract_expanded_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    brightness_std = np.std(gray)
    contrast = np.max(gray) - np.min(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    entropy = calc_entropy(gray)
    lbp_uniformity = calc_lbp_uniformity(gray)
    gray_mean = np.mean(gray)
    gray_std = np.std(gray)
    gray_skew = skew(gray.ravel())

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks: return None
        lm = results.multi_face_landmarks[0].landmark

        eye_L = get_xy(lm, 33, w, h)
        eye_R = get_xy(lm, 263, w, h)
        mouth_L = get_xy(lm, 61, w, h)
        mouth_R = get_xy(lm, 291, w, h)
        nose_top = get_xy(lm, 6, w, h)
        nose_tip = get_xy(lm, 195, w, h)
        chin = get_xy(lm, 152, w, h)
        jaw_L = get_xy(lm, 234, w, h)
        jaw_R = get_xy(lm, 454, w, h)
        eye_center = (eye_L + eye_R) / 2
        mouth_center = (get_xy(lm, 13, w, h) + get_xy(lm, 14, w, h)) / 2

        eye_dist = np.linalg.norm(eye_L - eye_R)
        mouth_width = np.linalg.norm(mouth_L - mouth_R)
        nose_len = np.linalg.norm(nose_top - nose_tip)
        face_width = np.linalg.norm(jaw_L - jaw_R)
        face_height = np.linalg.norm(chin - nose_top)
        mouth_height = np.linalg.norm(get_xy(lm,13,w,h) - get_xy(lm,14,w,h))
        mouth_to_eye_ratio = np.linalg.norm(mouth_center - eye_center)
        nose_to_mouth_dist = np.linalg.norm(nose_tip - mouth_center)

        feature_dict = {
            # ratio based
            "mouth_width_div_face_width": mouth_width / (face_width + 1e-6),
            "nose_len_div_face_height": nose_len / (face_height + 1e-6),
            "eye_dist_div_face_width": eye_dist / (face_width + 1e-6),
            "face_width_div_face_height": face_width / (face_height + 1e-6),
            "mouth_height_div_nose_to_mouth": mouth_height / (nose_to_mouth_dist + 1e-6),
            "mouth_to_eye_div_face_height": mouth_to_eye_ratio / (face_height + 1e-6),
            "mouth_width_div_nose_len": mouth_width / (nose_len + 1e-6),
            "eye_dist_div_nose_len": eye_dist / (nose_len + 1e-6),
            "brightness_std_div_contrast": brightness_std / (contrast + 1e-6),
            "laplacian_var_div_brightness_std": laplacian_var / (brightness_std + 1e-6),

            # symmetry based, normalized
            "eye_y_diff_norm": abs(eye_L[1] - eye_R[1]) / (eye_dist + 1e-6),
            "jaw_asymmetry_norm": abs(jaw_L[0] - jaw_R[0]) / (eye_dist + 1e-6),
            "mouth_corner_y_diff_norm": abs(mouth_L[1] - mouth_R[1]) / (eye_dist + 1e-6),
            "eyebrow_height_diff_norm": abs(get_xy(lm,65,w,h)[1] - get_xy(lm,295,w,h)[1]) / (eye_dist + 1e-6),
            "cheekbone_x_diff_norm": abs(get_xy(lm,127,w,h)[0] - get_xy(lm,356,w,h)[0]) / (eye_dist + 1e-6),

            # cv based
            "gray_mean": gray_mean,
            "gray_std": gray_std,
            "gray_skewness": gray_skew,
            "entropy": entropy,
            "lbp_uniformity": lbp_uniformity
        }

        return feature_dict

def main():
    image_dir = "../aaf_data/aglined_faces/"
    output_csv = "../aaf_features/aaf_20expanded_features_normalized.csv"
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))

    records = []
    for path in image_paths:
        result = extract_expanded_features(path)
        if result:
            result["filename"] = os.path.basename(path)
            records.append(result)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"saved: {output_csv} ({df.shape[0]} samples, {df.shape[1]-1} features)")

if __name__ == "__main__":
    main()
