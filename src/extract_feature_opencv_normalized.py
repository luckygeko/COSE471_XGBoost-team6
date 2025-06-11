import cv2
import numpy as np
import os
import pandas as pd
from glob import glob


def extract_opencv_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    brightness = np.mean(img_clahe)
    laplacian = cv2.Laplacian(img_clahe, cv2.CV_64F)
    wrinkle_score = laplacian.var()

    def local_binary_pattern(img):
        lbp_img = np.zeros_like(img)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                center = img[i, j]
                binary = ''
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]:
                    binary += '1' if img[i+dy, j+dx] > center else '0'
                lbp_img[i, j] = int(binary, 2)
        return lbp_img

    lbp = local_binary_pattern(img_clahe)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    texture_uniformity = np.std(hist)

    return brightness, wrinkle_score, texture_uniformity


def normalize_features(feature_list):
    data = np.array(feature_list)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    normalized = (data - min_vals) / (max_vals - min_vals + 1e-8)  # stablize
    return normalized


def main():
    image_dir = "../aaf_data/aglined_faces/"  ################################################### data path
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))

    feature_data = []
    filenames = []

    for i, path in enumerate(image_paths):
        try:
            b, w, t = extract_opencv_features(path)
            feature_data.append([b, w, t])
            filenames.append(os.path.basename(path))
        except:
            continue
        
        if (i + 1) % 10 == 0:
            print(f"{i + 1} images processed...")
        

    normalized_features = normalize_features(feature_data)

    df = pd.DataFrame(normalized_features, columns=["brightness", "wrinkle_score", "texture_uniformity"])
    df["filename"] = filenames

    df.to_csv("../aaf_features/aaf_opencv_normalized_features.csv", index=False) ####################################################### csv path
    print(df.head())
    

if __name__ == "__main__":
    main()