from extract_feature_opencv_normalized import extract_opencv_features
from extract_feature_mediapipe_normalized import extract_mediapipe_features
from extract_feature_mediapipe_extra import extract_additional_features
from moremore_features import extract_combined_features
from extract_20more_feature import extract_expanded_features


ordered_feature_names = [
    "nose_len", "mouth_width", "mouth_height", "eye_dist",
    "eye_ratio", "eye_slope", "face_width", "face_height", "face_ratio_x",
    "nose_to_mouth_dist", "mouth_to_jaw_ratio",
    "brightness", "wrinkle_score", "texture_uniformity",
    "face_ratio_y", "mouth_to_eye_ratio", "symmetry_score",
    "brightness_std", "contrast", "laplacian_var", "aspect_ratio",
    "mouth_width_div_face_width", "nose_len_div_face_height",
    "eye_dist_div_face_width", "face_width_div_face_height",
    "mouth_height_div_nose_to_mouth", "mouth_to_eye_div_face_height",
    "mouth_width_div_nose_len", "eye_dist_div_nose_len",
    "brightness_std_div_contrast", "laplacian_var_div_brightness_std",
    "eye_y_diff_norm", "jaw_asymmetry_norm", "mouth_corner_y_diff_norm",
    "eyebrow_height_diff_norm", "cheekbone_x_diff_norm",
    "gray_mean", "gray_std", "gray_skewness", "entropy", "lbp_uniformity"
]


def extract_41_features(image_path: str) -> dict:

    b, w, t = extract_opencv_features(image_path) 
    m1 = extract_mediapipe_features(image_path)  
    m2 = extract_additional_features(image_path)  
    m3 = extract_combined_features(image_path)  
    m4 = extract_expanded_features(image_path)  

    m3_keys = [
        "face_ratio_y", "mouth_to_eye_ratio", "symmetry_score",
        "brightness_std", "contrast", "laplacian_var", "aspect_ratio"
    ]
    m3_dict = dict(zip(m3_keys, m3))


    feature_dict = {
        "nose_len": m1["nose_len"],
        "mouth_width": m1["mouth_width"],
        "mouth_height": m1["mouth_height"],
        "eye_dist": m1["eye_dist"],
        "eye_ratio": m2["eye_ratio"],
        "eye_slope": m2["eye_slope"],
        "face_width": m2["face_width"],
        "face_height": m2["face_height"],
        "face_ratio_x": m2["face_ratio"],
        "nose_to_mouth_dist": m2["nose_to_mouth_dist"],
        "mouth_to_jaw_ratio": m2["mouth_to_jaw_ratio"],
        "brightness": b,
        "wrinkle_score": w,
        "texture_uniformity": t,
        **m3_dict,
        **m4  
    }


    ordered = {k: feature_dict[k] for k in ordered_feature_names}
    return ordered