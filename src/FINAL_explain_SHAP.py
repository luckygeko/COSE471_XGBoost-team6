def explain_shap(shap_df):
    
    explanation = "🔍 Top contributing facial features and their effects:\n"

    for _, row in shap_df.iterrows():
        f, v, shap_val = row['feature'], row['value'], row['shap']
        dir_txt = "more" if shap_val > 0 else "less"
        effect = "makes the person look older" if shap_val > 0 else "makes the person look younger"

        if f == "nose_len":
            reason = f"A {dir_txt} prominent nose {effect}."
        elif f == "mouth_width":
            reason = f"A {dir_txt} wide mouth {effect}."
        elif f == "mouth_height":
            reason = f"A {dir_txt} tall mouth opening {effect}."
        elif f == "eye_dist":
            reason = f"{dir_txt.capitalize()} space between the eyes {effect}."
        elif f == "eye_ratio":
            reason = f"{dir_txt.capitalize()} roundness of the eyes {effect}."
        elif f == "eye_slope":
            reason = f"{dir_txt.capitalize()} tilt of the eyes {effect}."
        elif f == "face_width":
            reason = f"A {dir_txt} wide-looking face {effect}."
        elif f == "face_height":
            reason = f"A {dir_txt} long-looking face {effect}."
        elif f == "face_ratio_x":
            reason = f"A {dir_txt} square-shaped face (wider than tall) {effect}."
        elif f == "nose_to_mouth_dist":
            reason = f"{dir_txt.capitalize()} distance between nose and lips {effect}."
        elif f == "mouth_to_jaw_ratio":
            reason = f"{dir_txt.capitalize()} gap from lips to chin {effect}."
        elif f == "brightness":
            reason = f"A {dir_txt} bright face overall {effect}."
        elif f == "wrinkle_score":
            reason = f"{dir_txt.capitalize()} visible wrinkles {effect}."
        elif f == "texture_uniformity":
            reason = f"{dir_txt.capitalize()} evenness of skin texture {effect}."
        elif f == "face_ratio_y":
            reason = f"A {dir_txt} vertically-stretched facial shape {effect}."
        elif f == "mouth_to_eye_ratio":
            reason = f"{dir_txt.capitalize()} spacing between lips and eyes {effect}."
        elif f == "symmetry_score":
            reason = f"{dir_txt.capitalize()} overall facial symmetry {effect}."
        elif f == "brightness_std":
            reason = f"{dir_txt.capitalize()} variation in face brightness {effect}."
        elif f == "contrast":
            reason = f"{dir_txt.capitalize()} contrast between light and dark areas {effect}."
        elif f == "laplacian_var":
            reason = f"{dir_txt.capitalize()} skin sharpness or wrinkle detail {effect}."
        elif f == "aspect_ratio":
            reason = f"A {dir_txt} tall face compared to its width {effect}."
        elif f == "mouth_width_div_face_width":
            reason = f"{dir_txt.capitalize()} mouth size compared to face width {effect}."
        elif f == "nose_len_div_face_height":
            reason = f"{dir_txt.capitalize()} nose size compared to face height {effect}."
        elif f == "eye_dist_div_face_width":
            reason = f"{dir_txt.capitalize()} eye spacing relative to face width {effect}."
        elif f == "face_width_div_face_height":
            reason = f"{dir_txt.capitalize()} width of the face compared to height {effect}."
        elif f == "mouth_height_div_nose_to_mouth":
            reason = f"{dir_txt.capitalize()} vertical mouth opening compared to nose-lip distance {effect}."
        elif f == "mouth_to_eye_div_face_height":
            reason = f"{dir_txt.capitalize()} lip-eye distance compared to face height {effect}."
        elif f == "mouth_width_div_nose_len":
            reason = f"{dir_txt.capitalize()} mouth width compared to nose length {effect}."
        elif f == "eye_dist_div_nose_len":
            reason = f"{dir_txt.capitalize()} eye spacing relative to nose length {effect}."
        elif f == "brightness_std_div_contrast":
            reason = f"{dir_txt.capitalize()} brightness variation versus contrast {effect}."
        elif f == "laplacian_var_div_brightness_std":
            reason = f"{dir_txt.capitalize()} skin detail versus brightness variation {effect}."
        elif f == "eye_y_diff_norm":
            reason = f"{dir_txt.capitalize()} uneven eye height {effect}."
        elif f == "jaw_asymmetry_norm":
            reason = f"{dir_txt.capitalize()} uneven jaw shape {effect}."
        elif f == "mouth_corner_y_diff_norm":
            reason = f"{dir_txt.capitalize()} uneven mouth corners {effect}."
        elif f == "eyebrow_height_diff_norm":
            reason = f"{dir_txt.capitalize()} eyebrow height imbalance {effect}."
        elif f == "cheekbone_x_diff_norm":
            reason = f"{dir_txt.capitalize()} cheekbone position imbalance {effect}."
        elif f == "gray_mean":
            reason = f"{dir_txt.capitalize()} average skin tone (brightness) {effect}."
        elif f == "gray_std":
            reason = f"{dir_txt.capitalize()} uneven skin brightness {effect}."
        elif f == "gray_skewness":
            reason = f"{dir_txt.capitalize()} imbalance in brightness across the face {effect}."
        elif f == "entropy":
            reason = f"{dir_txt.capitalize()} overall facial visual complexity {effect}."
        elif f == "lbp_uniformity":
            reason = f"{dir_txt.capitalize()} smoothness of skin pattern {effect}."
        else:
            reason = f"The feature '{f}' being {dir_txt} {effect}."

        explanation += f"- {reason} (SHAP: {shap_val:.3f})\n"

    return explanation
