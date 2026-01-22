### grain_analysis_modular/analysis.py
import cv2
import numpy as np
from utils import touches_border

def analyze_grains(image_path, scale_pixel, min_area_px=10, border_pad=0):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be loaded.")

    h, w = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    rejected_border = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= min_area_px:
            continue
        if touches_border(c, w, h, pad=border_pad):
            rejected_border.append(c)
            continue
        valid_contours.append(c)

    data_list = []
    for i, c in enumerate(valid_contours, start=1):
        area_px = cv2.contourArea(c)
        diameter_px = 2 * np.sqrt(area_px / np.pi)
        diameter_um = diameter_px * scale_pixel
        area_um2 = area_px * (scale_pixel ** 2)

        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        perimeter = cv2.arcLength(c, True)
        form_factor = 4 * np.pi * area_px / (perimeter ** 2) if perimeter != 0 else 0

        data_list.append({
            "Grain #": i,
            "Diameter (µm)": diameter_um,
            "Area (µm²)": area_um2,
            "X Coordinate": cx,
            "Y Coordinate": cy,
            "Form Factor": form_factor
        })

    grain_diameters = [d["Diameter (µm)"] for d in data_list]

    if len(grain_diameters) > 0:
        mean_diameter = np.mean(grain_diameters)
        median_diameter = np.median(grain_diameters)
        std_diameter = np.std(grain_diameters)
    else:
        mean_diameter = median_diameter = std_diameter = np.nan

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_image, valid_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(color_image, rejected_border, -1, (0, 0, 255), 2)

    return data_list, mean_diameter, median_diameter, std_diameter, len(valid_contours), len(rejected_border), color_image, image_path
