# analysis.py
# Flächenbasierte Korngrößenanalyse (ECD als Standard) – robust und abwärtskompatibel
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from utils import touches_border

def _as_gray(image_or_path: Union[str, np.ndarray]) -> Tuple[np.ndarray, Optional[str]]:
    """
    Nimmt entweder einen Bildpfad (str) oder ein bereits geladenes Bild (np.ndarray) entgegen
    und liefert ein Graustufenbild + den erkannten Pfad (oder None).
    """
    src_path = None
    if isinstance(image_or_path, str):
        src_path = image_or_path
        img = cv2.imread(image_or_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Image could not be loaded: {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        img = image_or_path
    else:
        raise TypeError("image_or_path must be a file path (str) or a numpy.ndarray image.")

    if img.ndim == 2:
        gray = img.copy()
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image shape.")

    return gray, src_path

def _grain_metrics(contour, microns_per_pixel: float, k_abs: float = 1.0) -> Optional[Dict[str, float]]:
    """
    Berechnet 2D-Metriken eines Korns aus der Kontur.
    k_abs ist eine *heuristische* Skalierung (Default 1.0 – aus). Keine physikalische 3D-Korrektur!
    """
    A_px = cv2.contourArea(contour)
    if A_px <= 0:
        return None

    P_px = cv2.arcLength(contour, True)

    # physikalische Einheiten
    A_um2 = A_px * (microns_per_pixel ** 2)  # µm²
    P_um  = P_px * microns_per_pixel        # µm

    # ECD (flächenäquivalenter Durchmesser) – Standard
    d_ECD = 2.0 * np.sqrt(A_um2 / np.pi)    # µm

    # Umfangs-äquivalenter Durchmesser (Heywood)
    d_Heywood = (P_um / np.pi) if P_um > 0 else np.nan

    # Zirkularität (1 = Kreis)
    C = (4.0 * np.pi * A_um2) / (P_um ** 2) if P_um > 0 else np.nan

    # Feret aus minAreaRect
    (cxr, cyr), (w_px, h_px), _ = cv2.minAreaRect(contour)
    feret_max = max(w_px, h_px) * microns_per_pixel
    feret_min = min(w_px, h_px) * microns_per_pixel
    aspect = (feret_max / feret_min) if feret_min > 0 else np.nan

    # Schwerpunkt (für Label/Export hilfreich)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # optionale (heuristische) Multiplikation – standardmäßig AUS
    d_ECD_abs = k_abs * d_ECD
    d_Heywood_abs = k_abs * d_Heywood

    return {
        "Diameter_ECD (µm)": float(d_ECD),
        "Diameter_Heywood (µm)": float(d_Heywood),
        "Diameter_ECD_abs (µm)": float(d_ECD_abs),
        "Diameter_Heywood_abs (µm)": float(d_Heywood_abs),
        "Area (µm²)": float(A_um2),
        "Perimeter (µm)": float(P_um),
        "Circularity": float(C) if C == C else np.nan,  # handle NaN
        "Feret_max (µm)": float(feret_max),
        "Feret_min (µm)": float(feret_min),
        "Aspect_Ratio": float(aspect) if aspect == aspect else np.nan,
        "X Coordinate": int(cx),
        "Y Coordinate": int(cy),
    }

def analyze_grains(
    image_or_path: Union[str, np.ndarray],
    scale_pixel: float,
    min_area_px: int = 10,
    border_pad: int = 0,
    k_abs: float = 1.0,
) -> Tuple[List[Dict[str, float]], float, float, float, int, int, np.ndarray, Optional[str]]:
    """
    Hauptfunktion der Flächenanalyse.
      - Segmentierung (Blur + Otsu + kleine Morphologie-Glättung)
      - Randkörner optional ausschließen
      - Pro Korn ECD & Formmaße berechnen
    Rückgabe (abwärtskompatibel):
      data_list, mean_ECD, median_ECD, std_ECD, n_valid, n_rejected, overlay_bgr, image_path_or_None
    """
    if scale_pixel is None or scale_pixel <= 0:
        raise ValueError("scale_pixel (µm/Pixel) muss > 0 sein.")

    gray, src_path = _as_gray(image_or_path)
    h, w = gray.shape[:2]

    # Vorverarbeitung + Otsu
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # leichte Morphologie zur Glättung
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours, rejected_border = [], []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        if touches_border(c, w, h, pad=border_pad):
            rejected_border.append(c)
            continue
        valid_contours.append(c)

    data_list: List[Dict[str, float]] = []
    for i, c in enumerate(valid_contours, start=1):
        m = _grain_metrics(c, microns_per_pixel=scale_pixel, k_abs=k_abs)
        if m is None:
            continue
        row = {"Grain #": i, **m}
        data_list.append(row)

    # Statistik (Standard auf Basis ECD 2D)
    ecd = np.array([d["Diameter_ECD (µm)"] for d in data_list], dtype=float)
    if ecd.size > 0:
        mean_d = float(np.nanmean(ecd))
        median_d = float(np.nanmedian(ecd))
        std_d = float(np.nanstd(ecd))
    else:
        mean_d = median_d = std_d = float("nan")

    # Ergebnis-Overlay
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if valid_contours:
        cv2.drawContours(overlay, valid_contours, -1, (0, 255, 0), 2)
    if rejected_border:
        cv2.drawContours(overlay, rejected_border, -1, (0, 0, 255), 2)

    return (
        data_list,
        mean_d,
        median_d,
        std_d,
        len(valid_contours),
        len(rejected_border),
        overlay,
        src_path,
    )
