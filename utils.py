### grain_analysis_modular/utils.py
import cv2 
def touches_border(contour, w, h, pad=0):
    x, y, cw, ch = cv2.boundingRect(contour)
    return (x <= pad or y <= pad or (x + cw) >= (w - 1 - pad) or (y + ch) >= (h - 1 - pad))
