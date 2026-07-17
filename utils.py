# utils.py
import cv2
import numpy as np

def touches_border(contour, width: int, height: int, pad: int = 0) -> bool:
    """
    True, wenn die Kontur den Bildrand (inkl. optionaler pad-Zone) berührt.
    Robust gegenüber konturformbedingten Ausreißern.
    """
    # Schneller Check per Bounding Box
    x, y, cw, ch = cv2.boundingRect(contour)
    if x <= pad or y <= pad or (x + cw) >= (width - 1 - pad) or (y + ch) >= (height - 1 - pad):
        return True

    # Exakterer Check: irgendein Punkt direkt auf Rand?
    pts = contour.reshape(-1, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]
    if (xs <= pad).any() or (ys <= pad).any() or (xs >= width - 1 - pad).any() or (ys >= height - 1 - pad).any():
        return True

    return False
