import cv2
import numpy as np
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

def prepare_image_for_analysis():
    """Bild laden, Maßstab setzen und Bild + Maßstab zurückgeben."""
    path, _ = QFileDialog.getOpenFileName(None, "Bild auswählen", "", "Bilder (*.png *.jpg *.jpeg *.tif *.tiff)")
    if not path:
        QMessageBox.warning(None, "Kein Bild ausgewählt", "Bitte wähle ein Bild aus.")
        return None, None

    image = cv2.imread(path)
    if image is None:
        QMessageBox.critical(None, "Fehler", "Bild konnte nicht geladen werden.")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    microns_per_pixel = get_scale_from_user(image_rgb)
    if microns_per_pixel is None:
        QMessageBox.warning(None, "Maßstab nicht gesetzt", "Der Maßstab wurde nicht korrekt gesetzt.")
        return None, None

    return image_rgb, microns_per_pixel


def get_scale_from_user(image_rgb):
    """
    Zeigt das Bild an und lässt den Nutzer zwei Punkte anklicken,
    fragt dann die reale Länge ab und berechnet den Maßstab (μm/Pixel).
    """
    coords = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, display_img.copy())
            if len(coords) == 2:
                cv2.destroyAllWindows()

    # Kopie für Anzeige
    display_img = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    window_name = "Maßstab setzen – klicke zwei Punkte an"
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, click_event)
    cv2.waitKey(0)

    if len(coords) != 2:
        return None

    pixel_dist = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))

    real_length, ok = QInputDialog.getDouble(
        None,
        "Reale Länge eingeben",
        "Reale Länge zwischen den Punkten (μm):",
        decimals=2,
        minimum=0.01,
        maximum=10000.0,
        value=100.0
    )

    if not ok or real_length <= 0:
        return None

    microns_per_pixel = real_length / pixel_dist
    return microns_per_pixel
