# gui_qt.py
# Qt-GUI mit PySide6 für die Flächenanalyse (ECD)
# Features:
#   - Maßstabsbestimmung (1-Klick & 2-Klick)
#   - Flächenanalyse
#   - Histogrammtyp-Auswahl (Anzahl / Flächensumme)
#   - Boxplot-Button
#   - Bildanzeige ohne Verzerrung (KeepAspectRatio)

from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import os
import math
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QMessageBox, QSpacerItem, QSizePolicy, QComboBox,
    QCheckBox, QDialog, QScrollArea, QFormLayout, QDoubleSpinBox, QInputDialog
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QMouseEvent
from PySide6.QtCore import Qt, QPoint, QSize

from analysis import analyze_grains
from plotting import show_results, show_boxplot, show_comparison

try:
    from export import export_results_to_csv  # type: ignore
except Exception:
    export_results_to_csv = None


def _cv_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    if bgr is None or bgr.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _load_image_as_qpixmap(path: str) -> Tuple[QPixmap, int, int]:
    img = _load_image_bgr(path)
    return _cv_to_qpixmap(img), img.shape[1], img.shape[0]


# ------------------------
# 2-Klick Maßstab
# ------------------------
class ScaleMeasureDialog(QDialog):
    def __init__(self, image_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Maßstab messen – 2 Punkte anklicken")
        self.orig_pixmap, self.orig_w, self.orig_h = _load_image_as_qpixmap(image_path)

        max_w, max_h = 1200, 900
        self.display_pixmap = self.orig_pixmap
        if self.orig_pixmap.width() > max_w or self.orig_pixmap.height() > max_h:
            self.display_pixmap = self.orig_pixmap.scaled(
                max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        self.scale_x = self.display_pixmap.width() / self.orig_w
        self.scale_y = self.display_pixmap.height() / self.orig_h

        self.label = QLabel()
        self.label.setPixmap(self.display_pixmap)
        self.label.setFixedSize(self.display_pixmap.size())

        scroll = QScrollArea()
        scroll.setWidget(self.label)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        info = QLabel("Klicke zwei Punkte (Enden der Skalenleiste) und gib die reale Distanz (µm) ein.")
        info.setWordWrap(True)

        form = QFormLayout()
        self.um_input = QDoubleSpinBox()
        self.um_input.setRange(1e-12, 1e12)
        self.um_input.setDecimals(6)
        self.um_input.setValue(100.0)
        form.addRow("Reale Distanz [µm]:", self.um_input)

        btn_row = QHBoxLayout()
        self.btn_clear = QPushButton("Zurücksetzen")
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_ok.setEnabled(False)
        btn_row.addWidget(self.btn_clear)
        btn_row.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)

        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(scroll)
        layout.addLayout(form)
        layout.addLayout(btn_row)

        self.points: List[QPoint] = []
        self.scale_umpx: Optional[float] = None

        self.label.mousePressEvent = self._on_click  # type: ignore
        self.btn_clear.clicked.connect(self._clear)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._finish)
        self.um_input.valueChanged.connect(self._update_ok_enabled)

        self._update_overlay()
        self._update_ok_enabled()

    def _on_click(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        x = max(0, min(self.display_pixmap.width() - 1, pos.x()))
        y = max(0, min(self.display_pixmap.height() - 1, pos.y()))
        self.points.append(QPoint(x, y))
        if len(self.points) > 2:
            self.points = self.points[-2:]
        self._update_overlay()
        self._update_ok_enabled()

    def _clear(self) -> None:
        self.points.clear()
        self._update_overlay()
        self._update_ok_enabled()

    def _update_overlay(self) -> None:
        pm = QPixmap(self.display_pixmap)
        painter = QPainter(pm)
        painter.setPen(QPen(Qt.GlobalColor.green, 2))
        for p in self.points:
            painter.drawEllipse(p, 4, 4)
        if len(self.points) == 2:
            painter.drawLine(self.points[0], self.points[1])
            dx = self.points[1].x() - self.points[0].x()
            dy = self.points[1].y() - self.points[0].y()
            painter.setPen(QPen(Qt.GlobalColor.yellow, 1))
            painter.drawText(self.points[1] + QPoint(8, -8), f"{math.hypot(dx, dy):.1f} px (Anzeige)")
        painter.end()
        self.label.setPixmap(pm)

    def _update_ok_enabled(self) -> None:
        ok = (len(self.points) == 2) and (self.um_input.value() > 0.0)
        self.btn_ok.setEnabled(ok)

    def _finish(self) -> None:
        if len(self.points) != 2:
            return
        dx_disp = self.points[1].x() - self.points[0].x()
        dy_disp = self.points[1].y() - self.points[0].y()
        dx_orig = dx_disp / self.scale_x
        dy_orig = dy_disp / self.scale_y
        px_dist = math.hypot(dx_orig, dy_orig)
        if px_dist <= 0:
            QMessageBox.warning(self, "Fehler", "Die Punkte liegen zu nahe beieinander.")
            return
        dist_um = float(self.um_input.value())
        self.scale_umpx = dist_um / px_dist
        self.accept()


# -----------------------------
# 1-Klick Auto-Balken
# -----------------------------
class AutoScaleBarDialog(QDialog):
    def __init__(self, image_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto: Skalenbalken (1 Klick)")
        self.img_bgr = _load_image_bgr(image_path)
        self.orig_h, self.orig_w = self.img_bgr.shape[:2]

        self.orig_pixmap = _cv_to_qpixmap(self.img_bgr)
        max_w, max_h = 1200, 900
        self.display_pixmap = self.orig_pixmap
        if self.orig_pixmap.width() > max_w or self.orig_pixmap.height() > max_h:
            self.display_pixmap = self.orig_pixmap.scaled(
                max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        self.scale_x = self.display_pixmap.width() / self.orig_w
        self.scale_y = self.display_pixmap.height() / self.orig_h

        self.label = QLabel()
        self.label.setPixmap(self.display_pixmap)
        self.label.setFixedSize(self.display_pixmap.size())

        info = QLabel("Klicke einmal auf oder sehr nahe an den Skalenbalken. Die Länge wird automatisch ermittelt.")
        info.setWordWrap(True)

        scroll = QScrollArea()
        scroll.setWidget(self.label)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_row = QHBoxLayout()
        self.btn_cancel = QPushButton("Abbrechen")
        btn_row.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        btn_row.addWidget(self.btn_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(scroll)
        layout.addLayout(btn_row)

        self.label.mousePressEvent = self._on_click  # type: ignore
        self.btn_cancel.clicked.connect(self.reject)

        self.scale_umpx: Optional[float] = None

    def _on_click(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        x_disp = max(0, min(self.display_pixmap.width() - 1, pos.x()))
        y_disp = max(0, min(self.display_pixmap.height() - 1, pos.y()))
        x = int(round(x_disp / self.scale_x))
        y = int(round(y_disp / self.scale_y))

        try:
            px_len = self._measure_bar_length_px_near(x, y)
        except Exception as e:
            QMessageBox.warning(self, "Automatik fehlgeschlagen",
                                f"Automatische Messung fehlgeschlagen: {e}\nBitte den 2-Klick-Fallback verwenden.")
            return

        if px_len is None or px_len <= 0:
            QMessageBox.information(self, "Nicht gefunden",
                                    "Konnte im Umfeld keinen Balken sicher erkennen.")
            return

        dist_um, ok = QInputDialog.getDouble(self, "Reale Balkenlänge (µm)",
                                             "Wie viele µm hat der Skalenbalken?",
                                             decimals=6, min=1e-12, max=1e12, value=100.0)
        if not ok:
            return

        self.scale_umpx = float(dist_um) / float(px_len)
        self.accept()

    def _measure_bar_length_px_near(self, x: int, y: int) -> Optional[float]:
        h, w = self.img_bgr.shape[:2]
        win = max(60, min(int(0.25 * min(h, w)), 400))
        x0, y0 = max(0, x - win // 2), max(0, y - win // 2)
        x1, y1 = min(w, x0 + win), min(h, y0 + win)
        roi = self.img_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        def pick_bar(mask: np.ndarray):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            cnt = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            w_rect, h_rect = rect[1]
            if min(w_rect, h_rect) == 0:
                return None
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
            if aspect < 3.0:
                return None
            return float(max(w_rect, h_rect))

        len1, len2 = pick_bar(th1), pick_bar(th2)
        return max(len1 or 0, len2 or 0)


# ------------------------
# Hauptfenster
# ------------------------
class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Korngrößenanalyse – Flächenmethode (ECD)")
        self.resize(1000, 740)

        self.image_path: Optional[str] = None
        self.overlay_bgr: Optional[np.ndarray] = None
        self.results: List[Dict] = []
        self.summary_text = ""
        self._last_drawn_size: Optional[QSize] = None

        layout = QVBoxLayout(self)

        # Datei & Maßstab
        file_row = QHBoxLayout()
        self.path_label = QLabel("Kein Bild geladen")
        self.btn_browse = QPushButton("Bild öffnen…")
        self.btn_browse.clicked.connect(self.on_browse)
        self.btn_measure_auto = QPushButton("Auto: Skalenbalken (1 Klick)")
        self.btn_measure_auto.clicked.connect(self.on_measure_auto)
        self.btn_measure = QPushButton("Maßstab messen")
        self.btn_measure.clicked.connect(self.on_measure)
        file_row.addWidget(self.path_label)
        file_row.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        file_row.addWidget(self.btn_measure_auto)
        file_row.addWidget(self.btn_measure)
        file_row.addWidget(self.btn_browse)

        # Parameter
        scale_row = QHBoxLayout()
        self.scale_edit = QLineEdit()
        self.scale_edit.setPlaceholderText("Skalierung µm/Pixel")
        self.scale_edit.setFixedWidth(180)
        self.border_edit = QLineEdit()
        self.border_edit.setPlaceholderText("Rand-Pad [px]")
        self.border_edit.setFixedWidth(150)
        self.kabs_checkbox = QCheckBox("Heuristische Skalierung k_abs anwenden")
        self.kabs_edit = QLineEdit("1.0")
        self.kabs_edit.setFixedWidth(80)
        self.kabs_edit.setEnabled(False)
        self.kabs_checkbox.toggled.connect(self.kabs_edit.setEnabled)
        scale_row.addWidget(QLabel("Skalierung:"))
        scale_row.addWidget(self.scale_edit)
        scale_row.addSpacing(12)
        scale_row.addWidget(QLabel("Rand-Pad:"))
        scale_row.addWidget(self.border_edit)
        scale_row.addSpacing(24)
        scale_row.addWidget(self.kabs_checkbox)
        scale_row.addWidget(self.kabs_edit)

        # Durchmesser-Definition
        def_row = QHBoxLayout()
        self.combo_key = QComboBox()
        self.combo_key.addItems([
            "Diameter_ECD (µm)",
            "Diameter_Heywood (µm)",
            "Diameter_ECD_abs (µm)",
            "Diameter_Heywood_abs (µm)",
        ])
        def_row.addWidget(QLabel("Plot-Durchmesser:"))
        def_row.addWidget(self.combo_key)
        def_row.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Histogramm-Typ
        hist_row = QHBoxLayout()
        self.combo_hist_mode = QComboBox()
        self.combo_hist_mode.addItems(["Anzahl (Count)", "Flächensumme (µm²)"])
        hist_row.addWidget(QLabel("Histogramm-Typ:"))
        hist_row.addWidget(self.combo_hist_mode)
        self.cb_correction = QCheckBox("3D-Korrektur (Saltykov)")
        hist_row.addSpacing(24)
        hist_row.addWidget(self.cb_correction)
        hist_row.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Analyse starten")
        self.btn_run.clicked.connect(self.on_run)
        self.btn_plot = QPushButton("Histogramm")
        self.btn_plot.clicked.connect(self.on_plot)
        self.btn_compare = QPushButton("2D ↔ 3D")
        self.btn_compare.clicked.connect(self.on_compare)
        self.btn_box = QPushButton("Boxplot")                    # << neu
        self.btn_box.clicked.connect(self.on_boxplot)            # << neu
        self.btn_export = QPushButton("CSV exportieren")
        self.btn_export.clicked.connect(self.on_export)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_plot)
        btn_row.addWidget(self.btn_compare)
        btn_row.addWidget(self.btn_box)                          # << neu
        btn_row.addWidget(self.btn_export)
        btn_row.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Bildanzeige (KeepAspectRatio)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("\n\nKein Ergebnisbild – bitte Bild laden und Analyse starten")
        self.image_label.setStyleSheet("QLabel { border: 1px solid #ccc; }")
        self.image_label.setMinimumSize(300, 220)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # Summary
        self.summary_label = QLabel("")
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.summary_label.setWordWrap(True)

        # Layout
        layout.addLayout(file_row)
        layout.addLayout(scale_row)
        layout.addLayout(def_row)
        layout.addLayout(hist_row)
        layout.addLayout(btn_row)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.summary_label)

    # Bild zeichnen (ohne Verzerrung)
    def _update_image_view(self) -> None:
        if self.overlay_bgr is None:
            return
        target_w = max(1, self.image_label.width())
        target_h = max(1, self.image_label.height())
        src_h, src_w = self.overlay_bgr.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        size = QSize(new_w, new_h)
        if self._last_drawn_size == size:
            return
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(self.overlay_bgr, (new_w, new_h), interpolation=interp)
        self.image_label.setPixmap(_cv_to_qpixmap(scaled))
        self._last_drawn_size = size

    # Slots
    def on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Bild auswählen", os.path.expanduser("~"),
                                              "Bilder (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;Alle Dateien (*)")
        if path:
            self.image_path = path
            self.path_label.setText(os.path.basename(path))
            self.image_label.setPixmap(QPixmap())
            self.summary_label.setText("")
            self._last_drawn_size = None

    def on_measure(self) -> None:
        if not self.image_path:
            QMessageBox.information(self, "Hinweis", "Bitte zuerst ein Bild laden.")
            return
        try:
            dlg = ScaleMeasureDialog(self.image_path, parent=self)
            if dlg.exec() == QDialog.DialogCode.Accepted and dlg.scale_umpx:
                self.scale_edit.setText(f"{dlg.scale_umpx:.8f}")
                QMessageBox.information(self, "Maßstab übernommen",
                                        f"Skalierung gesetzt auf {dlg.scale_umpx:.6g} µm/Pixel")
        except Exception as e:
            QMessageBox.critical(self, "Fehler beim Messen", str(e))

    def on_measure_auto(self) -> None:
        if not self.image_path:
            QMessageBox.information(self, "Hinweis", "Bitte zuerst ein Bild laden.")
            return
        try:
            dlg = AutoScaleBarDialog(self.image_path, parent=self)
            if dlg.exec() == QDialog.DialogCode.Accepted and dlg.scale_umpx:
                self.scale_edit.setText(f"{dlg.scale_umpx:.8f}")
                QMessageBox.information(self, "Maßstab übernommen",
                                        f"Skalierung gesetzt auf {dlg.scale_umpx:.6g} µm/Pixel")
        except Exception as e:
            QMessageBox.critical(self, "Automatik-Fehler", str(e))

    def _parse_float(self, line: QLineEdit, default: float = 0.0) -> float:
        try:
            return float(line.text().strip())
        except Exception:
            return default

    def on_run(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Hinweis", "Bitte zuerst ein Bild laden.")
            return
        scale = self._parse_float(self.scale_edit, default=0.0)
        if scale <= 0:
            QMessageBox.warning(self, "Fehler", "Bitte eine gültige Skalierung (µm/Pixel) angeben.")
            return
        border_pad = int(self._parse_float(self.border_edit, default=0.0))
        k_abs = self._parse_float(self.kabs_edit, default=1.0) if self.kabs_checkbox.isChecked() else 1.0

        try:
            results, mean_d, median_d, std_d, n_valid, n_rej, overlay, src = analyze_grains(
                self.image_path, scale_pixel=scale, min_area_px=10, border_pad=border_pad, k_abs=k_abs
            )
        except Exception as e:
            QMessageBox.critical(self, "Analysefehler", str(e))
            return

        self.results = results
        self.overlay_bgr = overlay
        self.summary_text = (
            f"<b>Ergebnis</b> – gültige Körner: {n_valid}, Rand verworfen: {n_rej}<br>"
            f"Mean ECD: {mean_d:.3f} µm &nbsp;|&nbsp; Median: {median_d:.3f} µm &nbsp;|&nbsp; Std: {std_d:.3f} µm<br>"
            f"Skalierung: {scale:.6g} µm/px &nbsp;|&nbsp; k_abs: {k_abs:.3f} (nur Anzeige, keine 3D-Korrektur)"
        )

        # Append 3D-corrected summary (Saltykov) and show in footer
        try:
            import numpy as np
            from stereology import saltykov_unfold, weighted_stats

            # Extract diameters from results
            data_list = self.results[0] if isinstance(self.results, tuple) else self.results
            diam = np.array([float(d.get("Diameter_ECD (µm)", np.nan)) for d in (data_list or [])], dtype=float)
            diam = diam[np.isfinite(diam)]
            if diam.size > 0:
                # Histogram (linear), unfold to 3D counts
                edges = np.histogram_bin_edges(diam, bins=30)
                counts_2d, _ = np.histogram(diam, bins=edges)
                centers = 0.5 * (edges[:-1] + edges[1:])
                counts_3d = saltykov_unfold(counts_2d)

                # Stats (mean/median/std) on 3D distribution
                mean3d, std3d, n3d = weighted_stats(centers, counts_3d)
                order = np.argsort(centers)
                x = centers[order]; w = counts_3d[order]
                cw = w.cumsum(); cutoff = w.sum()/2.0
                idx = int(np.searchsorted(cw, cutoff, side="left"))
                idx = min(max(idx, 0), len(x)-1)
                median3d = float(x[idx])

                # Expected section area (π/6 · d²) and section-diameter
                if counts_3d.sum() > 0:
                    m_d2 = float((counts_3d * (centers**2)).sum() / counts_3d.sum())
                    m_d4 = float((counts_3d * (centers**4)).sum() / counts_3d.sum())
                    var_d2 = max(m_d4 - m_d2**2, 0.0)
                    sec_mean = (np.pi/6.0) * m_d2
                    sec_std  = (np.pi/6.0) * np.sqrt(var_d2)
                    d_sec_mean = float(np.sqrt(6.0*sec_mean/np.pi))
                    d_sec_std  = 0.0 if sec_mean<=0 else float(0.5*np.sqrt(6.0/(np.pi*sec_mean))*sec_std)
                else:
                    sec_mean = sec_std = d_sec_mean = d_sec_std = 0.0

                # Append to footer and update label
                self.summary_text += (
                    f"<br><b>3D-korrigiert:</b> Mean ECD = {mean3d:.3f} µm &nbsp;|&nbsp; "
                    f"Median = {median3d:.3f} µm &nbsp;|&nbsp; Std = {std3d:.3f} µm &nbsp;|&nbsp; n_eff = {n3d}"
                    f"<br>⌀ Schnittfläche 3D = {sec_mean:.2f} ± {sec_std:.2f} µm²"
                    f"<br>⌀ Schnitt-Durchmesser 3D = {d_sec_mean:.3f} ± {d_sec_std:.3f} µm"
                )
                self.summary_label.setText(self.summary_text)
        except Exception as e:
            print("[3D-Footer] Fehler:", e)
        self._last_drawn_size = None
        self._update_image_view()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_image_view()

    def on_plot(self) -> None:
        if not self.results:
            QMessageBox.information(self, "Hinweis", "Keine Analyseergebnisse vorhanden.")
            return
        key = self.combo_key.currentText()
        mode = "count" if "Anzahl" in self.combo_hist_mode.currentText() else "area_sum"
        correction = "saltykov" if self.cb_correction.isChecked() else "none"
        try:
            stats = show_results(self.results, key=key, title="Korngrößenverteilung",
                                 units="µm", mode=mode, correction=correction, return_stats=True)
            if stats:
                tag = "3D korrigiert" if stats.get("corrected") else "2D roh"
                QMessageBox.information(
    self, "Kennwerte",
    f"{tag}\n\n"
    f"Mean = {stats['mean']:.3f} {stats['units']}\n"
    f"Std  = {stats['std']:.3f} {stats['units']}\n"
    f"n_eff = {stats['n_eff']}"
)
        except Exception as e:
            QMessageBox.critical(self, "Plot-Fehler", str(e))

    def on_boxplot(self) -> None:
        if not self.results:
            QMessageBox.information(self, "Hinweis", "Keine Analyseergebnisse vorhanden.")
            return
        key = self.combo_key.currentText()
        try:
            show_boxplot(self.results, key=key, title="Boxplot Korndurchmesser", units="µm")
        except Exception as e:
            QMessageBox.critical(self, "Boxplot-Fehler", str(e))

    def on_compare(self) -> None:
        if not self.results:
            QMessageBox.information(self, "Hinweis", "Keine Analyseergebnisse vorhanden.")
            return
        key = self.combo_key.currentText()
        mode = "count" if "Anzahl" in self.combo_hist_mode.currentText() else "area_sum"
        try:
            stats = show_comparison(self.results, key=key, units="µm", mode=mode, return_stats=True)
            if stats:
                s2 = stats["2D"]; s3 = stats["3D"]; u = stats["units"]
                delta = s3["mean"] - s2["mean"]
                QMessageBox.information(
                    self, "2D ↔ 3D – Kennwerte",
                    f"2D:  mean = {s2['mean']:.3f} {u}, std = {s2['std']:.3f} {u}, n = {s2['n']}\n"
                    f"3D:  mean = {s3['mean']:.3f} {u}, std = {s3['std']:.3f} {u}, n = {s3['n']}\n"
                    f"Δmean (3D-2D) = {delta:.3f} {u}  ({(delta/max(s2['mean'],1e-9))*100:.1f} %)"
                )
        except Exception as e:
            QMessageBox.critical(self, "Vergleich-Plot Fehler", str(e))


    def on_export(self) -> None:
        if not self.results:
            QMessageBox.information(self, "Hinweis", "Keine Analyseergebnisse vorhanden.")
            return
        if export_results_to_csv is None:
            QMessageBox.information(self, "Export", "Exportmodul nicht gefunden. (export.py)")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "CSV speichern", os.path.expanduser("~/ergebnisse.csv"),
                                                  "CSV-Datei (*.csv)")
        if not out_path:
            return
        try:
            export_results_to_csv(self.results, out_path)  # type: ignore
            QMessageBox.information(self, "Export", f"CSV gespeichert: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export-Fehler", str(e))


def launch_gui() -> None:
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()