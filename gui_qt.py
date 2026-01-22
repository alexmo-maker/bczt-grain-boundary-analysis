from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox
from image_io import prepare_image_for_analysis
from analysis import analyze_grains  # ✅ USA-Englisch
from plotting import show_results    # optional

class GrainAnalyzerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grain Size Analysis – Ceramics")
        self.setGeometry(100, 100, 300, 150)

        layout = QVBoxLayout()

        self.analyze_button = QPushButton("Start Analysis")
        self.analyze_button.clicked.connect(self.start_analysis)

        layout.addWidget(self.analyze_button)
        self.setLayout(layout)

    def start_analysis(self):
        image, scale = prepare_image_for_analysis()

        if image is None or scale is None:
            return  # Already handled in image_io

        try:
            results = analyze_grains(image, scale)  # ✅ amerikanisch
            show_results(results)
            QMessageBox.information(self, "Done", "Analysis completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{str(e)}")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = GrainAnalyzerGUI()
    window.show()
    sys.exit(app.exec())
