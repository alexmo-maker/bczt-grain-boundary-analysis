### grain_analysis_modular/plotting.py
import matplotlib.pyplot as plt
import cv2

def plot_histogram(diameters):
    if not diameters:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(diameters, bins=20, edgecolor='black')
    plt.xlabel("Grain Diameter (µm)")
    plt.ylabel("Count")
    plt.title("Grain Size Distribution - Histogram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_boxplot(diameters):
    if not diameters:
        return
    plt.figure(figsize=(6, 6))
    plt.boxplot(diameters, vert=True, patch_artist=True)
    plt.ylabel("Grain Diameter (µm)")
    plt.title("Grain Size Distribution - Boxplot")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_results_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title("Analyzed Image (Green = valid, Red = rejected)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
