# stereology.py
import numpy as np

def saltykov_unfold(counts_2d: np.ndarray) -> np.ndarray:
    """
    Schwartz–Saltykov-Entfaltung für lineare, gleich breite Klassen.
    Input: 2D-Anzahl je Klasse (np.histogram counts)
    Output: geschätzte 3D-Anzahl je Klasse (gleiche Klassierung)
    """
    counts_2d = np.asarray(counts_2d, dtype=float).copy()
    n = counts_2d.size
    counts_3d = np.zeros_like(counts_2d)

    # Näherungs-Koeffizienten (gängig, Beiträge > m=6 vernachlässigbar)
    def k(m: int) -> float:
        if m == 0: return 1.0
        if m == 1: return 0.267
        if m == 2: return 0.089
        if m == 3: return 0.036
        if m == 4: return 0.016
        if m == 5: return 0.007
        if m == 6: return 0.003
        return 0.0

    # Rücksubstitution von der größten Klasse abwärts
    for i in range(n - 1, -1, -1):
        s = counts_2d[i]
        for m, j in enumerate(range(i + 1, n), start=1):
            s -= k(m) * counts_3d[j]
        counts_3d[i] = max(s, 0.0)

    return counts_3d


def volume_weighted_from_counts(bin_centers: np.ndarray, counts_3d: np.ndarray) -> np.ndarray:
    """Volumen-gewichtetes Histogramm aus 3D-Anzahlen (Gewicht ∝ d^3)."""
    bin_centers = np.asarray(bin_centers, dtype=float)
    counts_3d = np.asarray(counts_3d, dtype=float)
    return counts_3d * np.power(bin_centers, 3)


def weighted_stats(bin_centers: np.ndarray, counts: np.ndarray):
    """
    Liefert (mean, std, n_eff) für ein Histogramm (bin-centers, counts).
    Std hier als gewichtete Populations-Std.
    """
    w = np.asarray(counts, dtype=float)
    x = np.asarray(bin_centers, dtype=float)
    W = w.sum()
    if W <= 0:
        return float("nan"), float("nan"), 0
    mu = (w * x).sum() / W
    var = (w * (x - mu) ** 2).sum() / W
    return float(mu), float(np.sqrt(var)), int(W)
