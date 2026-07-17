# plotting.py
from typing import Iterable, List, Dict, Union, Optional, Tuple
import numpy as np

# Matplotlib-Backend (Qt) + non-blocking Anzeige
import matplotlib
try:
    matplotlib.use("QtAgg")
except Exception:
    matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.ion()

from stereology import saltykov_unfold, volume_weighted_from_counts, weighted_stats

DEFAULT_DIAMETER_KEY = "Diameter_ECD (µm)"
AREA_KEY = "Area (µm²)"


def _extract_from_results(
    results: List[Dict[str, float]],
    *,
    key: str = DEFAULT_DIAMETER_KEY,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    diam, areas = [], []
    for d in results:
        v = d.get(key)
        if v is None or not np.isfinite(v) or v <= 0:
            continue
        diam.append(float(v))
        a = d.get(AREA_KEY)
        if a is not None and np.isfinite(a) and a > 0:
            areas.append(float(a))
    diam = np.asarray(diam, dtype=float)
    areas = np.asarray(areas, dtype=float) if len(areas) == len(diam) and len(diam) > 0 else None
    return diam, areas


def _extract_diameters_only(data: Union[Iterable[float], np.ndarray]) -> np.ndarray:
    arr = np.asarray(list(data), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    return arr


def show_results(
    results_or_diameters: Union[Iterable[float], List[Dict[str, float]]],
    *,
    title: str = "Korngrößenverteilung",
    key: str = DEFAULT_DIAMETER_KEY,
    bins: int = 30,
    units: str = "µm",
    show_stats: bool = True,
    mode: str = "count",
    correction: str = "none",
    return_stats: bool = True
):
    """Normales Histogramm mit optionaler stereologischer Korrektur."""
    if isinstance(results_or_diameters, list) and results_or_diameters and isinstance(results_or_diameters[0], dict):
        diam, _ = _extract_from_results(results_or_diameters, key=key)
    else:
        diam = _extract_diameters_only(results_or_diameters)

    if diam.size == 0:
        print("[plotting.show_results] Keine Daten zum Plotten.")
        return {}

    counts_2d, edges = np.histogram(diam, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if correction.lower() == "saltykov":
        counts_for_number = saltykov_unfold(counts_2d)
        corr_label = " (3D korrigiert)"
    else:
        counts_for_number = counts_2d
        corr_label = " (2D roh)"

    if mode == "count":
        y = counts_for_number
        y_label = f"Anzahl je Klasse{corr_label}"
    elif mode == "area_sum":
        y = volume_weighted_from_counts(centers, counts_for_number)
        y_label = f"Volumen-gewichtete Summe je Klasse{corr_label}"
    else:
        raise ValueError("mode muss 'count' oder 'area_sum' sein.")

    plt.figure(figsize=(6, 4))
    plt.bar(centers, y, width=(edges[1]-edges[0]), edgecolor="black")
    plt.xlabel(f"Durchmesser [{units}]")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    mean_c, std_c, n_eff = weighted_stats(centers, counts_for_number)
    if show_stats:
        print(f"[Stats{corr_label}] mean={mean_c:.3f} {units}, std={std_c:.3f} {units}")

    return {"mean": mean_c, "std": std_c, "n_eff": n_eff, "units": units, "corrected": correction.lower()=="saltykov"}


def show_comparison(
    results_or_diameters: Union[Iterable[float], List[Dict[str, float]]],
    *,
    title: str = "2D vs. 3D – Korngrößenverteilung",
    key: str = DEFAULT_DIAMETER_KEY,
    bins: int = 30,
    units: str = "µm",
    mode: str = "count",
    return_stats: bool = True
):
    """Vergleichsplot: 2D (gemessen) vs. 3D (Saltykov-korrigiert)."""
    if isinstance(results_or_diameters, list) and results_or_diameters and isinstance(results_or_diameters[0], dict):
        diam, _ = _extract_from_results(results_or_diameters, key=key)
    else:
        diam = _extract_diameters_only(results_or_diameters)

    if diam.size == 0:
        print("[plotting.show_comparison] Keine Daten zum Plotten.")
        return {}

    counts_2d, edges = np.histogram(diam, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts_3d = saltykov_unfold(counts_2d)

    if mode == "count":
        y2d, y3d = counts_2d, counts_3d
        y_label = "Anzahl je Klasse"
    elif mode == "area_sum":
        y2d = volume_weighted_from_counts(centers, counts_2d)
        y3d = volume_weighted_from_counts(centers, counts_3d)
        y_label = "Volumen-gewichtete Summe je Klasse"
    else:
        raise ValueError("mode muss 'count' oder 'area_sum' sein.")

    plt.figure(figsize=(7, 4.5))
    width = edges[1]-edges[0]
    plt.bar(centers, y2d, width=width, alpha=0.4, label="2D (gemessen)", edgecolor="black")
    plt.plot(centers, y3d, linewidth=2, label="3D (Saltykov)")
    plt.xlabel(f"Durchmesser [{units}]")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    mean2d, std2d, n2d = weighted_stats(centers, counts_2d)
    mean3d, std3d, n3d = weighted_stats(centers, counts_3d)
    out = {"2D": {"mean": mean2d, "std": std2d, "n": n2d},
           "3D": {"mean": mean3d, "std": std3d, "n": n3d},
           "units": units}
    if return_stats:
        print(f"[2D] mean={mean2d:.3f} {units}, std={std2d:.3f} {units}")
        print(f"[3D] mean={mean3d:.3f} {units}, std={std3d:.3f} {units}")
    return out


def show_boxplot(
    results_or_diameters: Union[Iterable[float], List[Dict[str, float]]],
    *,
    key: str = DEFAULT_DIAMETER_KEY,
    title: str = "Boxplot Korndurchmesser",
    units: str = "µm",
) -> None:
    """Einfacher Boxplot der Durchmesser."""
    if isinstance(results_or_diameters, list) and results_or_diameters and isinstance(results_or_diameters[0], dict):
        diam, _ = _extract_from_results(results_or_diameters, key=key)
    else:
        diam = _extract_diameters_only(results_or_diameters)

    if diam.size == 0:
        print("[plotting.show_boxplot] Keine Daten zum Plotten.")
        return

    plt.figure(figsize=(5, 5))
    plt.boxplot(diam, vert=True, labels=[units], showfliers=True)
    plt.ylabel(f"Korndurchmesser [{units}]")
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
