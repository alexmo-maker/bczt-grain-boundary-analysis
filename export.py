### grain_analysis_modular/export.py
import pandas as pd
import os
from tkinter import messagebox


def export_to_csv(data_list, path):
    df = pd.DataFrame(data_list)
    csv_path = os.path.splitext(path)[0] + "_results.csv"
    df.to_csv(csv_path, index=False)
    messagebox.showinfo("Exported", f"Data exported to:\n{csv_path}")


def export_results_to_csv(results, path):
    """
    Qt-GUI-kompatibler Export (kein tkinter-Popup, speichert exakt unter `path`).
    Akzeptiert entweder eine data_list (Liste von Dicts) oder das volle
    analyze_grains()-Tuple (data_list ist dann results[0]).
    """
    data_list = results[0] if isinstance(results, tuple) else results
    df = pd.DataFrame(data_list)
    df.to_csv(path, index=False)
