### grain_analysis_modular/export.py
import pandas as pd
import os
from tkinter import messagebox

def export_to_csv(data_list, path):
    df = pd.DataFrame(data_list)
    csv_path = os.path.splitext(path)[0] + "_results.csv"
    df.to_csv(csv_path, index=False)
    messagebox.showinfo("Exported", f"Data exported to:\n{csv_path}")

