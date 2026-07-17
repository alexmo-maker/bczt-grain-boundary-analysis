# bczt-grain-boundary-analysis
This script was developed and used for the grain boundary and grain size analysis presented in a Master's thesis on doped BCZT ceramics (Graz University of Technology, 2026).

# BCZT Grain Boundary Analysis

This repository contains a Python script used for grain boundary and grain size analysis
of BCZT ceramics based on optical and SEM micrographs.

## Description
The script performs:
- Image preprocessing (thresholding, filtering)
- Grain boundary detection
- Grain size statistics (mean grain size, distribution)
- Stereological correction (Schwartz–Saltykov unfolding) to estimate 3D grain size distribution
- Visualization of segmented grains

## Requirements
- Python 3.9+
- numpy
- opencv-python
- matplotlib
- pandas
- PySide6

## Usage
```bash
python main.py
