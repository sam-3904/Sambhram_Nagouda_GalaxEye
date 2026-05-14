# Sambhram_Nagouda_GalaxEye
This repository contains an end-to-end deep learning pipeline for pixel-level change detection using Electro-Optical and Synthetic Aperture Radar imagery. I developed this for the GalaxEye Space AI Research Internship. This project addresses the challenges of multi-modal data fusion and extreme class imbalance in disaster datasets.

# Project Description
This project implements an end-to-end deep learning pipeline for binary pixel-level change detection using co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery. The model identifies structural changes (damaged or destroyed buildings) between pre-event EO and post-event SAR images, a critical task for rapid disaster response.
The approach utilizes a Siamese Residual UNet architecture to bridge the modality gap between optical and radar data.

# Requirements
Python: 3.10+
Dependencies (pinned versions in requirements.txt):
torch==2.1.0
rasterio==1.3.8
opencv-python==4.8.0.76
numpy==1.24.3
scikit-learn==1.3.0

# Environment Setup
Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies
pip install -r requirements.txt

# Dataset Structure
/content/dataset/
├── train/
│   ├── pre-event/    # EO .tif files
│   ├── post-event/   # SAR .tif files
│   └── target/       # Annotation masks
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)

# Training
python train.py --config config.yaml

# Evaluation
python eval.py --data_path /path/to/test_data --weights ./best_model.pth

# References
Architecture: UNet: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al.).
Loss Logic: Dice Loss for Data-imbalanced NLP Strategies (Li et al.).
Literature consulted: Research on EO-SAR fusion techniques for disaster monitoring.
