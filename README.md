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
1. The Core Architecture: 
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI).
2. EO-SAR Fusion for Change Detection: 
Ebel, P., Meraner, A., Schmitt, M., & Zhu, X. X. (2021). Multi-sensor Data Fusion for Cloud Removal in Global and Unfiltered Sentinel-2 Imagery. IEEE Transactions on Geoscience and Remote Sensing.
3. Handling Class Imbalance:  
Li, X., Sun, X., Meng, Y., Liang, J., Wu, F., & Li, J. (2020). Dice Loss for Data-imbalanced NLP Strategies. arXiv preprint arXiv:1911.02855.
4. Remote Sensing Change Detection: 
Chen, H., & Shi, Z. (2020). A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection. Remote Sensing.
5. Siamese Network: 
Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully Convolutional Siamese Networks for Change Detection. In IEEE International Conference on Image Processing (ICIP).

