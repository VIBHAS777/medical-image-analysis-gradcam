# Automated Medical Image Analysis for Brain Tumor and Skin Disease Diagnosis Using Transfer Learning and Grad-CAM

[![Conference](https://img.shields.io/badge/Conference-3rd_ICITEICS_2026-blue.svg)](https://iciteics.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Presented at the **3rd International Conference on Information Technology, Electronics and Intelligent Communication Systems (ICITEICS 2026)**.

---

## 👥 Authors
- **Araveeti Vibhas Reddy** ([@VIBHAS777](https://github.com/VIBHAS777))
- **Hima Sagar B**
- **Irigela Bharghava Vardhan Reddy**
- **S Pavan Kumar**
- **Nalini Sampath**

---

## 🩺 Overview
Black-box AI models face significant skepticism in clinical healthcare workflows. This repository presents an explainable clinical computer vision architecture combining fine-tuned transfer learning backbones (**VGG16**, **ResNet-50**, and **EfficientNet**) with **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

The framework produces dual-domain diagnostic predictions across:
1. **Brain Tumor MRI Scans** (Glioma, Meningioma, Pituitary, Normal).
2. **Dermoscopic Skin Lesions** (Melanoma, Basal Cell Carcinoma, Nevus, Keratosis).

Grad-CAM localizes pathological tissue regions directly within the convolutional feature maps, offering transparent visual corroboration for radiologists and oncologists.

---

## 📁 Repository Structure
```
medical-image-analysis-gradcam/
├── paper/
│   ├── Automated_Medical_Image_Analysis_for_Brain_Tumor_and_Skin_Disease.pdf
│   └── Certificate_ICITEICS_2026.pdf
├── src/
│   ├── models.py          # Transfer learning model architectures
│   └── gradcam.py         # PyTorch Grad-CAM visual heatmapping engine
├── main.py                # Dual-domain evaluation & demo pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start
```bash
git clone https://github.com/VIBHAS777/medical-image-analysis-gradcam.git
cd medical-image-analysis-gradcam
pip install -r requirements.txt
python main.py
```

---

## 📜 Citation
```bibtex
@inproceedings{reddy2026medicalimage,
  title={Automated Medical Image Analysis for Brain Tumor and Skin Disease Diagnosis Using Transfer Learning and Grad-CAM},
  author={Reddy, Araveeti Vibhas and Sagar B, Hima and Reddy, Irigela Bharghava Vardhan and Kumar, S Pavan and Sampath, Nalini},
  booktitle={3rd International Conference on Information Technology, Electronics and Intelligent Communication Systems (ICITEICS)},
  year={2026}
}
```
