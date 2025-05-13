# Computer Vision Explorations 2025

A consolidated repository containing solutions, code implementations, detailed reports, and results for all four major computer vision projects conducted as part of CSE344/CSE544/ECE344/ECE544 coursework in 2025.

---

## 📂 Repository Structure

```
├── ASS1_Vision_Methods
│   ├── ... folders of each task (code + results + report)...
│
├── ASS2_Camera_and_Panorama
│   ├── ... folders of each task (code + results + report)...
│
├── ASS3_Vision_Language_Models
│   ├── ... folders of each task (code + results + report)...
│
├── ASS4_Generative_and_Segmentation
│   ├── ... folders of each task (code + results + report)...
│
└── README.md             # This file
```

> **Note:** Folder names use descriptive tags reflecting topic focus. Each contains the code, final report, and obtained results for that project.

---

## 🚀 Getting Started

These projects require Python 3.8 or above and the following libraries :

* PyTorch
* OpenCV
* torchvision
* numpy, scipy, matplotlib
* scikit-learn
* WandB (for experiment logging)
* ultralytics (YOLOv8)
* transformers (for CLIP/BLIP)
* open3d (for point cloud registration)

---

## 🔍 Project Overviews

### **1. Vision Methods**

* **Focus:** Label smoothing, Gaussian cross-entropy, dilated convolutions, CNN classifiers, segmentation, detection, multi-object tracking.
* **Deliverables:** Theory derivations, CNN training from scratch, fine‑tuning ResNet‑18, SegNet & DeepLabV3 segmentation, YOLOv8 detection & TIDE analysis, MOT17 tracking.

### **2. Camera and Panorama**

* **Focus:** 3D transformations, camera calibration, panorama stitching, point cloud registration.
* **Deliverables:** Coordinate transforms & Rodrigues’ formula, homography & intrinsic/extrinsic estimation, chessboard calibration & reprojection error, SIFT-based stitching, ICP LiDAR registration.

### **3. Vision Language Models**

* **Focus:** CLIP & CLIPS zero‑shot evaluation, BLIP visual question answering & captioning, referring image segmentation (LAVT & Matcher).
* **Deliverables:** Similarity comparisons, VQA answers, caption quality metrics, RIS & one‑shot segmentation demos.

### **4. Generative and Segmentation**

* **Focus:** GAN theory & DCGAN/StyleGAN3 implementation, advanced segmentation tasks.
* **Deliverables:** GAN proofs, DCGAN & StyleGAN3 code (latent interpolation, style mixing, fine‑tuning on custom data), segmentation experiments.

---

## 📄 Reports & Results

* **Report** PDFs are there for each task which include methodology, graphs, visualizations, and theoretical discussions.
* **Results** directories contain trained model checkpoints, plots (loss curves, confusion matrices, segmentation masks, etc.).


---

## 🔖 License

This repository and its contents are provided under the MIT License.

---

*Prepared by \[Aarya Gupta].*
