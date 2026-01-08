# Calibration-Free Motor Imagery Classification Using Domain-Adversarial EEGNet

**Official repository** for the paper:  
**Calibration-Free Motor Imagery Classification Based on Electroencephalography Using Domain-Adversarial Neural Networks**  
Zhuldyz Bagybek  

International Information Technology University (IITU), Almaty, Kazakhstan

## Abstract

We propose a calibration-free motor imagery classification approach based on a compact EEGNet feature extractor combined with domain-adversarial neural network (DANN) training. Evaluated in a strict leave-one-subject-out (LOSO) protocol on the BCI Competition IV dataset 2a, the method achieves:

- **Mean balanced accuracy**: 0.551 ± 0.085  
- **Mean Cohen's kappa**: 0.367 ± 0.176  

Both metrics significantly exceed chance level (0.25 for a 4-class task) without any subject-specific calibration data.

## Repository Contents

- `run_loso.py` — Full script for LOSO training, evaluation, t-SNE visualization, confusion matrices, and training dynamics.
- `requirements.txt` — Required Python packages.
- `artifacts/` — Example results from the reported experiment (metrics, figures).
- `data/README.md` — Instructions for downloading and preparing the dataset.

## Results

Per-subject and average performance (from the paper):

| Subject | Balanced Accuracy | Cohen's kappa |
|---------|-------------------|---------------|
| S01     | 0.531             | 0.375         |
| S02     | 0.542             | 0.389         |
| S03     | 0.483             | 0.310         |
| S04     | 0.528             | 0.056         |
| S05     | 0.729             | 0.639         |
| S06     | 0.444             | 0.259         |
| S07     | 0.639             | 0.519         |
| S08     | 0.604             | 0.472         |
| S09     | 0.462             | 0.282         |
| **Mean ± STD** | **0.551 ± 0.085** | **0.367 ± 0.176** |

All figures (confusion matrix, t-SNE by class/domain for best/worst subjects, training dynamics) are in `artifacts/`.

## Installation

```bash
pip install -r requirements.txt
