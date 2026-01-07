# Calibration-Free Motor Imagery Classification Using Domain-Adversarial EEGNet

This repository contains the official code for the paper:

**"Calibration-Free Motor Imagery Classification Based on Electroencephalography Using Domain-Adversarial Neural Networks"**  
Aigerim Aitim, Zhuldyz Bagybek

## Abstract

We propose a calibration-free approach for motor imagery classification using a compact EEGNet feature extractor combined with domain-adversarial training (DANN). Evaluated in a strict leave-one-subject-out protocol on BCI Competition IV dataset 2a, the method achieves **mean balanced accuracy of 0.551** and **Cohen's kappa of 0.367** — significantly above chance level without any subject-specific calibration.

## Results

- Mean balanced accuracy: **0.551 ± 0.085**
- Mean Cohen's kappa: **0.367 ± 0.176**

See full per-subject results in `artifacts/`

## Usage

1. Install dependencies:
   ```bash
  pip install -r requirements.txt
python run_loso.py
