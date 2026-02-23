# AI-Based Infrastructure Risk Detection Using Time-Series Deep Learning

## Project Overview

This project implements a multi-phase anomaly detection framework for large-scale infrastructure energy consumption data (2M+ time-series records).

The objective is to detect abnormal infrastructure behavior using progressively advanced machine learning models — moving from statistical outlier detection to deep temporal modeling.

The project demonstrates how anomaly detection systems can evolve from simple isolation-based methods to sequence-aware deep learning architectures capable of modeling complex behavioral patterns.

---

## Dataset

This project uses a large-scale public dataset from the UCI Machine Learning Repository:

UCI Individual Household Electric Power Consumption Dataset  
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

The dataset contains over 2 million time-series measurements of electric power usage.

Note: The raw dataset is not included in this repository due to size constraints.

---

## Architecture Overview

The anomaly detection pipeline consists of three progressive phases:

### Phase 1 – Isolation Forest (Statistical Baseline)
- Tree-based unsupervised anomaly detection
- Detects statistically isolated observations
- Serves as a lightweight baseline model

### Phase 2 – Dense Autoencoder (Deep Feature Modeling)
- Feedforward neural network
- Learns compressed representations of normal behavior
- Anomalies detected via reconstruction error
- Captures nonlinear relationships between variables

### Phase 3 – LSTM Autoencoder (Temporal Deep Learning)
- Sequence-based neural network using 30-timestep sliding windows
- Learns temporal dependencies in energy usage patterns
- Detects anomalies based on sequence reconstruction error
- Captures time-evolving irregularities

---

## Model Comparison

| Model | Modeling Type | Temporal Awareness | Detection Basis |
|-------|---------------|-------------------|------------------|
| Isolation Forest | Statistical | No | Isolation-Based Outliers |
| Dense Autoencoder | Deep Learning | No | Feature Reconstruction Error |
| LSTM Autoencoder | Deep Learning | Yes | Sequence Reconstruction Error |

All deep learning thresholds use percentile-based anomaly classification for objective detection.

---

## Key Technical Highlights

- End-to-end machine learning pipeline
- Large-scale time-series processing (2M+ rows)
- Sliding-window sequence generation
- Percentile-based anomaly thresholding
- Model persistence and reproducibility
- Comparative model evaluation
- Clean repository structure for production-style organization

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

---

## Applications

This framework can be extended to:

- Power grid monitoring
- Industrial IoT anomaly detection
- Smart building energy optimization
- Infrastructure risk management systems
- Time-series behavioral monitoring

---

## Repository Structure

```
AI-Infrastructure-Risk-Detection/
│
├── infra_anomaly_detection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Author

Sabina Bimbi

---

## Summary

This project demonstrates a structured progression from statistical anomaly detection to temporal deep learning modeling for infrastructure risk analysis.

It reflects practical experience in time-series modeling, unsupervised learning, neural networks, and production-oriented ML workflow design.
