
# Stationarity Analysis Library

This library provides tools for detecting stationarity loss in passive sonar signals. The methods used include statistical tests such as the Augmented Dickey-Fuller (ADF) and Phillips-Perron (PP) tests, as well as similarity estimates between probability distributions like Kullback-Leibler Divergence, Jensen-Shannon Divergence, and Wasserstein Distance. The goal is to improve the detection of subtle changes in Probability Density Functions (PDF) over time, which can signal a loss of stationarity.

## Features

- **Stationarity Tests**: Augmented Dickey-Fuller and Phillips-Perron tests to identify stationarity.
- **Distribution Similarity Estimates**: Kullback-Leibler Divergence, Jensen-Shannon Divergence, and Wasserstein Distance to detect transitions in PDFs.
- **Z-Score Anomaly Detection**: Detects abnormal transitions in signals based on z-score thresholds, marking potential points of non-stationarity.
- **Support for Synthetic and Real Data**: Analyze both synthetic signals with different PDFs and real-world data from the ShipsEar dataset.
- **Customizable Analysis Parameters**: Window size, overlap, PDF estimation, and z-score threshold.

## Installation

To install the library, clone the repository and install the dependencies:

```bash
git clone https://github.com/gabrielhblisboa/stationarity_analysis.git
cd stationarity_analysis
pip install -r requirements.txt
```

## Data

The library can handle both synthetic data and real sonar recordings. Real data for testing is sourced from the [ShipsEar dataset](https://example.com/shipsear). You can also generate synthetic signals for controlled experiments.

## Results

Our experiments show that similarity estimates between distributions, combined with the z-score anomaly detection, outperform traditional statistical tests in detecting subtle transitions in sonar signals.
