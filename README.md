
# Stationarity Analysis Library

This library provides tools for detecting stationarity loss in passive sonar signals. The methods used include statistical tests such as the Augmented Dickey-Fuller (ADF) and Phillips-Perron (PP) tests, as well as similarity estimates between probability distributions like Kullback-Leibler Divergence, Jensen-Shannon Divergence, and Wasserstein Distance. Additionally, an anomaly detection method based on z-score is implemented to identify significant deviations in the signal over time.

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

## Usage

Here is a basic example of how to use the library to analyze stationarity loss and detect anomalies in sonar signals:

```python
from stationarity_analysis import StationarityDetector

# Load your data (synthetic or real)
data = load_data('path_to_data')

# Initialize the detector with customizable parameters, including z-score threshold
detector = StationarityDetector(window_size=16384, overlap=0.75, z_score_threshold=2.5)

# Run the analysis with anomaly detection
results = detector.run_analysis(data)

# View results
print(results)
```

## Anomaly Detection

The library uses a z-score based anomaly detector to identify significant changes in the signal. Each segment of the signal is normalized (zero mean, unit variance) based on the training data, and any sample that exceeds the z-score threshold is flagged as an anomaly. This method helps to highlight transitions that could indicate a loss of stationarity in the sonar signal.

## Data

The library can handle both synthetic data and real sonar recordings. Real data for testing is sourced from the [ShipsEar dataset](https://example.com/shipsear). You can also generate synthetic signals for controlled experiments.

## Experiments

The following experiments are included in the library to replicate the results from the article:

1. **Stationarity Tests on Colored Noise**: Evaluate the ADF and PP tests on white, pink, and brown noise.
2. **Distribution Similarity on Colored Noise**: Use KL, JSD, and WD to detect transitions in noise signals.
3. **Overlap Effect**: Analyze how different overlap percentages affect detection performance.
4. **Real Audio Testing**: Apply the techniques to real-world underwater noise recordings.
5. **Anomaly Detection**: Detect non-stationarity transitions using z-score anomaly detection.

## Results

Our experiments show that similarity estimates between distributions, combined with the z-score anomaly detection, outperform traditional statistical tests in detecting subtle transitions in sonar signals. For more details, refer to the article [here](https://example.com/article).

## Contributing

We welcome contributions to improve the library. Please feel free to submit issues or pull requests on GitHub.

## License

This project is licensed under the MIT License.
