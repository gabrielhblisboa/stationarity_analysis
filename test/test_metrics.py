import os
import math
import numpy as np

import noise_synthesis.noise as syn_noise
import noise_synthesis.metrics as syn_metrics


def main():
    """Main function for the test program."""

    # Set up the directory for saving results
    base_dir = "./result"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Set parameters for synthetic noise generation
    fs = 48000
    n_samples = 100 * fs
    n_fft = 2048
    f_test_min = 1000
    f_test_max = 3000
    max_db = 0
    min_db = -20

    # Create a desired spectrum with two levels and a sinusoidal shape transition
    min_sample = int(f_test_min * n_fft / fs)
    max_sample = int(f_test_max * n_fft / fs)
    desired_spectrum = np.ones(n_fft // 2 + 1) * min_db
    for i in range(min_sample, max_sample):
        desired_spectrum[i] = max_db
    for i in range(max_sample, len(desired_spectrum)):
        desired_spectrum[i] = max_db - (max_db - min_db) * math.sin(
            math.pi / 2 * ((i - max_sample) / (len(desired_spectrum) - max_sample)))

    # Generate synthetic noise based on the desired spectrum
    frequencies = np.linspace(0, fs / 2, len(desired_spectrum))
    noise1 = syn_noise.generate_noise(frequencies, desired_spectrum, n_samples, fs)


    #white noise mean psd
    psd_linear = 10 ** (((max_db + min_db)/2) / 20)
    max_power = np.max(psd_linear) * fs/2
    std_dev = np.sqrt(max_power)
    noise2 = np.random.normal(0, std_dev, n_samples)

    for estimator in syn_metrics.DataEstimator:
        output_filename = f"{base_dir}/{str(estimator)}.png"
        estimator.plot(filename=output_filename,
                       window1=noise1,
                       label1='Test noise',
                       window2=noise2,
                       label2='White noise',
                       n_points=128)


    print(f"{'Estimator':<30}", end='')
    for type in syn_metrics.Metrics.Type:
        print(f"{type:<30}", end='')
    print()

    for estimator in syn_metrics.DataEstimator:
        print(f"{estimator:<30}", end='')
        for type in syn_metrics.Metrics.Type:
            metric = syn_metrics.Metrics(type=type, estimator=estimator, n_points=128)
            value = metric.calc_block(window1=noise1, window2=noise2)
            print(f"{value:<30}", end='')
        print() 

if __name__ == "__main__":
    main()