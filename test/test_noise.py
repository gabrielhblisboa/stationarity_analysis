
import os
import math
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.noise as syn_noise


def main():
    """Main function for the test program."""

    # Set up the directory for saving results
    base_dir = "./result"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Set parameters for synthetic noise generation
    output_spectrum = f"{base_dir}/generate_output.png"
    fs = 48000
    n_samples = 100 * fs
    n_fft = 2048
    f_test_min = 1000
    f_test_max = 3000
    max_db = 100
    min_db = 60

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
    noise = syn_noise.generate_noise(frequencies, desired_spectrum, n_samples, fs)

    # Estimate the spectrum of the generated noise
    fft_freq, fft_result = syn_noise.psd(signal=noise, fs=fs, window_size=4096, overlap=0.5)

    # Plot and save the spectra for comparison
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freq, fft_result, label='Test Spectrum')
    plt.plot(frequencies, desired_spectrum, linestyle='--', label='Desired Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.savefig(output_spectrum)
    plt.close()

    # Print analysis results
    print("Analysis:")
    print(f"\tDesired mean spectrum: {np.mean(desired_spectrum):.2f} dB ref 1μPa @1m/Hz")
    print(f"\tSynthetic mean spectrum: {np.mean(fft_result):.2f} dB ref 1μPa @1m/Hz")
    print(f"\tMean error: {np.mean(desired_spectrum) - np.mean(fft_result):.2f} dB ref 1μPa @1m/Hz")
    print(f"Data and spectrum exported in {base_dir}")

    # scipy_wav.write(output_file, fs, noise)

if __name__ == "__main__":
    main()