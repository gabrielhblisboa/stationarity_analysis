
import os
import math
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.noise as syn_noise
import noise_synthesis.signals as syn_signals


def main():
    """Main function for the test program."""

    # Set up the directory for saving results
    base_dir = "./result"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Set parameters for synthetic noise generation
    fs = 48000
    n_samples = 100 * fs
    baseline_psd_db = 10

    for signal in syn_signals.Signals:
        output_spectrum = f"{base_dir}/{signal}.png"

        signal.plot(filename=output_spectrum,
                    n_samples=n_samples,
                    fs=fs,
                    baseline_psd_db=baseline_psd_db)


    output_spectrum = f"{base_dir}/band_noise.png"
    plt.figure(figsize=(12, 6))

    for signal in [syn_signals.Signals.LOW, syn_signals.Signals.MEDIUM_LOW, syn_signals.Signals.MEDIUM_HIGH, syn_signals.Signals.HIGH]:

        noise = signal.generate(n_samples=n_samples,
                                fs=fs,
                                baseline_psd_db=baseline_psd_db)

        fft_freq, fft_result = syn_noise.psd(signal=noise, fs=fs)

        plt.semilogx(fft_freq, fft_result, label=str(signal))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB/Hz)')
    plt.legend()
    plt.savefig(output_spectrum)
    plt.close()


if __name__ == "__main__":
    main()