
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
    base_dir = "./result/signals/real"
    os.makedirs(base_dir, exist_ok = True)

    # Set parameters for synthetic noise generation
    fs = 52734
    n_samples = 100 * fs
    baseline_psd_db = 10

    for type in syn_signals.RealSignal.Type:
        signal = syn_signals.RealSignal(type=type)
        output_spectrum = f"{base_dir}/{signal}.png"

        signal.plot(filename=output_spectrum,
                    n_samples=n_samples,
                    fs=fs,
                    baseline_psd_db=baseline_psd_db)


    output_spectrum = f"{base_dir}/real.png"
    plt.figure(figsize=(12, 6))

    for type in syn_signals.RealSignal.Type:
        signal = syn_signals.RealSignal(type=type)

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