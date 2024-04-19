
import numpy as np
import matplotlib.pyplot as plt

import noise_synthesis.background_noise as syn_bg


def main():

    fs = 44100  # Example sampling frequency
    n_samples = 100 * fs  # One second of noise

    # Generate logarithmically spaced frequencies from 20 Hz to Nyquist frequency
    frequencies = np.logspace(np.log10(20), np.log10(fs / 2), num=100)
    # frequencies = np.logspace(0, np.log10(fs / 2), num=100)

    # Calculate intensities for colored noise
    ref_frequency = 20  # Reference frequency (20 Hz)


    intensities_pink = -3 * np.log2(frequencies / ref_frequency)

    intensities_brown = -6 * np.log2(frequencies / ref_frequency)

    # Now you can call your generate_noise function
    pink_noise = syn_bg.generate_noise(frequencies, intensities_pink, n_samples, fs)
    brown_noise = syn_bg.generate_noise(frequencies, intensities_brown, n_samples, fs)

    fft_freq, fft_result = syn_bg.psd(pink_noise, int(n_samples / 100), overlap=0.5, fs=fs)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))


    # Plot and save the spectra for comparison

    ax[0].plot(pink_noise, color='hotpink')
    ax[0].grid()

    ax[1].plot(brown_noise, color='brown')
    ax[1].grid()

    ax[2].semilogx(fft_freq, fft_result, label='Pink Noise', color='hotpink')

    fft_freq, fft_result = syn_bg.psd(brown_noise, int(n_samples / 100), overlap=0.5, fs=fs)
    ax[2].semilogx(fft_result, label='Brown Noise', color='brown')

    ax[2].set_xlabel('Frequency (Hz)')
    ax[2].set_ylabel('Amplitude (dB)')
    ax[2].grid()
    ax[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
