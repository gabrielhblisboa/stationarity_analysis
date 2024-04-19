import enum

import numpy as np
import matplotlib.pyplot as plt

import noise_synthesis.noise as syn_noise

class Signals(enum.Enum):
    WHITE = 0
    BROWN = 1
    PINK = 2
    LOW = 3
    MEDIUM_LOW = 4
    MEDIUM_HIGH = 5
    HIGH = 6

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower().replace("_", " ") + " noise"

    def generate(self, n_samples, fs, baseline_psd_db) -> np.array:

        if self == Signals.WHITE:
            return syn_noise.generate_noise(frequencies=np.linspace(0, fs/2, 5),
                                            psd_db=np.array([baseline_psd_db] * 5),
                                            n_samples=n_samples,
                                            fs=fs)

        if self == Signals.BROWN or self == Signals.PINK:
            ref_frequency = 2
            frequencies = np.logspace(np.log10(ref_frequency), np.log10(fs / 2), num=100)

            if self == Signals.BROWN:
                intensities = baseline_psd_db - 6 * np.log2(frequencies / ref_frequency)
            else:
                intensities = baseline_psd_db - 3 * np.log2(frequencies / ref_frequency)

            return syn_noise.generate_noise(frequencies=frequencies,
                                            psd_db=intensities,
                                            n_samples=n_samples,
                                            fs=fs)

        if self.value >= Signals.LOW.value and self.value <= Signals.HIGH.value:

            ref_frequency = 2
            min_freq = 20
            max_freq = 20000

            frequencies = np.logspace(np.log10(ref_frequency), np.log10(fs / 2), num=1000)
            divisions = np.logspace(np.log10(min_freq), np.log10(max_freq), num=5)

            intensities = np.full_like(frequencies, baseline_psd_db-60.0)

            indexes = []
            if self == Signals.LOW:
                indexes = frequencies <= divisions[1]
            elif self == Signals.MEDIUM_LOW:
                indexes = (frequencies > divisions[1]) & (frequencies <= divisions[2])
            elif self == Signals.MEDIUM_HIGH:
                indexes = (frequencies > divisions[2]) & (frequencies <= divisions[3])
            elif self == Signals.HIGH:
                indexes = frequencies > divisions[3]

            intensities[indexes] = baseline_psd_db

            return syn_noise.generate_noise(frequencies=frequencies,
                                            psd_db=intensities,
                                            n_samples=n_samples,
                                            fs=fs)

        raise NotImplementedError(f'method gen not implemented for {self}')

    def plot(self, filename, n_samples, fs, baseline_psd_db):

        noise = self.generate(n_samples=n_samples,
                                fs=fs,
                                baseline_psd_db=baseline_psd_db)

        fft_freq, fft_result = syn_noise.psd(signal=noise, fs=fs)

        plt.figure(figsize=(12, 6))
        plt.semilogx(fft_freq, fft_result, label='Test Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('SPL (dB/Hz)')
        plt.savefig(filename)
        plt.close()