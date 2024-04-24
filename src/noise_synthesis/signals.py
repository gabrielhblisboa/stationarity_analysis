import abc
import enum
import typing
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav_file

import noise_synthesis.noise as syn_noise
import noise_synthesis.metrics as syn_metrics

class Signal(abc.ABC):

    @abc.abstractmethod
    def generate(self, n_samples, fs, baseline_psd_db) -> np.array:
        pass

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


class SyntheticSignal(Signal):

    class Type(enum.Enum):
        WHITE = 0
        BROWN = 1
        PINK = 2
        LOW = 3
        MEDIUM_LOW = 4
        MEDIUM_HIGH = 5
        HIGH = 6

        def __str__(self) -> str:
            return str(self.name).rsplit(".", maxsplit=1)[-1].lower().replace("_", " ") + " noise"

    def __init__(self, type: Type) -> None:
        super().__init__()
        self.type = type

    def __str__(self) -> str:
        return str(self.type)

    def generate(self, n_samples, fs, baseline_psd_db) -> np.array:

        if self.type == SyntheticSignal.Type.WHITE:
            return syn_noise.generate_noise(frequencies=np.linspace(0, fs/2, 5),
                                            psd_db=np.array([baseline_psd_db] * 5),
                                            n_samples=n_samples,
                                            fs=fs)

        if self.type == SyntheticSignal.Type.BROWN or self.type == SyntheticSignal.Type.PINK:
            ref_frequency = 2
            frequencies = np.logspace(np.log10(ref_frequency), np.log10(fs / 2), num=100)

            if self.type == SyntheticSignal.Type.BROWN:
                intensities = baseline_psd_db - 6 * np.log2(frequencies / ref_frequency)
            else:
                intensities = baseline_psd_db - 3 * np.log2(frequencies / ref_frequency)

            return syn_noise.generate_noise(frequencies=frequencies,
                                            psd_db=intensities,
                                            n_samples=n_samples,
                                            fs=fs)

        if self.type.value >= SyntheticSignal.Type.LOW.value and self.type.value <= SyntheticSignal.Type.HIGH.value:

            ref_frequency = 2
            min_freq = 20
            max_freq = 20000

            frequencies = np.logspace(np.log10(ref_frequency), np.log10(fs / 2), num=1000)
            divisions = np.logspace(np.log10(min_freq), np.log10(max_freq), num=5)

            intensities = np.full_like(frequencies, baseline_psd_db-60.0)

            indexes = []
            if self.type == SyntheticSignal.Type.LOW:
                indexes = frequencies <= divisions[1]
            elif self.type == SyntheticSignal.Type.MEDIUM_LOW:
                indexes = (frequencies > divisions[1]) & (frequencies <= divisions[2])
            elif self.type == SyntheticSignal.Type.MEDIUM_HIGH:
                indexes = (frequencies > divisions[2]) & (frequencies <= divisions[3])
            elif self.type == SyntheticSignal.Type.HIGH:
                indexes = frequencies > divisions[3]

            intensities[indexes] = baseline_psd_db

            return syn_noise.generate_noise(frequencies=frequencies,
                                            psd_db=intensities,
                                            n_samples=n_samples,
                                            fs=fs)

        raise NotImplementedError(f'method gen not implemented for {self}')

class RealSignal(Signal):
    class Type(enum.Enum):
        FLOW = 0            #81     x1 gain
        RAIN = 1            #82     x32 gain
        WAVE = 2            #83     x64 gain
        WIND = 3            #84     x1 gain
        FISH_BOAT = 4       #47     x16 gain
        MUSSEL_BOAT = 5     #75     x64 gain
                                    #sensitivity = -193.5 dB

        def __str__(self) -> str:
            return str(self.name).rsplit(".", maxsplit=1)[-1].lower()

    def __init__(self, type: Type) -> None:
        super().__init__()
        self.type = type

    def __str__(self) -> str:
        return str(self.type)

    def generate(self, n_samples, fs, baseline_psd_db) -> np.array:

        filename = os.path.join(os.path.dirname(__file__), "data", f'{str(self.type)}.wav')

        if not os.path.exists(filename):
            raise UnboundLocalError(f'File to {self} not found: {filename}')

        file_fs, input = wav_file.read(filename)

        if file_fs != fs:
            raise UnboundLocalError(f'Desired fs {fs} not equals to file fs {file_fs}')

        #normalizando int32 para +-1
        input = input / 2**(32-1)-1
        input = input - np.mean(input)

        #normalizando o valor rms
        input = input / np.sqrt(np.mean(np.square(input)))

        #aplicando o ganho desejado
        input = input * 10**(baseline_psd_db/20)

        return input[:n_samples]
