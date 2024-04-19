import enum
import typing

import numpy as np
import matplotlib.pyplot as plt

import noise_synthesis.noise as syn_noise
import noise_synthesis.metrics as syn_metrics

class SyntheticSignals(enum.Enum):
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

        if self == SyntheticSignals.WHITE:
            return syn_noise.generate_noise(frequencies=np.linspace(0, fs/2, 5),
                                            psd_db=np.array([baseline_psd_db] * 5),
                                            n_samples=n_samples,
                                            fs=fs)

        if self == SyntheticSignals.BROWN or self == SyntheticSignals.PINK:
            ref_frequency = 2
            frequencies = np.logspace(np.log10(ref_frequency), np.log10(fs / 2), num=100)

            if self == SyntheticSignals.BROWN:
                intensities = baseline_psd_db - 6 * np.log2(frequencies / ref_frequency)
            else:
                intensities = baseline_psd_db - 3 * np.log2(frequencies / ref_frequency)

            return syn_noise.generate_noise(frequencies=frequencies,
                                            psd_db=intensities,
                                            n_samples=n_samples,
                                            fs=fs)

        if self.value >= SyntheticSignals.LOW.value and self.value <= SyntheticSignals.HIGH.value:

            ref_frequency = 2
            min_freq = 20
            max_freq = 20000

            frequencies = np.logspace(np.log10(ref_frequency), np.log10(fs / 2), num=1000)
            divisions = np.logspace(np.log10(min_freq), np.log10(max_freq), num=5)

            intensities = np.full_like(frequencies, baseline_psd_db-60.0)

            indexes = []
            if self == SyntheticSignals.LOW:
                indexes = frequencies <= divisions[1]
            elif self == SyntheticSignals.MEDIUM_LOW:
                indexes = (frequencies > divisions[1]) & (frequencies <= divisions[2])
            elif self == SyntheticSignals.MEDIUM_HIGH:
                indexes = (frequencies > divisions[2]) & (frequencies <= divisions[3])
            elif self == SyntheticSignals.HIGH:
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

class AmplitudeTransitionType(enum.Enum):
    ABRUPT = 0
    LINEAR = 1
    SINUSOIDAL = 2

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower().replace("_", " ")

    def apply(self, input: np.array, start_psd_db: float, end_psd_db: float, transition_samples: typing.Union[int, float] = None) -> np.array:

        n_samples = len(input)
        attenuation = 10 ** (-np.abs(start_psd_db - end_psd_db)/10)

        if transition_samples is None:
            transition_samples = int(0.1 * n_samples)
        elif not isinstance(transition_samples, int):
            transition_samples = int(transition_samples * n_samples)


        transition_start = int(n_samples * 0.33)
        transition_end = int(n_samples * 0.67)

        if self != AmplitudeTransitionType.ABRUPT:
            transition_start -= transition_samples//2
            transition_end += transition_samples//2

            for n in range(1, transition_samples):
                if self == AmplitudeTransitionType.LINEAR:
                    if end_psd_db > start_psd_db:
                        input[transition_start+n] *= (1-(n / transition_samples)) * (attenuation-1) + 1
                        input[transition_end-n] *= (1-(n / transition_samples)) * (attenuation-1) + 1
                    else:
                        input[transition_start+n] *= (n / transition_samples) * (attenuation-1) + 1
                        input[transition_end-n] *= (n / transition_samples) * (attenuation-1) + 1

                elif self == AmplitudeTransitionType.SINUSOIDAL:
                    if end_psd_db > start_psd_db:
                        input[transition_start+n] *= np.sin(np.pi/2 * (1-(n / transition_samples))) * (attenuation-1) + 1
                        input[transition_end-n] *= np.sin(np.pi/2 * (1-(n / transition_samples))) * (attenuation-1) + 1
                    else:
                        input[transition_start+n] *= np.sin(np.pi/2 * (n / transition_samples)) * (attenuation-1) + 1
                        input[transition_end-n] *= np.sin(np.pi/2 * (n / transition_samples)) * (attenuation-1) + 1

        if end_psd_db > start_psd_db:
            input[:transition_start] *= attenuation
            input[transition_end:] *= attenuation
        else:
            input[transition_start:transition_end] *= attenuation

        return input


class Experiment():

    def __init__(self,
                 signal: SyntheticSignals,
                 transition: AmplitudeTransitionType,
                 metric_list: typing.List[syn_metrics.Metrics],
                 start_psd_db: float,
                 end_psd_db: float) -> None:
        self.signal = signal
        self.transition = transition
        self.metric_list = metric_list
        self.start_psd_db = start_psd_db
        self.end_psd_db = end_psd_db

    def run(self, file_basename: str, complete_size: int, fs: float, window_size: int, overlap: float) -> None:
        signal = self.signal.generate(complete_size, fs, np.max([self.start_psd_db, self.end_psd_db]))
        signal = self.transition.apply(input = signal,
                                       start_psd_db=self.start_psd_db,
                                       end_psd_db=self.end_psd_db)

        for metric in self.metric_list:
            values, ref_sample = metric.calc_data(data=signal, window_size=window_size, overlap=overlap)

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            ax[0].plot(np.linspace(0,len(signal)/fs, len(signal)), signal)
            ax[0].set_xlabel('Time (s)')
            ax[0].set_ylabel('Intensity')

            ax[1].plot(np.array(ref_sample)/fs, values)
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Metric')

            plt.savefig(f'{file_basename}_{metric}.png')
            plt.close()