import enum
import typing

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav_file

import noise_synthesis.noise as syn_noise
import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals


class AmplitudeTransitionType(enum.Enum):
    ABRUPT = 0
    LINEAR = 1
    SINUSOIDAL = 2

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower().replace("_", " ")

    def apply(self, signal1: np.array, signal2: np.array, transition_samples: typing.Union[int, float] = None) -> np.array:

        output = np.copy(signal1)
        n_samples = len(output)

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
                    factor = (n / transition_samples)
                elif self == AmplitudeTransitionType.SINUSOIDAL:
                    factor = np.cos(np.pi/2 * (n / transition_samples))

                output[transition_start+n] = factor * signal1[transition_start+n] + \
                                            (1-factor) * signal2[transition_start+n]

                output[transition_end-n] = factor * signal1[transition_end-n] + \
                                            (1-factor) * signal1[transition_end-n]

        output[transition_start:transition_end] = signal2[transition_start:transition_end]

        return output


class Experiment():

    def __init__(self,
                 signal1: syn_signals.SyntheticSignal,
                 psd_signal1: float,
                 signal2: syn_signals.SyntheticSignal,
                 psd_signal2: float,
                 transition: AmplitudeTransitionType,
                 metric_list: typing.List[syn_metrics.Metrics]) -> None:
        self.signal1 = signal1
        self.psd_signal1 = psd_signal1
        self.signal2 = signal2
        self.psd_signal2 = psd_signal2
        self.transition = transition
        self.metric_list = metric_list

    def run(self, file_basename: str, complete_size: int, fs: float, window_size: int, overlap: float) -> None:

        signal1 = self.signal1.generate(complete_size, fs, self.psd_signal1)
        signal2 = self.signal2.generate(complete_size, fs, self.psd_signal2)

        signal = self.transition.apply(signal1=signal1, signal2=signal2)

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

        wav_file.write(f'{file_basename}.wav', fs, syn_noise.normalize(signal, type=1))