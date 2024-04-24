import enum
import typing

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav_file
import tikzplotlib as tikz

import noise_synthesis.noise as syn_noise
import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals


class AmplitudeTransitionType(enum.Enum):
    ABRUPT = 0
    LINEAR = 1
    SINUSOIDAL = 2

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower().replace("_", " ")

    def apply(self, signal1: np.array, signal2: np.array, transition_samples: typing.Union[int, float] = None) -> typing.Tuple[np.array, typing.List[int]]:

        output = np.copy(signal1)
        n_samples = len(output)

        if transition_samples is None:
            transition_samples = int(0.1 * n_samples)
        elif not isinstance(transition_samples, int):
            transition_samples = int(transition_samples * n_samples)


        transition_start = int(n_samples * 0.33)
        transition_end = int(n_samples * 0.67)

        if self == AmplitudeTransitionType.ABRUPT:
            limits = [transition_start, transition_end]
        else:
            transition_start -= transition_samples//2
            transition_end += transition_samples//2

            limits = [transition_start,
                      transition_start + transition_samples,
                      transition_end - transition_samples,
                      transition_end]

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

        return output, limits


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

    def _generate(self, file_basename: str, complete_size: int, fs: float) -> typing.Tuple[np.array, typing.List[int]]:
        signal1 = self.signal1.generate(complete_size, fs, self.psd_signal1)
        signal2 = self.signal2.generate(complete_size, fs, self.psd_signal2)
        return self.transition.apply(signal1=signal1, signal2=signal2)

    def run(self, file_basename: str, complete_size: int, fs: float, window_size: int, overlap: float, n_runs = 100) -> None:

        for metric in self.metric_list:
            results = []

            fig, ax = plt.subplots(figsize=(12, 8))

            for _ in range(n_runs):
                signal, limits = self._generate(file_basename=file_basename, complete_size=complete_size, fs=fs)
                values, ref_sample = metric.calc_data(data=signal, window_size=window_size, overlap=overlap)
                results.append(values)

            ax.boxplot(np.array(results))

            indices = np.linspace(0, len(ref_sample) - 1, 5, dtype=int)
            ax.set_xticks([i for i in indices])
            ax.set_xticklabels([f'{ref_sample[i]/fs:.2f}s' for i in indices])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Intensity')

            for limit in limits:
                # index = np.where(np.array(ref_sample) > limit)[0][0]
                index = np.abs(np.array(ref_sample) - limit).argmin() + 1
                ax.axvline(x=index, color='red', linewidth=1.5) #linestyle='--',

            plt.savefig(f'{file_basename}_{metric}.png')
            tikz.save(f'{file_basename}_{metric}.tex')
            plt.close()


    def save_sample(self, file_basename: str, complete_size: int, fs: float) -> None:
        signal, _ = self._generate(file_basename=file_basename, complete_size=complete_size, fs=fs)
        wav_file.write(f'{file_basename}.wav', fs, syn_noise.normalize(signal, type=1))
