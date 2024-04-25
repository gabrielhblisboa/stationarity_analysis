import enum
import typing
import tqdm

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
                      transition_end - transition_samples,
                      transition_start + transition_samples,
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
                 name: str,
                 signal1: syn_signals.SyntheticSignal,
                 psd_signal1: float,
                 signal2: syn_signals.SyntheticSignal,
                 psd_signal2: float,
                 transition: AmplitudeTransitionType,
                 window_size: int,
                 overlap: float,
                 metric_list: typing.List[syn_metrics.Metrics]) -> None:
        self.name = name
        self.signal1 = signal1
        self.psd_signal1 = psd_signal1
        self.signal2 = signal2
        self.psd_signal2 = psd_signal2
        self.transition = transition
        self.window_size = window_size
        self.metric_list = metric_list
        self.overlap = overlap

    def generate(self, complete_size: int, fs: float) -> typing.Tuple[np.array, typing.List[int]]:
        signal1 = self.signal1.generate(complete_size, fs, self.psd_signal1)
        signal2 = self.signal2.generate(complete_size, fs, self.psd_signal2)
        return self.transition.apply(signal1=signal1, signal2=signal2)

    def run(self, file_basename: str, complete_size: int, fs: float, n_runs = 100) -> None:

        for metric in self.metric_list:
            results = []

            fig, ax = plt.subplots(figsize=(12, 8))

            for _ in range(n_runs):
                signal, limits = self.generate(complete_size=complete_size, fs=fs)
                values, start_sample = metric.calc_data(data=signal, window_size=self.window_size, overlap=self.overlap)
                results.append(values)

            ax.boxplot(np.array(results))

            indices = np.linspace(0, len(start_sample) - 1, 5, dtype=int)
            ax.set_xticks([i for i in indices])
            ax.set_xticklabels([f'{start_sample[i]/fs:.2f}s' for i in indices])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Intensity')

            for limit in limits:
                index = np.where(np.array(start_sample) + self.window_size > limit)[0][0]
                ax.axvline(x=index, color='red', linewidth=1.5) #linestyle='--',

                index = np.where(np.array(start_sample) > limit)[0][0]
                ax.axvline(x=index, color='blue', linewidth=1.5)
                

            plt.savefig(f'{file_basename}_{metric}.png')
            tikz.save(f'{file_basename}_{metric}.tex')
            plt.close()

    def save_sample(self, file_basename: str, complete_size: int, fs: float) -> None:
        signal, _ = self.generate(complete_size=complete_size, fs=fs)
        wav_file.write(f'{file_basename}.wav', fs, syn_noise.normalize(signal, type=1))

class Comparator():
    def __init__(self, experiment_list) -> None:
        self.experiment_list = experiment_list

    def run(self, file_basename: str, complete_size: int, fs: float, n_runs = 100, error_bar = False) -> None:

        fig, ax = plt.subplots(figsize=(12, 8))

        for exp in tqdm.tqdm(self.experiment_list, leave=False, desc="Experiment"):

            for metric in exp.metric_list:

                results = []
                for _ in range(n_runs):
                    signal, limits = exp.generate(complete_size=complete_size, fs=fs)
                    values, start_sample = metric.calc_data(data=signal, window_size=exp.window_size, overlap=exp.overlap)
                    results.append(values)

                    if len(exp.metric_list) == 1:
                        label = f'{exp.name}'
                    else:
                        label = f'{exp.name}_metric[{metric}]'

                y = np.mean(np.array(results), axis=0)
                y_err = np.std(np.array(results), axis=0)
                y_err = y_err/(np.max(y) - np.min(y))
                y = (y-np.min(y))/(np.max(y) - np.min(y))
                y = y-np.mean(y)
                if error_bar:
                    ax.errorbar(np.array(start_sample)/fs, y, yerr=y_err, label=label, fmt='-o', capsize=5)
                else:
                    ax.plot(np.array(start_sample)/fs, y, label=label)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Intensity')

        # for limit in limits:
        #     ax.axvline(x=(limit - window_size)/fs, color='red', linewidth=1.5) #linestyle='--',
        #     ax.axvline(x=(limit)/fs, color='blue', linewidth=1.5)

        tikz.save(f'{file_basename}.tex')

        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        fig.tight_layout()
        plt.savefig(f'{file_basename}.png')
        plt.close()
