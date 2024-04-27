import enum
import typing
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav_file
import tikzplotlib as tikz

import noise_synthesis.noise as syn_noise
import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.detector as syn_detector


class Experiment():

    def __init__(self,
                 generator: syn_signals.Generator,
                 metrics: syn_metrics.Metrics,
                 detector: syn_detector.Detector,
                 window_size: int,
                 overlap: float,) -> None:
        self.generator = generator
        self.metrics = metrics
        self.detector = detector
        self.window_size = window_size
        self.overlap = overlap

    def boxplot(self, file_basename: str, complete_size: int, fs: float, n_runs: int) -> None:

        results = []

        fig, ax = plt.subplots(figsize=(12, 8))

        syn_noise.set_seed()
        for _ in tqdm.tqdm(range(n_runs), leave=False, desc="Run"):
            values, start_sample, limits = self.calculate(complete_size=complete_size, fs=fs)
            results.append(values)

        ax.boxplot(np.array(results))

        indices = np.linspace(0, len(start_sample) - 1, 5, dtype=int)
        ax.set_xticks([i for i in indices])
        ax.set_xticklabels([f'{start_sample[i]/fs:.2f}s' for i in indices])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Intensity')

        for limit in limits:
            start_index = np.where(np.array(start_sample) >= limit[0] - self.window_size)[0][0]
            end_index = np.where(np.array(start_sample) >= limit[1])[0][0]
            ax.axvline(x=start_index, color='red', linewidth=1.5) #linestyle='--',
            ax.axvline(x=end_index, color='blue', linewidth=1.5)
            

        plt.savefig(f'{file_basename}.png')
        tikz.save(f'{file_basename}.tex')
        plt.close()

    def calculate(self, complete_size: int, fs: float):

        signal, limits = self.generator.generate(complete_size=complete_size, fs=fs)

        values, start_sample = self.metrics.calc_data(data=signal,
                                                window_size=self.window_size,
                                                overlap=self.overlap)

        return values, start_sample, limits

    def execute(self, complete_size: int, fs: float, n_runs: int):
        TP, FP = [], []

        syn_noise.set_seed()
        for _ in tqdm.tqdm(range(n_runs), leave=False, desc="Run"):

            values, start_sample, limits = self.calculate(complete_size, fs)

            intervals = []
            for limit in limits:
                start_index = np.where(np.array(start_sample) >= limit[0] - self.window_size)[0][0]
                end_index = np.where(np.array(start_sample) >= limit[1])[0][0]
                intervals.append([start_index, end_index])

            tp, fp = self.detector.run(input_data=np.array(values), intervals=intervals)

            TP.extend([tp])
            FP.extend([fp])

        TP = np.array(TP)
        FP = np.array(FP)

        return f'{np.mean(TP)*100:.2f} ± {np.std(TP)*100:.2f}', \
                f'{np.mean(FP)*100:.2f} ± {np.std(FP)*100:.2f}'


class Comparator():

    def __init__(self) -> None:
        self.experiment_params = []
        self.experiment_list = []

    def add_exp(self, params_ids, experiment: Experiment) -> None:
        self.experiment_params.append(params_ids)
        self.experiment_list.append(experiment)

    def plot(self, file_basename: str, complete_size: int, fs: float, n_runs = 100, error_bar = False) -> None:

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

    def execute(self, complete_size: int, fs: float, n_runs = 100, label = "Experiment"):

        headers = []
        for param_pack in self.experiment_params:
            for key, value in param_pack.items():
                headers.append(key)

        headers = list(set(headers))

        columns = headers.copy()
        columns.extend(['TP', 'FP'])

        results_df = pd.DataFrame(columns=columns)

        for i in tqdm.tqdm(range(len(self.experiment_list)), leave=False, desc=label):

            tp, fp = self.experiment_list[i].execute(complete_size=complete_size,
                                                     fs=fs,
                                                     n_runs=n_runs)
            
            result_dict = {}
            for header in headers:
                if header in self.experiment_params[i]:
                    result_dict[header] = self.experiment_params[i][header]
                else:
                    result_dict[header] = ' - '

            result_dict['TP'] = tp
            result_dict['FP'] = fp

            results_df = pd.concat([results_df, pd.DataFrame(result_dict, index=[0])],
                                    ignore_index=True)

        return results_df