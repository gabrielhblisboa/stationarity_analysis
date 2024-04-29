
import os
import argparse
import math
import itertools
import tqdm
import random

import pandas as pd
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.noise as syn_noise
import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.experiment as syn_exp
import noise_synthesis.detector as syn_detector

import config

def main(n_runs: int):
    """Main function for the test program."""

    # Set parameters for synthetic noise generation
    base_dir = f"./result/{os.path.splitext(os.path.basename(__file__))[0]}"
    os.makedirs(base_dir, exist_ok = True)

    params = {
        'Estimador': syn_metrics.DataEstimator,
        'Diferença de Intensidade (dB)': [0, 3, 6, 10]
    }

    combinations = list(itertools.product(*params.values()))

    headers = list(dict(zip(params.keys(), combinations[0])).keys())
    columns = headers.copy()
    columns.extend(['Prob. Detecção', 'Falso Alarme'])
    results_df = pd.DataFrame(columns=columns)

    for i, combination in enumerate(tqdm.tqdm(combinations, desc='Experiments', leave=False)):
        param_pack = dict(zip(params.keys(), combination))

        metrics = syn_metrics.Metrics(type=syn_metrics.Metrics.Type.WASSERSTEIN,
                                    estimator=param_pack['Estimador'],
                                    n_points=config.selected_n_points if param_pack['Estimador'] == syn_metrics.DataEstimator.PDF else config.selected_fft_n_points)

        detector = syn_detector.Detector(memory_size=config.memory_size,
                                        threshold=config.threshold)
        
        TP, FP = [], []
        syn_noise.set_seed()

        ship_noise = [syn_signals.RealSignal.Type.FISH_BOAT,
                       syn_signals.RealSignal.Type.MUSSEL_BOAT,
                       syn_signals.RealSignal.Type.DREDGER]

        for _ in tqdm.tqdm(range(n_runs), leave=False, desc="Run"):

            noise2 = random.sample(ship_noise, 2)

            signal1=syn_signals.RealSignal(type=noise2[0])
            signal2=syn_signals.RealSignal(type=noise2[1])
            generator = syn_signals.Generator(signal1=signal1,
                                            psd_signal1=config.psd_db,
                                            signal2=signal2,
                                            psd_signal2=config.psd_db + param_pack['Diferença de Intensidade (dB)'],
                                            transition=syn_signals.AmplitudeTransitionType.ABRUPT)


            signal, limits = generator.generate(complete_size=config.n_samples, fs=config.fs)

            values, start_sample = metrics.calc_data(data=signal,
                                            window_size=config.window_size,
                                            overlap=config.selected_overlap)

            intervals = []
            for limit in limits:
                start_index = np.where(np.array(start_sample) >= limit[0] - config.window_size)[0][0]
                end_index = np.where(np.array(start_sample) >= limit[1])[0][0]
                intervals.append([start_index, end_index])

            tp, fp = detector.run(input_data=np.array(values), intervals=intervals)

            TP.extend([tp])
            FP.extend([fp])

        TP = np.array(TP)
        FP = np.array(FP)

        result_dict = {}
        for header in headers:
            if header in param_pack:
                result_dict[header] = param_pack[header]
            else:
                result_dict[header] = ' - '

        result_dict['Prob. Detecção'] = f'{np.mean(TP)*100:.2f} ± {np.std(TP)*100:.2f} \%'.replace('.',',')
        result_dict['Falso Alarme'] = f'{np.mean(FP)*100:.2f} ± {np.std(FP)*100:.2f} \%'.replace('.',',')

        results_df = pd.concat([results_df, pd.DataFrame(result_dict, index=[0])],
                                ignore_index=True)


    print(results_df)
    results_df.to_pickle(f"{base_dir}.pkl")
    results_df.style.hide(axis="index").to_latex(f"{base_dir}.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'Run experiment {os.path.splitext(os.path.basename(__file__))[0]}')
    parser.add_argument('--n_runs', default=config.n_runs, type=int, help='Number of runs')
    args = parser.parse_args()
    main(n_runs = args.n_runs)