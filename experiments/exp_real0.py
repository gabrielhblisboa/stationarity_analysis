
import os
import math
import itertools

import pandas as pd
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.experiment as syn_exp
import noise_synthesis.detector as syn_detector

import config

def main():
    """Main function for the test program."""

    # Set parameters for synthetic noise generation
    base_dir = f"./result/{os.path.splitext(os.path.basename(__file__))[0]}"
    os.makedirs(base_dir, exist_ok = True)

    params = {
        'Arquivo' : syn_signals.RealSignal.Type,
        # 'Estimador': [syn_metrics.DataEstimator.PDF],
        '': [syn_metrics.Metrics.Type.WASSERSTEIN, syn_metrics.Metrics.Type.JENSEN_SHANNON],
    }

    combinations = list(itertools.product(*params.values()))

    headers = list(dict(zip(params.keys(), combinations[0])).keys())
    columns = headers.copy()
    columns.extend(['Falso Alarme'])
    results_df = pd.DataFrame(columns=columns)

    for i, combination in enumerate(combinations):
        param_pack = dict(zip(params.keys(), combination))

        signal = syn_signals.RealSignal(type=param_pack['Arquivo'])

        data, _ = signal.complete()

        # metrics = syn_metrics.Metrics(type=param_pack[''],
        #                             estimator=param_pack['Estimador'],
        #                             n_points=config.selected_n_points if param_pack['Estimador'] == syn_metrics.DataEstimator.PDF else config.selected_fft_n_points)
        metrics = syn_metrics.Metrics(type=param_pack[''],
                                    estimator=syn_metrics.DataEstimator.PDF,
                                    n_points=config.selected_n_points if param_pack[''] == syn_metrics.Metrics.Type.WASSERSTEIN else config.selected_jensen_n_points)

        detector = syn_detector.Detector(memory_size=config.memory_size,
                                        threshold=config.threshold)

        values, _ = metrics.calc_data(data=data,
                                        window_size=config.window_size,
                                        overlap=config.selected_overlap)

        _, fp = detector.run(input_data=np.array(values), intervals=[])

        result_dict = {}
        for header in headers:
            if header in param_pack:
                result_dict[header] = param_pack[header]
            else:
                result_dict[header] = ' - '
        result_dict['Falso Alarme'] = f'{fp*100:.2f} \%'.replace('.',',')

        results_df = pd.concat([results_df, pd.DataFrame(result_dict, index=[0])],
                                ignore_index=True)

        print(param_pack, ': ', fp)

    results_df.to_pickle(f"{base_dir}.pkl")
    results_df.style.hide(axis="index").to_latex(f"{base_dir}.tex")


if __name__ == "__main__":
    main()