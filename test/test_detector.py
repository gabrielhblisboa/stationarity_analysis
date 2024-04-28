
import os
import math
import itertools

import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.experiment as syn_exp
import noise_synthesis.detector as syn_detector


def main():
    """Main function for the test program."""

    # Set parameters for synthetic noise generation
    fs = 52734
    n_samples = 6 * 3 * fs
    start_psd_db = -np.log10(fs/2)*20  # psd de um ruido branco de variancia 1
    end_psd_db = start_psd_db+6

    base_dir = f"./result/test/detector"
    os.makedirs(base_dir, exist_ok = True)

    # params = {
    #     'window_size': [1024, 2*1024, 4*1024, 8*1024, 16*1024],
    #     'n_points': [128, 256, 512, 1024],
    #     'Memory size': [64, 128],
    #     'Threshold': [3, 4, 5],
    # }
    params = {
        'window_size': [16*1024],
        'n_points': [64],
        'Memory size': [32],
        'Threshold': [2.5],
        'Overlap': [0.75],
        'Signal': [syn_signals.SyntheticSignal.Type.WHITE,
                   syn_signals.SyntheticSignal.Type.BROWN],
    }

    comp = syn_exp.Comparator()

    combinations = list(itertools.product(*params.values()))
    for combination in combinations:
        param_pack = dict(zip(params.keys(), combination))

        metrics = syn_metrics.Metrics(type=syn_metrics.Metrics.Type.WASSERSTEIN,
                                    estimator=syn_metrics.DataEstimator.PDF,
                                    n_points=param_pack['n_points'])
        signal = syn_signals.SyntheticSignal(type=param_pack['Signal'])
        generator = syn_signals.Generator(signal1=signal,
                                        psd_signal1=start_psd_db,
                                        signal2=signal,
                                        psd_signal2=end_psd_db,
                                        transition=syn_signals.AmplitudeTransitionType.ABRUPT)

        detector = syn_detector.Detector(memory_size=param_pack['Memory size'],
                                       threshold=param_pack['Threshold'])

        comp.add_exp(params_ids=param_pack,
                     experiment=syn_exp.Experiment(
                            detector=detector,
                            metrics=metrics,
                            generator=generator,
                            window_size=param_pack['window_size'],
                            overlap=param_pack['Overlap']))

    df = comp.execute(complete_size=n_samples, fs=fs, n_runs=100)
    df.to_pickle(f"{base_dir}/detector.pkl")
    df.to_latex(f"{base_dir}/detector.tex", index=False)
    print(df)


    plt.show()


if __name__ == "__main__":
    main()