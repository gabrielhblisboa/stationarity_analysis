
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
    n_samples = 18 * fs
    start_psd_db = -np.log10(fs/2)*20  # psd de um ruido branco de variancia 1
    end_psd_db = start_psd_db+3

    base_dir = f"./result/detector"
    os.makedirs(base_dir, exist_ok = True)

    metrics = syn_metrics.Metrics(type=syn_metrics.Metrics.Type.WASSERTEIN, estimator=syn_metrics.DataEstimator.PDF)
    signal = syn_signals.SyntheticSignal(type=syn_signals.SyntheticSignal.Type.WHITE)
    generator = syn_signals.Generator(signal1=signal,
                                      psd_signal1=start_psd_db,
                                      signal2=signal,
                                      psd_signal2=end_psd_db,
                                      transition=syn_signals.AmplitudeTransitionType.ABRUPT)

    params = {
        'Memory size': [32, 64, 128],
        'Threshold': [3, 4, 5],
    }

    param_pack_list = []
    experiment_list = []

    comp = syn_exp.Comparator()

    combinations = list(itertools.product(*params.values()))
    for combination in combinations:
        param_pack = dict(zip(params.keys(), combination))

        detector = syn_detector.Detector(memory_size=param_pack['Memory size'],
                                       threshold=param_pack['Threshold'])

        comp.add_exp(params_ids=param_pack,
                     experiment=syn_exp.Experiment(
                            detector=detector,
                            metrics=metrics,
                            generator=generator,
                            window_size=4*1024,
                            overlap=0.75))

    df = comp.execute(complete_size=n_samples, fs=fs, n_runs=20)
    print(df)


    plt.show()


if __name__ == "__main__":
    main()