
import os
import math
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals


def main():
    """Main function for the test program."""


    # Set parameters for synthetic noise generation
    fs = 48000
    n_samples = 3 * fs
    start_psd_db = -np.log10(fs/2)*20  # psd de um ruido branco de variancia 1
    end_psd_db = start_psd_db+3

    for type in syn_metrics.Metrics.Type:
        base_dir = f"./result/experiment/{type}"
        os.makedirs(base_dir, exist_ok = True)

        metric_list = []
        for estimator in syn_metrics.DataEstimator:
            metric_list.append(syn_metrics.Metrics(type=type, estimator=estimator))

        for transitions in syn_signals.AmplitudeTransitionType:
            for signal in syn_signals.SyntheticSignals:

                file_basename = f"{base_dir}/{transitions} {signal}"

                exp = syn_signals.Experiment(signal=signal,
                                            transition=transitions,
                                            metric_list=metric_list,
                                            start_psd_db=start_psd_db,
                                            end_psd_db=end_psd_db)

                exp.run(file_basename=file_basename, complete_size=n_samples, fs =fs, window_size=4096, overlap=0)


if __name__ == "__main__":
    main()