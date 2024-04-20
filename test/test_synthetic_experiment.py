
import os
import math
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.experiment as syn_exp


def main():
    """Main function for the test program."""

    # Set parameters for synthetic noise generation
    fs = 48000
    n_samples = 3 * fs
    start_psd_db = -np.log10(fs/2)*20  # psd de um ruido branco de variancia 1
    end_psd_db = start_psd_db+3

    for type in syn_metrics.Metrics.Type:
        base_dir = f"./result/experiment/synthetic/{type}"
        os.makedirs(base_dir, exist_ok = True)

        metric_list = []
        for estimator in syn_metrics.DataEstimator:
            metric_list.append(syn_metrics.Metrics(type=type, estimator=estimator))

        for transitions in syn_exp.AmplitudeTransitionType:
            for type in syn_signals.SyntheticSignal.Type:
                signal = syn_signals.SyntheticSignal(type=type)

                file_basename = f"{base_dir}/{transitions} {signal}"

                exp = syn_exp.Experiment(signal1=signal,
                                         psd_signal1=start_psd_db,
                                         signal2=signal,
                                         psd_signal2=end_psd_db,
                                         transition=transitions,
                                         metric_list=metric_list)

                exp.run(file_basename=file_basename,
                        complete_size=n_samples,
                        fs=fs,
                        window_size=4096,
                        overlap=0)


if __name__ == "__main__":
    main()