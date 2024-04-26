
import os
import math
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

    metric_list = []
    for estimator in [syn_metrics.DataEstimator.PDF]:
        metric_list.append(syn_metrics.Metrics(type=syn_metrics.Metrics.Type.WASSERTEIN, estimator=estimator))

    signal = syn_signals.SyntheticSignal(type=syn_signals.SyntheticSignal.Type.WHITE)

    file_basename = f"{base_dir}/{signal}"

    exp = syn_exp.Experiment(name='test_syn',
                                signal1=signal,
                                psd_signal1=start_psd_db,
                                signal2=signal,
                                psd_signal2=end_psd_db,
                                transition=syn_exp.AmplitudeTransitionType.ABRUPT,
                                window_size=4*1024,
                                overlap=0.75,
                                metric_list=metric_list)
    
    comp = syn_exp.Comparator(experiment_list=[exp])

    df = comp.detect(detector=syn_detector.Detector(memory_size=128, threshold=5),
                complete_size=n_samples,
                fs=fs,
                n_runs=100)

    print(df)


    plt.show()


if __name__ == "__main__":
    main()