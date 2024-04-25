
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
    fs = 52734
    n_samples = 9 * fs
    baseline_psd_db = 0

    base_dir = f"./result/test_exp/real/"
    os.makedirs(base_dir, exist_ok = True)

    metric_list = []
    for estimator in [syn_metrics.DataEstimator.PDF]:
        metric_list.append(syn_metrics.Metrics(type=syn_metrics.Metrics.Type.WASSERTEIN,
                                               estimator=estimator))
    metric_list.append(syn_metrics.StatisticTest(syn_metrics.StatisticTest.Type.ADF))

    signal_type_list = [[syn_signals.RealSignal.Type.FLOW, syn_signals.RealSignal.Type.RAIN],
                   [syn_signals.RealSignal.Type.FLOW, syn_signals.RealSignal.Type.WAVE],
                   [syn_signals.RealSignal.Type.FLOW, syn_signals.RealSignal.Type.WIND],
                   [syn_signals.RealSignal.Type.FLOW, syn_signals.RealSignal.Type.MUSSEL_BOAT],
                   [syn_signals.RealSignal.Type.FLOW, syn_signals.RealSignal.Type.FISH_BOAT]]

    for transitions in [syn_exp.AmplitudeTransitionType.ABRUPT]:
        for type1, type2 in signal_type_list:
            signal1 = syn_signals.RealSignal(type=type1)
            signal2 = syn_signals.RealSignal(type=type2)

            file_basename = f"{base_dir}/{transitions} {signal1} {signal2}"

            exp = syn_exp.Experiment(name='test_real',
                                     signal1=signal1,
                                     psd_signal1=baseline_psd_db,
                                     signal2=signal2,
                                     psd_signal2=baseline_psd_db,
                                     transition=transitions,
                                     window_size=4096,
                                     overlap=0,
                                     metric_list=metric_list)

            exp.run(file_basename=file_basename,
                    complete_size=n_samples,
                    fs=fs)


if __name__ == "__main__":
    main()