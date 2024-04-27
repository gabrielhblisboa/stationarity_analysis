
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
    baseline_psd_db = 0

    base_dir = f"./result/test/adf/"
    os.makedirs(base_dir, exist_ok = True)

    adf = syn_metrics.StatisticTest(syn_metrics.StatisticTest.Type.ADF)

    for n_samples in [1/4, 1, 5]:

        print(f'samples {n_samples}s')

        for signal_type in syn_signals.RealSignal.Type:

            signal = syn_signals.RealSignal(type=signal_type)

            data = signal.generate(n_samples=int(n_samples*fs),
                                fs=fs,
                                baseline_psd_db=baseline_psd_db)

            metrics, _ = adf.calc_data(data=data, window_size=4*1024, overlap=0.75)

            print(np.min(metrics), ' -> ', np.max(metrics))


if __name__ == "__main__":
    main()