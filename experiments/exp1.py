
import os
import argparse

import numpy as np

import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.experiment as syn_exp


def main(n_runs: int):
    """Main function for the test program."""

    # Set parameters for synthetic noise generation
    fs = 52734
    n_samples = int(1.2 * fs)
    psd_db = -np.log10(fs/2)*20  # psd de um ruido branco de variancia 1
    end_psd_db = psd_db + 6


    base_dir = f"./result/{os.path.splitext(os.path.basename(__file__))[0]}"
    os.makedirs(base_dir, exist_ok = True)

    metric_list = []
    for type in syn_metrics.Metrics.Type:
        metric_list.append(syn_metrics.Metrics(type=type, estimator=syn_metrics.DataEstimator.PDF, n_points=128))

    for type in syn_signals.SyntheticSignal.Type:
        signal = syn_signals.SyntheticSignal(type=type)

        file_basename = f"{base_dir}/{signal}"

        exp = syn_exp.Experiment(name='exp1',
                                 signal1=signal,
                                 psd_signal1=psd_db,
                                 signal2=signal,
                                 psd_signal2=end_psd_db,
                                 transition=syn_exp.AmplitudeTransitionType.ABRUPT,
                                 window_size=4*1024,
                                 overlap=0.5,
                                 metrics=metric_list)

        exp.run(file_basename=file_basename,
                complete_size=n_samples,
                fs=fs,
                n_runs=n_runs)

        exp.save_sample(file_basename=file_basename,
                complete_size=n_samples,
                fs=fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'Run experiment {os.path.splitext(os.path.basename(__file__))[0]}')
    parser.add_argument('--n_runs', default=100, type=int, help='Number of runs')
    args = parser.parse_args()
    main(n_runs = args.n_runs)