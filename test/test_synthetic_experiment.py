
import os
import tqdm
import numpy as np
import itertools

import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.experiment as syn_exp
import noise_synthesis.detector as syn_detector



def main(n_runs = 2):
    """Main function for the test program."""

    # Set parameters for synthetic noise generation
    fs = 52734
    n_samples = 3 * fs
    start_psd_db = -np.log10(fs/2)*20  # psd de um ruido branco de variancia 1
    end_psd_db = start_psd_db+3

    base_dir = f"./result/test/experiment/synthetic"
    os.makedirs(base_dir, exist_ok = True)

    params = {
        'Metrics': syn_metrics.Metrics.Type,
        'Estimator': syn_metrics.DataEstimator,
        'Transition': syn_signals.AmplitudeTransitionType,
        'Signal': syn_signals.SyntheticSignal.Type
    }

    combinations = list(itertools.product(*params.values()))
    for combination in tqdm.tqdm(combinations, desc='Experiments', leave=False):
        param_pack = dict(zip(params.keys(), combination))

        metrics = syn_metrics.Metrics(type=param_pack['Metrics'], estimator=param_pack['Estimator'])
        signal = syn_signals.SyntheticSignal(type=param_pack['Signal'])
        generator = syn_signals.Generator(signal1=signal,
                                        psd_signal1=start_psd_db,
                                        signal2=signal,
                                        psd_signal2=end_psd_db,
                                        transition=param_pack['Transition'])
        detector = syn_detector.Detector()
        experiment = syn_exp.Experiment(detector=detector,
                                      metrics=metrics,
                                      generator=generator,
                                      window_size=4*1024,
                                      overlap=0.75)

        file_basename = f"{base_dir}/"
        for _, value in param_pack.items():
            file_basename = f'{file_basename}{str(value)}'

        experiment.boxplot(file_basename = file_basename,
                           complete_size = n_samples,
                           fs = fs,
                           n_runs = n_runs)


if __name__ == "__main__":
    main()