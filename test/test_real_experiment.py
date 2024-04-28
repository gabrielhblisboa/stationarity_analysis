
import os
import tqdm
import itertools

import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.experiment as syn_exp
import noise_synthesis.detector as syn_detector



def main(n_runs = 5):
    """Main function for the test program."""

    # Set parameters for synthetic noise generation
    fs = 52734
    n_samples = 9 * fs
    baseline_psd_db = 0

    base_dir = f"./result/test/experiment/real"
    os.makedirs(base_dir, exist_ok = True)

    metric_list = []
    for estimator in [syn_metrics.DataEstimator.PDF]:
        metric_list.append(syn_metrics.Metrics(type=syn_metrics.Metrics.Type.WASSERSTEIN,
                                               estimator=estimator))
    metric_list.append(syn_metrics.StatisticTest(syn_metrics.StatisticTest.Type.ADF))

    signal_type_list = [syn_signals.RealSignal.Type.RAIN,
                   syn_signals.RealSignal.Type.WAVE,
                   syn_signals.RealSignal.Type.WIND,
                   syn_signals.RealSignal.Type.MUSSEL_BOAT,
                   syn_signals.RealSignal.Type.FISH_BOAT]

    params = {
        'Metrics': metric_list,
        'Signal': signal_type_list
    }

    combinations = list(itertools.product(*params.values()))
    for combination in tqdm.tqdm(combinations, desc='Experiments', leave=False):
        param_pack = dict(zip(params.keys(), combination))

        metrics = param_pack['Metrics']
        signal1=syn_signals.RealSignal(type=syn_signals.RealSignal.Type.FLOW)
        signal2 = syn_signals.RealSignal(type=param_pack['Signal'])
        generator = syn_signals.Generator(signal1=signal1,
                                        psd_signal1=baseline_psd_db,
                                        signal2=signal2,
                                        psd_signal2=baseline_psd_db,
                                        transition=syn_signals.AmplitudeTransitionType.ABRUPT)
        detector = syn_detector.Detector()
        experiment = syn_exp.Experiment(detector=detector,
                                      metrics=metrics,
                                      generator=generator,
                                      window_size=4*1024,
                                      overlap=0.75)

        file_basename = f"{base_dir}/"
        for _, value in param_pack.items():
            file_basename = f'{file_basename}{str(value)} '

        experiment.boxplot(file_basename = file_basename,
                           complete_size = n_samples,
                           fs = fs,
                           n_runs = n_runs)
        

if __name__ == "__main__":
    main()