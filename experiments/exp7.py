
import os
import argparse
import itertools
import tqdm

import numpy as np

import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.experiment as syn_exp
import noise_synthesis.detector as syn_detector

import config

def main(n_runs: int):
    """Main function for the test program."""

    base_dir = f"./result/{os.path.splitext(os.path.basename(__file__))[0]}"
    os.makedirs(base_dir, exist_ok = True)

    params = {
        'Transição': [syn_signals.AmplitudeTransitionType.LINEAR,
                      syn_signals.AmplitudeTransitionType.SIGMOIDAL
                      ],
        'Número de amostras': [0.5 * config.window_size,
                               1 * config.window_size,
                               2 * config.window_size,
                               4 * config.window_size,
                               8 * config.window_size,
                               16 * config.window_size],
        # 'Sinal': [syn_signals.SyntheticSignal.Type.LOW,
        #            syn_signals.SyntheticSignal.Type.MEDIUM_LOW,
        #            syn_signals.SyntheticSignal.Type.MEDIUM_HIGH,
        #            syn_signals.SyntheticSignal.Type.HIGH],
    }

    comp = syn_exp.Comparator()

    metrics = syn_metrics.Metrics(type=syn_metrics.Metrics.Type.WASSERSTEIN,
                                    estimator=syn_metrics.DataEstimator.PDF,
                                    n_points=config.selected_n_points)
    signal=syn_signals.SyntheticSignal(type=config.noise)
    generator = syn_signals.Generator(signal1=signal,
                                    psd_signal1=config.psd_db,
                                    signal2=signal,
                                    psd_signal2=config.psd_db + 3,
                                    transition=syn_signals.AmplitudeTransitionType.ABRUPT)
    detector = syn_detector.Detector(memory_size=config.memory_size,
                                        threshold=config.threshold)
    experiment = syn_exp.Experiment(output_base_name=f"{base_dir}/0",
                                    detector=detector,
                                    metrics=metrics,
                                    generator=generator,
                                    window_size=config.window_size,
                                    overlap=config.overlap)
    comp.add_exp(params_ids={'Transição': syn_signals.AmplitudeTransitionType.ABRUPT,
                             'Número de amostras': ' - '}, experiment=experiment)

    combinations = list(itertools.product(*params.values()))
    for i, combination in enumerate(combinations):
        param_pack = dict(zip(params.keys(), combination))

        metrics = syn_metrics.Metrics(type=syn_metrics.Metrics.Type.WASSERSTEIN,
                                      estimator=syn_metrics.DataEstimator.PDF,
                                      n_points=config.selected_n_points)
        signal=syn_signals.SyntheticSignal(type=config.noise)
        generator = syn_signals.Generator(signal1=signal,
                                        psd_signal1=config.psd_db,
                                        signal2=signal,
                                        psd_signal2=config.psd_db + 3,
                                        transition=param_pack['Transição'],
                                        transition_samples=int(param_pack['Número de amostras']))
        detector = syn_detector.Detector(memory_size=config.memory_size,
                                         threshold=config.threshold)
        experiment = syn_exp.Experiment(output_base_name=f"{base_dir}/{i+1}",
                                        detector=detector,
                                        metrics=metrics,
                                        generator=generator,
                                        window_size=config.window_size,
                                        overlap=config.overlap)

        comp.add_exp(params_ids=param_pack, experiment=experiment)

    df = comp.execute(complete_size=config.n_samples, fs=config.fs, n_runs=n_runs)
    df.to_pickle(f"{base_dir}.pkl")
    df.style.hide(axis="index").to_latex(f"{base_dir}.tex")
    print(df)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'Run experiment {os.path.splitext(os.path.basename(__file__))[0]}')
    parser.add_argument('--n_runs', default=config.n_runs, type=int, help='Number of runs')
    args = parser.parse_args()
    main(n_runs = args.n_runs)