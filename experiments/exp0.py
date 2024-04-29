
import os
import argparse
import itertools
import tqdm

import noise_synthesis.metrics as syn_metrics
import noise_synthesis.signals as syn_signals
import noise_synthesis.experiment as syn_exp
import noise_synthesis.detector as syn_detector

import config

def main(n_runs: int, test: int):
    """Main function for the test program."""

    base_dir = f"./result/{os.path.splitext(os.path.basename(__file__))[0]}"
    os.makedirs(base_dir, exist_ok = True)

    params = {
        'Teste': [syn_metrics.StatisticTest.Type.ADF,
                          syn_metrics.StatisticTest.Type.PhillipsPerron],
        'Signal': [syn_signals.SyntheticSignal.Type.WHITE,
                   syn_signals.SyntheticSignal.Type.BROWN,
                   syn_signals.SyntheticSignal.Type.PINK],
    }

    comp = syn_exp.Comparator()

    combinations = list(itertools.product(*params.values()))
    for i, combination in enumerate(combinations):
        param_pack = dict(zip(params.keys(), combination))

        metrics = syn_metrics.StatisticTest(type=param_pack['Teste'])
        signal=syn_signals.SyntheticSignal(type=param_pack['Signal'])
        generator = syn_signals.Generator(signal1=signal,
                                        psd_signal1=config.psd_db,
                                        signal2=signal,
                                        psd_signal2=config.end_psd_db,
                                        transition=syn_signals.AmplitudeTransitionType.ABRUPT)
        detector = syn_detector.Detector(memory_size=config.memory_size,
                                         threshold=config.threshold)
        experiment = syn_exp.Experiment(output_base_name=f"{base_dir}/{i}",
                                        detector=detector,
                                        metrics=metrics,
                                        generator=generator,
                                        window_size=config.window_size,
                                        overlap=config.overlap)

        file_basename = f"{base_dir}/"
        for _, value in param_pack.items():
            file_basename = f'{file_basename}{str(value)} '

        generator.save_sample(file_basename = file_basename,
                              complete_size = config.n_samples,
                              fs = config.fs)

        # experiment.boxplot(file_basename = file_basename,
        #                    complete_size = config.n_samples,
        #                    fs = config.fs,
        #                    n_runs = n_runs)

        comp.add_exp(params_ids=param_pack, experiment=experiment)


    df = comp.execute(complete_size=config.n_samples, fs=config.fs, n_runs=n_runs)
    df.to_pickle(f"{base_dir}.pkl")
    df.style.hide(axis="index").to_latex(f"{base_dir}.tex")
    print(df)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'Run experiment {os.path.splitext(os.path.basename(__file__))[0]}')
    parser.add_argument('--n_runs', default=config.n_runs, type=int, help='Number of runs')
    parser.add_argument('--test', default=-1, type=int, help='ADF(0), KPSS(1), PP(2)')
    args = parser.parse_args()
    main(n_runs = args.n_runs, test = args.test)