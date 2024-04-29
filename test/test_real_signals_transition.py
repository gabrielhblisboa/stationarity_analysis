
import os
import math
import itertools
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.noise as syn_noise
import noise_synthesis.signals as syn_signals


def main():
    """Main function for the test program."""

    # Set up the directory for saving results
    base_dir = "./result/test/signals/real_transition"
    os.makedirs(base_dir, exist_ok = True)

    # Set parameters for synthetic noise generation
    fs = 52734
    n_samples = 6 * 3 * fs
    baseline_psd_db = 10

    params = {
        'Arquivo1' : syn_signals.RealSignal.Type,
        'Arquivo2' : syn_signals.RealSignal.Type,
    }

    combinations = list(itertools.product(*params.values()))
    for i, combination in enumerate(combinations):
        param_pack = dict(zip(params.keys(), combination))

        if param_pack['Arquivo1'] == param_pack['Arquivo2']:
            continue

        signal1=syn_signals.RealSignal(type=param_pack['Arquivo1'])
        signal2=syn_signals.RealSignal(type=param_pack['Arquivo2'])
        generator = syn_signals.Generator(signal1=signal1,
                                        psd_signal1=baseline_psd_db,
                                        signal2=signal2,
                                        psd_signal2=baseline_psd_db,
                                        transition=syn_signals.AmplitudeTransitionType.ABRUPT)

        for a in range(3):
            signal, _ = generator.generate(complete_size=n_samples, fs=fs)
            scipy_wav.write(f"{base_dir}/{param_pack['Arquivo1']} -- {param_pack['Arquivo2']} ---- {a}.wav", fs, syn_noise.normalize(signal, type=1))
            plt.plot(signal)
            plt.savefig(f"{base_dir}/{param_pack['Arquivo1']} -- {param_pack['Arquivo2']} ---- {a}.png")
            plt.close()


if __name__ == "__main__":
    main()