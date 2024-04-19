
import os
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.noise as syn_noise
from kl_test import calculate_kl
import noise_mod


def main():

    fs = 48000  # Example sampling frequency
    n_samples = 5 * fs  # Five seconds of noise

    pdf_type = 'fft'
    sub_dir = 'white-noise'
    est_type = 'kl'

    # Set up the directory for saving results
    base_dir = f"./result/{sub_dir}/{est_type}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    white_noise = np.random.normal(0, 1, n_samples)

    # White Noise with amplitude variation (abrupt transition) -------------------------------------------------------
    transition_start = int(n_samples * 0.33)
    transition_end = int(n_samples * 0.67)

    mod_noise = white_noise

    mod_noise[transition_start:transition_end] = (
            mod_noise[transition_start:transition_end] * 3)

    calculate_kl(mod_noise, n_samples, 'White', 'amp-var', sub_dir,
                 transition_start=transition_start, transition_end=transition_end, est_type=est_type, pdf_type=pdf_type)

    out_wav = f"{base_dir}/white_noise_amp-var_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_noise.normalize(mod_noise, 1))

    # White Noise with amplitude variation (linear transition) -------------------------------------------------------
    mod_noise = noise_mod.transition(white_noise, n_samples, 'linear')
    calculate_kl(mod_noise, n_samples, 'White', 'linear-trans', sub_dir, est_type=est_type, pdf_type=pdf_type)

    out_wav = f"{base_dir}/white_noise_linear-trans_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_noise.normalize(mod_noise, 1))

    # # White noise with amplitude variation (sin transition) ---------------------------------------------------------
    mod_noise = noise_mod.transition(white_noise, n_samples, 'sin')
    calculate_kl(mod_noise, n_samples, 'White', 'sin-trans', sub_dir, est_type=est_type, pdf_type=pdf_type)

    out_wav = f"{base_dir}/white_noise_sin-trans_audio.wav"

    scipy_wav.write(out_wav, fs, syn_noise.normalize(mod_noise, 1))

    # White Noise with inverted samples in second block -----------------------------------------------------
    white_noise[int(n_samples * 0.4):int(n_samples * 0.6)] = (
            white_noise[int(n_samples * 0.4):int(n_samples * 0.6)][::-1])

    calculate_kl(white_noise, n_samples, 'White', 'inv', sub_dir, est_type=est_type, pdf_type=pdf_type)

    out_wav = f"{base_dir}/white_noise_inv_audio.wav"

    scipy_wav.write(out_wav, fs, syn_noise.normalize(white_noise, 1))


if __name__ == '__main__':
    main()


