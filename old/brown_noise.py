
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

    est_type = 'jensen-shannon'

    sub_dir = 'brown-noise'

    # Set up the directory for saving results
    base_dir = f"./result/{sub_dir}/{est_type}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Brown Noise ------------------------------------------------------------------------------------------

    # Generate logarithmically spaced frequencies from 20 Hz to Nyquist frequency
    frequencies = np.logspace(np.log10(20), np.log10(fs / 2), num=100)
    # frequencies = np.logspace(0, np.log10(fs / 2), num=100)

    # Calculate intensities for colored noise
    ref_frequency = 20  # Reference frequency (20 Hz)
    intensities = -6 * np.log2(frequencies / ref_frequency)

    brown_noise = syn_noise.generate_noise(frequencies, intensities, n_samples, fs)

    calculate_kl(brown_noise, n_samples, 'Brown', 'std', sub_dir, est_type=est_type)

    out_wav = f"{base_dir}/brown_noise_std_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_noise.normalize(brown_noise, 1))

    # plt.figure(figsize=(12, 5))
    # plt.plot(brown_noise, color='brown')
    # plt.title("Teste Ruido Rosa")
    # plt.show()

    # Brown Noise with multiple passages in filter -------------------------------------------------------------------

    block1 = syn_noise.generate_noise(frequencies, intensities, int(n_samples / 4), fs)
    block2 = syn_noise.generate_noise(frequencies, intensities, int(n_samples / 4), fs)
    block3 = syn_noise.generate_noise(frequencies, intensities, int(n_samples / 4), fs)
    block4 = syn_noise.generate_noise(frequencies, intensities, int(n_samples / 4), fs)

    aux1 = np.append(block1, block2)
    aux2 = np.append(aux1, block3)
    mult_noise = np.append(aux2, block4)

    calculate_kl(mult_noise, n_samples, 'Brown', 'mult', sub_dir, est_type=est_type)

    out_wav = f"{base_dir}/brown_noise_mult_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_noise.normalize(mult_noise, 1))

    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,5))
    #
    # ax[0].plot(mult_noise, color='brown')
    # ax[1].plot(brown_noise, color='brown')
    # plt.title("Teste Ruido Rosa")
    # plt.show()

    # Brown Noise with amplitude variation (abrupt transition) -------------------------------------------------------
    transition_start = int(n_samples * 0.25)
    transition_end = int(n_samples * 0.75)

    mod_noise = brown_noise

    mod_noise[transition_start:transition_end] = (
            mod_noise[transition_start:transition_end] * 10)

    calculate_kl(mod_noise, n_samples, 'Brown', 'amp-var', sub_dir,
                 transition_start=transition_start, transition_end=transition_end, est_type=est_type)

    out_wav = f"{base_dir}/brown_noise_amp-var_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_noise.normalize(mod_noise, 1))

    # Brown Noise with amplitude variation (linear transition) -------------------------------------------------------
    mod_noise = noise_mod.transition(brown_noise, n_samples, 'linear')
    calculate_kl(mod_noise, n_samples, 'Brown', 'linear-trans', sub_dir, est_type=est_type)

    out_wav = f"{base_dir}/brown_noise_linear-trans_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_noise.normalize(mod_noise, 1))

    # # Brown noise with amplitude variation (sin transition) ---------------------------------------------------------
    mod_noise = noise_mod.transition(brown_noise, n_samples, 'sin')
    calculate_kl(mod_noise, n_samples, 'Brown', 'sin-trans', sub_dir, est_type=est_type)

    out_wav = f"{base_dir}/brown_noise_sin-trans_audio.wav"

    scipy_wav.write(out_wav, fs, syn_noise.normalize(mod_noise, 1))

    # Brown Noise with inverted samples in second block -----------------------------------------------------
    brown_noise[int(n_samples * 0.4):int(n_samples * 0.6)] = (
            brown_noise[int(n_samples * 0.4):int(n_samples * 0.6)][::-1])

    calculate_kl(brown_noise, n_samples, 'Brown', 'inv', sub_dir, est_type=est_type)

    out_wav = f"{base_dir}/brown_noise_inv_audio.wav"

    scipy_wav.write(out_wav, fs, syn_noise.normalize(brown_noise, 1))


if __name__ == '__main__':
    main()


