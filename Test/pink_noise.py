
import os
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.background_noise as syn_bg
from kl_test import calculate_kl
import noise_mod


def main():

    fs = 48000  # Example sampling frequency
    n_samples = 5 * fs  # Five seconds of noise

    sub_dir = 'pink-noise'

    # Set up the directory for saving results
    base_dir = f"./result/{sub_dir}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Pink Noise ------------------------------------------------------------------------------------------

    # Generate logarithmically spaced frequencies from 20 Hz to Nyquist frequency
    frequencies = np.logspace(np.log10(20), np.log10(fs / 2), num=100)
    # frequencies = np.logspace(0, np.log10(fs / 2), num=100)

    # Calculate intensities for colored noise
    ref_frequency = 20  # Reference frequency (20 Hz)
    intensities = -3 * np.log2(frequencies / ref_frequency)

    pink_noise = syn_bg.generate_noise(frequencies, intensities, n_samples, fs)

    calculate_kl(pink_noise, n_samples, 'Pink', 'std', sub_dir)

    out_wav = f"{base_dir}/pink_noise_std_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(pink_noise, 1))

    # plt.figure(figsize=(12, 5))
    # plt.plot(pink_noise, color='hotpink')
    # plt.title("Teste Ruido Rosa")
    # plt.show()

    # Pink Noise with multiple passages in filter -------------------------------------------------------------------

    block1 = syn_bg.generate_noise(frequencies, intensities, int(n_samples / 4), fs)
    block2 = syn_bg.generate_noise(frequencies, intensities, int(n_samples / 4), fs)
    block3 = syn_bg.generate_noise(frequencies, intensities, int(n_samples / 4), fs)
    block4 = syn_bg.generate_noise(frequencies, intensities, int(n_samples / 4), fs)

    aux1 = np.append(block1, block2)
    aux2 = np.append(aux1, block3)
    mult_noise = np.append(aux2, block4)

    calculate_kl(mult_noise, n_samples, 'Pink', 'mult', sub_dir)

    out_wav = f"{base_dir}/pink_noise_mult_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(mult_noise, 1))

    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,5))
    #
    # ax[0].plot(mult_noise, color='hotpink')
    # ax[1].plot(pink_noise, color='hotpink')
    # plt.title("Teste Ruido Rosa")
    # plt.show()

    # Pink Noise with amplitude variation (abrupt transition) -------------------------------------------------------
    transition_start = int(n_samples * 0.25)
    transition_end = int(n_samples * 0.75)

    mod_noise = pink_noise

    mod_noise[transition_start:transition_end] = (
            mod_noise[transition_start:transition_end] * 3)

    calculate_kl(mod_noise, n_samples, 'Pink', 'amp-var', sub_dir,
                 transition_start=transition_start, transition_end=transition_end)

    out_wav = f"{base_dir}/pink_noise_amp-var_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(mod_noise, 1))

    # Pink Noise with amplitude variation (linear transition) -------------------------------------------------------
    mod_noise = noise_mod.transition(pink_noise, n_samples, 'linear')
    calculate_kl(mod_noise, n_samples, 'Pink', 'linear-trans', sub_dir)

    out_wav = f"{base_dir}/pink_noise_linear-trans_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(mod_noise, 1))

    # # Pink noise with amplitude variation (sin transition) ---------------------------------------------------------
    mod_noise = noise_mod.transition(pink_noise, n_samples, 'sin')
    calculate_kl(mod_noise, n_samples, 'Pink', 'sin-trans', sub_dir)

    out_wav = f"{base_dir}/pink_noise_sin-trans_audio.wav"

    scipy_wav.write(out_wav, fs, syn_bg.normalize(mod_noise, 1))

    # Pink Noise with inverted samples in second block -----------------------------------------------------
    pink_noise[int(n_samples * 0.4):int(n_samples * 0.6)] = (
            pink_noise[int(n_samples * 0.4):int(n_samples * 0.6)][::-1])

    calculate_kl(pink_noise, n_samples, 'Pink', 'inv', sub_dir)

    out_wav = f"{base_dir}/pink_noise_inv_audio.wav"

    scipy_wav.write(out_wav, fs, syn_bg.normalize(pink_noise, 1))


if __name__ == '__main__':
    main()


