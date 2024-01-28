
import os
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import noise_synthesis.background_noise as syn_bg
from kl_test import calculate_kl
import noise_mod


def main():

    sub_dir = 'sine'

    # Set up the directory for saving results
    base_dir = f"./result/{sub_dir}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

        # Parameters for the sine wave
    fs = 50000  # Sample rate in Hz
    duration = 2  # Duration in seconds
    frequency = 25  # Frequency of the sine wave in Hz

    samples = np.arange(fs * duration)
    n_samples = len(samples)
    step = 10000

    sin = np.sin(2 * np.pi * frequency * samples / fs)

    # Sine Wave with amplitude variation (abrupt) -------------------------------------------------------------------

    mod_sin = sin

    transition_start = int(n_samples * 0.25)
    transition_end = int(n_samples * 0.75)

    mod_sin[transition_start:transition_end] = (
            mod_sin[transition_start:transition_end] * 3)

    calculate_kl(mod_sin, n_samples, 'Sine', 'amp-var', sub_dir,
                 transition_start=transition_start, transition_end=transition_end, step=step)

    out_wav = f"{base_dir}/sine_amp-var_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(sin, 1))

    # Sine Wave with amplitude variation (linear) -------------------------------------------------------------------

    mod_sin = noise_mod.transition(sin, n_samples, 'linear', noise_type='sin')

    # plt.figure(figsize=(10, 4))
    # plt.plot(samples, mod_sin)
    # plt.title('Variable Amplitude Sine Wave')
    # plt.xlabel('Sample Number')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()

    calculate_kl(mod_sin, n_samples, 'Sine', 'linear-trans', sub_dir, step=step)

    out_wav = f"{base_dir}/sine_linear-trans_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(sin, 1))

    # Sine Wave with amplitude variation (sin) -------------------------------------------------------------------

    mod_sin = noise_mod.transition(sin, n_samples, 'sin', noise_type='sin')

    # plt.figure(figsize=(10, 4))
    # plt.plot(samples, mod_sin)
    # plt.title('Variable Amplitude Sine Wave')
    # plt.xlabel('Sample Number')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()

    calculate_kl(mod_sin, n_samples, 'Sine', 'sin-trans', sub_dir, step=step)

    out_wav = f"{base_dir}/sine_sin-trans_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(sin, 1))

    # Sine Wave with frequency variation ---------------------------------------------------------------------------
    frequency_middle = 50  # Frequency for the middle section in Hz

    middle_start = int(0.35 * len(samples))
    middle_end = int(0.65 * len(samples))

    # sine_wave = np.sin(2 * np.pi * frequency * samples / fs)
    #
    # plt.figure(figsize=(10, 4))
    # plt.plot(samples, sine_wave)
    # plt.title('Sine Wave per Sample Number')
    # plt.xlabel('Sample Number')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()

    # Frequency array - initially set to starting frequency
    variable_frequency = np.full(samples.shape, frequency)

    variable_frequency[middle_start:middle_end] = frequency_middle
    variable_frequency[middle_end] = frequency

    # Generate the variable frequency sine wave
    var_sin = np.sin(2 * np.pi * np.cumsum(variable_frequency) / fs)

    # # Plotting the variable frequency sine wave
    # plt.figure(figsize=(10, 4))
    # plt.plot(samples, var_sin)
    # plt.title('Variable Frequency Sine Wave')
    # plt.xlabel('Sample Number')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()

    step = 10000

    calculate_kl(var_sin, n_samples, 'Sine', 'var_freq', sub_dir, step=step)

    out_wav = f"{base_dir}/sine_var_freq_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(var_sin, 1))


if __name__ == '__main__':
    main()


