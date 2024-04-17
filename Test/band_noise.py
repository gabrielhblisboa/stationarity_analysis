
import numpy as np
import os
import matplotlib.pyplot as plt

import noise_synthesis.background_noise as syn_bg
import scipy.io.wavfile as scipy_wav
from kl_test import calculate_kl
import noise_mod

import scipy.fftpack


def test_noise(band_noise, n_samples, fs, noise_type):

    pdf_type = 'std'
    est_type = 'wasserstein'

    sub_dir = noise_type

    # Set up the directory for saving results
    base_dir = f"./result/{sub_dir}/{est_type}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Band Noise ------------------------------------------------------------------------------------------

    calculate_kl(band_noise, n_samples, noise_type, 'std', noise_type, pdf_type=pdf_type, est_type=est_type)

    out_wav = f"{base_dir}/{sub_dir}_std_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(band_noise, 1))

    # plt.figure(figsize=(12, 5))
    # plt.plot(band_noise, color='band')
    # plt.title("Teste Ruido Rosa")
    # plt.show()


    # Band Noise with amplitude variation (abrupt transition) -------------------------------------------------------
    transition_start = int(n_samples * 0.25)
    transition_end = int(n_samples * 0.75)

    mod_noise = band_noise

    mod_noise[transition_start:transition_end] = (
            mod_noise[transition_start:transition_end] * 3)

    calculate_kl(mod_noise, n_samples, noise_type, 'amp-var', sub_dir, pdf_type=pdf_type,
                 transition_start=transition_start, transition_end=transition_end, est_type=est_type)

    out_wav = f"{base_dir}/{sub_dir}_amp-var_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(mod_noise, 1))

    # Band Noise with amplitude variation (linear transition) -------------------------------------------------------
    mod_noise = noise_mod.transition(band_noise, n_samples, 'linear')
    calculate_kl(mod_noise, n_samples, noise_type, 'linear-trans', sub_dir, pdf_type=pdf_type, est_type=est_type)

    out_wav = f"{base_dir}/{sub_dir}_linear-trans_audio.wav"

    # Save the generated noise as a WAV file
    scipy_wav.write(out_wav, fs, syn_bg.normalize(mod_noise, 1))

    # # Band noise with amplitude variation (sin transition) ---------------------------------------------------------
    mod_noise = noise_mod.transition(band_noise, n_samples, 'sin')
    calculate_kl(mod_noise, n_samples, noise_type, 'sin-trans', sub_dir, pdf_type=pdf_type, est_type=est_type)

    out_wav = f"{base_dir}/{sub_dir}_sin-trans_audio.wav"

    scipy_wav.write(out_wav, fs, syn_bg.normalize(mod_noise, 1))

    # Band Noise with inverted samples in second block -----------------------------------------------------
    band_noise[int(n_samples * 0.4):int(n_samples * 0.6)] = (
            band_noise[int(n_samples * 0.4):int(n_samples * 0.6)][::-1])

    calculate_kl(band_noise, n_samples, noise_type, 'inv', sub_dir, pdf_type=pdf_type, est_type=est_type)

    out_wav = f"{base_dir}/{sub_dir}_inv_audio.wav"

    scipy_wav.write(out_wav, fs, syn_bg.normalize(band_noise, 1))


def test_generate_noise():
    fs = 40000  # Frequência de amostragem
    n_samples = 40000  # Número de amostras, 1 segundo de áudio

    min_freq = 20  # 20 Hz
    max_freq = 20000  # 20 kHz

    # Gerando 4 pontos que dividem a faixa de frequência em 4 partes iguais na escala logarítmica
    # frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), num=5)

    # Número total de frequências desejadas na lista
    total_frequencies = 1000

    # Gerando uma lista de frequências na escala logarítmica
    frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), total_frequencies)

    # Gerando os pontos de divisão
    divisions = np.logspace(np.log10(min_freq), np.log10(max_freq), num=5)

    # Separando as frequências em 4 listas
    list1 = frequencies[frequencies <= divisions[1]]
    list2 = frequencies[(frequencies > divisions[1]) & (frequencies <= divisions[2])]
    list3 = frequencies[(frequencies > divisions[2]) & (frequencies <= divisions[3])]
    list4 = frequencies[frequencies > divisions[3]]

    intensities_low = list1
    intensities_low_mid = list2
    intensities_mid_high = list3
    intensities_high = list4

    # Gera um array de frequências logaritmicamente espaçadas de 20 Hz até a frequência de Nyquist
    # frequencies = np.logspace(np.log10(20), np.log10(fs / 2), num=100)

    # # Inicializa o array de intensidades com -100 dB para todas as frequências
    # intensities_low = np.full_like(frequencies, -100.0)  # -100 dB representa uma atenuação muito alta
    # intensities_low_mid = np.full_like(frequencies, -100.0)
    # intensities_mid_high = np.full_like(frequencies, -100.0)
    # intensities_high = np.full_like(frequencies, -100.0)
    #
    # # Define a intensidade para 0 dB (ou 1 em escala linear) para frequências entre 20 Hz e 5 kHz
    # intensities_low[(frequencies >= 20) & (frequencies <= 5015)] = 0  # 0 dB
    # intensities_low_mid[(frequencies >= 5015) & (frequencies <= 10010)] = 0  # 0 dB
    # intensities_mid_high[(frequencies >= 10010) & (frequencies <= 15005)] = 0  # 0 dB
    # intensities_high[(frequencies >= 15005) & (frequencies <= 20000)] = 0  # 0 dB

    # Gerando ruído
    noise_low = syn_bg.generate_noise(frequencies, intensities_low, n_samples, fs)

    # Aplicando FFT ao ruído gerado
    fft_freq_low, fft_result_low = syn_bg.estimate_spectrum(noise_low, int(n_samples / 100), overlap=0.5, fs=fs)

    # Low-Mid Freq
    noise_low_mid = syn_bg.generate_noise(frequencies, intensities_low_mid, n_samples, fs)
    fft_freq_low_mid, fft_result_low_mid = (
        syn_bg.estimate_spectrum(noise_low_mid, int(n_samples / 100), overlap=0.5, fs=fs))

    # Mid-High Freq
    noise_mid_high = syn_bg.generate_noise(frequencies, intensities_mid_high, n_samples, fs)
    fft_freq_mid_high, fft_result_mid_high = (
        syn_bg.estimate_spectrum(noise_mid_high, int(n_samples / 100), overlap=0.5, fs=fs))

    # High
    noise_high = syn_bg.generate_noise(frequencies, intensities_high, n_samples, fs)
    fft_freq_high, fft_result_high = syn_bg.estimate_spectrum(noise_high, int(n_samples / 100), overlap=0.5, fs=fs)

    print(noise_low)
    plt.plot(noise_low)
    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,5))

    ax[0][0].plot(fft_freq_low, fft_result_low)
    ax[0][0].set_title('Low Frequency Band Noise')
    ax[0][0].set_xlabel('Frequency (Hz)')
    ax[0][0].set_ylabel('Amplitude (dB)')
    ax[0][0].grid(True, which='both', ls='-')

    ax[0][1].plot(fft_freq_low_mid, fft_result_low_mid)
    ax[0][1].set_title('Low to Mid Frequency Band Noise')
    ax[0][1].set_xlabel('Frequency (Hz)')
    ax[0][1].set_ylabel('Amplitude (dB)')
    ax[0][1].grid(True, which='both', ls='-')

    ax[1][0].plot(fft_freq_mid_high, fft_result_mid_high)
    ax[1][0].set_title('Mid to High Frequency Band Noise')
    ax[1][0].set_xlabel('Frequency (Hz)')
    ax[1][0].set_ylabel('Amplitude (dB)')
    ax[1][0].grid(True, which='both', ls='-')

    ax[1][1].plot(fft_freq_high, fft_result_high)
    ax[1][1].set_title('High Frequency Band Noise')
    ax[1][1].set_xlabel('Frequency (Hz)')
    ax[1][1].set_ylabel('Amplitude (dB)')
    ax[1][1].grid(True, which='both', ls='-')

    plt.tight_layout()
    plt.show()

# test_generate_noise()

def main():

    fs = 48000  # Frequência de amostragem
    n_samples = 5 * fs  # Número de amostras, 1 segundo de áudio

    min_freq = 20  # 20 Hz
    max_freq = 20000  # 20 kHz

    # Gerando 4 pontos que dividem a faixa de frequência em 4 partes iguais na escala logarítmica
    # frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), num=5)

    # Número total de frequências desejadas na lista
    total_frequencies = 1000

    # Gerando uma lista de frequências na escala logarítmica
    frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), total_frequencies)

    # Gerando os pontos de divisão
    divisions = np.logspace(np.log10(min_freq), np.log10(max_freq), num=5)

    # Inicializa o array de intensidades com -100 dB para todas as frequências
    intensities_low = np.full_like(frequencies, -100.0)  # -100 dB representa uma atenuação muito alta
    intensities_low_mid = np.full_like(frequencies, -100.0)
    intensities_mid_high = np.full_like(frequencies, -100.0)
    intensities_high = np.full_like(frequencies, -100.0)

    intensities_low[frequencies <= divisions[1]] = 0  # 0 dB
    intensities_low_mid[(frequencies > divisions[1]) & (frequencies <= divisions[2])] = 0  # 0 dB
    intensities_mid_high[(frequencies > divisions[2]) & (frequencies <= divisions[3])] = 0  # 0 dB
    intensities_high[frequencies > divisions[3]] = 0  # 0 dB

    # # Define a intensidade para 0 dB (ou 1 em escala linear) para frequências entre 20 Hz e 5 kHz
    # intensities_low[(frequencies >= 20) & (frequencies <= 5015)] = 0  # 0 dB
    # intensities_low_mid[(frequencies >= 5015) & (frequencies <= 10010)] = 0  # 0 dB
    # intensities_mid_high[(frequencies >= 10010) & (frequencies <= 15005)] = 0  # 0 dB
    # intensities_high[(frequencies >= 15005) & (frequencies <= 20000)] = 0  # 0 dB

    # Gerando ruído
    noise_low = syn_bg.generate_noise(frequencies, intensities_low, n_samples, fs)

    # Low-Mid Freq
    noise_low_mid = syn_bg.generate_noise(frequencies, intensities_low_mid, n_samples, fs)

    # Mid-High Freq
    noise_mid_high = syn_bg.generate_noise(frequencies, intensities_mid_high, n_samples, fs)

    # High
    noise_high = syn_bg.generate_noise(frequencies, intensities_high, n_samples, fs)

    test_noise(noise_low, n_samples, fs, 'low-freq')
    test_noise(noise_low_mid, n_samples, fs, 'low-mid-freq')
    test_noise(noise_mid_high, n_samples, fs, 'mid-high-freq')
    test_noise(noise_high, n_samples, fs, 'high-freq')

# test_generate_noise()

if __name__ == '__main__':
    main()
