
import numpy as np
import scipy.signal as scipy


def generate_noise(frequencies: np.array, intensities: np.array, n_samples: int, fs: float) -> np.array:
    """Generate background noise based on frequency and intensity information.

    Args:
        frequencies (np.array): Array of frequency values.
        intensities (np.array): Array of intensity values in dB/√Hz.
        n_samples (int): Number of samples to generate.
        fs (float): Sampling frequency.

    Returns:
        np.array: Generated background noise in μPa.

    Raises:
        UnboundLocalError: Raised if frequencies and intensities have different lengths.

    """

    if len(frequencies) != len(intensities):
        raise UnboundLocalError("for generate_noise frequencies and intensities must have the same length")

    # garantindo que as frequências inseridas contenham as frequência 0 e fs/2 (exigido pela scipy.firwin2)
    #   e estejam limitadas ao critério de nyquist
    index = np.argmax(frequencies > (fs / 2.0))
    if index > 0:
        if frequencies[index - 1] == (fs / 2):
            frequencies = frequencies[:index]
            intensities = intensities[:index]
        else:
            f = fs / 2
            i = intensities[index - 1] + (intensities[index] - intensities[index - 1]) * (
                        f - frequencies[index - 1]) / (frequencies[index] - frequencies[index - 1])

            frequencies = np.append(frequencies[:index], f)
            intensities = np.append(intensities[:index], i)
    else:
        if frequencies[-1] != (fs / 2):
            f = fs / 2
            i = intensities[-1] + (intensities[-1] - intensities[-2]) * (f - frequencies[-2]) / (
                        frequencies[-1] - frequencies[-2])

            frequencies = np.append(frequencies, f)
            intensities = np.append(intensities, i)

    if frequencies[0] != 0:
        frequencies = np.append(0, frequencies)
        intensities = np.append(intensities[0], intensities)

    # normalizando frequências entre 0 e 1
    if np.max(frequencies) > 1:
        frequencies = frequencies / (fs / 2)

    order = 2049
    noise = np.random.normal(0, 1.13, n_samples + order)
    # 1.13 ajustado manualmente com base na aplicação de teste background_noise.py para compensar um offset
    # gerando mais amostras que o desejado para eliminar a resposta transiente do filtro

    intensities = 10 ** ((intensities) / 20)

    coeficient = scipy.firwin2(order, frequencies, intensities, antisymmetric=False)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin2.html
    # antisymmetric=False, order=odd
    # filtro tipo 1 para que as frequências fs/2 e 0 não tenham que ser 0

    filter_gain = np.sum(coeficient)
    zeros, poles, gain = scipy.tf2zpk(coeficient, 1)

    # Imprimir resultados
    print("Ganho do filtro:", filter_gain)
    print("Zeros do filtro:", zeros)
    print("Polos do filtro:", poles)
    print("Ganho da função de transferência:", gain)

    out_noise = scipy.lfilter(coeficient, 1, noise)
    return out_noise[order:]


def estimate_spectrum(signal: np.array, window_size: int = 1024, overlap: float = 0.5, fs: float = 48000) \
        -> [np.array, np.array]:
    """Estimate the medium spectrum based on data.

    Args:
        signal (np.array): data in 1μPa.
        window_size (int): fft window size.
        overlap (float): overlap fft window, between 0 and 1.
        fs (float): Sampling frequency.

    Returns:
        np.array: Frequencies in Hz.
        np.array: Estimate spectrum in dB ref 1μPa @1m/Hz.
    """

    if overlap == 1:
        raise UnboundLocalError("Overlap cannot be 1")

    window_size = int(window_size)
    n_samples = signal.size
    novelty_samples = int(window_size * (1-overlap))

    fft_result = np.zeros(window_size//2)
    fft_freq = np.fft.fftfreq(window_size, 1/fs)[:window_size//2]

    i = 0
    n_means = 0
    while i + novelty_samples + window_size <= n_samples:
        fft_result = fft_result + np.abs(np.fft.fft(
                    signal[i:i+window_size], norm='ortho')
                [:window_size//2])
        i += novelty_samples
        n_means += 1

    fft_result = fft_result/n_means
    fft_result = 20 * np.log10(fft_result)

    return fft_freq, fft_result


def normalize(x, type=0):
    if type == 0: # normalize between 0 and 1
        return (x - np.min(x, axis=0))/(np.max(x, axis=0) - np.min(x, axis=0))
    if type == 1: # normalize -1 e 1, keeping 0 in place (librosa.util.normalize)
        return x/np.max(np.abs(x), axis=0)
    if type == 2:
        return x/np.linalg.norm(x, axis=0)
    raise UnboundLocalError("normalization {:d} not implemented".format(type))


