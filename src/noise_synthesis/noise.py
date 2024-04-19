
import numpy as np
import scipy.signal as scipy


def generate_noise(frequencies: np.array, psd_db: np.array, n_samples: int, fs: float) -> np.array:
    """Generate background noise based on frequency and intensity information.

    Args:
        frequencies (np.array): Array of frequency values.
        psd_db (np.array): Array of Power Spectral Density (PSD) values in dB ref 1μPa @1m/√Hz.
        n_samples (int): Number of samples to generate.
        fs (float): Sampling frequency.

    Returns:
        np.array: Generated background noise in μPa.

    Raises:
        UnboundLocalError: Raised if frequencies and intensities have different lengths.

    """

    if len(frequencies) != len(psd_db):
        raise UnboundLocalError("for generate_noise frequencies and intensities must have the same length")

    # garantindo que as frequências inseridas contenham as frequência 0 e fs/2 (exigido pela scipy.firwin2)
    #   e estejam limitadas ao critério de nyquist
    index = np.argmax(frequencies > (fs / 2.0))
    if index > 0:
        if frequencies[index - 1] == (fs / 2):
            frequencies = frequencies[:index]
            psd_db = psd_db[:index]
        else:
            f = fs / 2
            i = psd_db[index - 1] + (psd_db[index] - psd_db[index - 1]) * (
                        f - frequencies[index - 1]) / (frequencies[index] - frequencies[index - 1])

            frequencies = np.append(frequencies[:index], f)
            psd_db = np.append(psd_db[:index], i)
    else:
        if frequencies[-1] != (fs / 2):
            f = fs / 2
            i = psd_db[-1] + (psd_db[-1] - psd_db[-2]) * (f - frequencies[-2]) / (
                        frequencies[-1] - frequencies[-2])

            frequencies = np.append(frequencies, f)
            psd_db = np.append(psd_db, i)

    if frequencies[0] != 0:
        frequencies = np.append(0, frequencies)
        psd_db = np.append(psd_db[0], psd_db)


    psd_linear = 10 ** ((psd_db) / 20) # passando de dB para linear

    # calculando a potência total de ruído branco com psd equivalente a maior psd do sinal
    #    P = ∫ ​psd df  para o ruído branco => P = psd * Δf = psd * fs/2
    max_power = np.max(psd_linear) * fs/2

    # calculando o desvio padrão para a potência calculada
    #   potencia é => P = E[x^2]
    #   desvio padrão é => std = √(var(x)) = √(E[(x-μ)^2])
    #       como média (μ) é zero
    #       std = √P
    std_dev = np.sqrt(max_power)

    order = 1025
    noise = np.random.normal(0, std_dev, n_samples + order)
    # gerando mais amostras que o desejado para eliminar a resposta transiente do filtro

    # normalizando frequências entre 0 e 1(fs/2)
    if np.max(frequencies) > 1:
        frequencies = frequencies / (fs / 2)
    # normalizando ganho para cada psd
    intensities_norm = psd_linear/np.max(psd_linear)

    if np.min(intensities_norm) == 1:
        return noise[order:]

    coeficient = scipy.firwin2(order, frequencies, np.sqrt(intensities_norm), antisymmetric=False)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin2.html
    # antisymmetric=False, order=odd
    # filtro tipo 1 para que as frequências fs/2 e 0 não tenham que ser 0

    out_noise = scipy.lfilter(coeficient, 1, noise)
    return out_noise[order:]

def psd(signal: np.array, fs: float, window_size: int = 1024, overlap: float = 0.5) \
        -> [np.array, np.array]:
    """Estimate the power spectrum density (PSD) for input signal.

    Args:
        signal (np.array): data in some unity (μPa).
        fs (float): Sampling frequency, default frequency factor to fs.
        window_size (int): number of samples in each segment.
        overlap (float): overlap between segments, between 0 and 1.

    Returns:
        np.array: Frequencies in Hz.
        np.array: Estimate PSD in dB ref 1μPa @1m/√Hz. (eq. dB ref 1μPa^2 @1m/Hz).
    """
    # https://ieeexplore.ieee.org/document/1161901
    # http://resource.npl.co.uk/acoustics/techguides/concepts/siunits.html

    if overlap == 1:
        raise UnboundLocalError("Overlap expected as a float between 0 and 1")

    freqs, intensity = scipy.welch(x=signal,
                                fs=fs,
                                window='hann',
                                nperseg=window_size,
                                noverlap=int(window_size * overlap),
                                scaling='density',
                                axis=-1,
                                average='mean')

    intensity = 20 * np.log10(intensity)

    # removendo DC
    return freqs[1:], intensity[1:]

def normalize(x, type=0):
    if type == 0: # normalize between 0 and 1
        return (x - np.min(x, axis=0))/(np.max(x, axis=0) - np.min(x, axis=0))
    if type == 1: # normalize -1 e 1, keeping 0 in place (librosa.util.normalize)
        return x/np.max(np.abs(x), axis=0)
    if type == 2:
        return x/np.linalg.norm(x, axis=0)
    raise UnboundLocalError("normalization {:d} not implemented".format(type))
