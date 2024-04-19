
import numpy as np


def transition(noise, num_samples, trans_type, noise_type=None):

    p1 = noise[:int(num_samples * 0.25)]
    p2 = p1 * 3

    # pt12 = np.zeros(int(len(p1) / 2))
    # pt21 = np.zeros(int(len(p1) / 2))

    pt12 = np.zeros(len(noise[int(num_samples * 0.25):int(num_samples * 0.4)]))
    pt21 = np.zeros(len(noise[int(num_samples * 0.25):int(num_samples * 0.4)]))

    # first_block_end = 0.2
    # second_block_start = 0.35
    # second_block_end = 0.65
    # third_block_start = 0.8
    #
    # p1 = noise[:int(num_samples * first_block_end)]
    # # p2 = p1 * 3
    #
    # pt12 = np.zeros(len(noise[int(num_samples * first_block_end):int(num_samples * second_block_start)]))
    #
    # p2 = noise[int(num_samples * second_block_start):int(num_samples * second_block_end)] * 3
    #
    # pt21 = np.zeros(len(noise[int(num_samples * second_block_end):int(num_samples * third_block_start)]))
    #
    # p3 = noise[int(num_samples * third_block_start):]

    # for n in range(0, int(len(pt12))):
    #     if trans_type == 'linear':
    #         pt12[n] = p1[n] + (n / len(p1)) * (p2[n] - p1[n])
    #         pt21[-n] = p1[-n] - ((-1) * n / len(p1)) * (p2[-n] - p1[-n])
    #     elif trans_type == 'sin':
    #         pt12[n] = p1[n] + np.sin(np.pi * n/len(p1)) * (p2[n] - p1[n])
    #         pt21[-n] = p1[-n] - np.sin(np.pi * (-1) * n / len(p1)) * (p2[-n] - p1[-n])
    #     else:
    #         raise ValueError

    n_samples_block = int(len(pt12))

    for n in range(0, n_samples_block):
        if trans_type == 'linear':
            pt12[n] = p1[n] + (n / n_samples_block) * (p2[n] - p1[n])
            pt21[-n] = p1[-n] + (n / n_samples_block) * (p2[-n] - p1[-n])
        elif trans_type == 'sin':
            pt12[n] = p1[n] + np.sin((np.pi * n / n_samples_block) / 2) * (p2[n] - p1[n])
            pt21[-n] = p1[-n] + np.sin((np.pi * n / n_samples_block) / 2) * (p2[-n] - p1[-n])
        else:
            raise ValueError

    if noise_type == 'sin':
        pt12 = (-1) * pt12
        pt21 = (-1) * pt21

    aux1 = np.append(p1, pt12)
    aux2 = np.append(aux1, p2)
    aux3 = np.append(aux2, pt21)
    mod_noise = np.append(aux3, p1)

    return mod_noise


def modify_sine_wave(frequency, n_samples):
    """
    Modifies a sine wave signal to have an amplitude that increases to double its original amplitude,
    maintains the maximum amplitude for a portion of the signal, and then decreases back to the original amplitude.

    Args:
    signal (numpy array): Original sine wave signal.
    frequency (float): Frequency of the original sine wave signal.
    n_samples (int): Number of samples for the modified signal.

    Returns:
    numpy array: Modified sine wave signal.
    """
    def modified_amplitude(i, n, base_amplitude, max_amplitude):
        if i < n * 0.3:
            return base_amplitude + (max_amplitude - base_amplitude) * (i / (n * 0.3))
        elif i < n * 0.7:
            return max_amplitude
        else:
            return max_amplitude - (max_amplitude - base_amplitude) * ((i - n * 0.7) / (n * 0.3))

    t = np.linspace(0, 2 * np.pi, n_samples)
    modified_signal = np.array([np.sin(frequency * t_i) * modified_amplitude(i, n_samples, 1, 2) for i, t_i in enumerate(t)])

    return modified_signal

