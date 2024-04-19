
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.signal as sgn

"""
The three following functions represent the three main methods to calculate the KL Divergency of a certain noise.

std: standard KL Divergency between two windows of the signal.
sliding_window: uses a sliding window with steps equal to a certain percentage of the window size (novelty).
blocks: the windows are divided into a given number of blocks. There are four different ways you can calculate the KL
        Divergency between blocks here.
                --> 'std': standard KL by block.
                --> 'prog': progression by blocks - KL(1.k, 2.1) - the second window is locked at the first block.
                --> 'inv': inverted blocks - KL[1.(N-k), 2.k] - runs the first window blocks backwards.
                --> 'weights': same as the inverted blocks, but with added weights to the KL Divergencies, with bigger
                    weights assigned to the KL's in which the windows are closer.
                --> NOTE: the argument syn_type must be one of the following strings: 'std', 'prog', 'inv' or 'weights'.
                          The default value for syn_type is 'std'.
"""


def estimate_pdf(window1, window2, n_bins, pdf_type):

    if pdf_type == 'fft':

        fft_x1 = np.fft.fft(window1)
        fft_x2 = np.fft.fft(window2)

        x1_dist = np.abs(fft_x1)
        x2_dist = np.abs(fft_x2)

        x1_dist = np.where(x1_dist == 0, 1e-10, x1_dist)
        x2_dist = np.where(x2_dist == 0, 1e-10, x2_dist)

        edges = 0

        # plt.plot(x1_dist)
        # plt.plot(x2_dist)
        # plt.xscale('log')
        # plt.show()

    else:
        min_value = np.min([np.min(window1), np.min(window2)])
        max_value = np.max([np.max(window1), np.max(window2)])
        bins = np.linspace(min_value, max_value, n_bins)

        x1_dist, edges = np.histogram(window1, bins=bins, density=True)
        x2_dist, edges = np.histogram(window2, bins=bins, density=True)

        x1_dist = np.where(x1_dist == 0, 1e-10, x1_dist)
        x2_dist = np.where(x2_dist == 0, 1e-10, x2_dist)

    return x1_dist, x2_dist, edges


def kl_std(x1, x2, n_bins, pdf_type):

    x1_dist, x2_dist, edges = estimate_pdf(x1, x2, n_bins, pdf_type)

    # print(len(x1_dist))
    # print(len(x2_dist))

    result = np.zeros_like(x1_dist)
    for i in range(len(x2_dist)):
        result[i] = x1_dist[i] * np.log(x1_dist[i] / x2_dist[i])
    return np.sum(result)

    # result = scipy.special.kl_div(x1_dist, x2_dist)

    # return result


def kl_sliding_window(noise, num_samples, novelty, window_size, pdf_type):

    step = int(novelty * window_size)
    kl_divergences = []
    eq_sample = []

    for start in range(0, num_samples - window_size, step):

        window1 = noise[start:start + window_size]
        window2 = noise[start + step:start + step + window_size]

        kl_div = kl_std(window1, window2, 100, pdf_type)
        kl_divergences.append(kl_div)
        eq_sample.append(start + step)

    return kl_divergences, eq_sample


def kl_blocks(noise, num_samples, step, n_bins, num_blocks, pdf_type, syn_type='std'):

    kl_result = []

    if not type(syn_type) is str:
        raise TypeError("'syn_type' must be a string")

    alphas = np.linspace(0.5, 1, num_blocks)

    eq_sample = []

    for n in range(0, num_samples - step, step):

        window1 = noise[n:n + step]
        window2 = noise[n + step:n + step * 2]

        blocks1 = np.array_split(window1, num_blocks)
        blocks2 = np.array_split(window2, num_blocks)

        aux = []

        block_sample = n + step

        for i in range(num_blocks):
            if syn_type == 'inv':
                aux.append(kl_std(blocks1[-i], blocks2[i], n_bins, pdf_type))
            elif syn_type == 'weights':
                aux.append(alphas[-i] * kl_std(blocks1[-i], blocks2[i], n_bins, pdf_type))
            elif syn_type == 'prog':
                block_sample += len(blocks1[i])
                kl_result.append(kl_std(blocks1[i], blocks2[0], n_bins, pdf_type))
                eq_sample.append(block_sample)
            elif syn_type == 'std':
                aux.append(kl_std(blocks1[i], blocks2[i], n_bins, pdf_type))
            else:
                raise ValueError("'blocks' function argument 'syn_type' value not listed:", syn_type)

        if syn_type == 'prog':
            continue
        else:
            kl_result.append(np.sum(aux))
            eq_sample.append(n + step)

    return kl_result, eq_sample


def normalize(x, type=0):
    if type == 0: # normalize between 0 and 1
        return (x - np.min(x, axis=0))/(np.max(x, axis=0) - np.min(x, axis=0))
    if type == 1: # normalize -1 e 1, keeping 0 in place (librosa.util.normalize)
        return x/np.max(np.abs(x), axis=0)
    if type == 2:
        return x/np.linalg.norm(x, axis=0)
    raise UnboundLocalError("normalization {:d} not implemented".format(type))
