
import numpy as np

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


def std(x1, x2, n_bins):

    min_value = np.min([np.min(x1), np.min(x2)])
    max_value = np.max([np.max(x1), np.max(x2)])
    bins = np.linspace(min_value, max_value, n_bins)

    x1_dist, edges = np.histogram(x1, bins=bins, density=True)
    x2_dist, edges = np.histogram(x2, bins=bins, density=True)

    x1_dist = np.where(x1_dist == 0, 1e-10, x1_dist)
    x2_dist = np.where(x2_dist == 0, 1e-10, x2_dist)

    result = np.zeros_like(x1_dist)
    for i in range(len(x1_dist)):
        result[i] = x1_dist[i] * np.log(x1_dist[i] / x2_dist[i])
    return np.sum(result)


def sliding_window(noise, num_samples, novelty, window_size):

    step = int(novelty * window_size)
    kl_divergences = []
    eq_sample = []

    for start in range(0, num_samples - window_size, step):

        window1 = noise[start:start + window_size]
        window2 = noise[start + step:start + step + window_size]

        kl_div = std(window1, window2, 100)
        kl_divergences.append(kl_div)
        eq_sample.append(start + step)

    return kl_divergences, eq_sample


def blocks(noise, num_samples, step, n_bins, num_blocks, syn_type='std'):

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
                aux.append(std(blocks1[-i], blocks2[i], n_bins))
            elif syn_type == 'weights':
                aux.append(alphas[-i] * std(blocks1[-i], blocks2[i], n_bins))
            elif syn_type == 'prog':
                block_sample += len(blocks1[i])
                kl_result.append(std(blocks1[i], blocks2[0], n_bins))
                eq_sample.append(block_sample)
            elif syn_type == 'std':
                aux.append(std(blocks1[i], blocks2[i], n_bins))
            else:
                raise ValueError("'blocks' function argument 'syn_type' value not listed:", syn_type)

        if syn_type == 'prog':
            continue
        else:
            kl_result.append(np.sum(aux))
            eq_sample.append(n + step)

    return kl_result, eq_sample
