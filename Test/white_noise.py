import numpy as np
import matplotlib.pyplot as plt

from noise_synthesis import kl_divergence as kl

"""
Test Program for the noise_synthesis library using white noise with greater intensity between samples 40000 and 60000.
"""


def main():

    # Generating white noise
    num_samples = 100000
    white_noise = np.random.normal(0, 1, num_samples)
    white_noise[int(num_samples * 0.4):int(num_samples * 0.6)] = (
            white_noise[int(num_samples * 0.4):int(num_samples * 0.6)] * 2)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 5))

    ax[0][0].plot(white_noise)
    ax[0][0].set_title("White Noise x Sample Number")
    ax[0][0].set_xlabel("Sample Number")
    ax[0][0].set_ylabel("Magnitude")

    # KL (janela 1, janela 2)
    step = 1000
    kl_result = list()
    n_bins = 100

    for n in range(0, num_samples-step, step):
        kl_result.append(kl.std(white_noise[n:n + step], white_noise[n + step: n + (step * 2)], n_bins))
        # print(n, '----->', n+step, '------>', kl_result[-1])
        # print(n, ":", n + step, " --> ", n + step, ":", n + step * 2, " = ", kl_result[-1])

    ax[0][1].plot(kl_result)
    ax[0][1].set_ylabel('KL Divergence')
    ax[0][1].grid(True, which='both', ls='-')
    ax[0][1].set_title('KL Divergence')

    # KL with blocks
    num_blocks = 4
    # kl_blocks_error = kl.blocks(white_noise, num_samples, step, n_bins, num_blocks, 'normal')
    kl_blocks = kl.blocks(white_noise, num_samples, step, n_bins, num_blocks)

    ax[1][0].plot(kl_blocks)
    ax[1][0].set_ylabel('KL Divergence')
    ax[1][0].grid(True, which='both', ls='-')
    ax[1][0].set_title('KL Divergence Between Blocks')

    # KL progression by block
    kl_prog = kl.blocks(white_noise, num_samples, step, n_bins, num_blocks, 'prog')

    ax[1][1].plot(kl_prog)
    ax[1][1].set_ylabel('KL Divergence')
    ax[1][1].grid(True, which='both', ls='-')
    ax[1][1].set_title('KL Divergence Progression Between Blocks')

    # KL with inverted blocks
    kl_inv_blocks = kl.blocks(white_noise, num_samples, step, n_bins, num_blocks, 'inv')

    ax[2][0].plot(kl_inv_blocks)
    ax[2][0].set_ylabel('KL Divergence')
    ax[2][0].grid(True, which='both', ls='-')
    ax[2][0].set_title('KL Divergence Between Inverted Blocks')

    # KL accumulating inverted blocks (alpha is the weight  - increasing as the windows get closer)
    kl_inv_weight = kl.blocks(white_noise, num_samples, step, n_bins, num_blocks, 'weights')

    ax[2][1].plot(kl_inv_weight)
    ax[2][1].set_ylabel('KL Divergence')
    ax[2][1].grid(True, which='both', ls='-')
    ax[2][1].set_title('KL Divergence Accumulating Inverted Blocks')

    plt.tight_layout()
    plt.show()

    # KL sliding window (10-30-50% of the window)
    novelties = [0.1, 0.3, 0.5]
    window_size = 1024

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,5))

    ax[0].plot(white_noise)
    ax[0].set_title("White Noise x Sample Number")
    ax[0].set_xlabel("Sample Number")
    ax[0].set_ylabel("Magnitude")

    for novelty in novelties:
        kl_divergences, eq_sample = kl.sliding_window(white_noise, num_samples, novelty, window_size)
        ax[1].plot(eq_sample, kl_divergences, label=f'Step {novelty * 100}%')

    ax[1].legend()
    ax[1].set_xlabel("Sample Number")
    ax[1].set_ylabel("KL Divergence")
    ax[1].set_title("KL Divergence with Sliding Window")
    ax[1].grid(True, which='both', ls='-')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
