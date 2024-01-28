
import os
import numpy as np
import matplotlib.pyplot as plt

from noise_synthesis import kl_divergence as kl

"""
Test Program for the noise_synthesis library using white noise with greater intensity between samples 40000 and 60000.
"""


def calculate_kl(noise, num_samples, noise_color, noise_var, sub_dir,
                 transition_start=0, transition_end=0, step=10000):

    # Set up the directory for saving results
    base_dir = f"./result/{sub_dir}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    out_kl_png = f"{base_dir}/{noise_color}_noise_{noise_var}-kl.png"
    out_sw_png = f"{base_dir}/{noise_color}_noise_{noise_var}-sliding_window.png"

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

    ax[0][0].plot(noise)
    ax[0][0].set_title(noise_color + " Noise x Sample Number")
    ax[0][0].set_xlabel("Sample Number")
    ax[0][0].set_ylabel("Magnitude")
    ax[0][0].grid(True, which='both', ls='-')

    # KL (janela 1, janela 2)
    kl_result = list()
    n_bins = 100
    eq_samples = []

    for n in range(0, num_samples-step, step):
        # print(n+ (step * 2))
        kl_result.append(kl.std(noise[n:n + step], noise[n + step: n + (step * 2)], n_bins))
        eq_samples.append(n + step)
        # print(n, '----->', n+step, '------>', kl_result[-1])
        # print(n, ":", n + step, " --> ", n + step, ":", n + step * 2, " = ", kl_result[-1])

    ax[0][1].plot(eq_samples, kl_result)
    ax[0][1].set_ylabel('KL Divergence')
    ax[0][1].set_xlabel('Window Start Sample')
    ax[0][1].grid(True, which='both', ls='-')
    ax[0][1].set_title('KL Divergence')

    if transition_start != 0 and transition_end != 0:
        ax[0][1].axvline(x=transition_start, color='r', linestyle='dashed')
        ax[0][1].axvline(x=transition_end, color='r', linestyle='dashed')

    # KL with blocks
    num_blocks = 4
    # kl_blocks_error = kl.blocks(white_noise, num_samples, step, n_bins, num_blocks, 'normal')
    kl_blocks, eq_samples = kl.blocks(noise, num_samples, step, n_bins, num_blocks)

    ax[1][0].plot(eq_samples, kl_blocks)
    ax[1][0].set_ylabel('KL Divergence')
    ax[1][0].set_xlabel('Window Start Sample')
    ax[1][0].grid(True, which='both', ls='-')
    ax[1][0].set_title('KL Divergence Between Blocks')

    if transition_start != 0 and transition_end != 0:
        ax[1][0].axvline(x=transition_start, color='r', linestyle='dashed')
        ax[1][0].axvline(x=transition_end, color='r', linestyle='dashed')

    # KL progression by block
    kl_prog, eq_samples = kl.blocks(noise, num_samples, step, n_bins, num_blocks, 'prog')

    ax[1][1].plot(eq_samples, kl_prog)
    ax[1][1].set_ylabel('KL Divergence')
    ax[1][1].set_xlabel('Block Start Sample')
    ax[1][1].grid(True, which='both', ls='-')
    ax[1][1].set_title('KL Divergence Progression Between Blocks')

    if transition_start != 0 and transition_end != 0:
        ax[1][1].axvline(x=transition_start, color='r', linestyle='dashed')
        ax[1][1].axvline(x=transition_end, color='r', linestyle='dashed')

    # KL with inverted blocks
    kl_inv_blocks, eq_samples = kl.blocks(noise, num_samples, step, n_bins, num_blocks, 'inv')

    ax[2][0].plot(eq_samples, kl_inv_blocks)
    ax[2][0].set_ylabel('KL Divergence')
    ax[2][0].set_xlabel('Window Start Sample')
    ax[2][0].grid(True, which='both', ls='-')
    ax[2][0].set_title('KL Divergence Between Inverted Blocks')

    if transition_start != 0 and transition_end != 0:
        ax[2][0].axvline(x=transition_start, color='r', linestyle='dashed')
        ax[2][0].axvline(x=transition_end, color='r', linestyle='dashed')

    # KL accumulating inverted blocks (alpha is the weight  - increasing as the windows get closer)
    kl_inv_weight, eq_samples = kl.blocks(noise, num_samples, step, n_bins, num_blocks, 'weights')

    ax[2][1].plot(eq_samples, kl_inv_weight)
    ax[2][1].set_ylabel('KL Divergence')
    ax[2][1].set_xlabel('Window Start Sample')
    ax[2][1].grid(True, which='both', ls='-')
    ax[2][1].set_title('KL Divergence Accumulating Inverted Blocks')

    if transition_start != 0 and transition_end != 0:
        ax[2][1].axvline(x=transition_start, color='r', linestyle='dashed')
        ax[2][1].axvline(x=transition_end, color='r', linestyle='dashed')

    plt.tight_layout()
    plt.savefig(out_kl_png)
    plt.close()

    # KL sliding window (10-30-50% of the window)
    novelties = [0.1, 0.3, 0.5]
    window_size = 10000

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,5))

    ax[0].plot(noise)
    ax[0].set_title(noise_color + " Noise x Sample Number")
    ax[0].set_xlabel("Sample Number")
    ax[0].set_ylabel("Magnitude")

    for novelty in novelties:
        kl_divergences, eq_sample = kl.sliding_window(noise, num_samples, novelty, window_size)
        ax[1].plot(eq_sample, kl_divergences, label=f'Step {novelty * 100}%')

    ax[1].legend()
    ax[1].set_xlabel("Window Start Sample")
    ax[1].set_ylabel("KL Divergence")
    ax[1].set_title("KL Divergence with Sliding Window")
    ax[1].grid(True, which='both', ls='-')

    if transition_start != 0 and transition_end != 0:
        ax[1].axvline(x=transition_start, color='r', linestyle='dashed')
        ax[1].axvline(x=transition_end, color='r', linestyle='dashed')

    plt.tight_layout()
    plt.savefig(out_sw_png)
    plt.close()
