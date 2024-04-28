import enum
import typing
import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.special
import scipy.signal as sgn

import statsmodels.tsa.stattools as stats
import arch.unitroot as arch_root

import noise_synthesis.noise as syn_noise

class DataEstimator(enum.Enum):
    PDF = 0
    FFT = 1

    def __str__(self) -> str:
        return '\\ac{' + str(self.name).rsplit(".", maxsplit=1)[-1].lower() + '}'

    def _estimate_pdf(window1, window2, n_bins) -> typing.Tuple[np.array, np.array, np.array]:

        min_value = np.min([np.min(window1), np.min(window2)])
        max_value = np.max([np.max(window1), np.max(window2)])
        bins = np.linspace(min_value, max_value, n_bins)

        x1_dist, edges = np.histogram(window1, bins=bins, density=True)
        x2_dist, _ = np.histogram(window2, bins=bins, density=True)

        return x1_dist, x2_dist, edges

    def apply(self, window1, window2, n_points) -> typing.Tuple[np.array, np.array, np.array]:
        if self == DataEstimator.PDF:
            return DataEstimator._estimate_pdf(window1, window2, n_bins=n_points)

        if self == DataEstimator.FFT:
            frequencies, power1 = syn_noise.psd(signal=window1, fs=1, window_size=n_points*2)
            _, power2 = syn_noise.psd(signal=window2, fs=1, window_size=n_points*2)

            # return power1/np.sum(power1), power2/np.sum(power2), frequencies
            return power1, power2, frequencies

        raise NotImplementedError(f"apply {str(self)} not implemented")
    

    def plot(self, filename, window1, window2, n_points, label1 = "window 1", label2 = "window 2") -> None:
        y1, y2, x = self.apply(window1, window2, n_points)

        if self == DataEstimator.PDF:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            ax[0].bar(x[1:], y1, align="edge", width=np.diff(x))
            ax[0].set_title(f"PDF {label1}")
            ax[0].set_xlabel("Noise Magnitude")
            ax[0].set_ylabel("Probability")
            ax[0].grid(True, which='both', ls='-')

            ax[1].bar(x[1:], y2, align="edge", width=np.diff(x))
            ax[1].set_title(f"PDF {label2}")
            ax[1].set_xlabel("Noise Magnitude")
            ax[1].set_ylabel("Probability")
            ax[1].grid(True, which='both', ls='-')

        elif self == DataEstimator.FFT:
            plt.plot(x, y1, label=label1, color='blue')
            plt.plot(x, y2, label=label2, color='red')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (dB/Hz)')
            plt.legend()
            plt.grid()

        plt.savefig(filename)
        plt.close()

class Metrics():

    class Type(enum.Enum):
        KL_DIVERGENCE = 0
        SYMMETRIC_KL_DIVERGENCE = 1
        WASSERSTEIN = 2
        JENSEN_SHANNON = 3

        def __str__(self) -> str:
            labels = ['\\ac{kl}', '\\ac{kl} simétrica', '\\ac{jsd}', '\\ac{wasserstein}']
            return labels[self.value]

        def apply(self, pdf1, pdf2) -> float:

            if self == Metrics.Type.KL_DIVERGENCE:
                pdf1 = np.where(pdf1 < 1e-10, 1e-10, pdf1)
                pdf2 = np.where(pdf2 < 1e-10, 1e-10, pdf2)
                return np.sum(scipy.special.kl_div(pdf1, pdf2))

            if self == Metrics.Type.SYMMETRIC_KL_DIVERGENCE:
                return np.max([Metrics.Type.KL_DIVERGENCE.apply(pdf1, pdf2),
                               Metrics.Type.KL_DIVERGENCE.apply(pdf2, pdf1)])
        
            if self == Metrics.Type.WASSERSTEIN:
                return scipy.stats.wasserstein_distance(pdf1, pdf2)
            
            if self == Metrics.Type.JENSEN_SHANNON:
                return scipy.spatial.distance.jensenshannon(pdf1, pdf2)

            raise NotImplementedError(f"apply {str(self)} not implemented")

    def __init__(self, type: Type, estimator: DataEstimator, n_points: int = 1024) -> None:
        self.type = type
        self.estimator = estimator
        self.n_points = n_points

    def __str__(self) -> str:
        return f'{self.estimator} {self.type}'

    def calc_block(self, window1: np.array, window2: np.array) -> float:
        pdf1, pdf2, _ = self.estimator.apply(window1, window2, self.n_points)
        return self.type.apply(pdf1, pdf2)

    def calc_data(self, data: np.array, window_size: int, overlap: float = 0) -> typing.Tuple[np.array, np.array]:

        if isinstance(overlap, int):
            step = window_size - overlap
        else:
            step = int((1-overlap) * window_size)
        metrics = []
        eq_sample = []

        for start_sample in range(0, len(data) - window_size, step):

            window1 = data[start_sample:start_sample + window_size//2]
            window2 = data[start_sample + window_size//2:start_sample + window_size]

            metrics.append(self.calc_block(window1, window2))
            eq_sample.append(start_sample + window_size//2)

        return metrics, eq_sample

class StatisticTest(Metrics):

    class Type(enum.Enum):
        ADF = 0
        KPSS = 1
        PhillipsPerron = 2

        def __str__(self) -> str:
            labels = ['\\ac{adf}', '\\ac{kpss}', '\\ac{pp}']
            return labels[self.value]

    def __init__(self, type: Type) -> None:
        super().__init__(type, None, None)
        self.type = type

    def calc_block(self, window1: np.array, window2: np.array) -> float:
        if self.type == StatisticTest.Type.ADF:
            result = arch_root.ADF(np.concatenate((window1, window2)))
        elif self.type == StatisticTest.Type.KPSS:
            result = arch_root.KPSS(np.concatenate((window1, window2)))
        elif self.type == StatisticTest.Type.PhillipsPerron:
            result = arch_root.PhillipsPerron(np.concatenate((window1, window2)))
        else:
            raise NotImplementedError(f"calck block not implemented for {self.type}")

        # print('\tPhillipsPerron Statistic: %f' % result.stat)
        # print('\tp-value: %f' % result.pvalue)
        # print('\tCritical Values:')
        # for key, value in result.critical_values.items():
        #     print('\t\t%s: %.3f' % (key, value))
        return result.stat

    def __str__(self) -> str:
        return str(self.type)
