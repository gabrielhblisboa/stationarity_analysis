import numpy as np
import random

import noise_synthesis.signals as syn_signals

fs = 52734
n_samples = 6 * 3 * fs # qtd de segundos/bloco * numero de blocos * numero de amostras/s 
psd_db = -np.log10(fs/2)*20  # psd de um ruido branco de variancia 1
end_psd_db = psd_db + 6


window_size = 16*1024
overlap = 0.75
n_points = 64
memory_size = 32
threshold = 2.5
noise = syn_signals.SyntheticSignal.Type.PINK

n_runs=100
