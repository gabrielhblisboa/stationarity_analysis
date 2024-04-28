import typing
import tqdm
import numpy as np

import matplotlib.pyplot as plt

class Detector():

    def __init__(self, memory_size: int = 128, threshold: float = 3) -> None:
        self.memory_size = memory_size
        self.threshold = threshold

    def run(self, input_data: np.array, intervals: typing.List[typing.Tuple[int, int]]):
        tp = 0
        fp = 0

        if self.memory_size > len(input_data):
            raise UnboundLocalError(f"invalid data size memory_size[{self.memory_size}] and data with {len(input_data)} sample")

        results = []
        for i in range(self.memory_size, len(input_data)):
            window = input_data[i-self.memory_size:i]
            mean = np.mean(window)
            std = np.std(window)
            z_score = (np.max(input_data[i-1:i]) - mean) / std
            if z_score >= self.threshold:
                results.append(1)
            else:
                results.append(0)
        results = np.array(results)
        valid_index = []

        for interval in intervals:
            i = interval[0] - self.memory_size
            j = interval[1] - self.memory_size

            valid_index.extend([range(i,j)])

            if np.max(results[i:j]) == 1: # True positive
                tp += 1

        outside = [results[x] for x in range(len(results)) if x not in valid_index]
        fp = np.sum(np.diff(outside) == 1)/len(outside)
        tp /= len(intervals)

        return tp, fp
