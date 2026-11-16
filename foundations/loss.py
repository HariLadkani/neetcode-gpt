import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        result = []
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)

        for y, p in zip(y_true, y_pred):
            loss_per_prediction = y * np.log(p) + (1-y) * np.log(1-p)
            result.append(loss_per_prediction)

        result = np.array(result)    
        return round(-np.average(result), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        clipped_y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        result = []
        for y, p in zip(y_true, clipped_y_pred):
            logged_p = np.log(p)
            loss_per_prediction = np.sum(y * logged_p)
            result.append(loss_per_prediction)

        result = np.array(result)
        return round(-np.average(result), 4)

        
