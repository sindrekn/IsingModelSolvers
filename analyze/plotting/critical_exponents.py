import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def get_metric(name, N, T):
    metrics = {
        "direct":         lambda M: M[1],
        "susceptibility": lambda M: (M[2] - M[1]**2) * N / T,
        "specific_heat":  lambda M: (M[2] - M[1]**2) * N / T**2,
        "binder":         lambda M: 1 - M[4] / (3 * M[2]**2),
        "mag_dev_i":      lambda M, E, ME: ME[1] / M[1] - E[1]
    }
    return metrics[name]









