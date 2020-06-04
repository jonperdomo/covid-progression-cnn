import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import itertools

history_path = 'Models/Model_4_history.npy'
h = np.load(history_path, allow_pickle=True).item()
print("Success")
