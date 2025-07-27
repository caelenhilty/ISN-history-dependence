import numpy as np
import pandas as pd
import multiprocessing as mp
import time as time

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import network_model, util

# Load the fiducial network
Wji, pset, amp, dur, l_kernel, r_kernel = util.load_fiducial_network()

# calculate means of Wji, ignoring the diagonal
Wji_means = [np.mean(w[np.eye(w.shape[0], dtype=bool) == False]) for w in Wji]

numPairs = 5  # number of pairs of neurons in the network

def make_Wji_wrapper(CV):
    print(f"Generating Wji with CV={CV:.2f}")
    rng = np.random.default_rng()
    stds = [mean * CV for mean in Wji_means]
    start = time.time()
    Wji = network_model.makeWji_all_types(rng, numPairs, np.array(Wji_means), np.array(stds), verbose=True,
                                          std_tol=0.1)
    print(f"Finished Wji with CV={CV:.2f} in {time.time() - start:.2f} seconds")
    return Wji

make_Wji_wrapper(2)