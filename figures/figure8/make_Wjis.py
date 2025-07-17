import numpy as np
import pandas as pd
import multiprocessing as mp

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import network_model, util

# Load the fiducial network
Wji, pset, amp, dur, l_kernel, r_kernel = util.load_fiducial_network()

# calculate means of Wji, ignoring the diagonal
Wji_means = [np.mean(w[np.eye(w.shape[0], dtype=bool) == False]) for w in Wji]

# set up parameter sweep
CVs = np.logspace(-2, 0, 5)
numPairs = 5  # number of pairs of neurons in the network

def make_Wji_wrapper(CV, n_networks=10):
    rng = np.random.default_rng()
    stds = [mean * CV for mean in Wji_means]
    Wjis = [network_model.makeWji_all_types(rng, numPairs, np.array(Wji_means), np.array(stds),
                                           timeout=np.inf) for _ in range(n_networks)]
    return Wjis
if __name__ == "__main__":
    with mp.Pool(min(mp.cpu_count(), 10)) as pool:
        results = pool.map(make_Wji_wrapper, CVs)
    pool.close()
    pool.join()

    # Save the results
    data_dir = util.make_data_folder("figures/figure8")
    for i, CV in enumerate(CVs):
        Wjis = results[i]
        os.makedirs(data_dir + f'/CV_{CV:.2f}', exist_ok=True)
        for j, Wji in enumerate(Wjis):
            np.save(data_dir + f'/CV_{CV:.2f}/Wji_{j+1}.npy', Wji, allow_pickle=True)