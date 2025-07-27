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
    Wji = network_model.makeWji_all_types(rng, numPairs, np.array(Wji_means), np.array(stds),
                                          std_tol=0.1)
    # compute CV acheived
    means = [np.mean(w[np.eye(w.shape[0], dtype=bool) == False]) for w in Wji]
    stds = [np.std(w[np.eye(w.shape[0], dtype=bool) == False]) for w in Wji]
    CV_true = np.mean(np.array(stds)/np.array(means))
    print(f"Finished Wji with CV={CV:.2f}, true CV = {CV_true:.2f} in {time.time() - start:.3f} seconds")
    return Wji

if __name__ == "__main__":
    # set up parameter sweep
    CVs = np.array([0.25, 1, 2])
    # add zero to the front of the CVs for the zero-variance case
    CVs = np.insert(CVs, 0, 0.0)  # zero-variance case
    # repeat n_networks times
    n_networks = 10
    CVs = np.repeat(CVs, n_networks)
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(make_Wji_wrapper, CVs)
    pool.close()
    pool.join()

    # Save the results
    data_dir = util.make_data_folder("figures/figure8")
    for i, CV in enumerate(np.unique(CVs)):
        os.makedirs(data_dir + f'/CV_{CV:.2f}', exist_ok=True)
        for j, Wji in enumerate(results[i * n_networks : (i + 1) * n_networks]):
            np.save(data_dir + f'/CV_{CV:.2f}/Wji_{j+1}.npy', Wji, allow_pickle=True)