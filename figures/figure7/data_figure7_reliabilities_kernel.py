import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt

Wji, pset, _, _, l_kernel, r_kernel = util.load_fiducial_network()

# task parameters
numPairs = 5
dt = 1e-5
seq_len = 6
sequences = lrt.make_all_sequences(seq_len, ['L', 'R'])
equil_duration = 2

def run_model(dur, amp, l_kernel, r_kernel):
    if dur == 0:
        return 0.5  # return a default reliability for zero duration

    # define the stimulus
    stim_map = lrt.make_stim_map(numPairs, amp, dur, l_kernel, r_kernel, dt)

    # run the model
    FSM = lrt.make_FSM(numPairs, pset, Wji, stim_map, 2, dt=dt)
    reliability = lrt.FSM_reliability(sequences, FSM)
    return reliability

def get_kernels(i):
    rng = np.random.default_rng(i)
    
    # make kernels -- same mean
    l_kernel = rng.lognormal(0, 1, numPairs)
    l_kernel = l_kernel / np.sum(l_kernel) * numPairs
    l_kernel = np.round(l_kernel, 5)
    r_kernel = rng.lognormal(0, 1, numPairs)
    r_kernel = r_kernel / np.sum(r_kernel) * numPairs
    r_kernel = np.round(r_kernel, 5)

    return l_kernel, r_kernel

if __name__ == '__main__':
    import time

    # load reliabilities from kernel sweep
    n_kernels = 5
    reliabilities = np.load('figures/figure6/data_1/reliabilities.npy', allow_pickle=True)
    kernel_ids = np.argsort(reliabilities)[-n_kernels:]
    kernels = [get_kernels(i) for i in kernel_ids]

    # parameter sweep
    n = 20
    amp_range = np.logspace(max(np.log10(pset[-2]/np.max(kernels) + 0.1), 0), 2, n)
    dur_range = np.linspace(0, 0.05, n)
    dur_mesh, amp_mesh = np.meshgrid(dur_range, amp_range) # amp on y-axis, duration on x-axis
    dur_flat, amp_flat = dur_mesh.ravel(), amp_mesh.ravel()

    print(f"Running {len(dur_flat) * len(kernels)} tasks with {mp.cpu_count()} processes...")

    # iterator for all tasks
    def task_gen():
        for l_kernel, r_kernel in kernels:
            for dur, amp in zip(dur_flat, amp_flat):
                yield dur, amp, l_kernel, r_kernel

    start = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        reliabilities = np.array(pool.starmap(run_model, task_gen()))
    print(f"Completed in {time.time() - start:.2f} seconds.")
    pool.close()
    pool.join()
    
    # count the number of reliabilities.npy in the figures/figure6 directory
    data_dir = util.make_data_folder('figures/figure7', name='data_different_kernels')
    np.save(data_dir + '/reliabilities.npy', np.array(reliabilities))
    np.save(data_dir + '/dur_mesh.npy', dur_mesh)
    np.save(data_dir + '/amp_mesh.npy', amp_mesh)
    np.save(data_dir + '/kernels.npy', np.array(kernels))