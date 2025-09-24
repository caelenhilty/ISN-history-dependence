import numpy as np
import multiprocessing as mp

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt

# Load the fiducial network
Wji, pset, amp, dur, l_kernel, r_kernel = util.load_fiducial_network()

# calculate means of Wji, ignoring the diagonal
Wji_means = [np.mean(w[np.eye(w.shape[0], dtype=bool) == False]) for w in Wji]

# task parameters
numPairs = 5
dt = 1e-5
seq_len = 6
sequences = lrt.make_all_sequences(seq_len, ['L', 'R'])
equil_duration = 2

def run_model(Wji, dur, amp):
    if dur == 0:
        return 0.5  # return a default reliability for zero duration

    # define the stimulus
    stim_map = lrt.make_stim_map(numPairs, amp, dur, l_kernel, r_kernel, dt)

    # run the model
    FSM = lrt.make_FSM(numPairs, pset, Wji, stim_map, 2, dt=dt)
    reliability = lrt.FSM_reliability(sequences, FSM)
    return reliability

if __name__ == "__main__":
    data_dir = 'figures/figure7/backup'
    output_dir = util.make_data_folder(data_dir, name='raw_reliabilities')
    CVs = [0.05]

    # universal parameter sweep
    n = 20
    amp_min = max(min(pset[-2]/np.max(l_kernel), pset[-2]/np.max(r_kernel)), 1)  # pset[-2] is thetaE -- need to turn E on to get a response
    # amp_min is chosen so that at least one cue could elicit a response
    amp_range = np.logspace(np.log10(amp_min), 2, n)
    dur_range = np.linspace(0, 0.05, n)
    dur_mesh, amp_mesh = np.meshgrid(dur_range, amp_range) # amp on y-axis, duration on x-axis
    dur_flat, amp_flat = dur_mesh.ravel(), amp_mesh.ravel()
    
    def run_sweep(CV, i):
        # make the Wji matrix
        Wji = np.load(data_dir + f'/CV_{CV:.2f}/Wji_{i+1}.npy', allow_pickle=True)

        # run the model for each combination of duration and amplitude
        with mp.Pool(mp.cpu_count()) as pool:
            reliabilities = np.array(pool.starmap(run_model, zip([Wji]*len(dur_flat), dur_flat, amp_flat)))
        pool.close()
        pool.join()

        # reshape the results to match the input grid
        reliability_grid = reliabilities.reshape(dur_mesh.shape)
        
        # save the results
        np.save(output_dir + f'/CV_{CV:.2f}/reliabilities_{i+1}.npy', reliability_grid)
    
    n_networks = 10  # number of networks per CV
    for CV in CVs:
        os.makedirs(Path(output_dir + f'/CV_{CV:.2f}'), exist_ok=True)
        print(f"Running for CV={CV:.2f} with {n_networks} networks...")
        for i in range(n_networks):  # run n_networks for each CV
            print(f"Running for CV={CV:.2f}, network {i+1}/{n_networks}", end='\r')
            run_sweep(CV, i)
        print(f"\nFinished running for CV={CV:.2f}")
    
    # also run for CV = 0.00
    os.makedirs(Path(output_dir + '/CV_0.00'), exist_ok=True)
    run_sweep(0.0, 1)  # only one network for CV = 0.00

