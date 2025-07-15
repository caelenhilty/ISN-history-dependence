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

def run_model(dur, amp):
    if dur == 0:
        return 0.5  # return a default reliability for zero duration

    # define the stimulus
    stim_map = lrt.make_stim_map(numPairs, amp, dur, l_kernel, r_kernel, dt)

    # run the model
    FSM = lrt.make_FSM(numPairs, pset, Wji, stim_map, 2, dt=dt)
    reliability = lrt.FSM_reliability(sequences, FSM)
    return reliability

if __name__ == '__main__':
    import time
    # parameter sweep
    print("Running parameter sweep...")
    n = 50
    amp_range = np.logspace(0, 2, n)
    dur_range = np.linspace(0, 0.05, n)
    dur_mesh, amp_mesh = np.meshgrid(dur_range, amp_range) # amp on y-axis, duration on x-axis
    dur_flat, amp_flat = dur_mesh.ravel(), amp_mesh.ravel()
    print(f"Running {len(dur_flat)} simulations...")
    start = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        reliabilities = np.array(pool.starmap(run_model, zip(dur_flat, amp_flat)))
    print(f"Completed in {time.time() - start:.2f} seconds.")
    
    # count the number of reliabilities.npy in the figures/figure6 directory
    data_dir = util.make_data_folder('figures/figure7')
    np.save(data_dir + '/reliabilities.npy', np.array(reliabilities))
    np.save(data_dir + '/dur_mesh.npy', dur_mesh)
    np.save(data_dir + '/amp_mesh.npy', amp_mesh)

    # quick plot of results
    fig, ax = plt.subplots(layout='constrained')
    c = ax.pcolormesh(dur_mesh, amp_mesh, reliabilities.reshape(dur_mesh.shape), cmap='viridis', shading='auto')
    ax.set_yscale('log')
    ax.set_xlabel(r"$\tau_{dur}$")
    ax.set_ylabel(r"$I_{app}$")
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Accuracy')

    plt.show()