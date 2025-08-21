import numpy as np
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

# core parameters
rE_target = 5
rI_target = 10
Wji, pset, _, _, _, _ = util.load_fiducial_network()
max_duration = 6
dt = 1e-5

def trial(amp, dur, states):
    if dur == 0:
        return 0
    G, _ = network_model.get_state_transition_graph(Wji, pset, amp, dur, Eactive=rE_target, Iactive=rI_target, max_duration=max_duration, dt=dt, states=states)
    return network_model.longest_path(G)

if __name__ == '__main__':
    # parameter sweep setup
    n = 30
    amp_range = np.logspace(np.log10(pset[-2]), 2, n) # amplitude should be > thetaE
    dur_range = np.linspace(0, 0.05, n)
    dur_mesh, amp_mesh = np.meshgrid(dur_range, amp_range) # amp on y-axis, duration on x-axis
    dur_flat, amp_flat = dur_mesh.ravel(), amp_mesh.ravel()
    
    # find all states
    print("Finding states...")
    numPairs = Wji.shape[1]
    states = network_model.get_all_states(Wji, pset, numPairs, rE_target, rI_target, duration = max_duration, dt=dt)
    _, unique_idxs = np.unique(np.round(states, 0), axis=0, return_index=True)  # remove duplicates
    states = [np.array(state) for state in states[unique_idxs]]  # convert to list
    print("Found states")

    print(f"Starting {len(amp_flat)} trials with {mp.cpu_count()} processes...")
    try:
        with mp.Pool(processes=min(mp.cpu_count(), n**2)) as pool:
            results = pool.starmap(trial, zip(amp_flat, dur_flat, [states for _ in range(len(dur_flat))]))
    except KeyboardInterrupt:
        pool.terminate()
    pool.close()
    pool.join()
    
    results = np.array(results)
    data_dir = util.make_data_folder('figures/figure3')
    np.save(data_dir + '/longest_paths.npy', results)
    np.save(data_dir + '/dur_mesh.npy', dur_mesh)
    np.save(data_dir + '/amp_mesh.npy', amp_mesh)