import numpy as np
import multiprocessing as mp

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

# Load the fiducial network
Wji, pset, _, _, _, _ = util.load_fiducial_network()

# calculate means of Wji, ignoring the diagonal
Wji_means = [np.mean(w[np.eye(w.shape[0], dtype=bool) == False]) for w in Wji]

# task parameters
rE_target = 5
rI_target = 10
max_duration = 6
dt = 1e-5

def trial(Wji, amp, dur, states):
    if dur == 0:
        return 0
    G, _ = network_model.get_state_transition_graph(Wji, pset, amp, dur, 
                                                    Eactive=rE_target, Iactive=rI_target, 
                                                    max_duration=max_duration, dt=dt, 
                                                    states=states)
    return network_model.longest_path(G)

if __name__ == "__main__":
    data_dir = 'figures/figure8/data_3'
    output_dir = util.make_data_folder(data_dir, name='longest_paths')
    CVs = [0.05, 0.50, 1.00, 1.75]

    # universal parameter sweep
    n = 20
    amp_min = max(pset[-2], 1)  # pset[-2] is thetaE -- need to turn E on to get a response
    # amp_min is chosen so that at least one cue could elicit a response
    amp_range = np.logspace(np.log10(amp_min), 2, n)
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
    
    def run_sweep(CV, i):
        # make the Wji matrix
        Wji = np.load(data_dir + f'/CV_{CV:.2f}/Wji_{i+1}.npy', allow_pickle=True)

        # run the model for each combination of duration and amplitude
        with mp.Pool(mp.cpu_count()) as pool:
            results = np.array(pool.starmap(trial, 
                                            zip([Wji]*len(dur_flat), 
                                                amp_flat, dur_flat, 
                                                [states for _ in range(len(dur_flat))])))
        pool.close()
        pool.join()

        # reshape the results to match the input grid
        longest_paths = results.reshape(dur_mesh.shape)

        # save the results
        np.save(output_dir + f'/CV_{CV:.2f}/longest_paths_{i+1}.npy', longest_paths)
    
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

