import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt

Wji, pset, _, _, _, _ = util.load_fiducial_network()

kernels = np.load(f'data_different_kernels/kernels.npy')
l_kernel, r_kernel = kernels[4, 0], kernels[4, 1]

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

    # pcurve and decision dictionary computations -----------------------
    # trace all sequences on the graph
    nodes_dict = {node: [] for node in list(FSM.nodes)}
    for seq in sequences:
        # start at the beginning
        current_node = 1
        # trace the sequence
        for letter in seq:
            for edge in FSM.out_edges(current_node, keys=True):
                if FSM.edges[edge]['label'] == letter:
                    # move to the next node
                    current_node = edge[1]
                    break
        nodes_dict[current_node].append(seq)

    left_counts = np.zeros(seq_len + 1)
    left_choices = np.zeros(seq_len + 1)
    decision_dict = {}
    for node in nodes_dict:
        # is the node a L or R node? based on frequency
        l_cue_counts = [seq.count('L') for seq in nodes_dict[node]]
        num_above = sum([1 for count in l_cue_counts if count > seq_len/2])
        num_below = sum([1 for count in l_cue_counts if count < seq_len/2])
        if num_above == num_below:
            for count in l_cue_counts:
                left_counts[count] += 1     # record distribution of `counts` terminating in current node
                left_choices[count] += 0.5   # record how many times each `count` resulted in a left decision
            for seq in nodes_dict[node]:
                decision_dict[seq] = 'Tie'
        else:
            left_decision = num_below < num_above # True if L, False if R
            for count in l_cue_counts:
                left_counts[count] += 1     # record distribution of `counts` terminating in current node
                left_choices[count] += left_decision   # record how many times each `count` resulted in a left decision
            # update dictionary
            for seq in nodes_dict[node]:
                decision_dict[seq] = 'L' if left_decision else 'R'
    
    p_curve = left_choices/left_counts

    return reliability, p_curve, decision_dict

if __name__ == '__main__':
    import time
    # parameter sweep
    print("Running parameter sweep...")
    n = 50
    
    amp_min = max(pset[-2]/np.max(kernels[4]), 1)   # ensure minimum amplitude is at least 1
    # larger if thetaE is large, to ensure the stimulus is strong enough
    print(f"Minimum amplitude: {amp_min}")

    amp_range = np.logspace(np.log10(amp_min), 2, n)
    dur_range = np.linspace(0, 0.05, n)
    dur_mesh, amp_mesh = np.meshgrid(dur_range, amp_range) # amp on y-axis, duration on x-axis
    dur_flat, amp_flat = dur_mesh.ravel(), amp_mesh.ravel()
    print(f"Running {len(dur_flat)} simulations...")
    start = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = np.array(pool.starmap(run_model, zip(dur_flat, amp_flat)))
    print(f"Completed in {time.time() - start:.2f} seconds.")
    pool.close()
    pool.join()

    # unpack results
    reliabilities, p_curves, decision_dicts = zip(*results)
    reliabilities = np.array(reliabilities)
    p_curves = np.array(p_curves)
    decision_dicts = pd.DataFrame({i: decision_dicts[i] for i in range(len(decision_dicts))})
    
    # count the number of reliabilities.npy in the figures/figure6 directory
    data_dir = util.make_data_folder('figures/figure7')
    np.save(data_dir + '/reliabilities.npy', np.array(reliabilities))
    np.save(data_dir + '/dur_mesh.npy', dur_mesh)
    np.save(data_dir + '/amp_mesh.npy', amp_mesh)
    np.save(data_dir + '/reliabilities.npy', reliabilities)
    np.save(data_dir + '/p_curves.npy', p_curves)
    decision_dicts.to_csv(data_dir + '/decision_dicts.csv', index=False)

    # quick plot of results
    fig, ax = plt.subplots(layout='constrained')
    c = ax.pcolormesh(dur_mesh, amp_mesh, reliabilities.reshape(dur_mesh.shape), cmap='viridis', shading='auto')
    ax.set_yscale('log')
    ax.set_xlabel(r"$\tau_{dur}$")
    ax.set_ylabel(r"$I_{app}$")
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Accuracy')

    plt.show()