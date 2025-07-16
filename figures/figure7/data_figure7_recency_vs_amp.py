import numpy as np
import pandas as pd
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import left_right_task as lrt, util

# load reliabilities from kernel sweep
reliabilities = np.load('figures/figure6/data_1/reliabilities.npy', allow_pickle=True)

# task parameters
numPairs = 5
dt = 1e-5
seq_len = 6
sequences = lrt.make_all_sequences(seq_len, ['L', 'R'])
equil_duration = 2

Wji, pset, amp, dur, l_kernel, r_kernel = util.load_fiducial_network()

def run_trial(i, amp):
    rng = np.random.default_rng(i)
    
    # make kernels -- same mean
    l_kernel = rng.lognormal(0, 1, numPairs)
    l_kernel = l_kernel / np.sum(l_kernel) * numPairs
    l_kernel = np.round(l_kernel, 5)
    r_kernel = rng.lognormal(0, 1, numPairs)
    r_kernel = r_kernel / np.sum(r_kernel) * numPairs
    r_kernel = np.round(r_kernel, 5)
    
    # make stimuli
    stim_map = lrt.make_stim_map(numPairs, amp, dur, l_kernel, r_kernel, dt)
    
    # make FSM
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
    
    n_kernels = 10
    kernel_ids = np.argsort(reliabilities)[-n_kernels:]
    amps = amp * np.logspace(-0.5, 0.5, 11)
    kernel_mesh, amps_mesh = np.meshgrid(kernel_ids, amps)
    kernel_ids = kernel_mesh.flatten()
    amps = amps_mesh.flatten()

    print(f'Running {n_kernels * len(amps)} trials...')
    start = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(run_trial, zip(kernel_ids, amps))
    pool.close()
    pool.join()
    print(f"Completed in {time.time() - start:.2f} seconds.")
    
    reliabilities, p_curves, decision_dicts = zip(*results)
    reliabilities = np.array(reliabilities)
    p_curves = np.array(p_curves)
    decision_dicts = pd.DataFrame({i: decision_dicts[i] for i in range(len(decision_dicts))})

    # save results
    data_dir = util.make_data_folder('figures/figure7')
    np.save(data_dir + '/reliabilities.npy', reliabilities)
    np.save(data_dir + '/p_curves.npy', p_curves)
    np.save(data_dir + '/amps.npy', amps)
    np.save(data_dir + '/kernel_ids.npy', kernel_ids)
    decision_dicts.to_csv(data_dir + '/decision_dicts.csv', index=False)
    print(f"Saved results to {data_dir}")