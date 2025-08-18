import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt

Wji, pset, amp, dur, l_kernel, r_kernel = util.load_fiducial_network()

# task parameters
numPairs = 5
dt = 1e-5
seq_len = 6
sequences = lrt.make_all_sequences(seq_len, ['L', 'R'])
equil_duration = 2

def run_trial(i):
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
    import time as time

    n_trials = 1000

    print(f"Running {n_trials} trials... with {mp.cpu_count()} processes")
    try:
        start_time = time.time()
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(run_trial, range(n_trials))
        pool_time = time.time() - start_time
        
        print(f"Completed {n_trials} trials in {pool_time:.2f} seconds")
    except KeyboardInterrupt:
        pool.terminate()
    pool.close()
    pool.join()
    
    # unpack results
    p_curves = np.zeros((n_trials, seq_len + 1))
    reliabilities = np.zeros(n_trials)
    print("Unpacking results...")
    for i, (reliability, p_curve, decision_dict) in tqdm(enumerate(results)):
        reliabilities[i] = reliability
        p_curves[i] = p_curve
        if i == 0:
            all_decision_dicts = {i: decision_dict}
        else:
            all_decision_dicts[i] = decision_dict
    df_decisions = pd.DataFrame(all_decision_dicts)
    print(np.histogram(reliabilities, bins=20))

    # save results
    # make a data folder
    data_dir = util.make_data_folder('figures/figure5', name='psycho_data')
    print(f"Saving results to {data_dir}...")
    # save the dataframe
    df_decisions.to_csv(f'{data_dir}/decision_dict.csv')
    np.save(f'{data_dir}/p_curves.npy', p_curves)
    np.save(f'{data_dir}/reliabilities.npy', reliabilities)