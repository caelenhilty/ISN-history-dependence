import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

dt = 1e-5
dur = 0.01     # fix dur at a slightly longer value
CV = 0.5    
numPairs=5

def get_Wji(i):
    rng = np.random.default_rng(i)
    mean_Wji_range = np.logspace(-3, 0, 100).round(5)
    mean_Wji = np.array([-rng.choice(mean_Wji_range), -rng.choice(mean_Wji_range), rng.choice(mean_Wji_range), rng.choice(mean_Wji_range)])
    std_Wji = np.abs(CV * mean_Wji)
    return network_model.makeWji_all_types(rng, numPairs, mean_Wji, std_Wji)

def trial(i, pset, stim_map, data_dir):
    Wji = get_Wji(i)
    
    try:
        FSM = lrt.make_FSM(numPairs, pset, Wji, stim_map, 2, dt=dt, 
                        apply_to_I_and_E=True, raise_unstable=True)
    except:
        return 0.0
    
    # check stability of all nodes 
    stable = True
    for _, out_degree in FSM.out_degree:
        if not out_degree:
            stable = False
    
    sequences = lrt.make_all_sequences(6, ['L', 'R'])        
    reliability = lrt.FSM_reliability(sequences, FSM)
    
    if stable and reliability > 0.73:
        np.save(data_dir + f"/Wji_{i}_reliability_{reliability:.2f}", Wji)

    return reliability

if __name__ == '__main__':
    _, pset, amp, _, l_kernel, r_kernel = util.load_fiducial_network()
    data_dir = util.make_data_folder('figures/RSO', name='dales_law_RSO')
    
    # make stimulus map
    l_stim = np.ones((numPairs)) * amp * l_kernel
    l_stim = np.repeat(l_stim[:, np.newaxis], int(dur/dt), axis=1)
    r_stim = np.ones((numPairs)) * amp * r_kernel
    r_stim = np.repeat(r_stim[:, np.newaxis], int(dur/dt), axis=1)
    stim_map = {'L': l_stim, 'R': r_stim}
    
    n_trials = 100
    def task_generator():
        for i in range(n_trials):
            yield (i, pset, stim_map, data_dir)
    
    start = time.time()
    print(f"starting {n_trials} trials with {mp.cpu_count()} processes")
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(trial, task_generator())
    
    print(f"maximum reliability: {np.max(results)}")
    print(f"elapsed time: {time.time() - start:.1f} seconds for {n_trials} trials. Avg: {((time.time() - start) / n_trials):.1f} seconds per trial")
    