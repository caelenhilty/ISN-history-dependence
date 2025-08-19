import numpy as np
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

# core parameters
rE_target = 10
rI_target = 5
thetaE = 5.34
thetaI = 82.43
max_duration = 6
dt = 1e-5

def trial(Wji, i, numPairs, pset):
    states = network_model.get_all_states(Wji, pset, numPairs, rE_target, rI_target, dt=dt, duration=max_duration)
    n_states = len(np.unique(np.round(states, 0), axis=0))
    return i, numPairs, n_states

if __name__ == '__main__':
    Wji, pset, _, _, _, _ = util.load_fiducial_network()

    # core parameters
    Wji_means = [np.mean(w[np.eye(w.shape[0], dtype=bool) == False]) for w in Wji]
    Wji_stds = np.abs([np.std(w[np.eye(w.shape[0], dtype=bool) == False]) for w in Wji])
    CV = np.mean(np.array(Wji_stds) / np.abs(np.array(Wji_means)))
    
    numPairs_range = np.arange(1, 11, 2)  # Range of number of pairs to test
    n_trials = 5
    
    def yield_next_task():
        for numPairs in numPairs_range:
            for i in range(n_trials):
                means = np.array(Wji_means) * 5 / numPairs  # scale to match the number of pairs (was 5 originally)
                Wji = network_model.makeWji_all_types(np.random.default_rng(i), numPairs, means, np.abs(means * CV))
                yield (Wji, i, numPairs, pset)

    # count states for each parameter set
    try:
        print('Counting states for each parameter set...')
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(trial, yield_next_task())
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    pool.close()
    pool.join()
    
    results = np.array(results)
    
    # save data
    data_dir = util.make_data_folder('figures/figure4', name='n_states_vs_n_pairs')
    np.save(data_dir + '/counts.npy', results)