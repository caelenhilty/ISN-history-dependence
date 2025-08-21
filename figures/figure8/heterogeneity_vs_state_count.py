import numpy as np
import multiprocessing as mp

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, network_model

rE_target = 5
rI_target = 10
dt = 1e-5
max_duration = 6

def trial(i, CV, pset, data_dir):
    Wji = np.load(data_dir + f'/CV_{CV:.2f}/Wji_{i}.npy', allow_pickle=True)
    numPairs = Wji.shape[1]
    states = network_model.get_all_states(Wji, pset, numPairs, rE_target, rI_target, dt=dt, duration=max_duration)
    n_states = len(np.unique(np.round(states, 0), axis=0))
    return i, CV, n_states

if __name__ == '__main__':
    _, pset, _, _, _, _ = util.load_fiducial_network()

    data_dir = 'figures/figure8/data_3'
    output_dir = util.make_data_folder(data_dir, name='state_counts')
    CVs = [0.05, 0.5, 1.0, 1.75]

    def yield_next_task():
        for CV in CVs:
            for i in range(1, 11):
                yield CV, i

    print('Counting states for each parameter set...')
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(trial, [(i, CV, pset, data_dir) for CV, i in yield_next_task()])
    pool.close()
    pool.join()
    
    results = np.array(results)
    np.save(output_dir + '/state_counts.npy', results)