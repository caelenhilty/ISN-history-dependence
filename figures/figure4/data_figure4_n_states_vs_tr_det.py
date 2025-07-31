import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
numPairs = 5
max_duration = 12
dt = 1e-5

def trial(Wji, pset):
    if Wji is None or pset is None:
        return 0  # Skip if Wji or pset is None
    return network_model.count_homogeneous_ISN_states(Wji, pset, numPairs, 
                                                      rE_target, rI_target, 
                                                      dt=dt, duration=max_duration)

if __name__ == '__main__':
    Wji, pset, _, _, _, _ = util.load_fiducial_network()

    # core parameters
    Wji_means = [np.mean(w[np.eye(w.shape[0], dtype=bool) == False]) for w in Wji]
    Wji_means = np.array(Wji_means) * 5 / numPairs  # scale to match the number of pairs
    Wji = network_model.makeWji_all_types(np.random.default_rng(), numPairs, Wji_means, np.zeros(4))
    
    # outer loop over parameters
    n = 30
    traces = np.logspace(0, 5, n) * -1
    determinants = np.logspace(5, 7, n)
    trace_mesh, determinant_mesh = np.meshgrid(traces, determinants)
    trace_mesh_, determinant_mesh_ = trace_mesh.ravel(), determinant_mesh.ravel()
    
    WEE_mesh = np.zeros_like(trace_mesh_)
    WEI_mesh = np.zeros_like(trace_mesh_)
    WIE_mesh = np.zeros_like(trace_mesh_)
    WII_mesh = np.zeros_like(trace_mesh_)
    for i, (tr, det) in enumerate(tqdm(zip(trace_mesh_, determinant_mesh_), total=trace_mesh_.size, mininterval=1)):
        target = util.make_target(rE_target, rI_target, tr, det, thetaE, thetaI)
        x, valid = util.get_solution(target, method='hybr')
        if valid:
            WEE_mesh[i], WEI_mesh[i], WIE_mesh[i], WII_mesh[i] = x
        else:
            WEE_mesh[i], WEI_mesh[i], WIE_mesh[i], WII_mesh[i] = np.nan, np.nan, np.nan, np.nan
    
    print(f'Number of NaN values: {np.sum(np.isnan(WEE_mesh))}')
    
    # define task generator
    def yield_next_task():
        for WEE, WEI, WIE, WII in zip(WEE_mesh, WEI_mesh, WIE_mesh, WII_mesh):
            if not np.isnan(WEE):
                new_pset = network_model.pack_parameters(pset[0], pset[1], pset[2], pset[3], pset[4], pset[5], pset[6], pset[7], 
                                                         WEE, WEI, WIE, WII,
                                                         thetaE, thetaI)
                yield (Wji, new_pset)
            else:
                yield (None, None)
    
    # count states for each parameter set
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(trial, yield_next_task())
    pool.close()
    pool.join()
    
    results = np.array(results)
    
    # save data
    data_dir = util.make_data_folder('figures/figure4', name='n_states_vs_tr_det')
    np.save(data_dir + '/WEE_mesh.npy', WEE_mesh)
    np.save(data_dir + '/WEI_mesh.npy', WEI_mesh)
    np.save(data_dir + '/WIE_mesh.npy', WIE_mesh)
    np.save(data_dir + '/WII_mesh.npy', WII_mesh)
    np.save(data_dir + '/trace_mesh.npy', trace_mesh)
    np.save(data_dir + '/determinant_mesh.npy', determinant_mesh)
    np.save(data_dir + '/counts.npy', results)
    
    # quick plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(trace_mesh, determinant_mesh, results.reshape(trace_mesh.shape), shading='auto')
    plt.colorbar(label='Number of stable states')
    plt.savefig(data_dir + '/counts.png', dpi=300)