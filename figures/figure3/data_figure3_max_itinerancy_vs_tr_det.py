import numpy as np
import multiprocessing as mp
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

# core parameters
numPairs = 5
max_duration = 6
dt = 1e-5

def trial(amp, dur, states, pset):
    if dur == 0:
        return 0
    G, _ = network_model.get_state_transition_graph(Wji, pset, amp, dur, Eactive=rE_target, Iactive=rI_target, max_duration=max_duration, dt=dt, states=states)
    return network_model.longest_path(G)

if __name__ == '__main__':
    Wji, pset, _, _, _, _ = util.load_fiducial_network()
    thetaE, thetaI = pset[-2], pset[-1]
    rE_target = 5
    rI_target = 10

    # bounds on traces and determinants
    traces = np.logspace(0, 3, 3) * -1
    determinants = np.logspace(1, 6, 3)
    # set tr0 and det0 as the log-space midpoints
    tr_0 = traces[3 // 2]
    det_0 = determinants[3 // 2]
    # compute a line through the midpoint
    # perpendicular to tr = -2 sqrt(det)
    n_samples = 10
    norm = np.sqrt(det_0)
    det_sample = np.logspace(np.log10(min(determinants)), np.log10(max(determinants)), n_samples)
    trace_sample = norm*(det_sample - det_0) + tr_0
    # clip det_sample and trace_sample so that trace_sample does not exceed
    # the bounds of traces
    valid_indices = np.where((trace_sample >= min(traces)) & (trace_sample <= max(traces)))[0]
    det_sample = det_sample[valid_indices]
    trace_sample = trace_sample[valid_indices]

    WEE_mesh = np.zeros_like(trace_sample)
    WEI_mesh = np.zeros_like(trace_sample)
    WIE_mesh = np.zeros_like(trace_sample)
    WII_mesh = np.zeros_like(trace_sample)
    for i, (tr, det) in enumerate(tqdm(zip(trace_sample, det_sample), total=trace_sample.size, mininterval=1)):
        target = util.make_target(rE_target, rI_target, tr, det, thetaE, thetaI)
        x, valid = util.get_solution(target, method='hybr')
        if valid:
            WEE_mesh[i], WEI_mesh[i], WIE_mesh[i], WII_mesh[i] = x
        else:
            WEE_mesh[i], WEI_mesh[i], WIE_mesh[i], WII_mesh[i] = np.nan, np.nan, np.nan, np.nan
    
    print(f'Number of NaN values: {np.sum(np.isnan(WEE_mesh))}')