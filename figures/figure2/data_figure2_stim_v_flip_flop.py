import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import network_model, util

# core parameters
rE_target = 10
rI_target = 5
thetaE = 5.34
thetaI = 82.43

max_duration = 12
dt = 1e-5

# simulation wrapper function
def bistable_no_depression(stimulus_duration, stimulus_amplitude, 
                           dt, max_duration:int,
                           WEE, WEI, WIE, WII, thetaE, thetaI, tauE=10e-3, tauI=10e-3,
                           initial_conditions=[0, 0]):
    duration = max_duration
    rmax = 100
    total_duration = duration + stimulus_duration + 1 # 1 second of equilibration pre-stimulus
    
    IappI = np.zeros(int(total_duration/dt))
    IappE = np.zeros(int(total_duration/dt))
    IappE[int(1/dt):int((stimulus_duration + 1)/dt)] = stimulus_amplitude
    IappI[int(1/dt):int((stimulus_duration + 1)/dt)] = stimulus_amplitude

    rE, rI = network_model.simulateISP(dt, total_duration, rmax, tauE, tauI, 
                            WEE, WEI, WIE, WII, thetaE, thetaI,
                            IappI, IappE, rE0=initial_conditions[0], rI0=initial_conditions[1])   
    # check stability
    stable = np.allclose(rE[int((max_duration - 0.1)/dt):], rE[-1], atol=0.1, rtol=0)
    
    if not stable: # flag as unstable with negative values
        rE *= -1
        rI *= -1
    return rE, rI    
    
def trial(stim_amp, stim_dur, WEE, WEI, WIE, WII):
    rE, rI = bistable_no_depression(stim_dur, stim_amp, 
                                            dt, max_duration,
                                            WEE, WEI, WIE, WII, thetaE, thetaI)
    if np.any(rE < 0) or rE[-1] == 100 or rI[-1] == 100: # if not stable, go to next stimulus
        return 0
    on = (int((rE[-1] > 0.1) and (rI[-1] > 0.1))) # check if ON
    # run again
    if on:
        rE, rI = bistable_no_depression(stim_dur, stim_amp, 
                                        dt, max_duration,
                                        WEE, WEI, WIE, WII, thetaE, thetaI, 
                                        initial_conditions=[rE[-1], rI[-1]])
        if np.any(rE < 0): # if not stable, go to next stimulus
            return 0
        on = (int((rE[-1] > 0.1) and (rI[-1] > 0.1))) # check if ON
        if not on:
            return 1
    return 0

if __name__ == '__main__':
    data_dir = 'figures/figure2/trace_vs_det_1'
    WEE_mesh = np.load(data_dir + '/WEE_mesh.npy', allow_pickle=True)
    WEI_mesh = np.load(data_dir + '/WEI_mesh.npy', allow_pickle=True)
    WIE_mesh = np.load(data_dir + '/WIE_mesh.npy', allow_pickle=True)
    WII_mesh = np.load(data_dir + '/WII_mesh.npy', allow_pickle=True)
    n = int(np.sqrt(WEE_mesh.shape[0]))
    WEE_mesh = WEE_mesh.reshape((n, n))
    WEI_mesh = WEI_mesh.reshape((n, n))
    WIE_mesh = WIE_mesh.reshape((n, n))
    WII_mesh = WII_mesh.reshape((n, n))
    
    # outer loop over selected points
    selected_points = [(1, 3), (25, 20),(10, 35)]
    
    # inner loop over stimulus parameters
    m = 50
    stimulus_durations = np.logspace(-3, 0, m)
    stimulus_amplitudes = np.logspace(0, 2, m)
    STIM_DUR, STIM_AMP = np.meshgrid(stimulus_durations, stimulus_amplitudes)
    STIM_DUR_, STIM_AMP_ = STIM_DUR.ravel(), STIM_AMP.ravel()
    
    for i, (x, y) in tqdm(enumerate(selected_points)):
        WEE = WEE_mesh[x, y]
        WEI = WEI_mesh[x, y]
        WIE = WIE_mesh[x, y]
        WII = WII_mesh[x, y]
        
        # run trials in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(trial, [(STIM_AMP_[j], STIM_DUR_[j], WEE, WEI, WIE, WII) 
                                           for j in range(len(STIM_AMP_))])
        
        # reshape results
        results = np.array(results).reshape(STIM_DUR.shape)
        
        # save results
        np.save(data_dir + f'/sample_({x},{y})_stim_sweep.npy', results)
    
    
    