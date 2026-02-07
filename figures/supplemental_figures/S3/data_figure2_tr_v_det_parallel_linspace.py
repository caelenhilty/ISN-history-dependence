import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

# core parameters
rE_target = 5
rI_target = 10
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
    
def trial(stim_amps, stim_durs, WEE, WEI, WIE, WII) -> np.array:
    # store if turned on then off in a binary array
    response = np.zeros((len(stim_amps)))
    for i, (amp, dur) in enumerate(zip(stim_amps, stim_durs)):
        rE, rI = bistable_no_depression(dur, amp, 
                                        dt, max_duration,
                                        WEE, WEI, WIE, WII, thetaE, thetaI)
        if np.any(rE < 0) or rE[-1] == 100 or rI[-1] == 100: # if not stable, go to next stimulus
            continue
        on = (int((rE[-1] > 0.1) and (rI[-1] > 0.1))) # check if ON
        # run again
        if on:
            rE, rI = bistable_no_depression(dur, amp, 
                                            dt, max_duration,
                                            WEE, WEI, WIE, WII, thetaE, thetaI, 
                                            initial_conditions=[rE[-1], rI[-1]])
            if np.any(rE < 0): # if not stable, go to next stimulus
                continue
            on = ((rE[-1] > 0.1) and (rI[-1] > 0.1)) # check if ON
            if not on:
                response[i] = 1
    return response

if __name__ == '__main__':
    # load parameter meshes
    data_dir = 'figures/figure2/data'
    WEE_mesh = np.load(data_dir + '/WEE_mesh.npy', allow_pickle=True)
    WEI_mesh = np.load(data_dir + '/WEI_mesh.npy', allow_pickle=True)
    WIE_mesh = np.load(data_dir + '/WIE_mesh.npy', allow_pickle=True)
    WII_mesh = np.load(data_dir + '/WII_mesh.npy', allow_pickle=True)
    traces = np.load(data_dir + '/trace_mesh.npy', allow_pickle=True)
    determinants = np.load(data_dir + '/determinant_mesh.npy', allow_pickle=True)
    
    # reshape
    n = 50
    trace_mesh = traces.reshape((n, n))
    determinant_mesh = determinants.reshape((n, n))
    WEE_mesh = WEE_mesh.reshape((n, n))
    WEI_mesh = WEI_mesh.reshape((n, n))
    WIE_mesh = WIE_mesh.reshape((n, n))
    WII_mesh = WII_mesh.reshape((n, n))

    # count nan values
    print(f"Warning: {np.sum(np.isnan(WEE_mesh))} NaN values found in parameter mesh.")
    


    # pick out points of interest
    selected_points = [(1, 5), (15, 15),(10, 35)]
    max_dur_pows = [0, np.log10(0.4), 0]
    # inner loop over stimulus parameters
    m = 300
    amp_meshes, dur_meshes = [], []
    for pow in max_dur_pows:
        stimulus_durations = np.linspace(10**-3, 10**pow, m)
        stimulus_amplitudes = np.logspace(0, 2, m)
        STIM_DUR, STIM_AMP = np.meshgrid(stimulus_durations, stimulus_amplitudes)
        STIM_DUR_, STIM_AMP_ = STIM_DUR.ravel(), STIM_AMP.ravel()
        amp_meshes.append(STIM_AMP_)
        dur_meshes.append(STIM_DUR_)

    def task_generator():
        for i, (x, y) in enumerate(selected_points):
            WEE = WEE_mesh[x, y]
            WEI = WEI_mesh[x, y]
            WIE = WIE_mesh[x, y]
            WII = WII_mesh[x, y]
            yield (amp_meshes[i], dur_meshes[i], WEE, WEI, WIE, WII)
    
    try:
        n_processes = min(mp.cpu_count(), len(selected_points))
        print(f"Starting pool with {n_processes} processes")
        with mp.Pool(processes=n_processes) as pool:
            results = pool.starmap(trial, task_generator())
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, terminating workers")
        pool.terminate()
        sys.exit()    

    areas = np.array(results)
    
    # save data
    data_dir = util.make_data_folder('figures/figure2', name='data_linspace')
    np.save(data_dir + '/areas.npy', areas)
    np.save(data_dir + '/stim_durations.npy', stimulus_durations)
    np.save(data_dir + '/stim_amplitudes.npy', stimulus_amplitudes)