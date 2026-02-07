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
    # # check stability
    # stable = np.allclose(rE[int((max_duration - 0.1)/dt):], rE[-1], atol=0.1, rtol=0)
    
    # if not stable: # flag as unstable with negative values
    #     rE *= -1
    #     rI *= -1
    return rE, rI

def off_to_on(rE, rI, rE_star, rI_star) -> bool:
    # checks if the system turned on stably
    # necessary criteria:
    # (1) final rates must be above 0.1 Hz
    if not (rE[-1] > 0.1 and rI[-1] > 0.1):
        return False
    # (2) rates must not diverge to 100 Hz (indicating instability)
    if rE[-1] == 100 or rI[-1] == 100:
        return False
    # AND one of the following must also be true:
    # (1) rates are stable in the last 100 ms of the simulation
    # check first because it's faster
    if np.allclose(rE[int((max_duration - 0.1)/dt):], rE[-1], atol=0.5, rtol=0) and \
        np.allclose(rI[int((max_duration - 0.1)/dt):], rI[-1], atol=0.5, rtol=0):
        return True
    # (2) Poincare map approaches rE_star, rI_star
    sampling_start = -int(2/dt) # sample the last 2 seconds of the simulation
    rE_sample, rI_sample = rE[sampling_start:], rI[sampling_start:]

    # find all the times rI increases through rI_star
    crossings = np.where((rI_sample[:-1] < rI_star) & (rI_sample[1:] >= rI_star))[0]
    if len(crossings) < 2: # need at least 2 crossings to check for decay
        return False
    
    # interpolate rE, t at the crossing times
    alphas = (rI_star - rI_sample[crossings]) / (rI_sample[crossings + 1] - rI_sample[crossings])
    valid_alphas = (alphas >= 0) & (alphas <= 1)
    if not np.any(valid_alphas):
        return False
    alphas = alphas[valid_alphas]
    crossings = crossings[valid_alphas]
    rE_interp = rE_sample[crossings] + alphas * (rE_sample[crossings + 1] - rE_sample[crossings])
    t_cross = (crossings + alphas)

    # compute distance from the fixed point at the crossing times
    # should have no rI component, so just look at rE distance from rE_star
    # (equivalent to distance formula in the limit as rI approaches rI_star)
    radii = np.abs(rE_interp - rE_star) 

    # fit radius across crossings, should be exponential decay if stable
    r_floor = 1e-10
    keep = radii > r_floor
    if np.count_nonzero(keep) < 2:
        return False

    log_r = np.log(radii[keep])
    t_fit = t_cross[keep]

    slope, _ = np.polyfit(t_fit, log_r, 1)
    return slope < 0

def on_to_off(rE, rI) -> bool:
    # check if all below 0.1 Hz in the last 100 ms of the simulation
    return np.all((rE[int((max_duration - 0.1)/dt):] < 0.1) & (rI[int((max_duration - 0.1)/dt):] < 0.1))

def trial(stim_amps, stim_durs, WEE, WEI, WIE, WII, rE_star, rI_star) -> np.array:
    # store if turned on then off in a binary array
    response = np.zeros((len(stim_amps)))
    for i, (amp, dur) in enumerate(zip(stim_amps, stim_durs)):
        rE, rI = bistable_no_depression(dur, amp, 
                                        dt, max_duration,
                                        WEE, WEI, WIE, WII, thetaE, thetaI)
        if off_to_on(rE, rI, rE_star, rI_star):
            rE, rI = bistable_no_depression(dur, amp, 
                                            dt, max_duration,
                                            WEE, WEI, WIE, WII, thetaE, thetaI, 
                                            initial_conditions=[rE_star, rI_star])
            if on_to_off(rE, rI):
                response[i] = 1
    return response

if __name__ == '__main__':
    # outer loop over parameters
    n = 50
    traces = np.logspace(0, 4.5, n) * -1
    determinants = np.logspace(5, 6.5, n)
    trace_mesh, determinant_mesh = np.meshgrid(traces, determinants)
    trace_mesh_, determinant_mesh_ = trace_mesh.ravel(), determinant_mesh.ravel()
    areas = np.zeros_like(trace_mesh_)

    WEE_mesh = np.zeros_like(trace_mesh_)
    WEI_mesh = np.zeros_like(trace_mesh_)
    WIE_mesh = np.zeros_like(trace_mesh_)
    WII_mesh = np.zeros_like(trace_mesh_)
    rE_star_mesh = np.zeros_like(trace_mesh_)
    rI_star_mesh = np.zeros_like(trace_mesh_)
    for i, (tr, det) in enumerate(tqdm(zip(trace_mesh_, determinant_mesh_), total=trace_mesh_.size, mininterval=1)):
        target = util.make_target(rE_target, rI_target, tr, det, thetaE, thetaI)
        x, valid = util.get_solution(target, method='hybr')
        rE_star, rI_star, _, _ = target(x) # get the fixed point rates from the solution
        if valid:
            WEE_mesh[i], WEI_mesh[i], WIE_mesh[i], WII_mesh[i] = x
            rE_star_mesh[i], rI_star_mesh[i] = rE_star, rI_star
        else:
            WEE_mesh[i], WEI_mesh[i], WIE_mesh[i], WII_mesh[i] = np.nan, np.nan, np.nan, np.nan
            rE_star_mesh[i], rI_star_mesh[i] = np.nan, np.nan
            
    # count nan values
    print(f"Warning: {np.sum(np.isnan(WEE_mesh))} NaN values found in parameter mesh.")

    # inner loop over stimulus parameters
    m = 50
    stimulus_durations = np.logspace(-3, 0, m)
    stimulus_amplitudes = np.logspace(0, 2, m)
    STIM_DUR, STIM_AMP = np.meshgrid(stimulus_durations, stimulus_amplitudes)
    STIM_DUR_, STIM_AMP_ = STIM_DUR.ravel(), STIM_AMP.ravel()

    def task_generator():
        for WEE, WEI, WIE, WII, rE_star, rI_star in zip(WEE_mesh, WEI_mesh, WIE_mesh, WII_mesh, rE_star_mesh, rI_star_mesh):
            yield (STIM_AMP_, STIM_DUR_, WEE, WEI, WIE, WII, rE_star, rI_star)
    
    max_processes = 50
    n_processes = min(mp.cpu_count(), max_processes)
    try:
        print(f"Starting pool with {n_processes} processes")
        with mp.Pool(processes=n_processes) as pool:
            results = pool.starmap(trial, task_generator())
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, terminating workers")
        pool.terminate()
        sys.exit()    

    areas = np.array(results)
    
    # save data
    data_dir = util.make_data_folder('figures/figure2', name='data_v2')
    np.save(data_dir + '/WEE_mesh.npy', WEE_mesh)
    np.save(data_dir + '/WEI_mesh.npy', WEI_mesh)
    np.save(data_dir + '/WIE_mesh.npy', WIE_mesh)
    np.save(data_dir + '/WII_mesh.npy', WII_mesh)
    np.save(data_dir + '/trace_mesh.npy', trace_mesh_)
    np.save(data_dir + '/determinant_mesh.npy', determinant_mesh_)
    np.save(data_dir + '/areas.npy', areas)
    np.save(data_dir + '/stim_durations.npy', stimulus_durations)
    np.save(data_dir + '/stim_amplitudes.npy', stimulus_amplitudes)
    np.save(data_dir + '/rE_star.npy', rE_star_mesh)
    np.save(data_dir + '/rI_star.npy', rI_star_mesh)