import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.figure2.__dev.data_figure2_stim_v_flip_flop import trial
from model import util, left_right_task as lrt, network_model

# core parameters
rE_target = 5
rI_target = 10

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
    
def trial(stim_amps, stim_durs, WEE, WEI, WIE, WII, thetaE, thetaI):
    area = 0
    for amp, dur in zip(stim_amps, stim_durs):
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
            on = (rE[-1] > 0.1) and (rI[-1] > 0.1) # check if ON
            if not on:
                area += 1
                continue
    return area

if __name__ == '__main__':
    # outer loop over parameters
    n = 15
    traces = np.logspace(0, 5, n) * -1
    determinants = np.logspace(4, 7, n)
    trace_mesh, determinant_mesh = np.meshgrid(traces, determinants)
    trace_mesh_, determinant_mesh_ = trace_mesh.ravel(), determinant_mesh.ravel()
    areas = np.zeros_like(trace_mesh_)

    # inner loop over stimulus parameters
    m = 25
    stimulus_durations = np.logspace(-3, 0, m)
    stimulus_amplitudes = np.logspace(0, 2, m)
    STIM_DUR, STIM_AMP = np.meshgrid(stimulus_durations, stimulus_amplitudes)
    STIM_DUR_, STIM_AMP_ = STIM_DUR.ravel(), STIM_AMP.ravel()

    data_dir = util.make_data_folder('figures/figure2', name='search_data')

    psets = np.load('figures/RSO_figure1/psets.npy', allow_pickle=True)
    best_result = None
    best_mean = -np.inf
    for pset in psets[:40]:
        thetaE, thetaI = pset[-2], pset[-1]

        WEE_mesh = np.zeros_like(trace_mesh_)
        WEI_mesh = np.zeros_like(trace_mesh_)
        WIE_mesh = np.zeros_like(trace_mesh_)
        WII_mesh = np.zeros_like(trace_mesh_)
        for i, (tr, det) in enumerate(tqdm(zip(trace_mesh_, determinant_mesh_), total=trace_mesh_.size, mininterval=1)):
            target = util.make_target(rE_target, rI_target, tr, det, thetaE, thetaI)
            x, valid = util.get_solution(target, method='hybr', maxiter=1000)
            if valid:
                WEE_mesh[i], WEI_mesh[i], WIE_mesh[i], WII_mesh[i] = x
            else:
                WEE_mesh[i], WEI_mesh[i], WIE_mesh[i], WII_mesh[i] = np.nan, np.nan, np.nan, np.nan
                
        # count nan values
        try:
            assert np.sum(np.isnan(WEE_mesh)) == 0
        except AssertionError:
            print("Warning: NaN values found in parameter mesh.")

        def task_generator():
            for WEE, WEI, WIE, WII in zip(WEE_mesh, WEI_mesh, WIE_mesh, WII_mesh):
                yield (STIM_AMP_, STIM_DUR_, WEE, WEI, WIE, WII, thetaE, thetaI)
        
        try:
            print(f"Starting pool with {mp.cpu_count()} processes")
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.starmap(trial, task_generator())
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught, terminating workers")
            pool.terminate()
            sys.exit()    

        areas = np.array(results)
        if np.mean(areas) > best_mean:
            best_mean = np.mean(areas)
            best_result = (thetaE, thetaI, areas)
        np.save(data_dir + f'/areas_tE{thetaE:.1f}_tI{thetaI:.1f}_mean{np.mean(areas):.1f}.npy', areas)

    # plot the best result
    if best_result is not None:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(determinant_mesh, trace_mesh, best_result[2].reshape(trace_mesh.shape), shading='auto')
        plt.xscale('log')
        plt.yscale('symlog')
        det_line = determinant_mesh[:,0]
        trace_line = -2*np.sqrt(det_line)
        idx_max = np.argmax(trace_line < np.min(trace_mesh)) if np.min(trace_line) < np.min(trace_mesh) else -1
        idx_min = np.argmax(trace_line > np.max(trace_mesh)) if np.max(trace_line) > np.max(trace_mesh) else 0
        det_line = det_line[idx_min:idx_max]
        trace_line = trace_line[idx_min:idx_max]
        plt.plot(det_line, trace_line, 'r--', label=r'$tr^2 = 4 \Delta$')
        plt.colorbar(label='Area')
        plt.savefig(data_dir + f'/best_result_tE{thetaE:.1f}_tI{thetaI:.1f}_mean{np.mean(areas):.1f}.png')
        plt.show()