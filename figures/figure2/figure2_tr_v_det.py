import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

plt.style.use('seaborn-v0_8-talk')

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
                           initial_conditions=[0, 0], step=1):
    
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
    stable = np.abs(rE[-1]-rE[(int(-0.1/dt))]) < 0.1 and np.abs(rE[-1]-rE[(int((-0.05)/dt))]) < 0.1 \
        and np.abs(rE[-1]-rE[(int((-0.075)/dt))]) < 0.1
    
    if not stable: # flag as unstable with negative values
        rE *= -1
        rI *= -1
    return rE, rI

# outer loop over parameters
n = 30
traces = np.logspace(0, 5, n) * -1
determinants = np.logspace(5, 7, n)
trace_mesh, determinant_mesh = np.meshgrid(traces, determinants)
trace_mesh_, determinant_mesh_ = trace_mesh.ravel(), determinant_mesh.ravel()
areas = np.zeros_like(trace_mesh_)

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
        
# count nan values
try:
    assert np.sum(np.isnan(WEE_mesh)) == 0
except AssertionError:
    # plot WEE mesh
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout='tight')
    im = ax.pcolormesh(determinant_mesh, trace_mesh, WEE_mesh.reshape(trace_mesh.shape), cmap='viridis')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    # add line where tr**2 = 4*det
    det_line = determinants[:]
    trace_line = -2*np.sqrt(det_line)
    idx_max = np.argmax(trace_line < np.min(traces)) if np.min(trace_line) < np.min(traces) else -1
    idx_min = np.argmax(trace_line > np.max(traces)) if np.max(trace_line) > np.max(traces) else 0
    det_line = det_line[idx_min:idx_max]
    trace_line = trace_line[idx_min:idx_max]
    ax.plot(det_line, trace_line, 'r--', label=r'$tr^2 = 4 \cdot det$')
    plt.show()

# inner loop over stimulus parameters
m = 50
stimulus_durations = np.logspace(-3, 0, m)
stimulus_amplitudes = np.logspace(0, 2, m)
STIM_DUR, STIM_AMP = np.meshgrid(stimulus_durations, stimulus_amplitudes)
STIM_DUR_, STIM_AMP_ = STIM_DUR.ravel(), STIM_AMP.ravel()

for i, (WEE, WEI, WIE, WII) in enumerate(tqdm(zip(WEE_mesh, WEI_mesh, WIE_mesh, WII_mesh), total=WEE_mesh.size, mininterval=1)):
    if np.isnan(WEE):
        continue
    for j, (stim_dur, stim_amp) in enumerate(zip(STIM_DUR_, STIM_AMP_)):
        rE, rI = bistable_no_depression(stim_dur, stim_amp, 
                                        dt, max_duration,
                                        WEE, WEI, WIE, WII, thetaE, thetaI)
        if np.any(rE < 0) or rE[-1] == 100 or rI[-1] == 100: # if not stable, go to next stimulus
            continue
        on = (int((rE[-1] > 0.1) and (rI[-1] > 0.1))) # check if ON
        # run again
        if on:
            rE, rI = bistable_no_depression(stim_dur, stim_amp, 
                                            dt, max_duration,
                                            WEE, WEI, WIE, WII, thetaE, thetaI, 
                                            initial_conditions=[rE[-1], rI[-1]])
            if np.any(rE < 0): # if not stable, go to next stimulus
                continue
            on = (int((rE[-1] > 0.1) and (rI[-1] > 0.1))) # check if ON
            if not on:
                areas[i] += 1

# save data
data_dir = util.make_data_folder('figures/figure2', name='trace_vs_det')
np.save(data_dir + '/WEE_mesh.npy', WEE_mesh)
np.save(data_dir + '/WEI_mesh.npy', WEI_mesh)
np.save(data_dir + '/WIE_mesh.npy', WIE_mesh)
np.save(data_dir + '/WII_mesh.npy', WII_mesh)
np.save(data_dir + '/areas.npy', areas)

# compute tolerance
areas = areas/(m**2)
Lx = np.max(determinants) - np.min(determinants)
Ly = np.max(traces) - np.min(traces)
log_radius = np.sqrt(areas * Lx * Ly / np.pi)
tolerances = 10**(log_radius) 
# reports fold-change in parameter space, assuming a circular area

# plot areas vs determinants and traces
fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout='tight')
im = ax.pcolormesh(determinant_mesh, trace_mesh, tolerances.reshape(trace_mesh.shape), cmap='viridis')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Tolerance (%)')
# add line where tr**2 = 4*det
det_line = determinants[:]
trace_line = -2*np.sqrt(det_line)
idx_max = np.argmax(trace_line < np.min(traces)) if np.min(trace_line) < np.min(traces) else -1
idx_min = np.argmax(trace_line > np.max(traces)) if np.max(trace_line) > np.max(traces) else 0
det_line = det_line[idx_min:idx_max]
trace_line = trace_line[idx_min:idx_max]
ax.plot(det_line, trace_line, 'r--', label=r'$tr^2 = 4 \cdot det$')
# axis scaling
ax.set_xscale('symlog')
ax.set_yscale('symlog')
ax.set(ylabel=r'Trace', xlabel=r'Determinant $\Delta$')
ax.legend()
fig.savefig(data_dir + '/figure2.png')