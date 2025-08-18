import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import pandas as pd
import csv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import network_model, plot_style

def bistable_no_depression(pset, stimulus_timing, stimulus_duration, stimulus_amplitude, duration, dt, initial_conditions=[0, 0]):
    """
    A wrapper function for simulating the bistable ISP.
    """
    rmax = 100
    tauE, tauI, _, _, _, _, _, _, WEE, WEI, WIE, WII, thetaE, thetaI = network_model.unpack_parameters(pset)
    
    IappI = np.zeros(int(duration/dt))
    IappE = np.zeros(int(duration/dt))
    IappE[int(stimulus_timing/dt):int((stimulus_timing+stimulus_duration)/dt)] = stimulus_amplitude
    IappI[int(stimulus_timing/dt):int((stimulus_timing+stimulus_duration)/dt)] = stimulus_amplitude

    rE, rI = network_model.simulateISP(dt, duration, rmax, tauE, tauI, 
                            WEE, WEI, WIE, WII, thetaE, thetaI,
                            IappI, IappE, rE0=initial_conditions[0], rI0=initial_conditions[1])   
    
    return rE, rI

def get_basins(pset, amplitude, n=500, rE_max=50, rI_max=50):
    """
    Get the basins of attraction for the bistable ISP.
    Samples initial conditions and finds the corresponding final states.
    
    Parameters
    ----------
    pset : np.array
        Parameter set for the network model.
    amplitude : float
        Amplitude of the stimulus.
    n : int
        Number of samples for initial conditions.
    rE_max : float
        Maximum initial value for excitatory population activity.
    rI_max : float
        Maximum initial value for inhibitory population activity.
        
    Returns
    -------
    X : np.ndarray
        Meshgrid for initial conditions (rE0).
    Y : np.ndarray
        Meshgrid for initial conditions (rI0).
    color_mappings : np.ndarray
        Mapping of final states to unique IDs for colormap.
    steady_states : np.ndarray
        Unique steady states found in the final states.
    """
    dt = 1e-4

    # meshgrid for initial conditions
    x = np.linspace(0, rE_max, n) # rE0
    y = np.linspace(0, rI_max, n) # rI0
    X, Y = np.meshgrid(x, y)
    X_, Y_ = X.ravel(), Y.ravel()

    # find final states for each initial condition
    final_states = np.zeros((n*n, 2))
    duration = 6
    for j, (rE0, rI0) in enumerate(zip(X_, Y_)):
        rE, rI = bistable_no_depression(pset, 0, duration, amplitude, duration, dt, initial_conditions=[rE0, rI0])
        final_states[j] = [rE[-1], rI[-1]]
        
    # find unique steady states
    steady_states, indices = np.unique(np.round(final_states,1), axis=0, return_index=True)
    steady_states = final_states[indices]
    
    # map final states to steady states for colormap
    color_mappings = np.ones((n*n))*-1    # will break the colormap if any -1s are left (intended behavior)
    for i, state in enumerate(final_states):
        for j, steady_state in enumerate(steady_states):
            if np.allclose(state, steady_state, atol=1e-1):
                color_mappings[i] = j
    
    return X, Y, color_mappings, steady_states

# load best psets from manual search results
best_psets = np.load('figures/figure1/best_psets.npy', allow_pickle=True)
best_amp_durs = pd.read_csv('figures/figure1/best_amp_durs.csv')
best_amp_durs = best_amp_durs.values

# generate data
trial_duration = 6
dt = 1e-5
timing = 0.5
pset = np.array(best_psets[5], dtype=np.float32)
amp, dur = best_amp_durs[5]

x_max, y_max = 15, 30
X, Y, autonomous_basins, auto_states = get_basins(pset, 0, n=100, rE_max=x_max, rI_max=y_max)
_, _, stim_basins, stim_states = get_basins(pset, amp, n=100, rE_max=x_max, rI_max=y_max)
off_to_on = bistable_no_depression(pset, timing, dur, amp, trial_duration, dt)
on_to_off = bistable_no_depression(pset, timing, dur, amp, trial_duration, dt, initial_conditions=[5, 10])
off_to_on = np.array(off_to_on)
on_to_off = np.array(on_to_off)

# plotting setup
def plot_trace(ax, traces, duration=2, dt = 1e-5, x_max=2, y_max=20, x_min=0):
    time = np.arange(0, duration, dt)
    ax.plot(time-timing, traces[0], label=r'$r_E$', c=plot_style.excit_red)
    ax.plot(time-timing, traces[1], label=r'$r_I$', c=plot_style.inh_blue)
    ax.set_xlabel('Time (s)')
    ax.set(xlim=(x_min,x_max), ylim=(0,y_max))
    ax.axvspan(0, dur, alpha=0.75, color='0.8')
    
def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    Add an arrow to a line.
    Credit: https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot

    Parameters
    ----------
    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        start_ind = len(xdata) // 2
    else:
        start_ind = int(len(xdata) * position)
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size
    )

def plot_basins(ax, X, Y, colors, states, trajectory, xmax=20, ymax=50):
    ax.contourf(X, Y, colors.reshape(X.shape), levels=len(states), cmap='Pastel1')
    ax.set(xlim=(0, xmax), ylim=(0, ymax))
    ax.plot(trajectory[0], trajectory[1], color='black')
    # add a point at the beginning and end of the trajectory
    ax.scatter(trajectory[0][0], trajectory[1][0], color='black', marker='o', zorder=10, s=20)
    ax.scatter(trajectory[0][-1], trajectory[1][-1], color='black', marker='o', zorder=10, s=20)
    # get line
    line = ax.lines[-1]
    ax.scatter(states[:, 0], states[:, 1], color='red', label='steady states', marker='x', zorder=10)
    return line

# plot
px = 1/plt.rcParams['figure.dpi']   # convert pixel to inches
fig = plt.figure(layout='tight', figsize=(plot_style.MAX_WIDTH*px, plot_style.MAX_WIDTH*px*0.6))
axd = fig.subplot_mosaic(
    """
    AaCD
    EFGH
    """,
    height_ratios=[1, 1]
)
# hide axis A and I
axd['A'].axis('off')
axd['a'].axis('off')
# add labels
label_dict = {
    'A':'A1   stimulus on', 'a':'A2   stimulus off',
    'C':'B1', "D":'B2',   
    'E':'C1   stimulus on', "F":'C2   stimulus. off', 'G':'C3   stimulus. on', 'H':'C4   stimulus. off'
    }
for label, ax in axd.items():
    ax.set_title(label_dict[label], loc='left', fontweight='bold')

# plot standard traces ---------------------------------------------
axd['C'].sharey(axd['D'])
plot_trace(axd['C'], off_to_on, duration=6, x_min=-0.15, x_max = 0.15, y_max=30)
plot_trace(axd['D'], on_to_off, duration=6, x_min=-0.15, x_max = 0.15, y_max=30)
axd['C'].set_ylabel('Firing Rate (Hz)')
axd['C'].legend(loc='upper left')
axd['C'].annotate(r'$-/+$', xy=(0.775, 0.825), xycoords='axes fraction', ha='center', fontsize=18)
axd['D'].annotate(r'$+/-$', xy=(0.775, 0.825), xycoords='axes fraction', ha='center', fontsize=18)

# plot traces with arrows over basins -----------------------------------------------
line = plot_basins(axd['E'], X, Y, stim_basins, stim_states, off_to_on[:, int(timing/dt):int((timing+dur)/dt)], xmax=x_max, ymax=y_max)
for p in np.linspace(0.5, 1, 3, endpoint=False):
    add_arrow(line, position=p, direction='right', size=12)
line = plot_basins(axd['F'], X, Y, autonomous_basins, auto_states, off_to_on[:, int((timing+dur)/dt):], xmax=x_max, ymax=y_max)
for p in [0.0001, 0.0004, 0.0006, 0.001]:
    add_arrow(line, position=p, direction='right', size=12)
line = plot_basins(axd['G'], X, Y, stim_basins, stim_states, on_to_off[:, int(timing/dt):int((timing+dur)/dt)], xmax=x_max, ymax=y_max)
for p in np.linspace(0.25, 1, 3, endpoint=False):
    add_arrow(line, position=p, direction='right', size=12)
line = plot_basins(axd['H'], X, Y, autonomous_basins, auto_states, on_to_off[:, int((timing+dur)/dt):], xmax=x_max, ymax=y_max)
for p in [0.0003, 0.0005]:
    add_arrow(line, position=p, direction='right', size=12)
# add annotations to basins
annotations = [r'$-/+$', r'$-/+$', r'$+/-$', r'$+/-$']
for ax, annotation in zip([axd['E'], axd['F'], axd['G'], axd['H']], annotations):
    ax.annotate(annotation, xy=(0.8, 0.025), xycoords='axes fraction', ha='center', fontsize=18)
# add xlabels to basins
for ax in [axd['E'], axd['F'], axd['G'], axd['H']]:
    ax.set_xlabel(r'$r_E (Hz)$')
# basins share y-label
axd['E'].set_ylabel(r'$r_I (Hz)$')

plt.savefig('figures/figure1/figure1_raw.png', dpi=600, bbox_inches='tight')