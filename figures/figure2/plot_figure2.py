import numpy as np
import matplotlib.pyplot as plt
import csv as csv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import plot_style

# load data
data_dir = 'figures/figure2/data'
raw_areas = np.load(data_dir + '/areas.npy', allow_pickle=True)
# areas has shape (num_simulations, num_amp, num_dur)
# flatten last two dimensions
areas = raw_areas.reshape(raw_areas.shape[0], -1)
areas = np.mean(areas, axis=1) # mean over stimuli
traces = np.load(data_dir + '/trace_mesh.npy', allow_pickle=True)
determinants = np.load(data_dir + '/determinant_mesh.npy', allow_pickle=True)
n = int(np.sqrt(len(areas)))
trace_mesh = traces.reshape((n, n))
determinant_mesh = determinants.reshape((n, n))
WEE_mesh = np.load(data_dir + '/WEE_mesh.npy', allow_pickle=True)
areas = np.where(np.isnan(WEE_mesh), np.nan, areas)

stimulus_durations = np.load(data_dir + '/stim_durations.npy', allow_pickle=True)
stimulus_amplitudes = np.load(data_dir + '/stim_amplitudes.npy', allow_pickle=True)
m = len(stimulus_durations)
dur_mesh, amp_mesh = np.meshgrid(stimulus_durations, stimulus_amplitudes)

# convert areas to tolerance
tolerances = []
for data in raw_areas:
    area = np.sum(data)
    if area == 0:
        tolerances.append(0)
        continue
    mask = np.array(data == 1).reshape(dur_mesh.shape)
    dur_max = np.max(dur_mesh[mask])
    dur_min = np.min(dur_mesh[mask])
    Lx = dur_max / dur_min
    amp_max = np.max(amp_mesh[mask])
    amp_min = np.min(amp_mesh[mask])
    Ly = amp_max / amp_min
    
    get_idx = lambda arr, val: np.argmin(np.abs(arr - val))
    n_stimuli = (get_idx(stimulus_durations, dur_max) - get_idx(stimulus_durations, dur_min) + 1) * \
                (get_idx(stimulus_amplitudes, amp_max) - get_idx(stimulus_amplitudes, amp_min) + 1)
    
    area_norm = area / n_stimuli
    fold_area = area_norm * Lx * Ly     # convert area to fold-fold change
    tolerances.append(2*np.sqrt(fold_area/np.pi))   # diameter of the "circle" in fold-fold space

tolerance = np.array(tolerances).reshape((n,n))

# samples
selected_points = [(1, 5), (15, 15),(10, 35)] # adjust to match data_figure2_stim_v_flip_flop.py

# plot
px = 1/plt.rcParams['figure.dpi']   # convert pixel to inches
fig = plt.figure(layout='constrained', figsize=(plot_style.MAX_WIDTH*px, plot_style.MAX_WIDTH*px))
axd = fig.subplot_mosaic(
    """
    Aaaaaa
    BBCCDD
    """,
    width_ratios=[0.5, 1, 0.5, 1, 0.5, 1],
    height_ratios = [1, 0.4]
)
# B, C, and D share the same y-axis
axd['B'].sharey(axd['C'])
axd['A'].set_title('A', loc='left', fontweight='bold')
for label, ax in axd.items():
    if label == 'd':
        continue
    if label == 'a':
        # plot the determinant vs trace mesh
        c = ax.pcolormesh(determinant_mesh, trace_mesh, tolerance.reshape(trace_mesh.shape), shading='auto', cmap='viridis')
        for i, point in enumerate(selected_points):
            x, y = point
            ax.text(determinant_mesh[x,0], traces[y], chr(66 + i), color='red', fontsize=10, ha='center', va='center')
        ax.set_xscale('symlog')
        ax.set_yscale('symlog')
        ax.set(ylabel=r'Trace', xlabel=r'Determinant $\Delta$')
        # add line where tr**2 = 4*det
        det_line = determinant_mesh[:,0]
        trace_line = -2*np.sqrt(det_line)
        idx_max = np.argmax(trace_line < np.min(traces)) if np.min(trace_line) < np.min(traces) else -1
        idx_min = np.argmax(trace_line > np.max(traces)) if np.max(trace_line) > np.max(traces) else 0
        det_line = det_line[idx_min:idx_max]
        trace_line = trace_line[idx_min:idx_max]
        ax.plot(det_line, trace_line, 'r--', label=r'$tr^2 = 4 \Delta$')
        ax.legend()
        cbar = fig.colorbar(c, cax=axd['A'])
        cbar.set_label('Tolerance')
        cbar.ax.yaxis.set_label_position('left')
    elif label in ['B', 'C', 'D']:  
        idx = ord(label) - ord('B')
        x, y = selected_points[idx][0], selected_points[idx][1]
        sweep_results = raw_areas.reshape(n, n, m, m)[x, y, :, :]
        ax.set_title(label + f"    tolerance = {tolerance[x, y]:.1f}", loc='left', fontweight='bold')
        c = ax.pcolormesh(dur_mesh, amp_mesh, sweep_results.reshape((m, m)), shading='auto', cmap='viridis')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\tau_{dur} (s)$')
        if label == 'B':
            ax.set_ylabel(r'$I_{app}$')
            
plt.savefig('figures/figure2/figure2.png', dpi=600)
plt.savefig('figures/figure2/figure2.tiff', dpi=600)