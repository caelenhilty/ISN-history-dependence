import numpy as np
import matplotlib.pyplot as plt
import csv as csv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import plot_style

# load data
data_dir = 'figures/figure2/data'
areas = np.load(data_dir + '/areas.npy', allow_pickle=True)
traces = np.load(data_dir + '/trace_mesh.npy', allow_pickle=True)
determinants = np.load(data_dir + '/determinant_mesh.npy', allow_pickle=True)
n = int(np.sqrt(len(areas)))
trace_mesh = traces.reshape((n, n))
determinant_mesh = determinants.reshape((n, n))

stimulus_durations = np.load(data_dir + '/stim_durations.npy', allow_pickle=True)
stimulus_amplitudes = np.load(data_dir + '/stim_amplitudes.npy', allow_pickle=True)
dur_mesh, amp_mesh = np.meshgrid(stimulus_durations, stimulus_amplitudes)

# covert areas to tolerance
m = len(stimulus_durations)
n_stimuli = m ** 2
norm_areas = areas / n_stimuli
Lx = np.max(stimulus_durations)/np.min(stimulus_durations)
Ly = np.max(stimulus_amplitudes)/np.min(stimulus_amplitudes)
fold_areas = norm_areas.reshape((n, n)) * (Lx * Ly) # Area in fold-fold change space
tolerance = np.sqrt(fold_areas / np.pi) # radius expresses the fold change in terms of a circle's radius

# samples
selected_points = [(1, 3), (25, 20),(10, 35)] # adjust to match data_figure2_stim_v_flip_flop.py

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
        sweep_results = np.load(data_dir + f'/sample_({x},{y})_stim_sweep.npy', allow_pickle=True)
        ax.set_title(label + f"    tolerance = {tolerance[x, y]:.1f}", loc='left', fontweight='bold')
        c = ax.pcolormesh(dur_mesh, amp_mesh, sweep_results.reshape((m, m)), shading='auto', cmap='viridis')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\tau_{dur} (s)$')
        if label == 'B':
            ax.set_ylabel(r'$I_{app}$')
            
plt.savefig('figures/figure2/figure2.png', dpi=600)
plt.savefig('figures/figure2/figure2.tiff', dpi=600)