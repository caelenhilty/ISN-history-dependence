import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import plot_style

# load data

tr_det_data_dir = 'figures/figure4/data_n_states_vs_tr_det'
tr_det_counts = np.load(tr_det_data_dir + '/counts.npy', allow_pickle=True)
trace_mesh = np.load(tr_det_data_dir + '/trace_mesh.npy', allow_pickle=True)
determinant_mesh = np.load(tr_det_data_dir + '/determinant_mesh.npy', allow_pickle=True)
WEE_mesh = np.load(tr_det_data_dir + '/WEE_mesh.npy', allow_pickle=True)
tr_det_counts = np.where(np.isnan(WEE_mesh), np.nan, tr_det_counts)

nPairs_counts = np.load('figures/figure4/data_n_states_vs_n_pairs/counts.npy', allow_pickle=True)

def plot_tr_vs_det(ax):
    c = ax.pcolormesh(determinant_mesh, trace_mesh, tr_det_counts.reshape(trace_mesh.shape), shading='auto')
    ax.set(ylabel=r'Trace', xlabel=r'Determinant $\Delta$')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    
    det_line = determinant_mesh[:,0]
    trace_line = -2*np.sqrt(det_line)
    idx_max = np.argmax(trace_line < np.min(trace_mesh)) if np.min(trace_line) < np.min(trace_mesh) else -1
    idx_min = np.argmax(trace_line > np.max(trace_mesh)) if np.max(trace_line) > np.max(trace_mesh) else 0
    det_line = det_line[idx_min:idx_max]
    trace_line = trace_line[idx_min:idx_max]
    ax.plot(det_line, trace_line, 'r--', label=r'$tr^2 = 4 \Delta$')
    ax.legend()
    
    return c

def plot_n_pairs_vs_states(ax):
    ns = nPairs_counts[:,1]
    counts = nPairs_counts[:,2]
    ax.scatter(ns, counts, c='k')
        
    # exponential fit
    def exp_func(x, a, b):
        return a * np.power(b, x)
    exp_coeffs, _ = curve_fit(exp_func, ns, counts, p0=(1,2))
    
    fit_ns = np.linspace(min(ns-1), max(ns+1), 100)
    a, b = exp_coeffs
    ax.plot(fit_ns, exp_func(fit_ns, *exp_coeffs), 'r--', label=r"$y = $" + fr"${a:.2f}$" + '\u00D7' + fr"${b:.2f}^N$")
    ax.set_xlabel('# of Pairs (N)')
    ax.set_ylabel('State Count')
    ax.set_yscale('log')
    ax.legend()
    
px = 1/plt.rcParams['figure.dpi']   # convert pixel to inches
fig = plt.figure(layout='constrained', figsize=(plot_style.MAX_WIDTH*px, plot_style.MAX_HEIGHT*px*0.4))
axd = fig.subplot_mosaic(
    """
    ABx
    """,
    width_ratios=[1, 1, 0.1]
)
# add titles
axd['A'].set_title('A', loc='left', fontweight='bold')
axd['B'].set_title('B', loc='left', fontweight='bold')

c = plot_tr_vs_det(axd['B'])
cbar = fig.colorbar(c, cax=axd['x'], location='right')

plot_n_pairs_vs_states(axd['A'])

plt.savefig('figures/figure4/figure4.png')
plt.savefig('figures/figure4/figure4.tiff', dpi=600)