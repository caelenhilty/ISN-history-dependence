import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import networkx as nx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import plot_style, network_model, util

# load data
data_dir = 'figures/figure4/data'
amp_mesh = np.load(data_dir + '/amp_mesh.npy')
dur_mesh = np.load(data_dir + '/dur_mesh.npy')
longest_paths = np.load(data_dir + '/longest_paths.npy')

amp_dur_pairs = np.array(list(zip(amp_mesh.ravel(), dur_mesh.ravel())))
longest_idxs = np.argsort(longest_paths)[::-1]
amp, dur = amp_dur_pairs[longest_idxs][0]

states = np.load(data_dir + '/states.npy', allow_pickle=True)

# make an example trace
Wji, pset, _, _, _, _ = util.load_fiducial_network()
numPairs = 5
rE_target = 5
rI_target = 10
max_duration = 6
dt =1e-5

# make the sample network (subplot C)
G, states, unstable_states, edge_list, edge_rates, IappE = network_model.get_detailed_state_transition_graph(Wji, pset, amp, dur, 
                                                Eactive=rE_target, Iactive=rI_target, 
                                                max_duration=max_duration, dt=dt, states=states)

longest_path = network_model.get_longest_path(G)
longest_edges = [edge for edge in G.edges() if edge[0] in longest_path and edge[1] in longest_path]
pre_states = [edge[0] for edge in longest_edges]
post_states = [edge[1] for edge in longest_edges]

# first sort longest_edges by their order in the path: start by finding the initial state
r0_index = [pre for pre in pre_states if pre not in post_states][0]
assert r0_index != -1  # make sure the initial state is not unstable
current_state = r0_index
sorted_longest_edges = []
while True:
    next_edge = [edge for edge in longest_edges if edge[0] == current_state]
    if len(next_edge) == 0:
        break
    next_edge = next_edge[0]
    sorted_longest_edges.append(next_edge)
    current_state = next_edge[1]

# stitch together longest path from edge rates
rates = np.zeros((numPairs, 2, 0))
for i, edge in enumerate(sorted_longest_edges):
    # find index of edge in edge_list
    edge_idx = edge_list.index(edge)
    edge_rate = edge_rates[edge_idx]
    rates = np.concatenate((rates, edge_rate), axis=-1)

# make IappE for the stitched rates
raw_time_len = rates.shape[-1]
IappE = np.tile(IappE[0], len(longest_edges))
IappE = IappE[:raw_time_len]
    

def plot_FSM(G, ax):
    weak_components = list(nx.weakly_connected_components(G))
    colors = [i for i in range(len(weak_components))]
    node_colors_dict = {node: colors[i] for i, comp in enumerate(weak_components) for node in comp}
    node_colors = [node_colors_dict[node] for node in list(G)]
    edge_colors = [node_colors_dict[edge[0]] for edge in G.edges()]
    
    longest_path = network_model.get_longest_path(G)
    edge_widths = [2 if edge[0] in longest_path and edge[1] in longest_path else 1 for edge in G.edges()]
    
    pos = nx.arf_layout(G, pos=nx.planar_layout(G))
    nx.draw_networkx_nodes(G, ax=ax, nodelist=longest_path, pos = pos, node_color='k',
                        node_size=400)
    nx.draw_networkx_nodes(G, ax=ax, pos = pos, node_color = node_colors)
    nx.draw_networkx_edges(G, ax=ax, pos = pos, arrows=True, 
                        edge_color=edge_colors, width=edge_widths)
    # nx.draw_networkx_labels(G, pos, ax=ax)
    
    ax.spines['bottom'].set_color('red')
    ax.spines['top'].set_color('red') 
    ax.spines['right'].set_color('red')
    ax.spines['left'].set_color('red')
    
def plot_stim_sweep(ax):
    c = ax.pcolormesh(dur_mesh, amp_mesh, longest_paths.reshape((30, 30)), shading='auto')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau_{dur} (s)$')
    ax.set_ylabel(r'$I_{app}$')
    return c

def draw_rect_pcolormesh(ax, i, j, X, Y, color='red', alpha=1, linewidth=0.1):
    dx = X[j + 1, i + 1] - X[j, i]
    dy = Y[j + 1, i + 1] - Y[j, i]
    x0 = X[j, i] - dx/2
    y0 = Y[j, i] - dy/2
    rect = plt.Rectangle((x0, y0), dx, dy, color=color, alpha=alpha,
                         linewidth=linewidth, fill=False, zorder=10)
    ax.add_patch(rect)
    
def plot_trace(rates, IappE, ax1, ax2):
    raw_time_len = rates.shape[-1]
    # new_rates should contain the rates of both units in each pair, in order
    new_rates = np.zeros((2*numPairs, raw_time_len))
    for i in range(numPairs):
        new_rates[2*i] = rates[i, 0]
        new_rates[2*i + 1] = rates[i, 1]
    # plot firing rates
    vals = ax1.imshow(new_rates[:, :], cmap = 'afmhot', interpolation='nearest', aspect='auto', vmin=0, vmax= int(np.max(new_rates))+1)
    ax1.set_ylabel("Pair ID")
    ax1.set_yticks(np.arange(0.5, 2*numPairs, 2), labels=np.arange(numPairs), minor=False)
    ax1.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax1.set_xticks(np.arange(0, raw_time_len, int(4/dt)), labels=[])
    ax1.tick_params(which='minor', length=0)

    # plot Iapp
    x2ticks = np.arange(0, raw_time_len, int(4/dt))
    ax2.set_xticks(x2ticks, labels = [f'{s:.0f}' for s in np.round(np.linspace(0, raw_time_len*dt, len(x2ticks)), 0)])
    ax2.plot(IappE)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("$I_{app}$")
    
    return vals
    
px = 1/plt.rcParams['figure.dpi']   # convert pixel to inches
fig = plt.figure(layout='constrained', figsize=(plot_style.MAX_WIDTH*px, plot_style.MAX_WIDTH*px*0.7))
axd = fig.subplot_mosaic(
    """
    axBX
    axCC
    SxCC
    """,
    width_ratios=[1, 0.1, 1, 0.1],
    height_ratios=[3,2,1]
)
# figure titles
label_dict ={'a':'A',
             'B':'B',
             'C':'C'}
for name, ax in axd.items():
    label = label_dict.get(name, None)
    if label is not None:
        ax.set_title(label, loc='left', fontweight='bold')

# share axes
axd['a'].sharex(axd['S'])

# plot A
vals = plot_trace(rates, IappE, axd['a'], axd['S'])
fig.colorbar(vals, cax=axd['x'], label='Firing Rate (Hz)')

# plot B
c = plot_stim_sweep(axd['B'])
fig.colorbar(c, cax=axd['X'], label='Longest Path Length (L)')
draw_rect_pcolormesh(axd['B'], 1, 11, dur_mesh, amp_mesh, linewidth=1)

# plot C
plot_FSM(G, axd['C'])

fig.set_constrained_layout_pads(h_pad=0.0, hspace=0.0, w_pad=0.02, wspace=0.02)

plt.savefig('figures/figure4/figure4.png', dpi=600)
plt.savefig('figures/figure4/figure4.tiff', dpi=600)