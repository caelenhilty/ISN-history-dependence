import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import networkx as nx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import left_right_task as lrt, network_model, util, plot_style

Wji, pset, amp, dur, l_kernel, r_kernel = util.load_fiducial_network()

# task parameters
numPairs = 5
dt = 1e-5
seq_len = 6
sequences = lrt.make_all_sequences(seq_len, ['L', 'R'])
equil_duration = 2

# make example FSM
l_stim = np.ones((numPairs)) * amp * l_kernel
l_stim = np.repeat(l_stim[:, np.newaxis], int(dur/dt), axis=1)
r_stim = np.ones((numPairs)) * amp * r_kernel
r_stim = np.repeat(r_stim[:, np.newaxis], int(dur/dt), axis=1)
stim_map = {'L': l_stim, 'R': r_stim}
FSM = lrt.make_FSM(numPairs, pset, Wji, stim_map, 2, dt=dt)

# get psychometric curve for example FSM
def psychometric_curve(sequences, FSM):
    seq_len = len(sequences[0])
    # trace all sequences on the graph
    nodes_dict = {node: [] for node in list(FSM.nodes)}
    for seq in sequences:
        # start at the beginning
        current_node = 1
        # trace the sequence
        for letter in seq:
            for edge in FSM.out_edges(current_node, keys=True):
                if FSM.edges[edge]['label'] == letter:
                    # move to the next node
                    current_node = edge[1]
                    break
        nodes_dict[current_node].append(seq.count('L'))

    left_counts = np.zeros(seq_len + 1)
    left_choices = np.zeros(seq_len + 1)
    for node in nodes_dict:
        # is the node a L or R node? based on frequency
        num_above = sum([1 for seq in nodes_dict[node] if seq > seq_len/2])
        num_below = sum([1 for seq in nodes_dict[node] if seq < seq_len/2])
        if num_above == num_below:
            continue
        node_label = num_below < num_above # True if L, False if R
        for seq in nodes_dict[node]:
            left_counts[seq] += 1
            left_choices[seq] += node_label

    return left_choices / left_counts

p_left = psychometric_curve(sequences, FSM)

# get example trace
def make_trace(Wji, pset, amp, dur, l_kernel, r_kernel, sequence = "RRLLLR", dt=1e-5):
    test_duration = 12
    r0_test = np.zeros((numPairs, 2))
    rates = network_model.simulateISN(Wji, numPairs, r0_test, pset, 
                              np.zeros((numPairs, int(test_duration/dt))), np.zeros((numPairs, int(test_duration/dt))), 
                              dt, test_duration)
    stable = np.allclose(rates[:,:,int((test_duration-1-2*dt)/dt)], rates[:,:,-1], atol=0.1, rtol=0., equal_nan=False) and \
        np.allclose(rates[:,:,int(-0.1/dt)], rates[:,:,-1], atol=0.1, rtol=0., equal_nan=False) and \
        np.allclose(rates[:,:,int((test_duration-1.1)/dt)], rates[:,:,-1], atol=0.1, rtol=0., equal_nan=False)
    assert stable, "r0 did not stabilize"
    r0 = rates[:,:,-1]
    
    # make stimulus map
    l_stim = np.ones((numPairs)) * amp * l_kernel
    l_stim = np.repeat(l_stim[:, np.newaxis], int(dur/dt), axis=1)
    r_stim = np.ones((numPairs)) * amp * r_kernel
    r_stim = np.repeat(r_stim[:, np.newaxis], int(dur/dt), axis=1)
    stim_map = {'L': l_stim, 'R': r_stim}
    
    dt1 = dt
    rates, IappE = lrt.encode_sequence(numPairs, pset, Wji, r0, sequence, 
                                       stim_map, 2, dt=dt1, return_timeseries=True)
    
    return rates, IappE

rates, IappE = make_trace(Wji, pset, amp, dur, l_kernel, r_kernel, sequence = "RLRLLL", dt=dt)
# make sure shapes match (rare bug)
raw_time_len = min(rates.shape[-1], IappE.shape[-1])
rates, IappE = rates[:,:,:raw_time_len], IappE[:, :raw_time_len]

def plot_FSM(FSM, ax, color_map={'L': 'blue', 'R': 'red'}, connection_map={'L': 'arc3,rad=0.1', 'R': 'arc3,rad=0.2'}, node_size=800, font_size=None):
    """Hard coded for this example"""
    
    if font_size is None:
        font_size = plt.rcParams['font.size']
    left_edges = []
    right_edges = []
    l_self_edges = []
    r_self_edges = []
    for edge in FSM.edges:
        if FSM.edges[edge]['label'] == 'L':
            if edge[0] == edge[1]:
                l_self_edges.append(edge)  
            else:
                left_edges.append(edge)
        else:
            if edge[0] == edge[1]:
                r_self_edges.append(edge)
            else:
                right_edges.append(edge)
    
    pos = nx.kamada_kawai_layout(FSM)
    nx.draw_networkx_nodes(FSM, pos, ax=ax, node_size=node_size, 
                           node_color=['red', 'red', 'red', 'blue'])
    nx.draw_networkx_edges(FSM, pos, edgelist=left_edges, edge_color=color_map['L'], connectionstyle=connection_map['L'], ax=ax,
                           node_size=node_size)
    nx.draw_networkx_edges(FSM, pos, edgelist=right_edges, edge_color=color_map['R'], connectionstyle=connection_map['R'], ax=ax,
                           node_size=node_size)
    nx.draw_networkx_edges(FSM, pos, edgelist=l_self_edges,
                           edge_color='blue',
                           connectionstyle='arc3,rad=0.1',
                           ax=ax, node_size=node_size
                        )
    nx.draw_networkx_edges(FSM, pos, edgelist=r_self_edges,
                           edge_color='red',
                           ax=ax, node_size=node_size
                        )
    nx.draw_networkx_labels(FSM, pos, ax=ax, font_size=font_size, font_color='white')
    
def plot_trace(rates, IappE, fig, ax1, ax2):
    # new_rates should contain the rates of both units in each pair, in order
    new_rates = np.zeros((2*numPairs, raw_time_len))
    for i in range(numPairs):
        new_rates[2*i] = rates[i, 0]
        new_rates[2*i + 1] = rates[i, 1]
    # plot firing rates
    vals = ax1.imshow(new_rates[:, :], cmap = 'afmhot', interpolation='nearest', aspect='auto', vmin=0, vmax= int(np.max(new_rates))+1)
    ax1.set_ylabel("Unit ID")
    ax1.set_yticks(np.arange(0.5, 2*numPairs, 2), labels=np.arange(numPairs), minor=False)
    ax1.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax1.set_xticks(np.arange(0, raw_time_len, int(1/dt)), labels=[])
    ax1.tick_params(which='minor', length=0)
    fig.colorbar(vals, ax=ax1, fraction=0.046, pad=0.04, label="Firing Rate (Hz)")

    # plot Iapp
    x2ticks = np.arange(0, raw_time_len, int(2/dt))
    ax2.set_xticks(x2ticks, labels = np.round(np.linspace(0, raw_time_len*dt, len(x2ticks)), 0))
    ax2.plot(IappE[0])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("$I_{app}$")
    
import scipy

def plot_p_curves(ax, data_dir='figures/figure5/psycho_data', seq_len=6):
    data = np.load(data_dir + '/p_curves.npy', allow_pickle=True)
    reliabilities = np.load(data_dir + '/reliabilities.npy', allow_pickle=True)
    good_p_curves = data[reliabilities > 0.73]
    ax.plot(np.arange(seq_len + 1), good_p_curves.T, color='gray', alpha=0.05)
    mean_p = np.mean(good_p_curves, axis=0)
    sem_p = scipy.stats.sem(good_p_curves, axis=0)
    ax.plot(np.arange(seq_len + 1), mean_p, '-o', color='blue', label='Mean')
    ax.fill_between(np.arange(seq_len + 1), mean_p - 1.96 * sem_p, mean_p + 1.96 * sem_p, color='blue', alpha=0.2, label='95% CI')
    ax.set_xlabel('Number of L cues in sequence')
    ax.set_ylabel('P(choose L)')

# make figure
px = 1/plt.rcParams['figure.dpi']   # convert pixel to inches
fig = plt.figure(layout='constrained', figsize=(plot_style.MAX_WIDTH*px, plot_style.MAX_HEIGHT*px))
axd = fig.subplot_mosaic(
    """
    AB
    Ab
    Ac
    CD
    """,
    height_ratios=[0.25,4,1,4]
)
for label, ax in axd.items():
    if label in ('b', 'c'): continue
    ax.set_title(label, loc='left', fontweight='bold')
axd['A'].axis('off')    # hide A
axd['B'].axis('off')    # hide B
# share axes between b and c
axd['b'].sharex(axd['c'])

# plot trace 
plot_trace(rates, IappE, fig, axd['b'], axd['c'])

# plot FSM
plot_FSM(FSM, axd['C'], connection_map={'L': 'arc3,rad=0.2', 'R': 'arc3,rad=0.2'})
# add a label in the bottom right corner
axd['C'].text(0.6, 0.7, 'R cue', transform=axd['C'].transAxes, ha='center', va='bottom', 
              fontsize=plt.rcParams['font.size'], c='red')
axd['C'].text(0.825, 0.7, 'L cue', transform=axd['C'].transAxes, ha='center', va='bottom', 
              fontsize=plt.rcParams['font.size'], c='blue')
axd['C'].text(0.425, 0.6, 'R decision', transform=axd['C'].transAxes, ha='center', va='bottom', 
              fontsize=plt.rcParams['font.size'], c='red', weight='semibold')
axd['C'].text(0.35, 0.025, 'L decision', transform=axd['C'].transAxes, ha='center', va='bottom', 
              fontsize=plt.rcParams['font.size'], c='blue', weight='semibold')
# plot reliabilities ------------------------------------------------------------
plot_p_curves(axd['D'])
axd['D'].plot(np.arange(7), p_left, color='red', alpha=0.5, label='fiducial')
ax.legend()

plt.savefig('figures/figure5/figure5_raw.png')