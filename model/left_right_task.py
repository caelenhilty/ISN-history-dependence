import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import networkx as nx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import network_model 

def make_all_sequences(sequence_length:int, alphabet:list):
    """
    Generate all possible sequences of a given length, given an alphabet.
    """
    if sequence_length == 1:
        return alphabet
    else:
        return [i + j for i in alphabet for j in make_all_sequences(sequence_length - 1, alphabet)]

def make_stim_map(numPairs:int, amp:float, dur:float, 
                  l_kernel:np.ndarray, r_kernel:np.ndarray, dt:float):
    """
    Create a stimulus map for the left and right stimuli.
    
    Parameters:
    ----------
    numPairs : int
        Number of pairs of neurons in the network.
    amp : float
        Amplitude of the stimulus.
    dur : float
        Duration of the stimulus in seconds.
    l_kernel : np.array
        Left kernel for the stimulus.
    r_kernel : np.array
        Right kernel for the stimulus.
    dt : float
        Time step of the simulation.

    Returns:
    -------
    dict
        A dictionary mapping 'L' and 'R' to the stimuli to encode.
    """
    l_stim = np.ones((numPairs)) * amp * l_kernel
    l_stim = np.repeat(l_stim[:, np.newaxis], int(dur/dt), axis=1)
    r_stim = np.ones((numPairs)) * amp * r_kernel
    r_stim = np.repeat(r_stim[:, np.newaxis], int(dur/dt), axis=1)
    return {'L': l_stim, 'R': r_stim}
   
def calculate_sequence_duration(sequence, stim_map, equil_duration, dt):
    """
    Calculate the duration of a sequence in time steps.
    
    Parameters:
    ----------
    sequence : Iterable
        A sequence of stimuli to encode.
    stim_map : dict
        A dictionary mapping 'L' and 'R' to the stimuli to encode.
    equil_duration : int
        Duration of the equilibration period.
    dt : float
        Time step of the simulation.
        
    Returns:
    -------
    int
        The duration of the sequence in time steps.
        
    """
    duration = 2 * int(equil_duration/dt) # one at the beginning and one at the end
    for cue in sequence:
        duration += stim_map[cue].shape[1] + int(equil_duration/dt)
    return duration  # not in seconds, but in time steps

def encode_sequence(numPairs, pset, Wji, r0,
                    sequence, stim_map, equil_duration, 
                    sigma_internal_noise:float = 0, 
                    sigma_stim_OFF_noise:float = 0, 
                    sigma_stim_ON_noise:float = 0, 
                    rng=np.random.default_rng(),
                    dt = 0.5e-3, return_timeseries=False,
                    apply_to_E_and_I=False):

    """ Uses the model to encode a sequence of stimuli and returns the final state of the network. 
    
    Parameters:
    ----------
    sequence : np.array
        A sequence of stimuli to encode.
    stim_map : dict
        A dictionary mapping 'L' and 'R' to the stimuli to encode.
    numPairs : int
        Number of pairs of neurons in the network.
    pset : dict
        A dictionary containing the parameters of the model.
    equil_duration : int
        Duration of the equilibration period.
    rng : np.random.default_rng
        Random number generator.
    
    Returns:
    -------
    np.array
        The final state of the network after encoding the sequence.
        Raw firing rate values rounded to the nearest hundredth.
        
    """

    # create input currents
    equil_tsteps = int(equil_duration/dt)
    IappE = sigma_stim_OFF_noise*rng.normal(0, 1, (numPairs, equil_tsteps))
    for cue in sequence:
        # add stimulus to input current
        stim = stim_map[cue]
        IappE = np.append(IappE, stim+sigma_stim_ON_noise*rng.normal(0, 1, stim.shape), axis=1)            
        # add silence after stimulus -- still have default noise
        IappE = np.append(IappE, sigma_stim_OFF_noise*rng.normal(0, 1, (numPairs, equil_tsteps)), axis=1)            
    # make matching sigma_internal_noise_array
    sigma_internal_noise_array = sigma_internal_noise * np.ones((numPairs, 2, IappE.shape[1]))
    # another equil duration for the network to equilibrate in final state -- no noise here
    IappE = np.append(IappE, np.zeros((numPairs, equil_tsteps)), axis=1)
    sigma_internal_noise_array = np.append(sigma_internal_noise_array, np.zeros((numPairs, 2, equil_tsteps)), axis=2)
    
    # create input currents for inhibitory neurons
    if apply_to_E_and_I:
        IappI = np.copy(IappE)
    else:
        IappI = np.zeros((numPairs, IappE.shape[1]))
        
    # calculate total duration
    total_duration = IappE.shape[1] * dt
    
    # run the model
    rates = network_model.simulateISN_noisy(Wji, numPairs, r0,
                    pset, IappE, IappI, 
                    dt, total_duration,
                    sigma_internal_noise_array)
    
    # save the final state of the network
    # check stability?
    stable = np.allclose(rates[:, :, int((total_duration-1)/dt):-1], rates[:, :, -1][..., np.newaxis], rtol=0, atol=0.1)
    
    if return_timeseries: 
        return rates, IappE
    
    if not stable or np.any(rates[:,:,-1]>=100):
        return np.ones((numPairs, 2)) * -1 # conservative placeholder for instability
        # conservative because it maps all oscillatory (or chaotic states) to the same state
    
    # return the final state of the network
    final_state = np.round(rates[:,:,-1], 2)
    return final_state

def make_FSM(numPairs, pset, Wji,
            stim_map, equil_duration,
            dt = 0.5e-3, return_states=False,
            apply_to_I_and_E = False, raise_unstable = False):

    # find r0 -- start from quiescence, see where it stabilizes
    test_duration = 6
    r0_test = np.zeros((numPairs, 2))
    rates = network_model.simulateISN(Wji, numPairs, r0_test, pset, 
                              np.zeros((numPairs, int(test_duration/dt))), np.zeros((numPairs, int(test_duration/dt))), 
                              dt, test_duration)
    stable = np.allclose(rates[:, :, int((test_duration-1)/dt):-1], rates[:, :, -1][..., np.newaxis], rtol=0, atol=0.1)
    assert stable, "r0 did not stabilize"
    r0 = rates[:,:,-1]
    
    # breadth first search to make a graph of the FSM
    alphabet = make_all_sequences(1, stim_map.keys())
    all_states = [r0]
    states_names = [1]
    queue = [(r0,1)]
    FSM = nx.MultiDiGraph()
    FSM.add_node(1)

    while queue:
        current_state, current_name = queue.pop(0)
        # check if current_state is negative anywhere
        if np.any(current_state < 0):
            if raise_unstable:
                raise Exception("Unstable state")
            continue
        for letter in alphabet: # for every "letter" in the FSM alphabet
            next_state = encode_sequence(numPairs, pset, Wji, current_state,
                                        letter, stim_map, equil_duration,
                                        dt=dt, apply_to_E_and_I=apply_to_I_and_E)
            # check if next_state is in states_names already (is visited)
            for i, state in enumerate(all_states):
                if np.allclose(next_state, state, atol=0.1, rtol=0., equal_nan=False):
                    FSM.add_edge(current_name, states_names[i], label=letter)
                    break
            else:
                # mark as visited
                states_names.append(len(states_names)+1)
                all_states.append(next_state)
                # add to queue
                queue.append((next_state, states_names[-1]))
                # add to G
                FSM.add_node(states_names[-1])
                FSM.add_edge(current_name, states_names[-1], label=letter)
    
    if return_states: return FSM, all_states
    return FSM

def plot_FSM(FSM, color_map={'L': 'blue', 'R': 'red'}, connection_map={'L': 'arc3,rad=0.1', 'R': 'arc3,rad=0.2'},node_size=800, font_size=18):
    left_edges = []
    right_edges = []
    for edge in FSM.edges:
        if FSM.edges[edge]['label'] == 'L':
            left_edges.append(edge)
        else:
            right_edges.append(edge)
            
    # plot FSM
    fig = plt.figure(figsize=(5,5), layout='constrained')
    ax = fig.add_subplot(111)
    pos = nx.kamada_kawai_layout(FSM)
    nx.draw_networkx_nodes(FSM, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_edges(FSM, pos, edgelist=left_edges, edge_color=color_map['L'], connectionstyle=connection_map['L'], ax=ax,
                           node_size=node_size)
    nx.draw_networkx_edges(FSM, pos, edgelist=right_edges, edge_color=color_map['R'], connectionstyle=connection_map['R'], ax=ax,
                           node_size=node_size)
    nx.draw_networkx_labels(FSM, pos, ax=ax, font_size=font_size, font_color='white')
    
    return fig
    
def FSM_reliability(sequences, FSM, start_node=1):
    """
    Test the reliability of the FSM on a set of sequences.
    Reliability is defined as the proportion of sequences that 
    are correctly classified by the FSM (not counting draws)
    
    Parameters:
    ----------
    sequences : list
        A list of sequences to test the FSM on.
    FSM : nx.DiGraph
        The finite state machine to test.
    start_node : int
        The starting node of the FSM.
    
    Returns:
    -------
    float
        The reliability of the FSM on the set of sequences.
    
    """
    
    
    seq_len = len(sequences[0])
    
    # trace all sequences on the graph
    nodes_dict = {node: [] for node in list(FSM.nodes)}
    for seq in sequences:
        # start at the beginning
        current_node = start_node
        # trace the sequence
        for letter in seq:
            for edge in FSM.out_edges(current_node, keys=True):
                if FSM.edges[edge]['label'] == letter:
                    # move to the next node
                    current_node = edge[1]
                    break
        nodes_dict[current_node].append(seq.count('R'))
    
    # calculate reliability
    reliability = 0
    num_valid = 0
    for node in nodes_dict:
        num_above = sum([1 for seq in nodes_dict[node] if seq > seq_len/2])
        num_below = sum([1 for seq in nodes_dict[node] if seq < seq_len/2])
        reliability += max(num_above, num_below)
        num_valid += num_below + num_above
        
    reliability /= num_valid
                     
    return reliability

def make_pFSM(numPairs, pset, Wji,
            stim_map, equil_duration,
            sigma_internal_noise, sigma_stim_OFF_noise=0, sigma_stim_ON_noise=0, n_trials=100,
            dt = 0.5e-3):

    # find r0 -- start from quiescence, see where it stabilizes
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
    
    # breadth first search to make a graph of the pFSM
    alphabet = make_all_sequences(1, stim_map.keys())
    all_states = [r0]
    states_names = [1]
    queue = [(r0,1)]
    pFSM = nx.MultiDiGraph()
    pFSM.add_node(1)

    while queue:
        current_state, current_name = queue.pop(0)
        # check if current_state is negative anywhere
        if np.any(current_state < 0):
            # handle oscillatory states by adding self-loop -- conservatively mark as a sink state, which can only hurt reliability
            # add self-loop for all letters
            for letter in alphabet:
                pFSM.add_edge(current_name, current_name, label=letter, weight=1) # weight is 1 because it's deterministic
            continue
        for letter in alphabet: # for every "letter" in the pFSM alphabet
            next_states = {} # store next states and counts
            for seed in range(n_trials):
                rng = np.random.default_rng(seed)
                next_state = encode_sequence(numPairs, pset, Wji, current_state,
                                            letter, stim_map, equil_duration,
                                            sigma_internal_noise=sigma_internal_noise, 
                                            sigma_stim_OFF_noise=sigma_stim_OFF_noise, 
                                            sigma_stim_ON_noise=sigma_stim_ON_noise,
                                            rng=rng, dt=dt)
                # check if next_state is in states_names already (is visited)
                for i, state in enumerate(all_states):
                    if np.allclose(next_state, state, atol=0.1, rtol=0., equal_nan=False):
                        name = states_names[i]
                        next_states[name] = next_states.get(name, 0) + 1
                        break
                else:
                    name = len(states_names)+1
                    # mark as visited
                    states_names.append(name)
                    all_states.append(next_state)
                    # add to queue
                    queue.append((next_state, name))
                    # add to next_states
                    next_states[name] = 1
                
            # update graph
            next_states = {k: v/n_trials for k, v in next_states.items()}
            for name, count in next_states.items():
                pFSM.add_edge(current_name, name, label=letter, weight=count)
                
    return pFSM, all_states

def plot_pFSM(pFSM, color_map={'L': 'blue', 'R': 'red'}, connection_map={'L': 'arc3,rad=0.1', 'R': 'arc3,rad=0.2'},node_size=800, font_size=18):
    left_edges = []
    left_weights = []
    right_edges = []
    right_weights = []
    for edge in pFSM.edges:
        if pFSM.edges[edge]['label'] == 'L':
            left_edges.append(edge)
            left_weights.append(pFSM.edges[edge]['weight'])
        else:
            right_edges.append(edge)
            right_weights.append(pFSM.edges[edge]['weight'])
            
    # plot pFSM
    fig = plt.figure(figsize=(5,5), layout='constrained')
    ax = fig.add_subplot(111)
    pos = nx.circular_layout(pFSM)
    nx.draw_networkx_nodes(pFSM, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_edges(pFSM, pos, edgelist=left_edges, edge_color=color_map['L'], connectionstyle=connection_map['L'], ax=ax,
                           node_size=node_size, width=left_weights)
    nx.draw_networkx_edges(pFSM, pos, edgelist=right_edges, edge_color=color_map['R'], connectionstyle=connection_map['R'], ax=ax,
                           node_size=node_size, width=right_weights)
    nx.draw_networkx_labels(pFSM, pos, ax=ax, font_size=font_size, font_color='white')
    
    return fig

def pFSM_reliability(sequences, pFSM, start_node=1):
    seq_len = len(sequences[0])
    
    # trace each sequence to every possible final state using a modified BFS
    final_states = {node: {} for node in list(pFSM.nodes)}
    for seq in sequences:
        # don't trace sequences where L and R are tied
        if seq.count('L') == seq.count('R'):
            continue
        nodes = [(start_node, 1)]
        for letter in seq:
            next_nodes = []
            for node, prob in nodes:
                for edge in pFSM.edges(node, data=True):
                    if edge[2]['label'] == letter:
                        next_nodes.append((edge[1], prob*edge[2]['weight']))
            nodes = next_nodes
        for node, prob in nodes:
            final_states[node][seq] = final_states[node].get(seq, 0) + prob
    
    # calculate reliability
    num_valid = np.sum([seq.count('L') != seq.count('R') for seq in sequences])
    reliability = 0
    for node in final_states:
        sum_right = sum([final_states[node][seq] for seq in final_states[node] if seq.count('L') < seq.count('R')])
        sum_left = sum([final_states[node][seq] for seq in final_states[node] if seq.count('L') > seq.count('R')])
        reliability += max(sum_right, sum_left)
    reliability /= num_valid
    
    return reliability