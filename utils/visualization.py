import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from typing import List, Tuple

def visualize_quantum_state(state: qt.Qobj) -> plt.Figure:
    """
    Visualize a quantum state using a bar plot of its probability amplitudes.
    
    Args:
        state: Quantum state to visualize
        
    Returns:
        Matplotlib figure containing the visualization
    """
    # Get density matrix if not already
    if not state.isoper:
        rho = state * state.dag()
    else:
        rho = state
    
    # Get diagonal elements (probabilities)
    probs = np.real(rho.diag())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot probabilities
    x = np.arange(len(probs))
    ax.bar(x, probs, width=0.6, color='indigo', alpha=0.7)
    
    # Format axes
    ax.set_xlabel('Basis State')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    
    # If we have fewer than 16 states, label them in binary
    if len(probs) <= 16:
        binary_labels = [format(i, f'0{int(np.log2(len(probs)))}b') for i in range(len(probs))]
        ax.set_xticks(x)
        ax.set_xticklabels(binary_labels, rotation=45)
    else:
        # Otherwise, just show a few labels
        step = max(1, len(probs) // 8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"{i}" for i in x[::step]])
    
    plt.tight_layout()
    return fig

def plot_bloch_sphere(states: List[qt.Qobj], labels: List[str] = None) -> plt.Figure:
    """
    Visualize qubit states on the Bloch sphere.
    
    Args:
        states: List of quantum states to visualize
        labels: Optional list of labels for the states
        
    Returns:
        Matplotlib figure containing the Bloch sphere
    """
    # Create Bloch sphere
    b = qt.Bloch()
    
    # Define colors for different states
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    
    # Add states to the Bloch sphere
    for i, state in enumerate(states):
        color = colors[i % len(colors)]
        b.add_states(state, color)
    
    # Add labels if provided
    if labels:
        b.point_labels = labels
    
    # Customize appearance
    b.point_size = [30] * len(states)
    b.point_marker = ['o'] * len(states)
    
    # Render the Bloch sphere
    fig = plt.figure(figsize=(6, 6))
    b.render(fig=fig)
    
    return fig

def plot_recovery_fidelity(fidelities: List[float], t_max: float) -> plt.Figure:
    """
    Plot the fidelity of the quantum state over time during recovery.
    
    Args:
        fidelities: List of fidelity values
        t_max: Maximum evolution time
        
    Returns:
        Matplotlib figure showing the fidelity over time
    """
    times = np.linspace(0, t_max, len(fidelities))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, fidelities, 'b-', lw=2)
    ax.plot(times, fidelities, 'ro', ms=4)
    
    # Add threshold line for successful recovery
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Success Threshold')
    
    # Add threshold line for failed recovery
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Failure Threshold')
    
    # Format axes
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Fidelity')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_fractal_encoding(classical_key: np.ndarray, quantum_key: qt.Qobj) -> plt.Figure:
    """
    Visualize the fractal encoding of a classical key.
    
    Args:
        classical_key: The classical binary key
        quantum_key: The quantum encoding of the key
        
    Returns:
        Matplotlib figure showing the encoding visualization
    """
    n = len(classical_key)
    
    # Get the probability amplitudes from the quantum state
    if not quantum_key.isoper:
        rho = quantum_key * quantum_key.dag()
    else:
        rho = quantum_key
    probs = np.real(rho.diag())
    phases = np.angle(quantum_key.full().flatten())
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the classical key as binary sequence
    markerline, stemlines, baseline = ax1.stem(np.arange(n), classical_key, basefmt=' ')
    plt.setp(markerline, 'color', 'k', 'marker', 'o')
    plt.setp(stemlines, 'color', 'k', 'linestyle', '-')
    ax1.set_xlabel('Bit Position')
    ax1.set_ylabel('Bit Value')
    ax1.set_title('Classical Key')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Plot the quantum state probabilities with phase information
    num_states = min(16, len(probs))  # Limit to first 16 states for clarity
    x = np.arange(num_states)
    
    # Plot probability bars
    bars = ax2.bar(x, probs[:num_states], width=0.6, color='indigo', alpha=0.7)
    
    # Color the bars according to phase
    for i, bar in enumerate(bars):
        if i < len(phases):
            # Map phase (-π to π) to a color
            phase_color = plt.cm.hsv((phases[i] + np.pi) / (2 * np.pi))
            bar.set_color(phase_color)
    
    # Format axis
    ax2.set_xlabel('Basis State')
    ax2.set_ylabel('Probability')
    ax2.set_title('Quantum Fractal Encoding')
    ax2.set_ylim(0, 1)
    
    # Add binary labels
    binary_labels = [format(i, f'0{n}b') for i in range(num_states)]
    ax2.set_xticks(x)
    ax2.set_xticklabels(binary_labels, rotation=45)
    
    # Add colorbar to show phase mapping
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(-np.pi, np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', pad=0.1)
    cbar.set_label('Phase')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    
    plt.tight_layout()
    return fig

def plot_resilience_comparison(error_rates: np.ndarray, qshe_fidelities: np.ndarray, 
                              classical_fidelities: np.ndarray) -> plt.Figure:
    """
    Plot a comparison of resilience between Q-SHE and classical encryption.
    
    Args:
        error_rates: Array of error rates
        qshe_fidelities: Recovery fidelities for Q-SHE
        classical_fidelities: Recovery fidelities for classical ECC
        
    Returns:
        Matplotlib figure showing the comparison
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot Q-SHE resilience
    ax.plot(error_rates, qshe_fidelities, 'b-', lw=2, marker='o', label='Q-SHE')
    
    # Plot classical ECC resilience
    ax.plot(error_rates, classical_fidelities, 'r--', lw=2, marker='s', label='Classical ECC')
    
    # Add recovery threshold line
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Recovery Threshold')
    
    # Format axes
    ax.set_xlabel('Error Rate')
    ax.set_ylabel('Recovery Fidelity')
    ax.set_xlim(0, max(error_rates))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    
    # Add title
    ax.set_title('Resilience Comparison: Q-SHE vs Classical ECC')
    
    plt.tight_layout()
    return fig

def plot_overhead_comparison(key_sizes: np.ndarray) -> plt.Figure:
    """
    Plot a comparison of computational overhead between Q-SHE and classical methods.
    
    Args:
        key_sizes: Array of key sizes in bits
        
    Returns:
        Matplotlib figure showing the overhead comparison
    """
    # Calculate overhead for Q-SHE (theoretical): O(log n)
    qshe_overhead = np.log2(key_sizes)
    
    # Calculate overhead for classical ECC (theoretical): O(n)
    classical_overhead = key_sizes
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot overheads
    ax.plot(key_sizes, qshe_overhead, 'b-', lw=2, marker='o', label='Q-SHE: O(log n)')
    ax.plot(key_sizes, classical_overhead, 'r-', lw=2, marker='s', label='Classical ECC: O(n)')
    
    # Format axes
    ax.set_xlabel('Key Size (bits)')
    ax.set_ylabel('Computational Overhead')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add title
    ax.set_title('Overhead Comparison: Q-SHE vs Classical ECC')
    
    plt.tight_layout()
    return fig
