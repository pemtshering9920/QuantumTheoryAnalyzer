import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from utils.quantum_simulation import (
    create_fractal_key,
    simulate_ciphertext_dynamics,
    apply_error,
    create_recovery_hamiltonian,
    simulate_recovery_hamiltonian
)
from utils.visualization import (
    visualize_quantum_state,
    plot_recovery_fidelity,
    plot_bloch_sphere
)

st.set_page_config(
    page_title="Q-SHE Error Correction",
    page_icon="🛠️",
    layout="wide"
)

def main():
    st.title("Error Correction in Quantum Self-Healing Encryption")
    
    st.markdown("""
    This interactive demonstration shows how Q-SHE automatically corrects errors that would corrupt 
    classical encryption systems. Explore different error types, error rates, and recovery mechanisms.
    """)
    
    # Error introduction and correction demo
    st.header("Interactive Error Correction Demo")
    
    # Parameters for the simulation
    col1, col2 = st.columns(2)
    
    with col1:
        plaintext_length = st.slider(
            "Plaintext Length (bits)", 
            min_value=2, 
            max_value=8, 
            value=4,
            help="Length of the plaintext in bits"
        )
        
        error_type = st.selectbox(
            "Error Type",
            ["Random Bit Flips", "Localized Burst Error", "Targeted Attack"],
            help="Type of error to introduce to the encrypted data"
        )
    
    with col2:
        error_rate = st.slider(
            "Error Rate (%)", 
            min_value=5, 
            max_value=50, 
            value=20,
            help="Percentage of qubits affected by errors"
        ) / 100
        
        recovery_time = st.slider(
            "Recovery Time (t)", 
            min_value=0.0, 
            max_value=10.0, 
            value=5.0, 
            step=0.5,
            help="Time to run the recovery Hamiltonian"
        )
    
    # Generate random plaintext for demonstration
    if 'plaintext' not in st.session_state:
        st.session_state.plaintext = np.random.randint(0, 2, size=plaintext_length)
    
    # Button to regenerate plaintext
    if st.button("Generate New Plaintext"):
        st.session_state.plaintext = np.random.randint(0, 2, size=plaintext_length)
        st.session_state.pop('simulation_results', None)
    
    # Display current plaintext
    plaintext_str = ''.join(map(str, st.session_state.plaintext))
    st.markdown(f"**Current Plaintext:** {plaintext_str}")
    
    # Run simulation button
    if st.button("Run Error Correction Simulation"):
        with st.spinner("Simulating error correction process..."):
            # Run the simulation with custom error application based on selected type
            initial_state, corrupted_state, recovered_state, fidelity_over_time = simulate_error_correction(
                st.session_state.plaintext,
                error_rate,
                recovery_time,
                error_type
            )
            
            # Store results in session state
            st.session_state.simulation_results = {
                'initial_state': initial_state,
                'corrupted_state': corrupted_state,
                'recovered_state': recovered_state,
                'fidelity_over_time': fidelity_over_time,
                'error_rate': error_rate,
                'recovery_time': recovery_time,
                'error_type': error_type
            }
            
            # Calculate key metrics
            final_fidelity = fidelity_over_time[-1]
            initial_fidelity = 1.0  # By definition
            corruption_fidelity = qt.fidelity(initial_state, corrupted_state)
            
            # Save result to database
            from utils.database import save_simulation_result
            result_id = save_simulation_result(
                simulation_type="error_correction",
                key_size=len(st.session_state.plaintext),
                error_rate=error_rate,
                recovery_time=recovery_time,
                initial_fidelity=initial_fidelity,
                corrupted_fidelity=corruption_fidelity,
                recovered_fidelity=final_fidelity,
                error_type=error_type,
                description=f"Error correction simulation with {error_type} errors",
                parameters={
                    "plaintext": st.session_state.plaintext.tolist(),
                    "time_steps": len(fidelity_over_time)
                }
            )
            
            # Success message
            if final_fidelity > 0.9:
                st.success(f"Simulation complete! Recovery successful with fidelity {final_fidelity:.4f} (Result ID: {result_id})")
            elif final_fidelity > 0.5:
                st.warning(f"Simulation complete! Partial recovery with fidelity {final_fidelity:.4f} (Result ID: {result_id})")
            else:
                st.error(f"Simulation complete! Recovery failed with fidelity {final_fidelity:.4f} (Result ID: {result_id})")
    
    # Display simulation results if available
    if 'simulation_results' in st.session_state:
        results = st.session_state.simulation_results
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Recovery Process", "Quantum States", "Analysis"])
        
        with tab1:
            st.subheader("Recovery Process Visualization")
            
            # Plot recovery fidelity over time
            fig = plot_recovery_fidelity(results['fidelity_over_time'], results['recovery_time'])
            st.pyplot(fig)
            
            # Display recovery metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                initial_fidelity = 1.0  # By definition
                st.metric(
                    label="Initial Fidelity",
                    value=f"{initial_fidelity:.4f}",
                    delta=None
                )
            
            with col2:
                # Calculate corruption fidelity
                corruption_fidelity = qt.fidelity(results['initial_state'], results['corrupted_state'])
                st.metric(
                    label="After Error",
                    value=f"{corruption_fidelity:.4f}",
                    delta=f"{corruption_fidelity - initial_fidelity:.4f}",
                    delta_color="inverse"
                )
            
            with col3:
                final_fidelity = results['fidelity_over_time'][-1]
                st.metric(
                    label="After Recovery",
                    value=f"{final_fidelity:.4f}",
                    delta=f"{final_fidelity - corruption_fidelity:.4f}",
                    delta_color="normal"
                )
            
            # Recovery information
            recovery_status = "Success" if final_fidelity > 0.9 else "Partial" if final_fidelity > 0.5 else "Failed"
            improvement = ((final_fidelity - corruption_fidelity) / (1 - corruption_fidelity) * 100) if corruption_fidelity < 1 else 0
            
            st.markdown(f"""
            ### Recovery Summary
            
            - **Error Type:** {results['error_type']}
            - **Error Rate:** {results['error_rate']:.2%}
            - **Recovery Status:** {recovery_status}
            - **Recovery Improvement:** {improvement:.1f}%
            - **Theoretical Bound:** {1 - results['error_rate']**2 * (1 - np.exp(-0.5 * results['recovery_time'])):.4f}
            """)
        
        with tab2:
            st.subheader("Quantum State Visualization")
            
            # Create three columns for state visualizations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Initial Ciphertext State")
                fig = visualize_quantum_state(results['initial_state'])
                st.pyplot(fig)
            
            with col2:
                st.markdown(f"#### Corrupted State ({results['error_type']})")
                fig = visualize_quantum_state(results['corrupted_state'])
                st.pyplot(fig)
            
            with col3:
                st.markdown("#### Recovered State")
                fig = visualize_quantum_state(results['recovered_state'])
                st.pyplot(fig)
            
            # If the state is small enough, show Bloch sphere representation
            if plaintext_length <= 2:
                st.markdown("#### Bloch Sphere Representation")
                
                # Prepare states for Bloch sphere (for small states only)
                if results['initial_state'].dims[0][0] <= 4:  # Only for small states
                    # Extract the first qubit state for Bloch sphere visualization
                    states_for_bloch = []
                    labels = []
                    
                    # Add initial state
                    states_for_bloch.append(results['initial_state'])
                    labels.append("Initial")
                    
                    # Add corrupted state
                    states_for_bloch.append(results['corrupted_state'])
                    labels.append("Corrupted")
                    
                    # Add recovered state
                    states_for_bloch.append(results['recovered_state'])
                    labels.append("Recovered")
                    
                    # Plot Bloch sphere
                    fig = plot_bloch_sphere(states_for_bloch, labels)
                    st.pyplot(fig)
                else:
                    st.info("Bloch sphere visualization available only for small quantum states (≤ 2 qubits)")
            else:
                st.info("Bloch sphere visualization available only for small quantum states (≤ 2 qubits)")
        
        with tab3:
            st.subheader("Error Correction Analysis")
            
            # Compare actual recovery with theoretical bounds
            st.markdown("#### Comparison with Theoretical Bounds")
            
            # Calculate theoretical recovery curve
            times = np.linspace(0, results['recovery_time'], 100)
            epsilon = results['error_rate']
            gamma = 0.5  # Simplified recovery rate parameter
            
            theoretical_fidelity = 1 - epsilon**2 * (1 - np.exp(-gamma * times))
            
            # Create plot comparing actual vs theoretical recovery
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Actual fidelity points
            actual_times = np.linspace(0, results['recovery_time'], len(results['fidelity_over_time']))
            ax.plot(actual_times, results['fidelity_over_time'], 'b-', marker='o', label='Actual Recovery', markersize=5)
            
            # Theoretical curve
            ax.plot(times, theoretical_fidelity, 'r--', label='Theoretical Bound')
            
            # Add threshold lines
            ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Success Threshold')
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Partial Recovery')
            
            # Format plot
            ax.set_xlabel('Time (t)')
            ax.set_ylabel('Fidelity')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title('Actual vs Theoretical Recovery')
            
            st.pyplot(fig)
            
            # Error localization visualization
            st.markdown("#### Error Localization")
            
            # Create a simplified visualization of error localization
            n_qubits = results['initial_state'].dims[0][0]
            
            fig, ax = plt.subplots(figsize=(10, 3))
            
            # Generate random "error locations" based on error rate
            np.random.seed(42)  # For consistent visualization
            qubit_positions = np.arange(n_qubits)
            error_locations = np.random.choice(n_qubits, size=int(n_qubits * results['error_rate']), replace=False)
            
            # Create error heatmap
            error_values = np.zeros(n_qubits)
            error_values[error_locations] = 1
            
            # Create a grid showing error locations
            ax.imshow([error_values], cmap='Reds', aspect='auto', interpolation='none')
            
            # Add scale indicators
            scale_positions = []
            scale_labels = []
            
            for k in range(int(np.log2(n_qubits)) + 1):
                scale_size = n_qubits // (2**k) if k > 0 else n_qubits
                if scale_size >= 1:
                    scale_positions.append(scale_size // 2)
                    scale_labels.append(f"Scale {k}")
            
            # Add scale indicators as vertical lines
            for pos in scale_positions[1:]:  # Skip the first (full system)
                ax.axvline(x=pos - 0.5, color='blue', linestyle='--', alpha=0.5)
            
            # Format plot
            ax.set_xticks(np.arange(n_qubits))
            ax.set_xticklabels([f"Q{i}" for i in range(n_qubits)])
            ax.set_yticks([])
            ax.set_title('Error Localization across Scales')
            
            # Add an annotation for each scale
            for i, (pos, label) in enumerate(zip(scale_positions, scale_labels)):
                ax.annotate(label, xy=(pos, -0.5), xytext=(pos, -0.3), ha='center',
                           arrowprops=dict(arrowstyle='->', color='blue'))
            
            st.pyplot(fig)
            
            # Theory explanation
            with st.expander("Theoretical Explanation"):
                st.markdown("""
                ### Error Localization in Q-SHE
                
                The fractal encoding in Q-SHE ensures that errors remain localized at different scales, 
                allowing for efficient correction:
                
                1. **Error Decomposition**: Errors are decomposed across different scales in the fractal structure
                
                2. **Scale-Local Correction**: The recovery Hamiltonian corrects errors at each scale independently
                
                3. **Information Propagation**: Correct information propagates from undamaged scales to damaged ones
                
                4. **Autonomy**: The entire process happens automatically under Hamiltonian evolution
                
                According to the Error Localization Theorem, for any ε-local error E (acting on ≤ εn qubits), the corrupted state satisfies:
                
                $$\\min_{\\text{scale } k} \\|\\text{Tr}_{-k}(E|K_{\\text{fract}}\\rangle\\langle K_{\\text{fract}}|E^†) - \\rho_k\\|_1 \\leq \\epsilon^{2^k} \\log n$$
                
                This means that errors remain localized at appropriate scales, allowing the recovery Hamiltonian to identify and correct them efficiently.
                """)
    
    # Types of errors section
    st.header("Types of Errors in Quantum Systems")
    
    st.markdown("""
    Quantum encryption systems face several types of errors that can corrupt the quantum state. 
    Q-SHE is designed to be resilient against these various error types:
    """)
    
    # Create tabs for different error types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Bit Flip Errors", 
        "Phase Errors", 
        "Depolarizing Errors", 
        "Erasure Errors"
    ])
    
    with tab1:
        st.markdown("""
        ### Bit Flip Errors
        
        **Description**: Analogous to classical bit flips, these errors flip a qubit from |0⟩ to |1⟩ or vice versa.
        
        **Mathematical Representation**: Bit flip errors are represented by the Pauli-X operator:
        
        $$X|0\\rangle = |1\\rangle, \\quad X|1\\rangle = |0\\rangle$$
        
        **Impact on Encryption**: Bit flips directly corrupt the encoded information, changing the logical value of qubits.
        
        **Q-SHE Resilience**: The fractal encoding in Q-SHE distributes information across multiple scales, making the system resilient against localized bit flips.
        """)
        
        # Simple visualization of bit flip
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        
        # Before bit flip
        ax[0].bar([0, 1], [0.9, 0.1], color='blue', alpha=0.7)
        ax[0].set_title("Before Bit Flip")
        ax[0].set_xticks([0, 1])
        ax[0].set_xticklabels(["|0⟩", "|1⟩"])
        ax[0].set_ylim(0, 1)
        
        # After bit flip
        ax[1].bar([0, 1], [0.1, 0.9], color='red', alpha=0.7)
        ax[1].set_title("After Bit Flip")
        ax[1].set_xticks([0, 1])
        ax[1].set_xticklabels(["|0⟩", "|1⟩"])
        ax[1].set_ylim(0, 1)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.markdown("""
        ### Phase Errors
        
        **Description**: Phase errors change the relative phase between quantum states, affecting superpositions.
        
        **Mathematical Representation**: Phase errors are represented by the Pauli-Z operator:
        
        $$Z|0\\rangle = |0\\rangle, \\quad Z|1\\rangle = -|1\\rangle$$
        
        **Impact on Encryption**: Phase errors can corrupt the quantum interference patterns essential for quantum encryption.
        
        **Q-SHE Resilience**: The recovery Hamiltonian in Q-SHE includes terms that detect and correct phase inconsistencies.
        """)
        
        # Simple visualization of phase error using a Bloch sphere-like representation
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        
        # Create simplified Bloch sphere representation
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Before phase error
        ax[0].plot(np.cos(theta), np.sin(theta), 'b-', alpha=0.3)
        ax[0].arrow(0, 0, 0.7, 0.7, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        ax[0].set_title("Before Phase Error")
        ax[0].set_xlim(-1.2, 1.2)
        ax[0].set_ylim(-1.2, 1.2)
        ax[0].set_aspect('equal')
        ax[0].grid(True, alpha=0.3)
        ax[0].axis('off')
        
        # After phase error
        ax[1].plot(np.cos(theta), np.sin(theta), 'b-', alpha=0.3)
        ax[1].arrow(0, 0, 0.7, -0.7, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax[1].set_title("After Phase Error")
        ax[1].set_xlim(-1.2, 1.2)
        ax[1].set_ylim(-1.2, 1.2)
        ax[1].set_aspect('equal')
        ax[1].grid(True, alpha=0.3)
        ax[1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.markdown("""
        ### Depolarizing Errors
        
        **Description**: Depolarizing errors represent a combination of bit flip and phase errors, pushing the quantum state toward a completely mixed state.
        
        **Mathematical Representation**: A depolarizing channel with error probability p transforms a state ρ as:
        
        $$\\mathcal{E}(\\rho) = (1-p)\\rho + \\frac{p}{3}(X\\rho X + Y\\rho Y + Z\\rho Z)$$
        
        **Impact on Encryption**: These are among the most damaging errors as they reduce the purity of quantum states.
        
        **Q-SHE Resilience**: The multi-scale structure of Q-SHE enables it to recover from depolarizing errors by preserving information redundantly across scales.
        """)
        
        # Simple visualization of depolarizing error
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        
        # Before depolarizing
        bars = ax[0].bar([0, 1, 2, 3], [0.85, 0.05, 0.05, 0.05], color='blue', alpha=0.7)
        ax[0].set_title("Pure State")
        ax[0].set_xticks([0, 1, 2, 3])
        ax[0].set_xticklabels(["|ψ⟩", "X|ψ⟩", "Y|ψ⟩", "Z|ψ⟩"])
        ax[0].set_ylim(0, 1)
        
        # After depolarizing
        ax[1].bar([0, 1, 2, 3], [0.4, 0.2, 0.2, 0.2], color='red', alpha=0.7)
        ax[1].set_title("Mixed State (Depolarized)")
        ax[1].set_xticks([0, 1, 2, 3])
        ax[1].set_xticklabels(["|ψ⟩", "X|ψ⟩", "Y|ψ⟩", "Z|ψ⟩"])
        ax[1].set_ylim(0, 1)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab4:
        st.markdown("""
        ### Erasure Errors
        
        **Description**: Erasure errors occur when qubits are completely lost or when their state becomes unknown.
        
        **Mathematical Representation**: An erasure channel replaces a qubit with a flagged error state |e⟩:
        
        $$\\mathcal{E}(|\\psi\\rangle) = \\sqrt{1-p}|\\psi\\rangle + \\sqrt{p}|e\\rangle$$
        
        **Impact on Encryption**: Erasures can create gaps in the quantum ciphertext, potentially breaking the structure.
        
        **Q-SHE Resilience**: The autonomous recovery in Q-SHE can reconstruct lost information from the remaining intact ciphertext due to its fractal redundancy.
        """)
        
        # Simple visualization of erasure
        fig, ax = plt.subplots(figsize=(8, 3))
        
        # Create a qubit array showing erasure
        n_qubits = 8
        qubit_states = np.ones(n_qubits)
        erased_positions = [2, 5]
        qubit_states[erased_positions] = 0
        
        # Plot qubits as circles with different colors
        for i in range(n_qubits):
            if i in erased_positions:
                circle = plt.Circle((i, 0), 0.3, color='white', ec='red', lw=2, alpha=0.3)
                ax.add_patch(circle)
                ax.text(i, 0, "?", ha='center', va='center', color='red')
            else:
                circle = plt.Circle((i, 0), 0.3, color='blue', alpha=0.7)
                ax.add_patch(circle)
                ax.text(i, 0, str(np.random.randint(0, 2)), ha='center', va='center', color='white')
        
        # Format plot
        ax.set_xlim(-0.5, n_qubits - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Erasure Errors: Qubits 2 and 5 are lost')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Recovery process explanations
    st.header("The Self-Healing Process")
    
    # Create tabs for different aspects of the recovery process
    tab1, tab2, tab3 = st.tabs([
        "Fractal Error Correction", 
        "Recovery Hamiltonian", 
        "Time Evolution"
    ])
    
    with tab1:
        st.markdown("""
        ### Fractal Error Correction
        
        The cornerstone of Q-SHE is its fractal structure that enables autonomous error correction:
        
        1. **Multi-Scale Encoding**: Information is encoded redundantly across multiple scales
        2. **Error Localization**: Errors remain localized to specific scales
        3. **Cross-Scale Correction**: Intact scales help correct corrupted scales
        4. **Hierarchical Recovery**: Correction proceeds from larger to smaller scales
        
        This fractal approach allows Q-SHE to correct errors without explicitly detecting their location or type.
        """)
        
        # Visualization of fractal correction
        st.markdown("#### Fractal Structure Visualization")
        
        # Create a visual representation of fractal scales
        n = 8  # Number of qubits
        scales = int(np.log2(n)) + 1
        
        fig, axes = plt.subplots(scales, 1, figsize=(10, 6), sharex=True)
        
        for k in range(scales):
            scale_size = n // (2**k) if k > 0 else n
            scale_data = np.zeros(n)
            
            for i in range(0, n, max(1, scale_size)):
                end = min(i + scale_size, n)
                scale_data[i:end] = 1
            
            # Add some variation for visualization purposes
            if k > 0:
                scale_data *= (scales - k) / scales
            
            axes[k].bar(range(n), scale_data, width=0.7, 
                       color='indigo', alpha=0.7)
            axes[k].set_ylabel(f"Scale {k}")
            axes[k].set_yticks([])
            
            # Highlight example error at this scale
            if k == 1:  # Example error at scale 1
                error_pos = 2
                error_width = 2
                axes[k].bar(range(error_pos, error_pos + error_width), 
                           scale_data[error_pos:error_pos + error_width] * 0,
                           width=0.7, color='red', alpha=0.5)
                
                # Show correction
                axes[k].annotate("Error", xy=(error_pos + error_width/2, 0.5), 
                               xytext=(error_pos + error_width/2 + 2, 0.5),
                               arrowprops=dict(arrowstyle='<-', color='red'))
                
                # Show how other scales help correct
                axes[0].annotate("Helps correct", xy=(error_pos + error_width/2, 0.5), 
                               xytext=(error_pos + error_width/2, 0.8),
                               arrowprops=dict(arrowstyle='->', color='green'))
        
        axes[-1].set_xlabel("Qubit Position")
        axes[-1].set_xticks(range(n))
        axes[-1].set_xticklabels([f"Q{i}" for i in range(n)])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add explanation of the visualization
        st.markdown("""
        The visualization above shows how information is distributed across different scales in the fractal structure. 
        When an error occurs at one scale (shown in red), information from other scales can help correct it.
        This multi-scale redundancy is what gives Q-SHE its self-healing properties.
        """)
    
    with tab2:
        st.markdown("""
        ### Recovery Hamiltonian
        
        The autonomous recovery in Q-SHE is driven by a specially designed Hamiltonian:
        
        $$H_{\\text{repair}} = -\\Delta \\sum_{i=1}^{m} Z_{\\text{Anc}_i} \\otimes \\prod_{j\\in N(i)} Z_j - \\lambda \\sum_{\\langle i,j\\rangle} \\left(X_i X_j + Y_i Y_j + \\frac{\\gamma}{2}(Z_i + Z_j)\\right)$$
        
        This Hamiltonian has two key components:
        
        1. **Error Detection Term**: The first term detects inconsistencies between ancilla qubits and data qubits
        
        2. **Error Correction Term**: The second term propagates correct information across the system
        
        Valid ciphertexts form the ground state of this Hamiltonian, so natural evolution drives corrupted states back toward valid ones.
        """)
        
        # Interactive demo of Hamiltonian parameters
        st.markdown("#### Hamiltonian Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta = st.slider("Δ (Energy Scale)", 0.5, 5.0, 2.0, 0.1, key="delta_param")
            st.markdown("""
            **Δ (Delta)**: Controls the energy penalty for inconsistencies between ancilla and data qubits. 
            Higher values make the system more sensitive to errors but can slow down correction.
            """)
        
        with col2:
            lambda_param = st.slider("λ (Coupling Strength)", 0.5, 3.0, 1.0, 0.1, key="lambda_param")
            st.markdown("""
            **λ (Lambda)**: Determines the strength of interactions that propagate information across the system.
            Higher values accelerate recovery but can cause overshooting if too large.
            """)
        
        with col3:
            gamma = st.slider("γ (Spectral Gap)", 0.1, 1.0, 0.5, 0.1, key="gamma_param")
            st.markdown("""
            **γ (Gamma)**: Controls the spectral gap that separates valid and invalid states.
            This parameter balances the correction of bit-flip vs. phase errors.
            """)
        
        # Visualize energy landscape
        st.markdown("#### Energy Landscape Visualization")
        
        # Create a simple visualization of the energy landscape
        # This is a simplified 2D projection of a high-dimensional landscape
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        # Create a potential well with minimum at (0,0)
        # Scale depth and width based on selected parameters
        Z = delta * (X**2 + Y**2) + lambda_param * np.sin(np.pi * X) * np.sin(np.pi * Y) + gamma * (np.cos(np.pi * X) + np.cos(np.pi * Y))
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)
        
        # Add a point for the ground state
        ax.scatter([0], [0], [Z[50, 50]], color='green', s=100, label='Ground State')
        
        # Add a point for a sample corrupted state
        error_x, error_y = 1.2, 0.8
        error_z = delta * (error_x**2 + error_y**2) + lambda_param * np.sin(np.pi * error_x) * np.sin(np.pi * error_y) + gamma * (np.cos(np.pi * error_x) + np.cos(np.pi * error_y))
        ax.scatter([error_x], [error_y], [error_z], color='red', s=100, label='Corrupted State')
        
        # Add a simple trajectory showing recovery
        trajectory_x = np.linspace(error_x, 0, 20)
        trajectory_y = np.linspace(error_y, 0, 20)
        trajectory_z = []
        
        for tx, ty in zip(trajectory_x, trajectory_y):
            tz = delta * (tx**2 + ty**2) + lambda_param * np.sin(np.pi * tx) * np.sin(np.pi * ty) + gamma * (np.cos(np.pi * tx) + np.cos(np.pi * ty))
            trajectory_z.append(tz)
        
        ax.plot(trajectory_x, trajectory_y, trajectory_z, 'b--', label='Recovery Path')
        
        # Format plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Energy')
        ax.set_title('Energy Landscape of Recovery Hamiltonian')
        ax.legend()
        
        st.pyplot(fig)
        
        # Add explanation of the energy landscape
        st.markdown("""
        The 3D visualization above represents a simplified energy landscape of the recovery Hamiltonian. 
        Valid ciphertexts (green point) sit at the minimum energy, while corrupted states (red point) have higher energy.
        
        The natural dynamics of the quantum system cause it to evolve toward lower energy states along the blue path,
        effectively healing the ciphertext without external intervention.
        
        **Key Properties:**
        
        - **Gapped Spectrum**: The energy difference between valid and invalid states (controlled by γ)
        - **Basin of Attraction**: The region around the ground state that will converge to it
        - **Recovery Path**: Trajectory followed during self-healing (affected by λ)
        """)
    
    with tab3:
        st.markdown("""
        ### Time Evolution
        
        The self-healing process in Q-SHE relies on the natural time evolution of quantum systems:
        
        $$|\\psi(t)\\rangle = e^{-iH_{\\text{repair}}t}|\\psi(0)\\rangle$$
        
        This evolution follows the Schrödinger equation:
        
        $$i\\frac{d}{dt}|\\psi(t)\\rangle = H_{\\text{repair}}|\\psi(t)\\rangle$$
        
        As time progresses, the corrupted state evolves toward the ground state of the Hamiltonian, which corresponds to the original ciphertext.
        """)
        
        # Interactive visualization of time evolution
        st.markdown("#### Time Evolution Visualization")
        
        # Create time slider for interactive visualization
        evolution_time = st.slider("Evolution Time", 0.0, 10.0, 0.0, 0.5, key="evolution_time")
        
        # Generate a simple animation-like sequence showing recovery over time
        # We'll use the existing simulation results if available, otherwise generate a simple model
        if 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results
            
            # Calculate fidelity at the selected time
            times = np.linspace(0, results['recovery_time'], len(results['fidelity_over_time']))
            current_fidelity_index = min(len(times) - 1, max(0, int(evolution_time / results['recovery_time'] * len(times))))
            current_fidelity = results['fidelity_over_time'][current_fidelity_index]
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot fidelity curve with current position
            ax1.plot(times, results['fidelity_over_time'], 'b-', lw=2)
            ax1.plot(times[current_fidelity_index], current_fidelity, 'ro', ms=10)
            ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Success Threshold')
            ax1.set_xlabel('Time (t)')
            ax1.set_ylabel('Fidelity')
            ax1.set_ylim(0, 1.05)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title('Recovery Fidelity Over Time')
            
            # Create a simplified state vector visualization
            # Linearly interpolate between corrupted and recovered state
            if evolution_time == 0:
                current_state = results['corrupted_state']
            elif evolution_time >= results['recovery_time']:
                current_state = results['recovered_state']
            else:
                # This is a simplified model - in reality, quantum evolution is more complex
                alpha = evolution_time / results['recovery_time']
                
                # Get probability amplitudes from both states
                if not results['corrupted_state'].isoper:
                    corrupted_amplitudes = results['corrupted_state'].full().flatten()
                    recovered_amplitudes = results['recovered_state'].full().flatten()
                else:
                    # If density matrices, use diagonals as approximate visualization
                    corrupted_amplitudes = np.sqrt(np.real(results['corrupted_state'].diag()))
                    recovered_amplitudes = np.sqrt(np.real(results['recovered_state'].diag()))
                
                # Interpolate (this is just for visualization, not physically accurate)
                interpolated = (1-alpha) * corrupted_amplitudes + alpha * recovered_amplitudes
                
                # Normalize
                interpolated = interpolated / np.linalg.norm(interpolated)
                
                # Create a qobj from the interpolated amplitudes
                dims = results['corrupted_state'].dims
                current_state = qt.Qobj(interpolated.reshape(-1, 1), dims=dims)
            
            # Visualize the current state
            if not current_state.isoper:
                rho = current_state * current_state.dag()
            else:
                rho = current_state
            
            probs = np.real(rho.diag())
            
            # Plot state probabilities
            x = np.arange(len(probs))
            ax2.bar(x, probs, width=0.6, color='indigo', alpha=0.7)
            ax2.set_xlabel('Basis State')
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            
            # If we have fewer than 16 states, label them in binary
            if len(probs) <= 16:
                binary_labels = [format(i, f'0{int(np.log2(len(probs)))}b') for i in range(len(probs))]
                ax2.set_xticks(x)
                ax2.set_xticklabels(binary_labels, rotation=45)
            else:
                # Otherwise, just show a few labels
                step = max(1, len(probs) // 8)
                ax2.set_xticks(x[::step])
                ax2.set_xticklabels([f"{i}" for i in x[::step]])
            
            ax2.set_title(f'Quantum State at t={evolution_time:.1f}')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display recovery metrics at current time
            st.markdown(f"""
            **Recovery Metrics at t={evolution_time:.1f}:**
            
            - Current Fidelity: {current_fidelity:.4f}
            - Recovery Status: {"Success" if current_fidelity > 0.9 else "In Progress" if current_fidelity > 0.5 else "Early Stage"}
            - Time to 90% Recovery: {find_recovery_time(times, results['fidelity_over_time'], 0.9) if any(f >= 0.9 for f in results['fidelity_over_time']) else "N/A"}
            """)
        else:
            # Create a simple model if no simulation results are available
            st.info("Run a simulation first to see detailed time evolution visualization.")
            
            # Generic recovery curve
            t = np.linspace(0, 10, 100)
            generic_fidelity = 0.4 + 0.6 * (1 - np.exp(-0.3 * t))
            
            current_index = min(len(t) - 1, max(0, int(evolution_time / 10 * len(t))))
            current_fidelity = generic_fidelity[current_index]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t, generic_fidelity, 'b-', lw=2)
            ax.plot(t[current_index], current_fidelity, 'ro', ms=10)
            ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Success Threshold')
            ax.set_xlabel('Time (t)')
            ax.set_ylabel('Fidelity (Generic Model)')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title('Generic Recovery Curve (Example Only)')
            
            st.pyplot(fig)
        
        # Add explanation of time evolution
        with st.expander("Details of Quantum Time Evolution"):
            st.markdown("""
            ### The Mathematics of Quantum Evolution
            
            The time evolution of quantum states is governed by the Schrödinger equation:
            
            $$i\\hbar \\frac{d}{dt}|\\psi(t)\\rangle = H|\\psi(t)\\rangle$$
            
            For time-independent Hamiltonians like our recovery Hamiltonian, the solution is:
            
            $$|\\psi(t)\\rangle = e^{-iHt/\\hbar}|\\psi(0)\\rangle = \\sum_n e^{-iE_nt/\\hbar}|E_n\\rangle\\langle E_n|\\psi(0)\\rangle$$
            
            where $|E_n\\rangle$ are the eigenstates of $H$ with eigenvalues $E_n$.
            
            In Q-SHE, this evolution has several important properties:
            
            1. **Energy Minimization**: The system evolves toward lower energy states
            2. **Ground State Attraction**: Valid ciphertexts form the ground state, so evolution naturally corrects errors
            3. **Lie-Robinson Bounds**: Information propagates at a bounded velocity, controlled by the parameter λ
            4. **Exponential Convergence**: Recovery fidelity approaches 1 exponentially in time
            
            The recovery time depends on the spectral gap γ and system size, with larger systems requiring longer healing times but providing greater security.
            """)

# Helper functions for the error correction simulations

def simulate_error_correction(plaintext, error_rate, recovery_time, error_type):
    """
    Simulate the error correction process with different types of errors.
    
    Args:
        plaintext: Binary array representing the plaintext
        error_rate: Fraction of qubits to corrupt
        recovery_time: Time to run the recovery Hamiltonian
        error_type: Type of error to introduce
        
    Returns:
        Tuple containing initial state, corrupted state, recovered state, and fidelity over time
    """
    # Generate random key for the plaintext
    key = np.random.randint(0, 2, size=len(plaintext))
    
    # Create initial ciphertext state
    initial_state = simulate_ciphertext_dynamics(plaintext, key)
    
    # Apply specific error type
    if error_type == "Random Bit Flips":
        corrupted_state = apply_random_bit_flips(initial_state, error_rate)
    elif error_type == "Localized Burst Error":
        corrupted_state = apply_localized_burst_error(initial_state, error_rate)
    elif error_type == "Targeted Attack":
        corrupted_state = apply_targeted_attack(initial_state, error_rate)
    else:
        # Default to standard error application
        corrupted_state = apply_error(initial_state, error_rate)
    
    # Create recovery Hamiltonian
    H_repair = create_recovery_hamiltonian(initial_state.dims[0][0])
    
    # Simulate time evolution
    n_steps = 50
    times = np.linspace(0, recovery_time, n_steps)
    result = qt.mesolve(H_repair, corrupted_state, times, [], [])
    
    # Get recovered state
    recovered_state = result.states[-1]
    
    # Calculate fidelity over time
    fidelity_over_time = [qt.fidelity(initial_state, state) for state in result.states]
    
    return initial_state, corrupted_state, recovered_state, fidelity_over_time

def apply_random_bit_flips(state, error_rate):
    """Apply random X (bit flip) errors to the quantum state."""
    n_qubits = state.dims[0][0]
    
    # Determine number of qubits to corrupt
    n_errors = max(1, int(n_qubits * error_rate))
    error_positions = np.random.choice(n_qubits, size=n_errors, replace=False)
    
    # Apply X (bit flip) errors
    corrupted_state = state
    for pos in error_positions:
        # Create X operator for this position
        op_list = [qt.identity(2)] * n_qubits
        op_list[pos] = qt.sigmax()
        error_operator = qt.tensor(op_list)
        
        # Apply the error
        corrupted_state = error_operator * corrupted_state
    
    return corrupted_state

def apply_localized_burst_error(state, error_rate):
    """Apply errors concentrated in a localized region (burst error)."""
    n_qubits = state.dims[0][0]
    
    # Determine burst size and location
    burst_size = max(1, int(n_qubits * error_rate))
    start_pos = np.random.randint(0, n_qubits - burst_size + 1)
    
    # Apply errors to consecutive qubits
    corrupted_state = state
    for pos in range(start_pos, start_pos + burst_size):
        # Choose a random Pauli error (X, Y, or Z)
        error_type = np.random.choice(['X', 'Y', 'Z'])
        
        if error_type == 'X':
            error_op = qt.sigmax()
        elif error_type == 'Y':
            error_op = qt.sigmay()
        else:
            error_op = qt.sigmaz()
        
        # Create the full error operator
        op_list = [qt.identity(2)] * n_qubits
        op_list[pos] = error_op
        error_operator = qt.tensor(op_list)
        
        # Apply the error
        corrupted_state = error_operator * corrupted_state
    
    return corrupted_state

def apply_targeted_attack(state, error_rate):
    """
    Apply a targeted attack that aims to maximize damage 
    by applying correlated errors to important qubits.
    """
    n_qubits = state.dims[0][0]
    
    # Determine number of qubits to corrupt
    n_errors = max(1, int(n_qubits * error_rate))
    
    # For a targeted attack, we focus on the first few qubits 
    # (In a real attack, this would target the most important qubits based on analysis)
    error_positions = range(min(n_errors, n_qubits))
    
    # Apply correlated errors
    corrupted_state = state
    
    # First apply Z errors to create phase issues
    for pos in error_positions:
        op_list = [qt.identity(2)] * n_qubits
        op_list[pos] = qt.sigmaz()
        error_operator = qt.tensor(op_list)
        corrupted_state = error_operator * corrupted_state
    
    # Then apply an entangling error across the first two qubits (if possible)
    if n_qubits >= 2 and n_errors >= 2:
        # Create a simple entangling error (like CNOT followed by Z)
        # This is simplified for demonstration
        op_list = [qt.identity(2)] * n_qubits
        op_list[0] = qt.sigmax()
        op_list[1] = qt.sigmaz()
        entangling_error = qt.tensor(op_list)
        corrupted_state = entangling_error * corrupted_state
    
    return corrupted_state

def find_recovery_time(times, fidelities, target_fidelity):
    """Find the time required to reach a target fidelity."""
    for i, fidelity in enumerate(fidelities):
        if fidelity >= target_fidelity:
            return f"{times[i]:.2f}"
    return "Not reached"

if __name__ == "__main__":
    main()
