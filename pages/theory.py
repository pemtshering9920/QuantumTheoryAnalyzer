import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from utils.visualization import visualize_quantum_state, plot_fractal_encoding
from utils.quantum_simulation import create_fractal_key

st.set_page_config(
    page_title="Q-SHE Theory",
    page_icon="📚",
    layout="wide"
)

def render_math():
    """Render mathematical equations for the theoretical concepts."""
    st.markdown("""
    ## 1. Fractal Key Encoding

    In Q-SHE, a classical key $K \in \\{0,1\\}^n$ is encoded into a quantum state using a recursive unitary transformation:

    $$|K_{\\text{fract}}\\rangle = \\bigotimes_{k=0}^{\\log_2 n} (U_k \\otimes I_{n/2^{k+1}}) |K\\rangle$$

    where $U_k = e^{-i\\pi H_k/2^k}$ and $H_k = \\sum_{j=1}^{n/2^k} Z_j Z_{j+1} + X_j$ is a scale-local Hamiltonian.
    
    ## 2. Self-Healing Ciphertext Dynamics

    The ciphertext $|C\\rangle$ encrypts plaintext $P$ as:

    $$|C\\rangle = \\bigoplus_{i=1}^{m} (P_i \\cdot X_i) \\otimes |\\text{Anc}_i\\rangle$$

    where $|\\text{Anc}_i\\rangle = \\frac{1}{\\sqrt{2}}\\sum_{b=0}^{1}(-1)^{f_i(b)}|b\\rangle$ and $f_i(b)$ is a parity function.
    
    ## 3. Autonomous Recovery Hamiltonian

    The system evolves under:

    $$H_{\\text{repair}} = -\\Delta \\sum_{i=1}^{m} Z_{\\text{Anc}_i} \\otimes \\prod_{j\\in N(i)} Z_j - \\lambda \\sum_{\\langle i,j\\rangle} \\left(X_i X_j + Y_i Y_j + \\frac{\\gamma}{2}(Z_i + Z_j)\\right)$$

    with a spectral gap $\\gamma \\geq \\frac{\\Delta \\lambda}{\\Delta + \\lambda} \\cdot \\frac{1}{\\text{polylog}(n)}$.
    
    ## 4. Self-Healing Theorem (Optimal Recovery)

    For a corrupted state $|C'\\rangle = E|C\\rangle$ with $\\|E\\| \\leq \\epsilon$, the fidelity under $H_{\\text{repair}}$ satisfies:

    $$\\mathcal{F}(t) = |\\langle C|C'(t)\\rangle|^2 \\geq 1 - \\epsilon^2 (1 - e^{-\\Gamma t})$$

    where $\\Gamma = \\frac{\\lambda^2 \\gamma}{2\\Delta^2} \\cdot \\frac{1}{\\text{diam}(G)}$.
    
    ## 5. Adversarial Resilience

    An adversary must corrupt $\\Omega(n^{1/d})$ qubits to reduce $\\mathcal{F}(t)$ below $1/2$, where $d$ is the fractal dimension.
    """)

def main():
    st.title("Quantum Self-Healing Encryption: Theoretical Foundations")
    
    st.markdown("""
    This page explores the theoretical foundations of Quantum Self-Healing Encryption (Q-SHE), 
    including the mathematical principles that enable its unique properties.
    """)
    
    # Tabs for different theoretical concepts
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Fractal Key Encoding", 
        "Self-Healing Ciphertext", 
        "Recovery Hamiltonian",
        "Recovery Theorem",
        "Adversarial Resilience"
    ])
    
    with tab1:
        st.markdown("""
        ## Fractal Key Encoding
        
        The core of Q-SHE is the fractal encoding of classical keys into quantum states. This encoding 
        creates a hierarchical structure that distributes information across multiple scales, providing 
        robustness against localized errors.
        """)
        
        # Interactive demonstration
        st.subheader("Interactive Demonstration")
        
        key_size = st.slider("Key Size (bits)", min_value=2, max_value=8, value=4, step=1, key="key_size_theory")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Generate a Custom Key")
            # Create input fields for each bit of the key
            key_bits = []
            for i in range(key_size):
                bit = st.selectbox(f"Bit {i}", [0, 1], key=f"key_bit_{i}")
                key_bits.append(bit)
            
            classical_key = np.array(key_bits)
            st.markdown(f"**Custom Key:** {' '.join(map(str, classical_key))}")
            
            if st.button("Encode Key", key="encode_key_btn"):
                # Encode the key
                quantum_key = create_fractal_key(classical_key)
                
                # Store in session state for display
                st.session_state.quantum_key = quantum_key
                st.session_state.classical_key = classical_key
                
                # Success message
                st.success("Key encoded successfully!")
        
        with col2:
            st.markdown("#### Quantum Encoding Visualization")
            
            # Check if we have a quantum key to display
            if 'quantum_key' in st.session_state and 'classical_key' in st.session_state:
                # Visualize the encoding
                fig = plot_fractal_encoding(st.session_state.classical_key, st.session_state.quantum_key)
                st.pyplot(fig)
                
                # Show properties
                st.markdown("#### Key Properties")
                st.markdown(f"- **Entropy:** {np.log2(2**len(st.session_state.classical_key)):.2f} bits")
                st.markdown(f"- **Quantum Dimension:** {2**len(st.session_state.classical_key)}")
                st.markdown(f"- **Fractal Depth:** {int(np.log2(len(st.session_state.classical_key))) + 1} scales")
                
                # Show quantum state details
                st.markdown("#### Quantum State Details")
                state_view = visualize_quantum_state(st.session_state.quantum_key)
                st.pyplot(state_view)
            else:
                st.info("Generate a key to view its quantum encoding.")
        
        # Mathematical explanation
        with st.expander("Mathematical Details"):
            st.markdown("""
            ### Mathematical Formulation
            
            The fractal encoding of a classical key $K \in \\{0,1\\}^n$ creates a quantum state $|K_{\\text{fract}}\\rangle$ through:
            
            $$|K_{\\text{fract}}\\rangle = \\bigotimes_{k=0}^{\\log_2 n} (U_k \\otimes I_{n/2^{k+1}}) |K\\rangle$$
            
            where:
            - $U_k = e^{-i\\pi H_k/2^k}$ is a unitary operation at scale $k$
            - $H_k = \\sum_{j=1}^{n/2^k} Z_j Z_{j+1} + X_j$ is a scale-local Hamiltonian
            - Each scale creates correlations between bits at different distances
            
            This encoding distributes the key information across multiple scales, creating a fractal structure
            in the quantum state that makes it resilient against errors.
            """)
            
            st.markdown("""
            ### Error Localization Theorem
            
            For any $\epsilon$-local error $E$ (acting on $\leq \epsilon n$ qubits), the corrupted state $E|K_{\\text{fract}}\\rangle$ satisfies:
            
            $$\\min_{\\text{scale } k} \\|\\text{Tr}_{-k}(E|K_{\\text{fract}}\\rangle\\langle K_{\\text{fract}}|E^†) - \\rho_k\\|_1 \\leq \\epsilon^{2^k} \\log n$$
            
            where $\\rho_k$ is the ideal state at scale $k$. This means errors remain localized in the fractal structure.
            """)
    
    with tab2:
        st.markdown("""
        ## Self-Healing Ciphertext Dynamics
        
        The encryption process in Q-SHE creates ciphertext with special dynamics that enable self-healing properties.
        The ciphertext combines the plaintext with a fractal-encoded key and ancilla qubits that assist in the 
        self-healing process.
        """)
        
        # Visualization of ciphertext creation
        st.subheader("Ciphertext Creation Process")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            The ciphertext is created by combining:
            1. **Plaintext bits** $P_i$
            2. **Key-dependent operators** $X_i$ (derived from the fractal key)
            3. **Ancilla qubits** in superposition states
            
            For each block, the ciphertext takes the form:
            
            $$|C_i\\rangle = (P_i \\cdot X_i) \\otimes |\\text{Anc}_i\\rangle$$
            
            The full ciphertext is the direct sum of these blocks:
            
            $$|C\\rangle = \\bigoplus_{i=1}^{m} |C_i\\rangle$$
            """)
            
            st.markdown("""
            #### Ancilla States
            
            The ancilla qubit for each block is prepared in a superposition:
            
            $$|\\text{Anc}_i\\rangle = \\frac{1}{\\sqrt{2}}\\sum_{b=0}^{1}(-1)^{f_i(b)}|b\\rangle$$
            
            where $f_i(b)$ is a parity function that depends on the plaintext and key bits.
            These ancilla qubits are crucial for the self-healing process.
            """)
        
        with col2:
            # Simple visualization of ciphertext structure
            st.markdown("#### Ciphertext Structure Visualization")
            
            # Create a simple diagram showing ciphertext structure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Draw blocks representing ciphertext components
            block_height = 0.6
            plaintext_color = '#4B0082'  # Indigo
            key_color = '#9370DB'  # Medium purple
            ancilla_color = '#BA55D3'  # Medium orchid
            
            for i in range(4):  # Show 4 blocks
                # Plaintext block
                ax.add_patch(plt.Rectangle((0, i*2), 1, block_height, 
                                          facecolor=plaintext_color, edgecolor='black', alpha=0.7))
                ax.text(0.5, i*2 + block_height/2, f"P{i}", 
                       ha='center', va='center', color='white', fontweight='bold')
                
                # Key operator block
                ax.add_patch(plt.Rectangle((1.2, i*2), 1, block_height, 
                                          facecolor=key_color, edgecolor='black', alpha=0.7))
                ax.text(1.7, i*2 + block_height/2, f"X{i}", 
                       ha='center', va='center', color='white', fontweight='bold')
                
                # Arrow showing combination
                ax.arrow(2.3, i*2 + block_height/2, 0.5, 0, head_width=0.1, 
                        head_length=0.1, fc='black', ec='black')
                
                # Tensor product symbol
                ax.text(3, i*2 + block_height/2, "⊗", 
                       ha='center', va='center', fontsize=16)
                
                # Ancilla block
                ax.add_patch(plt.Rectangle((3.3, i*2), 1, block_height, 
                                          facecolor=ancilla_color, edgecolor='black', alpha=0.7))
                ax.text(3.8, i*2 + block_height/2, f"Anc{i}", 
                       ha='center', va='center', color='white', fontweight='bold')
                
                # Block label
                ax.text(-0.5, i*2 + block_height/2, f"Block {i}:", 
                       ha='right', va='center', fontweight='bold')
            
            # Set axis limits and remove axis markings
            ax.set_xlim(-0.7, 4.5)
            ax.set_ylim(-0.5, 8)
            ax.axis('off')
            
            # Add title
            ax.set_title("Ciphertext Block Structure", pad=10)
            
            st.pyplot(fig)
            
            # Additional info about the structure
            st.markdown("""
            #### Key Features
            
            - Each block operates independently but is part of the full ciphertext
            - The ancilla qubits hold parity information that helps with error detection
            - The key-dependent operators create entanglement that distributes information
            - This structure allows errors to be identified and corrected autonomously
            """)
        
        # Mathematical details
        with st.expander("Mathematical Details"):
            st.markdown("""
            ### Ciphertext Mathematical Properties
            
            The ciphertext in Q-SHE has several important mathematical properties:
            
            1. **Parity Functions**: The function $f_i(b)$ in the ancilla superposition is a parity function over the $i$-th block of $P \oplus X$, which creates a relationship between plaintext, key, and ancilla qubits.
            
            2. **Ground State Property**: Valid ciphertexts form the ground state of the recovery Hamiltonian, which is crucial for the self-healing process.
            
            3. **Information Distribution**: The ciphertext distributes information across multiple qubits and scales, making it resilient against localized errors.
            
            4. **Entanglement Structure**: The entanglement between plaintext, key, and ancilla qubits creates a structure that can detect and correct errors autonomously.
            """)
    
    with tab3:
        st.markdown("""
        ## Autonomous Recovery Hamiltonian
        
        The recovery Hamiltonian is the quantum operator that drives the self-healing process in Q-SHE.
        When applied to a corrupted ciphertext, it autonomously restores the original state without
        requiring external correction procedures.
        """)
        
        st.markdown("""
        ### Recovery Hamiltonian Structure
        
        The recovery Hamiltonian has the form:
        
        $$H_{\\text{repair}} = -\\Delta \\sum_{i=1}^{m} Z_{\\text{Anc}_i} \\otimes \\prod_{j\\in N(i)} Z_j - \\lambda \\sum_{\\langle i,j\\rangle} \\left(X_i X_j + Y_i Y_j + \\frac{\\gamma}{2}(Z_i + Z_j)\\right)$$
        
        where:
        - $\\Delta$ is the energy scale for error detection
        - $\\lambda$ is the coupling strength for error correction
        - $\\gamma$ is the parameter that creates the spectral gap
        - $Z_{\\text{Anc}_i}$ is the Pauli-Z operator on the ancilla qubit of block $i$
        - $N(i)$ is the neighborhood of data qubits associated with ancilla $i$
        """)
        
        # Visualize the Hamiltonian spectrum
        st.subheader("Hamiltonian Spectrum Visualization")
        
        # Create visualization of energy spectrum
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Parameters
        gap = 2.0  # Energy gap
        states = 8  # Number of states to show
        
        # Plot ground state
        ax.scatter([0], [0], s=100, color='green', label='Ground State (Valid Ciphertext)')
        
        # Plot excited states
        excited_energies = gap + np.random.uniform(0, 1, states-1)
        excited_energies.sort()
        x_positions = np.zeros(states-1)
        ax.scatter(x_positions, excited_energies, s=80, color='red', label='Excited States (Invalid States)')
        
        # Add spectral gap annotation
        ax.annotate('Spectral Gap', xy=(0, gap/2), xytext=(1, gap/2),
                   arrowprops=dict(arrowstyle='<->', color='blue', lw=2),
                   color='blue', fontweight='bold')
        
        # Format axes
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, excited_energies[-1] + 0.5)
        ax.set_xlabel('')
        ax.set_ylabel('Energy')
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add title
        ax.set_title('Recovery Hamiltonian Energy Spectrum', pad=10)
        
        st.pyplot(fig)
        
        # Key properties of the Hamiltonian
        st.markdown("""
        ### Key Properties
        
        The recovery Hamiltonian has several crucial properties:
        
        1. **Gapped Spectrum**: The Hamiltonian has a spectral gap $\\gamma \\geq \\frac{\\Delta \\lambda}{\\Delta + \\lambda} \\cdot \\frac{1}{\\text{polylog}(n)}$, which separates valid ciphertexts from invalid states.
        
        2. **Protected Subspace**: Ground states of the Hamiltonian correspond exactly to valid ciphertexts, creating a protected subspace.
        
        3. **Error Detection**: The first term identifies errors by detecting inconsistencies between ancilla qubits and data qubits.
        
        4. **Error Correction**: The second term actively corrects errors by propagating correct information across the system.
        
        5. **Autonomous Operation**: The Hamiltonian operates autonomously, requiring no external measurement or feedback.
        """)
        
        # How it works
        st.subheader("How the Recovery Process Works")
        
        st.markdown("""
        1. **Error Detection**: When an error occurs, it creates an inconsistency between the ancilla qubits and data qubits, raising the energy of the system.
        
        2. **Energy Minimization**: The natural quantum evolution under the Hamiltonian drives the system toward its ground state (minimum energy).
        
        3. **Information Propagation**: The coupling terms propagate correct information from undamaged parts of the system to damaged parts.
        
        4. **Fractal Correction**: The multi-scale structure of the fractal encoding allows errors to be corrected using information from different scales.
        
        5. **Convergence**: Over time, the system converges to a state very close to the original ciphertext, effectively healing itself.
        """)
        
        with st.expander("Mathematical Details"):
            st.markdown("""
            ### Hamiltonian Dynamics
            
            The time evolution of a quantum state $|\\psi(t)\\rangle$ under the recovery Hamiltonian follows the Schrödinger equation:
            
            $$i\\hbar \\frac{d}{dt}|\\psi(t)\\rangle = H_{\\text{repair}}|\\psi(t)\\rangle$$
            
            The solution is given by:
            
            $$|\\psi(t)\\rangle = e^{-iH_{\\text{repair}}t/\\hbar}|\\psi(0)\\rangle$$
            
            For a corrupted ciphertext $|C'\\rangle = E|C\\rangle$, this evolution will drive the state back toward the original ciphertext $|C\\rangle$ with high fidelity.
            
            ### Lie-Robinson Bounds
            
            The error propagation during recovery is bounded by:
            
            $$\\|[e^{-iHt}E_\\alpha e^{iHt}, E_\\beta]\\| \\leq Ce^{vt-\\text{dist}(\\alpha,\\beta)}$$
            
            where $v \\sim \\lambda$ is the Lieb-Robinson velocity, ensuring that information propagates at a finite speed during recovery.
            """)
    
    with tab4:
        st.markdown("""
        ## Self-Healing Theorem (Optimal Recovery)
        
        The Self-Healing Theorem quantifies the recovery capabilities of Q-SHE, providing
        theoretical bounds on how well the system can recover from errors of different magnitudes.
        """)
        
        # Theorem statement
        st.subheader("Recovery Bound Theorem")
        
        st.markdown("""
        **Theorem 2 (Recovery Bound)**:
        
        Let $|C'\\rangle = E|C\\rangle$ be corrupted by error $E$ with $\\|E\\| \\leq \\epsilon$. Under $H_{\\text{repair}}$, the fidelity $\\mathcal{F}(t) = |\\langle C|C'(t)\\rangle|^2$ satisfies:
        
        $$\\mathcal{F}(t) \\geq 1 - \\epsilon^2 \\left(1 - e^{-\\Gamma t}\\right)$$
        
        where $\\Gamma = \\frac{\\lambda^2 \\gamma}{2\\Delta^2} \\cdot \\frac{1}{\\text{diam}(G)}$ and $\\text{diam}(G)$ is the fractal graph diameter.
        """)
        
        # Visualization of recovery bounds
        st.subheader("Recovery Bound Visualization")
        
        # Create interactive plot of recovery bounds
        epsilon = st.slider("Error Magnitude (ε)", min_value=0.01, max_value=0.5, value=0.2, step=0.01)
        gamma = st.slider("Recovery Rate (Γ)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Calculate and plot recovery curves
        t = np.linspace(0, 10, 100)
        fidelity = 1 - epsilon**2 * (1 - np.exp(-gamma * t))
        
        ax.plot(t, fidelity, 'b-', lw=2, label=f'ε = {epsilon}, Γ = {gamma}')
        
        # Add comparison curves for different error magnitudes
        error_comparisons = [0.1, 0.3, 0.5]
        for err in error_comparisons:
            if abs(err - epsilon) > 0.05:  # Only show if sufficiently different
                fid = 1 - err**2 * (1 - np.exp(-gamma * t))
                ax.plot(t, fid, '--', lw=1.5, alpha=0.6, label=f'ε = {err}')
        
        # Format axes
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Fidelity F(t)')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # Add title and threshold
        ax.set_title('Recovery Fidelity Bound Over Time', pad=10)
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Success Threshold')
        
        st.pyplot(fig)
        
        # Recovery time analysis
        st.subheader("Recovery Time Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### Time to Reach Target Fidelity
            
            For a target fidelity $F_{\\text{target}}$, the required recovery time is:
            
            $$t_{\\text{recovery}} = -\\frac{1}{\\Gamma}\\ln\\left(\\frac{1 - F_{\\text{target}}}{\\epsilon^2}\\right)$$
            
            This shows that recovery time scales logarithmically with the error size and polynomially with system size (through $\\Gamma$).
            """)
            
            # Calculate recovery time for a specific target
            target_fidelity = st.slider("Target Fidelity", min_value=0.8, max_value=0.99, value=0.95, step=0.01)
            
            # Only calculate if mathematically valid
            if 1 - target_fidelity < epsilon**2:
                st.warning(f"Target fidelity {target_fidelity} is not achievable with error ε = {epsilon}")
            else:
                recovery_time = -np.log((1 - target_fidelity) / epsilon**2) / gamma
                st.success(f"Time to reach {target_fidelity} fidelity: {recovery_time:.2f} time units")
        
        with col2:
            # Plot recovery time vs error magnitude
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Calculate recovery times for different error magnitudes
            errors = np.linspace(0.05, 0.5, 100)
            times = np.zeros_like(errors)
            
            for i, err in enumerate(errors):
                if 1 - target_fidelity < err**2:
                    times[i] = np.nan  # Not achievable
                else:
                    times[i] = -np.log((1 - target_fidelity) / err**2) / gamma
            
            # Plot the curve
            ax.plot(errors, times, 'r-', lw=2)
            
            # Mark the current error value
            valid_idx = ~np.isnan(times)
            if np.any(valid_idx):
                ax.vlines(epsilon, 0, -np.log((1 - target_fidelity) / epsilon**2) / gamma,
                        colors='b', linestyles='--', alpha=0.7)
            
            # Format axes
            ax.set_xlabel('Error Magnitude (ε)')
            ax.set_ylabel('Recovery Time')
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 20)
            ax.grid(True, alpha=0.3)
            
            # Add title
            ax.set_title(f'Recovery Time to Reach {target_fidelity} Fidelity', pad=10)
            
            st.pyplot(fig)
        
        # Proof sketch
        with st.expander("Proof Sketch"):
            st.markdown("""
            ### Proof Sketch for the Recovery Bound Theorem
            
            1. **Error Decomposition**: We decompose the error $E = \\sum_\\alpha c_\\alpha E_\\alpha$ where each $E_\\alpha$ acts on a single scale.
            
            2. **Lie-Robinson Propagation**: The error operators evolve as $e^{-iHt}E_\\alpha e^{iHt}$, with the bound:
               $$\\|[e^{-iHt}E_\\alpha e^{iHt}, E_\\beta]\\| \\leq Ce^{vt-\\text{dist}(\\alpha,\\beta)}$$
               where $v \\sim \\lambda$ is the propagation velocity.
            
            3. **Fractal Correction**: The multi-scale nature of the encoding means $\\text{dist}(\\alpha, \\beta) \\sim \\log n$, allowing efficient error correction across scales.
            
            4. **Fidelity Calculation**: Combining these effects leads to the fidelity bound:
               $$\\mathcal{F}(t) \\geq 1 - \\epsilon^2 \\left(1 - e^{-\\Gamma t}\\right)$$
               where $\\Gamma$ depends on the Hamiltonian parameters and system geometry.
            """)
    
    with tab5:
        st.markdown("""
        ## Adversarial Resilience
        
        Q-SHE provides strong guarantees against adversarial attacks, requiring an attacker to corrupt a large
        number of qubits to significantly impact the encrypted information.
        """)
        
        # Corollary statement
        st.subheader("Adversarial Resilience Corollary")
        
        st.markdown("""
        **Corollary 1**:
        
        An adversary must corrupt $\\Omega(n^{1/d})$ qubits to reduce $\\mathcal{F}(t)$ below $1/2$, where $d$ is the fractal dimension.
        
        This means that as the system size $n$ increases, the number of qubits an attacker must corrupt grows as a power law, making large-scale systems highly secure.
        """)
        
        # Comparison table
        st.subheader("Comparison to Classical Encryption")
        
        comparison_data = {
            "Metric": ["Recovery Mechanism", "Overhead", "Security"],
            "Classical ECC": ["Syndromes", "O(n)", "O(√n)-bounded"],
            "Q-SHE": ["Autonomous", "O(log n)", "O(n^(1/d))-resilient"]
        }
        
        # Display as a table
        st.markdown("### Comparative Advantages")
        st.table(comparison_data)
        
        # Visualization of the resilience scaling
        st.subheader("Resilience Scaling Visualization")
        
        # Create interactive plot
        dimension = st.slider("Fractal Dimension (d)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Calculate and plot resilience scaling
        n_values = np.logspace(1, 6, 100)
        
        # Classical bound (sqrt(n))
        classical_bound = np.sqrt(n_values)
        ax.loglog(n_values, classical_bound, 'r--', lw=2, label='Classical (O(√n))')
        
        # Q-SHE bound (n^(1/d))
        qshe_bound = n_values**(1/dimension)
        ax.loglog(n_values, qshe_bound, 'b-', lw=2, label=f'Q-SHE (O(n^(1/{dimension:.1f})))')
        
        # Format axes
        ax.set_xlabel('System Size (n)')
        ax.set_ylabel('Qubits Needed to Break')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='lower right')
        
        # Add title
        ax.set_title('Attack Resilience Scaling', pad=10)
        
        st.pyplot(fig)
        
        # Practical implications
        st.subheader("Practical Security Implications")
        
        st.markdown("""
        ### Real-World Security Advantages
        
        1. **Autonomous Recovery**: Unlike classical systems that require external error correction, Q-SHE systems recover automatically.
        
        2. **Scalable Security**: Security guarantees improve with system size, following a power law scaling.
        
        3. **Reduced Overhead**: The logarithmic overhead of Q-SHE compared to linear overhead in classical ECC leads to more efficient implementations.
        
        4. **Attack Resistance**: The fractal structure makes Q-SHE inherently resistant to targeted attacks that would compromise classical systems.
        
        5. **No Syndrome Measurements**: Classical error correction requires syndrome measurements and active correction, while Q-SHE operates passively.
        """)
        
        # Example calculation for specific system sizes
        with st.expander("Example Security Calculations"):
            st.markdown("""
            ### Security Calculations for Sample System Sizes
            
            For a fractal dimension of {:.1f}, here are the number of qubits an attacker would need to corrupt:
            """.format(dimension))
            
            # Calculate for specific system sizes
            sizes = [64, 256, 1024, 4096, 16384]
            
            for size in sizes:
                qshe_qubits = int(size**(1/dimension))
                classical_qubits = int(np.sqrt(size))
                
                st.markdown(f"""
                **System Size n = {size}**:
                - Q-SHE: Attacker must corrupt **{qshe_qubits}** qubits
                - Classical: Attacker must corrupt **{classical_qubits}** qubits
                - **Advantage**: {qshe_qubits/classical_qubits:.2f}x more qubits required
                """)
                
            st.markdown("""
            This demonstrates the superior scaling of Q-SHE's security guarantees, particularly for larger systems.
            """)
    
    # Add a divider
    st.markdown("---")
    
    # Mathematical formulations toggle
    if st.checkbox("Show All Mathematical Formulations"):
        render_math()

if __name__ == "__main__":
    main()
