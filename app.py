import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.quantum_simulation import (
    create_fractal_key, 
    simulate_ciphertext_dynamics,
    simulate_recovery_hamiltonian
)
from utils.visualization import (
    visualize_quantum_state,
    plot_recovery_fidelity,
    plot_fractal_encoding
)

st.set_page_config(
    page_title="Q-SHE Simulator",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Display logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("""
        <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="45" fill="#4B0082" stroke="#FFF" stroke-width="2"/>
            <text x="50" y="65" font-family="Arial" font-size="40" text-anchor="middle" fill="white">Q</text>
        </svg>
        """, unsafe_allow_html=True)
    
    with col2:
        st.title("Quantum Self-Healing Encryption (Q-SHE) Simulator")
        st.subheader("Interactive Visualization and Simulation Tool")

    # Introduction section
    st.markdown("""
    ## Introduction to Q-SHE
    
    Quantum Self-Healing Encryption (Q-SHE) represents a revolutionary approach to data security, 
    leveraging quantum mechanical principles to create encryption that can autonomously recover 
    from errors or attacks.
    
    This simulator allows you to:
    - Explore the theoretical foundations of Q-SHE
    - Visualize quantum fractal key encoding
    - Observe self-healing ciphertext dynamics
    - Simulate error introduction and autonomous recovery
    - Compare Q-SHE with classical encryption methods
    """)

    # Quick demonstration section
    st.header("Quick Demonstration")
    
    with st.expander("Fractal Key Encoding Demo", expanded=True):
        st.markdown("""
        Fractal Key Encoding is the foundation of Q-SHE, where a classical key is transformed into 
        a quantum state through recursive unitary operations across multiple scales.
        """)
        
        key_size = st.slider("Key Size (bits)", min_value=2, max_value=8, value=4, step=1)
        
        if st.button("Generate Fractal Key"):
            # Generate a random key of the selected size
            classical_key = np.random.randint(0, 2, size=key_size)
            key_str = ''.join(map(str, classical_key))
            st.write(f"Classical Key: {key_str}")
            
            # Simulate fractal encoding
            quantum_key = create_fractal_key(classical_key)
            
            # Visualize the encoding
            fig = plot_fractal_encoding(classical_key, quantum_key)
            st.pyplot(fig)
            
            # Display key properties
            st.write("### Key Properties")
            st.write(f"- Entropy: {np.log2(2**key_size):.2f} bits")
            st.write(f"- Quantum Dimension: {2**key_size}")
            st.write(f"- Error Localization Bound: ε²ᵏ log n")
    
    with st.expander("Self-Healing Demonstration", expanded=False):
        st.markdown("""
        Q-SHE ciphertexts can autonomously recover from errors through the interaction of 
        quantum states with a specially designed recovery Hamiltonian.
        """)
        
        # Simulation parameters
        plaintext_length = st.slider("Plaintext Length", 2, 6, 4)
        error_rate = st.slider("Error Rate (%)", 0, 100, 20) / 100
        recovery_time = st.slider("Recovery Time (t)", 0.0, 10.0, 5.0, 0.1)
        
        if st.button("Simulate Error Recovery"):
            # Generate random plaintext
            plaintext = np.random.randint(0, 2, size=plaintext_length)
            plaintext_str = ''.join(map(str, plaintext))
            st.write(f"Plaintext: {plaintext_str}")
            
            # Simulate encryption, error introduction, and recovery
            initial_state, corrupted_state, recovered_state, fidelity_over_time = simulate_recovery_hamiltonian(
                plaintext, error_rate, recovery_time
            )
            
            # Plot the recovery process
            fig = plot_recovery_fidelity(fidelity_over_time, recovery_time)
            st.pyplot(fig)
            
            # Visualize states
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Initial Ciphertext State")
                fig1 = visualize_quantum_state(initial_state)
                st.pyplot(fig1)
            
            with col2:
                st.write("Corrupted State")
                fig2 = visualize_quantum_state(corrupted_state)
                st.pyplot(fig2)
            
            with col3:
                st.write("Recovered State")
                fig3 = visualize_quantum_state(recovered_state)
                st.pyplot(fig3)
            
            # Display recovery metrics
            final_fidelity = fidelity_over_time[-1]
            st.write(f"### Recovery Metrics")
            st.write(f"- Final Fidelity: {final_fidelity:.4f}")
            st.write(f"- Theoretical Bound: {1 - error_rate**2 * (1 - np.exp(-0.5 * recovery_time)):.4f}")
            
            recovery_success = "Success" if final_fidelity > 0.9 else "Partial" if final_fidelity > 0.5 else "Failed"
            st.write(f"- Recovery Status: {recovery_success}")
    
    # Navigation section
    st.header("Explore More")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        [📚 Theoretical Foundations](/theory)
        
        Explore the mathematical theory behind Q-SHE, including fractal encoding and error localization theorems.
        """)
        
    with col2:
        st.markdown("""
        [🔑 Encryption & Decryption](/encryption_demo)
        
        Step through the complete encryption and decryption process using Q-SHE.
        """)
        
    with col3:
        st.markdown("""
        [🛠️ Error Correction](/error_correction)
        
        Simulate different types of errors and observe the self-healing process in action.
        """)
        
    with col4:
        st.markdown("""
        [📊 Comparison](/comparison)
        
        Compare Q-SHE with classical encryption methods in terms of security, overhead, and resilience.
        """)

if __name__ == "__main__":
    main()
