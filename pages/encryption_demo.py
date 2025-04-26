import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from utils.encryption import (
    encrypt_message, 
    decrypt_message, 
    generate_random_key,
    simulate_attack,
    heal_ciphertext
)
from utils.visualization import (
    visualize_quantum_state,
    plot_bloch_sphere,
    plot_fractal_encoding
)

st.set_page_config(
    page_title="Q-SHE Encryption Demo",
    page_icon="🔑",
    layout="wide"
)

def main():
    st.title("Quantum Self-Healing Encryption Demo")
    
    st.markdown("""
    This interactive demo lets you encrypt messages using Quantum Self-Healing Encryption (Q-SHE),
    visualize the encryption process, and see how the self-healing mechanism works against attacks.
    """)
    
    # Main encryption demo
    st.header("Encryption & Decryption Process")
    
    # Message input
    message = st.text_input("Enter a message to encrypt:", value="QUANTUM")
    key_size = st.slider("Key size (bits):", min_value=2, max_value=8, value=4, step=1)
    
    # Initialize session state for storing encryption details
    if 'encryption_details' not in st.session_state:
        st.session_state.encryption_details = None
    if 'attack_results' not in st.session_state:
        st.session_state.attack_results = None
    if 'healing_results' not in st.session_state:
        st.session_state.healing_results = None
    
    # Encrypt button
    if st.button("Encrypt Message"):
        with st.spinner("Encrypting message..."):
            # Perform encryption
            encryption_details = encrypt_message(message, key_size)
            st.session_state.encryption_details = encryption_details
            st.session_state.attack_results = None
            st.session_state.healing_results = None
            st.success("Message encrypted successfully!")
    
    # Show encryption results if available
    if st.session_state.encryption_details:
        details = st.session_state.encryption_details
        
        # Display encryption parameters
        st.subheader("Encryption Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Original Message:** {details['message']}")
            st.markdown(f"**Key Size:** {len(details['key'])} bits")
            
            # Display key (truncated if too long)
            key_str = ''.join(map(str, details['key']))
            if len(key_str) > 32:
                key_display = key_str[:16] + "..." + key_str[-16:]
            else:
                key_display = key_str
            st.markdown(f"**Encryption Key:** {key_display}")
            
            # Show binary plaintext
            plaintext_str = ''.join(map(str, details['plaintext']))
            if len(plaintext_str) > 64:
                plaintext_display = plaintext_str[:32] + "..." + plaintext_str[-32:]
            else:
                plaintext_display = plaintext_str
            st.markdown(f"**Binary Plaintext:** {plaintext_display}")
        
        with col2:
            # Plot the quantum key encoding
            st.markdown("**Quantum Key Visualization:**")
            fig = plot_fractal_encoding(details['key'], details['quantum_key'])
            st.pyplot(fig)
        
        # Visualization of ciphertext
        st.subheader("Ciphertext Visualization")
        
        # Display first ciphertext block
        if len(details['ciphertext_blocks']) > 0:
            st.markdown("**First Ciphertext Block:**")
            fig = visualize_quantum_state(details['ciphertext_blocks'][0])
            st.pyplot(fig)
            
            # If there are multiple blocks, show a summary
            if len(details['ciphertext_blocks']) > 1:
                st.markdown(f"**Total Ciphertext Blocks:** {len(details['ciphertext_blocks'])}")
                st.markdown("Each block encodes a portion of the plaintext with the quantum key.")
        
        # Decryption section
        st.subheader("Decryption")
        
        if st.button("Decrypt Message"):
            with st.spinner("Decrypting message..."):
                # Perform decryption
                decrypted = decrypt_message(details)
                st.success(f"Decrypted Message: {decrypted}")
        
        # Attack simulation section
        st.header("Attack Simulation")
        
        st.markdown("""
        This section simulates an attack on the encrypted message and demonstrates 
        the self-healing capability of Q-SHE.
        """)
        
        attack_strength = st.slider("Attack Strength (%):", min_value=5, max_value=50, value=20, step=5) / 100
        
        if st.button("Simulate Attack"):
            with st.spinner("Simulating attack..."):
                # Simulate attack on the ciphertext
                attack_results = simulate_attack(details, attack_strength)
                st.session_state.attack_results = attack_results
                st.session_state.healing_results = None
                
                # Display attack results
                avg_fidelity = attack_results['average_fidelity']
                damage_level = "Severe" if avg_fidelity < 0.5 else "Moderate" if avg_fidelity < 0.8 else "Minor"
                st.warning(f"Attack simulation complete. Damage level: {damage_level}")
                st.markdown(f"Average Fidelity After Attack: {avg_fidelity:.4f}")
        
        # Display attack results if available
        if st.session_state.attack_results:
            attack_results = st.session_state.attack_results
            
            # Visualize attacked ciphertext
            st.subheader("Corrupted Ciphertext Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Original state
                st.markdown("**Original Ciphertext (First Block):**")
                fig1 = visualize_quantum_state(details['ciphertext_blocks'][0])
                st.pyplot(fig1)
            
            with col2:
                # Corrupted state
                st.markdown("**Corrupted Ciphertext (First Block):**")
                fig2 = visualize_quantum_state(attack_results['corrupted_blocks'][0])
                st.pyplot(fig2)
            
            # Block fidelities
            st.markdown("### Block Fidelities After Attack")
            
            # Plot block fidelities
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(attack_results['block_fidelities']))
            ax.bar(x, attack_results['block_fidelities'], width=0.6, color='red', alpha=0.7)
            ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Success Threshold')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Failure Threshold')
            ax.set_xlabel('Block Index')
            ax.set_ylabel('Fidelity')
            ax.set_ylim(0, 1.05)
            ax.set_title('Ciphertext Block Fidelities After Attack')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Self-healing section
            st.header("Self-Healing Process")
            
            healing_time = st.slider("Healing Time:", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
            
            if st.button("Initiate Self-Healing"):
                with st.spinner("Healing in progress..."):
                    # Apply healing to corrupted ciphertext
                    healing_results = heal_ciphertext(attack_results, healing_time)
                    st.session_state.healing_results = healing_results
                    
                    # Display healing results
                    avg_healing = healing_results['average_healing_fidelity']
                    healing_success = "Successful" if avg_healing > 0.9 else "Partial" if avg_healing > 0.5 else "Failed"
                    st.success(f"Healing process complete. Result: {healing_success}")
                    st.markdown(f"Average Fidelity After Healing: {avg_healing:.4f}")
                    
                    # Save results to database
                    from utils.database import save_simulation_result
                    
                    # Extract key metrics
                    key_size = len(st.session_state.encryption_details['key'])
                    error_rate = attack_strength
                    recovery_time = healing_time
                    initial_fidelity = 1.0
                    corrupted_fidelity = attack_results['average_fidelity']
                    recovered_fidelity = avg_healing
                    
                    # Save to database
                    result_id = save_simulation_result(
                        simulation_type="encryption",
                        key_size=key_size,
                        error_rate=error_rate,
                        recovery_time=recovery_time,
                        initial_fidelity=initial_fidelity,
                        corrupted_fidelity=corrupted_fidelity,
                        recovered_fidelity=recovered_fidelity,
                        error_type="Attack Simulation",
                        description=f"Encryption demo with attack strength {attack_strength} and healing time {healing_time}",
                        parameters={
                            "original_message": st.session_state.encryption_details['message'],
                            "blocks_count": len(st.session_state.encryption_details['ciphertext_blocks']),
                            "healing_efficiency": blocks_healed / failed_before * 100 if failed_before > 0 else 100
                        }
                    )
                    
                    st.markdown(f"*Result saved to database with ID: {result_id}*")
            
            # Display healing results if available
            if st.session_state.healing_results:
                healing_results = st.session_state.healing_results
                
                # Visualize healed ciphertext
                st.subheader("Healed Ciphertext Visualization")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Original state
                    st.markdown("**Original Ciphertext:**")
                    fig1 = visualize_quantum_state(details['ciphertext_blocks'][0])
                    st.pyplot(fig1)
                
                with col2:
                    # Corrupted state
                    st.markdown("**Corrupted Ciphertext:**")
                    fig2 = visualize_quantum_state(attack_results['corrupted_blocks'][0])
                    st.pyplot(fig2)
                
                with col3:
                    # Healed state
                    st.markdown("**Healed Ciphertext:**")
                    fig3 = visualize_quantum_state(healing_results['healed_blocks'][0])
                    st.pyplot(fig3)
                
                # Healing fidelities
                st.markdown("### Healing Performance")
                
                # Compare original, attacked, and healed fidelities
                fig, ax = plt.subplots(figsize=(10, 5))
                
                x = np.arange(len(healing_results['healing_fidelities']))
                width = 0.3
                
                # Plot original fidelities (all 1.0)
                ax.bar(x - width, np.ones(len(x)), width, color='green', alpha=0.7, label='Original')
                
                # Plot attacked fidelities
                ax.bar(x, attack_results['block_fidelities'], width, color='red', alpha=0.7, label='After Attack')
                
                # Plot healed fidelities
                ax.bar(x + width, healing_results['healing_fidelities'], width, color='blue', alpha=0.7, label='After Healing')
                
                # Add reference lines
                ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Failure Threshold')
                
                # Format axes
                ax.set_xlabel('Block Index')
                ax.set_ylabel('Fidelity')
                ax.set_ylim(0, 1.05)
                ax.set_title('Comparison of Fidelities Throughout the Process')
                ax.legend()
                ax.grid(alpha=0.3)
                
                st.pyplot(fig)
                
                # Success metrics
                st.markdown("### Healing Success Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Calculate improvement
                    avg_attack = attack_results['average_fidelity']
                    avg_healing = healing_results['average_healing_fidelity']
                    improvement = (avg_healing - avg_attack) / (1 - avg_attack) * 100 if avg_attack < 1 else 0
                    
                    st.markdown(f"**Average Fidelity After Attack:** {avg_attack:.4f}")
                    st.markdown(f"**Average Fidelity After Healing:** {avg_healing:.4f}")
                    st.markdown(f"**Relative Improvement:** {improvement:.1f}%")
                    
                    # Recovery status
                    if avg_healing > 0.95:
                        st.success("**Recovery Status:** Complete")
                    elif avg_healing > 0.8:
                        st.info("**Recovery Status:** Substantial")
                    elif avg_healing > 0.5:
                        st.warning("**Recovery Status:** Partial")
                    else:
                        st.error("**Recovery Status:** Failed")
                
                with col2:
                    # Calculate number of blocks successfully healed
                    success_threshold = 0.9
                    failed_before = sum(f < success_threshold for f in attack_results['block_fidelities'])
                    failed_after = sum(f < success_threshold for f in healing_results['healing_fidelities'])
                    blocks_healed = failed_before - failed_after
                    
                    total_blocks = len(healing_results['healing_fidelities'])
                    
                    st.markdown(f"**Total Blocks:** {total_blocks}")
                    st.markdown(f"**Blocks Damaged by Attack:** {failed_before}")
                    st.markdown(f"**Blocks Successfully Healed:** {blocks_healed}")
                    st.markdown(f"**Blocks Still Damaged:** {failed_after}")
                    
                    # Recovery efficiency
                    if failed_before > 0:
                        healing_efficiency = blocks_healed / failed_before * 100
                        st.markdown(f"**Healing Efficiency:** {healing_efficiency:.1f}%")
                
                # Theoretical comparison
                with st.expander("Theoretical Analysis"):
                    st.markdown("""
                    ### Comparison with Theoretical Bounds
                    
                    According to the Self-Healing Theorem, the expected fidelity after healing for time $t$ is:
                    
                    $$\\mathcal{F}(t) \\geq 1 - \\epsilon^2 (1 - e^{-\\Gamma t})$$
                    
                    where $\\epsilon$ is the error magnitude and $\\Gamma$ is the recovery rate.
                    """)
                    
                    # Estimate parameters from results
                    epsilon_est = np.sqrt(1 - attack_results['average_fidelity'])
                    
                    # Estimate Gamma by fitting
                    healing_time_val = healing_time
                    observed_fidelity = healing_results['average_healing_fidelity']
                    
                    if observed_fidelity < 1:
                        gamma_est = -np.log((1 - observed_fidelity) / epsilon_est**2) / healing_time_val
                    else:
                        gamma_est = 1.0  # Default if perfect recovery
                    
                    st.markdown(f"""
                    **Estimated Parameters**:
                    - Error Magnitude (ε): {epsilon_est:.4f}
                    - Recovery Rate (Γ): {gamma_est:.4f}
                    
                    **Theoretical Prediction**:
                    - Predicted Fidelity: {1 - epsilon_est**2 * (1 - np.exp(-gamma_est * healing_time_val)):.4f}
                    - Actual Fidelity: {observed_fidelity:.4f}
                    """)
                    
                    # Plot theoretical vs actual
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    t = np.linspace(0, healing_time_val * 1.5, 100)
                    predicted_fidelity = 1 - epsilon_est**2 * (1 - np.exp(-gamma_est * t))
                    
                    ax.plot(t, predicted_fidelity, 'b-', lw=2, label='Theoretical Bound')
                    ax.scatter([healing_time_val], [observed_fidelity], color='red', s=100, label='Observed Result')
                    ax.scatter([0], [attack_results['average_fidelity']], color='orange', s=100, label='Initial Damage')
                    
                    ax.set_xlabel('Healing Time (t)')
                    ax.set_ylabel('Fidelity')
                    ax.set_ylim(min(attack_results['average_fidelity'] - 0.1, 0), 1.05)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_title('Theoretical vs. Observed Recovery')
                    
                    st.pyplot(fig)
    
    # Step-by-step explanation
    with st.expander("Step-by-Step Encryption Process"):
        st.markdown("""
        ### Q-SHE Encryption Process
        
        1. **Plaintext Preparation**:
           - Convert the message to binary
           - Divide into blocks that match the key size
        
        2. **Key Generation and Encoding**:
           - Generate a random binary key
           - Apply fractal encoding to create a quantum key state
        
        3. **Ciphertext Creation**:
           - For each plaintext block:
             - Apply key-dependent transformations
             - Create ancilla qubits with parity functions
             - Combine transformed plaintext with ancilla qubits
        
        4. **Self-Healing Properties**:
           - The ciphertext structure distributes information across multiple scales
           - Ancilla qubits maintain consistency checks for error detection
           - The fractal structure allows errors to be corrected using information from undamaged parts
        
        5. **Decryption Process**:
           - Apply the inverse of the encryption transformations
           - Measure the resulting state to recover the plaintext
           - If errors occurred, the self-healing properties can correct them before decryption
        """)
    
    # Advanced options and settings
    with st.expander("Advanced Options"):
        st.markdown("### Advanced Encryption Settings")
        
        st.markdown("""
        **Theoretical Parameter Adjustments**
        
        These parameters would affect real Q-SHE implementations but are simplified in this simulation:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("Delta (Energy Scale)", 0.5, 5.0, 2.0, 0.1, disabled=True)
            st.slider("Lambda (Coupling Strength)", 0.5, 3.0, 1.0, 0.1, disabled=True)
        
        with col2:
            st.slider("Gamma (Spectral Gap)", 0.1, 1.0, 0.5, 0.1, disabled=True)
            st.slider("Fractal Dimension", 1.0, 3.0, 2.0, 0.1, disabled=True)
        
        st.markdown("""
        **Note**: In a full quantum implementation, these parameters would directly affect the 
        recovery Hamiltonian and the system's resilience to errors. In this simulation, we use 
        simplified models to demonstrate the core concepts.
        """)

if __name__ == "__main__":
    main()
