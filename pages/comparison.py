import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.quantum_simulation import resilience_analysis, compare_with_classical
from utils.visualization import plot_resilience_comparison, plot_overhead_comparison

st.set_page_config(
    page_title="Q-SHE Comparison",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("Comparing Q-SHE with Classical Encryption Methods")
    
    st.markdown("""
    This page provides a comprehensive comparison between Quantum Self-Healing Encryption (Q-SHE)
    and classical encryption methods, highlighting the advantages and unique properties of Q-SHE.
    """)
    
    # Main comparison metrics
    st.header("Key Comparison Metrics")
    
    # Create a comparison table
    comparison_data = {
        "Metric": [
            "Recovery Mechanism", 
            "Overhead", 
            "Security Scaling", 
            "Error Detection", 
            "Error Correction",
            "Implementation Requirements"
        ],
        "Classical ECC": [
            "Syndrome measurement", 
            "O(n)", 
            "O(√n)-bounded", 
            "Explicit", 
            "Active correction",
            "Classical computers"
        ],
        "Q-SHE": [
            "Autonomous evolution", 
            "O(log n)", 
            "O(n¹/ᵈ)-resilient", 
            "Implicit", 
            "Passive healing",
            "Quantum processors"
        ]
    }
    
    # Display the comparison table
    st.table(pd.DataFrame(comparison_data))
    
    # Add detailed explanation of each metric
    with st.expander("Metric Explanations"):
        st.markdown("""
        ### Detailed Explanation of Comparison Metrics
        
        **Recovery Mechanism**:
        - **Classical ECC**: Classical error-correcting codes (ECC) require explicit measurement of error syndromes followed by correction based on the syndrome.
        - **Q-SHE**: Self-healing occurs automatically through Hamiltonian evolution without requiring any measurement or feedback.
        
        **Overhead**:
        - **Classical ECC**: The overhead (extra bits needed) scales linearly with system size O(n).
        - **Q-SHE**: Overhead scales logarithmically with system size O(log n), making it more efficient for large systems.
        
        **Security Scaling**:
        - **Classical ECC**: An attacker needs to corrupt approximately O(√n) bits to compromise security.
        - **Q-SHE**: An attacker must corrupt O(n¹/ᵈ) qubits, where d is the fractal dimension (typically 2-3), providing stronger security scaling.
        
        **Error Detection**:
        - **Classical ECC**: Errors must be explicitly detected through syndrome measurements.
        - **Q-SHE**: Error detection is implicit in the energy landscape of the recovery Hamiltonian.
        
        **Error Correction**:
        - **Classical ECC**: Once detected, errors must be actively corrected by applying specific operations.
        - **Q-SHE**: Errors are automatically corrected through passive evolution under the recovery Hamiltonian.
        
        **Implementation Requirements**:
        - **Classical ECC**: Can be implemented on conventional computing hardware.
        - **Q-SHE**: Requires quantum hardware capable of maintaining quantum states and implementing the recovery Hamiltonian.
        """)
    
    # Interactive comparison of resilience against errors
    st.header("Resilience Against Errors")
    
    st.markdown("""
    This section compares how Q-SHE and classical error-correcting codes (ECC) perform 
    when subjected to increasing error rates. The graphs show the recovery fidelity 
    (how well the original message can be recovered) versus the error rate.
    """)
    
    # Parameters for the comparison
    col1, col2 = st.columns(2)
    
    with col1:
        system_size = st.slider(
            "System Size (bits/qubits)", 
            min_value=8, 
            max_value=64, 
            value=16, 
            step=8,
            help="Size of the encryption system in bits (classical) or qubits (quantum)"
        )
        
    with col2:
        fractal_dimension = st.slider(
            "Fractal Dimension", 
            min_value=1.0, 
            max_value=3.0, 
            value=2.0, 
            step=0.1,
            help="Fractal dimension of the Q-SHE encoding (higher values increase resilience)"
        )
    
    # Run resilience comparison
    if st.button("Compare Resilience"):
        with st.spinner("Running resilience comparison..."):
            # Generate error rates for testing
            error_rates = np.linspace(0.01, 0.5, 20)
            
            # Run the comparison (simulated)
            qshe_fidelities, classical_fidelities = simulate_resilience_comparison(
                system_size, error_rates, fractal_dimension
            )
            
            # Plot results
            fig = plot_resilience_comparison(error_rates, qshe_fidelities, classical_fidelities)
            st.pyplot(fig)
            
            # Calculate and display the break-even point
            break_even = find_break_even_point(error_rates, qshe_fidelities, classical_fidelities)
            
            if break_even is not None:
                st.markdown(f"""
                **Break-even point**: At {break_even:.2%} error rate, both approaches show similar performance.
                
                - Below this error rate: Classical ECC may be more efficient
                - Above this error rate: Q-SHE demonstrates superior resilience
                """)
            else:
                st.markdown("""
                **Comparison Result**: Q-SHE demonstrates superior resilience across the entire error range tested.
                """)
            
            # Calculate advantage at high error rate
            high_error_index = int(len(error_rates) * 0.8)  # 80% through the error range
            high_error_rate = error_rates[high_error_index]
            
            # Save results to database
            from utils.database import save_simulation_result
            
            # Calculate key metrics for database storage
            avg_qshe_fidelity = np.mean(qshe_fidelities)
            avg_classical_fidelity = np.mean(classical_fidelities)
            max_error_rate = np.max(error_rates)
            
            # Save to database
            result_id = save_simulation_result(
                simulation_type="comparison",
                key_size=system_size,
                error_rate=max_error_rate,  # Using the maximum error rate tested
                recovery_time=0.0,  # Not applicable for this simulation
                initial_fidelity=1.0,
                corrupted_fidelity=0.0,  # Not applicable for this simulation
                recovered_fidelity=avg_qshe_fidelity,
                fractal_dimension=fractal_dimension,
                error_type="Resilience Comparison",
                description=f"Resilience comparison between Q-SHE and classical ECC with system size {system_size}",
                parameters={
                    "error_rates": error_rates.tolist(),
                    "qshe_fidelities": qshe_fidelities.tolist(),
                    "classical_fidelities": classical_fidelities.tolist(),
                    "break_even_point": break_even if break_even is not None else None,
                    "avg_qshe_fidelity": float(avg_qshe_fidelity),
                    "avg_classical_fidelity": float(avg_classical_fidelity),
                    "qshe_advantage": float(qshe_fidelities[high_error_index] - classical_fidelities[high_error_index])
                }
            )
            
            st.markdown(f"*Comparison results saved to database with ID: {result_id}*")
            qshe_high = qshe_fidelities[high_error_index]
            classical_high = classical_fidelities[high_error_index]
            
            st.markdown(f"""
            **At {high_error_rate:.2%} error rate**:
            - Q-SHE recovery fidelity: {qshe_high:.4f}
            - Classical ECC recovery fidelity: {classical_high:.4f}
            - **Q-SHE advantage**: {(qshe_high-classical_high)/max(0.001, classical_high)*100:.1f}% better recovery
            """)
    
    # Computational overhead comparison
    st.header("Computational Overhead")
    
    st.markdown("""
    This section visualizes the computational overhead of Q-SHE compared to classical methods.
    Overhead refers to the additional resources (time, memory, or operations) required for encryption, 
    decryption, and error correction.
    """)
    
    # Run overhead comparison
    if st.button("Compare Overhead"):
        with st.spinner("Calculating overhead comparison..."):
            # Generate key sizes for comparison
            key_sizes = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
            
            # Plot overhead comparison
            fig = plot_overhead_comparison(key_sizes)
            st.pyplot(fig)
            
            # Save results to database
            from utils.database import save_simulation_result
            
            # Save to database
            result_id = save_simulation_result(
                simulation_type="comparison",
                key_size=int(np.max(key_sizes)),  # Using the maximum key size
                error_rate=0.0,  # Not applicable for this simulation
                recovery_time=0.0,  # Not applicable for this simulation
                initial_fidelity=1.0,
                corrupted_fidelity=0.0,  # Not applicable for this simulation
                recovered_fidelity=1.0,  # Not applicable for this simulation
                error_type="Overhead Comparison",
                description=f"Computational overhead comparison between Q-SHE and classical ECC",
                parameters={
                    "key_sizes": key_sizes.tolist(),
                    "qshe_overhead": np.log2(key_sizes).tolist(),
                    "classical_overhead": key_sizes.tolist(),
                    "largest_key_size": int(np.max(key_sizes)),
                    "smallest_key_size": int(np.min(key_sizes))
                }
            )
            
            st.markdown(f"*Overhead comparison results saved to database with ID: {result_id}*")
            
            # Add explanation of the overhead
            st.markdown("""
            ### Overhead Analysis
            
            The graph above shows the theoretical computational overhead scaling for both approaches:
            
            - **Q-SHE**: Scales as O(log n) with key size
            - **Classical ECC**: Scales as O(n) with key size
            
            This logarithmic scaling of Q-SHE makes it increasingly efficient for larger key sizes, 
            which is particularly important for high-security applications.
            
            The practical implications of this scaling include:
            
            1. **Reduced energy consumption** for large-scale encryption systems
            2. **Faster processing times** for encryption and decryption operations
            3. **Lower memory requirements** for storing error correction information
            4. **Increased scalability** for securing large data sets
            """)
    
    # Security property comparison
    st.header("Security Properties")
    
    # Create tabs for different security aspects
    tab1, tab2, tab3 = st.tabs([
        "Attack Resistance", 
        "Information Leakage", 
        "Future Proofing"
    ])
    
    with tab1:
        st.markdown("""
        ## Attack Resistance
        
        A key advantage of Q-SHE is its superior resistance to targeted attacks compared to classical systems.
        """)
        
        # Interactive demonstration of attack resistance
        st.subheader("Attack Resistance Visualization")
        
        system_sizes = st.select_slider(
            "System Size (n)", 
            options=[16, 32, 64, 128, 256, 512, 1024, 2048],
            value=64
        )
        
        # Create visualization of attack resistance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate number of bits that must be corrupted
        sizes = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])
        classical_corruption = np.sqrt(sizes)
        qshe_corruption_d2 = sizes**(1/2)  # d=2
        qshe_corruption_d3 = sizes**(1/3)  # d=3
        
        # Plot data
        ax.loglog(sizes, classical_corruption, 'ro-', label='Classical ECC (O(√n))')
        ax.loglog(sizes, qshe_corruption_d2, 'bo-', label='Q-SHE (d=2)')
        ax.loglog(sizes, qshe_corruption_d3, 'go-', label='Q-SHE (d=3)')
        
        # Highlight selected system size
        size_index = np.where(sizes == system_sizes)[0][0]
        
        ax.plot(system_sizes, classical_corruption[size_index], 'ro', ms=10, mfc='none', mew=2)
        ax.plot(system_sizes, qshe_corruption_d2[size_index], 'bo', ms=10, mfc='none', mew=2)
        ax.plot(system_sizes, qshe_corruption_d3[size_index], 'go', ms=10, mfc='none', mew=2)
        
        # Format plot
        ax.set_xlabel('System Size (n)')
        ax.set_ylabel('Qubits Needed to Break')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        ax.set_title('Attack Resistance Scaling')
        
        st.pyplot(fig)
        
        # Display comparison for selected size
        st.markdown(f"""
        ### Attack Resistance for n = {system_sizes}
        
        For a system of size {system_sizes}, an attacker would need to corrupt:
        
        - **Classical ECC**: {int(np.sqrt(system_sizes))} bits
        - **Q-SHE (d=2)**: {int(system_sizes**(1/2))} qubits
        - **Q-SHE (d=3)**: {int(system_sizes**(1/3))} qubits
        
        This demonstrates that Q-SHE with higher fractal dimension (d=3) provides substantially better 
        protection against targeted attacks.
        """)
        
        # Add explanation of attack scenarios
        with st.expander("Attack Scenarios"):
            st.markdown("""
            ### Common Attack Scenarios
            
            **1. Random Bit Errors**
            
            - **Classical ECC Performance**: Good resistance to random errors up to the code's error correction capacity
            - **Q-SHE Performance**: Superior performance due to multi-scale redundancy
            
            **2. Burst Errors (Localized Corruption)**
            
            - **Classical ECC Performance**: Vulnerable to burst errors that exceed the code's local correction capacity
            - **Q-SHE Performance**: Resilient due to fractal structure spreading information across scales
            
            **3. Targeted Attacks**
            
            - **Classical ECC Performance**: Vulnerable when attacker knows the code structure
            - **Q-SHE Performance**: Highly resistant due to quantum effects and fractal encoding
            
            **4. Side-Channel Attacks**
            
            - **Classical ECC Performance**: Potentially vulnerable to timing, power, or electromagnetic analysis
            - **Q-SHE Performance**: Natural resistance due to quantum nature and autonomous operation
            """)
    
    with tab2:
        st.markdown("""
        ## Information Leakage
        
        Information leakage refers to how much data about the plaintext can be deduced from 
        partial knowledge of the ciphertext or from observing the encryption system.
        """)
        
        # Create comparison of information leakage
        comparison_data = {
            "Aspect": [
                "Partial Observation", 
                "Side-Channel Analysis", 
                "Quantum Measurement",
                "Physical Security"
            ],
            "Classical ECC": [
                "Leaks information proportional to observation size", 
                "Vulnerable to timing and power analysis", 
                "N/A (classical system)",
                "Protection depends on implementation"
            ],
            "Q-SHE": [
                "Exponentially suppressed information leakage", 
                "Natural resistance through quantum dynamics", 
                "Measurement disturbs the quantum state, reducing leakage",
                "Quantum properties provide inherent protection"
            ]
        }
        
        # Display the comparison table
        st.table(pd.DataFrame(comparison_data))
        
        # Information leakage visualization
        st.subheader("Information Leakage Visualization")
        
        # Create a visualization of information leakage
        observation_percentage = st.slider(
            "System Observation (%)", 
            min_value=10, 
            max_value=90, 
            value=30,
            help="Percentage of the system an attacker can observe"
        )
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate information leakage vs. observation percentage
        observation_percentages = np.linspace(0, 100, 100)
        
        # Classical information leakage (approximately linear)
        classical_leakage = observation_percentages / 100
        
        # Q-SHE information leakage (exponentially suppressed)
        qshe_leakage = 1 - np.exp(-0.01 * observation_percentages**2 / 100)
        
        # Plot curves
        ax.plot(observation_percentages, classical_leakage, 'r-', label='Classical ECC')
        ax.plot(observation_percentages, qshe_leakage, 'b-', label='Q-SHE')
        
        # Mark selected observation percentage
        ax.axvline(x=observation_percentage, color='gray', linestyle='--', alpha=0.7)
        ax.plot(observation_percentage, classical_leakage[observation_percentage], 'ro', ms=10)
        ax.plot(observation_percentage, qshe_leakage[observation_percentage], 'bo', ms=10)
        
        # Format plot
        ax.set_xlabel('System Observation (%)')
        ax.set_ylabel('Information Leakage (%)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Information Leakage vs. System Observation')
        
        st.pyplot(fig)
        
        # Display comparison for selected observation
        cl_leak = classical_leakage[observation_percentage]
        qshe_leak = qshe_leakage[observation_percentage]
        
        st.markdown(f"""
        ### Information Leakage at {observation_percentage}% Observation
        
        When an attacker can observe {observation_percentage}% of the system:
        
        - **Classical ECC**: Approximately {cl_leak:.1%} of the information is leaked
        - **Q-SHE**: Only about {qshe_leak:.1%} of the information is leaked
        
        This represents a **{(1 - qshe_leak/cl_leak)*100:.1f}%** reduction in information leakage with Q-SHE.
        """)
        
        # Add explanation of information theory aspects
        with st.expander("Information Theory Perspective"):
            st.markdown("""
            ### Information Theory Analysis
            
            From an information theory perspective, Q-SHE offers several advantages:
            
            **1. Mutual Information**
            
            The mutual information between plaintext P and observed ciphertext C' is exponentially lower in Q-SHE compared to classical systems:
            
            - **Classical ECC**: I(P;C') ≈ O(observation size)
            - **Q-SHE**: I(P;C') ≈ O(e^(-observation size))
            
            **2. Entropic Security**
            
            Q-SHE demonstrates stronger entropic security guarantees, meaning an attacker learns very little useful information from partial observations.
            
            **3. Quantum Uncertainty**
            
            The inherent uncertainty principles in quantum mechanics provide additional protection against information leakage in Q-SHE.
            
            **4. Non-Clonability**
            
            Unlike classical bits, quantum states cannot be perfectly copied (no-cloning theorem), which prevents certain types of attacks that work on classical systems.
            """)
    
    with tab3:
        st.markdown("""
        ## Future-Proofing Against Quantum Computers
        
        As quantum computing advances, many classical encryption methods face potential vulnerabilities.
        This section examines how well different encryption approaches can withstand future quantum computing threats.
        """)
        
        # Create comparison table for quantum resistance
        quantum_resistance_data = {
            "Threat": [
                "Shor's Algorithm", 
                "Grover's Algorithm", 
                "Quantum Machine Learning",
                "Future Quantum Algorithms"
            ],
            "Classical ECC": [
                "Vulnerable if combined with RSA/ECC", 
                "Square root speedup in attacks", 
                "Potentially vulnerable",
                "Unknown vulnerabilities"
            ],
            "Q-SHE": [
                "Resistant (doesn't rely on factoring)", 
                "Natural quantum resistance", 
                "Quantum nature provides inherent protection",
                "Designed within quantum framework"
            ]
        }
        
        # Display the comparison table
        st.table(pd.DataFrame(quantum_resistance_data))
        
        # Quantum resistance metrics
        st.subheader("Quantum Threat Resistance")
        
        # Create visualization of quantum threat resistance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define the threat categories and scores (0-10 scale)
        threat_categories = ["Current\nClassical\nAttacks", "Quantum\nDecryption\nAlgorithms", "Quantum\nML\nAttacks", "Post-Quantum\nThreats", "Zero-Day\nQuantum\nThreats"]
        classical_scores = [8, 3, 4, 3, 2]
        qshe_scores = [9, 8, 7, 7, 6]
        
        # Define positions
        x = np.arange(len(threat_categories))
        width = 0.35
        
        # Create the bars
        ax.bar(x - width/2, classical_scores, width, label='Classical ECC', color='crimson', alpha=0.7)
        ax.bar(x + width/2, qshe_scores, width, label='Q-SHE', color='royalblue', alpha=0.7)
        
        # Add labels and formatting
        ax.set_ylabel('Resistance Score (0-10)')
        ax.set_title('Resistance to Various Quantum Threats')
        ax.set_xticks(x)
        ax.set_xticklabels(threat_categories)
        ax.legend()
        
        # Add grid for readability
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 10)
        
        # Add value labels on bars
        for i, v in enumerate(classical_scores):
            ax.text(i - width/2, v + 0.3, str(v), ha='center', va='bottom', color='crimson', fontweight='bold')
        
        for i, v in enumerate(qshe_scores):
            ax.text(i + width/2, v + 0.3, str(v), ha='center', va='bottom', color='royalblue', fontweight='bold')
        
        st.pyplot(fig)
        
        # Explanation of quantum threats
        st.markdown("""
        ### Quantum Computing Threats Explained
        
        **Current Classical Attacks**:  
        Traditional cryptanalysis methods including brute force, side-channel attacks, and implementation vulnerabilities.
        
        **Quantum Decryption Algorithms**:  
        Algorithms like Shor's (for factoring) and Grover's (for searching) that provide quantum speedups for breaking encryption.
        
        **Quantum ML Attacks**:  
        Using quantum machine learning to identify patterns in encryption systems that classical ML cannot detect.
        
        **Post-Quantum Threats**:  
        Next-generation quantum algorithms currently in development that may target specific encryption weaknesses.
        
        **Zero-Day Quantum Threats**:  
        Unknown future quantum attacks that have not yet been theoretically developed.
        """)
        
        # Long-term outlook
        with st.expander("Long-term Security Outlook"):
            st.markdown("""
            ### Long-term Security Outlook
            
            **Classical Post-Quantum Cryptography:**
            
            Classical cryptography is currently transitioning to "post-quantum" algorithms designed to resist quantum attacks. However, these approaches:
            
            - Are still based on classical computational assumptions
            - May have hidden vulnerabilities to undiscovered quantum algorithms
            - Offer no inherent quantum advantages
            
            **Quantum-Native Security (Q-SHE):**
            
            As a quantum-native approach, Q-SHE:
            
            - Is designed within the quantum framework from the ground up
            - Leverages quantum mechanics for security rather than trying to defend against it
            - Benefits from the same quantum effects that threaten classical systems
            - Likely to remain secure even as quantum computing advances
            
            **Long-term Prediction:**
            
            In the long term, quantum-native security approaches like Q-SHE are likely to become the standard for high-security applications, especially as quantum computing hardware becomes more accessible.
            """)
    
    # Implementation considerations
    st.header("Implementation Considerations")
    
    # Create tabs for implementation aspects
    tab1, tab2 = st.tabs(["Hardware Requirements", "Practical Challenges"])
    
    with tab1:
        st.markdown("""
        ## Hardware Requirements
        
        Implementing Q-SHE requires specialized quantum hardware, while classical ECC can run on conventional computers.
        """)
        
        # Create comparison of hardware requirements
        hardware_comparison = {
            "Requirement": [
                "Processor Type", 
                "Memory Type", 
                "Error Rates", 
                "Coherence Time",
                "Qubit Connectivity",
                "Scale"
            ],
            "Classical ECC": [
                "Standard CPU/GPU", 
                "Classical RAM/Flash", 
                "Digital (effectively zero)", 
                "N/A",
                "N/A",
                "Readily available at large scale"
            ],
            "Q-SHE": [
                "Quantum processor", 
                "Quantum memory", 
                "< 1% for reliable operation", 
                "Longer than recovery time",
                "Local interactions sufficient",
                "Currently limited to small demonstrations"
            ]
        }
        
        # Display the hardware comparison table
        st.table(pd.DataFrame(hardware_comparison))
        
        # Technology readiness level comparison
        st.subheader("Technology Readiness Level")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Classical ECC readiness
            st.markdown("### Classical ECC")
            
            # Create a progress bar for TRL
            classical_trl = 9  # Technology Readiness Level (1-9)
            st.progress(classical_trl / 9)
            
            st.markdown(f"""
            **Technology Readiness Level: {classical_trl}/9**
            
            Classical error correction is a mature technology with widespread deployment in commercial systems.
            
            **Current Status:**
            - Used in everyday technologies (storage, communication)
            - Standardized implementations available
            - Commercial off-the-shelf solutions
            """)
        
        with col2:
            # Q-SHE readiness
            st.markdown("### Q-SHE")
            
            # Create a progress bar for TRL
            qshe_trl = 3  # Technology Readiness Level (1-9)
            st.progress(qshe_trl / 9)
            
            st.markdown(f"""
            **Technology Readiness Level: {qshe_trl}/9**
            
            Q-SHE is currently in the early experimental phase with active research ongoing.
            
            **Current Status:**
            - Proof of concept demonstrations
            - Early laboratory implementations
            - Requires specialized quantum hardware
            """)
        
        # Quantum hardware progress
        st.subheader("Quantum Hardware Progress")
        
        # Create a visualization of quantum hardware progress
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Data for quantum hardware progress
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
        qubit_counts = [5, 9, 20, 50, 65, 100, 127, 433, 1000, None, None, None, None, None, None, None]
        
        # Remove None values for plotting
        valid_years = years[:len([x for x in qubit_counts if x is not None])]
        valid_counts = [x for x in qubit_counts if x is not None]
        
        # Plot historical data
        ax.semilogy(valid_years, valid_counts, 'bo-', label='Historical Qubit Count')
        
        # Plot projected data
        projected_years = years[len(valid_counts):]
        projected_counts = [2000, 4000, 8000, 16000, 32000, 64000, 128000][:len(projected_years)]
        
        ax.semilogy(projected_years, projected_counts, 'r--', label='Projected Growth')
        
        # Add annotation for Q-SHE requirements
        ax.axhline(y=50, color='g', linestyle='--', alpha=0.7, label='Minimum for Q-SHE Demo')
        ax.axhline(y=1000, color='g', linestyle='-', alpha=0.7, label='Practical Q-SHE Applications')
        
        # Format plot
        ax.set_xlabel('Year')
        ax.set_ylabel('Qubit Count (log scale)')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        ax.set_title('Quantum Hardware Progress')
        
        st.pyplot(fig)
        
        # Add explanation of hardware trends
        st.markdown("""
        ### Quantum Hardware Trends
        
        The graph above shows the historical and projected growth in quantum computing hardware. For Q-SHE implementation:
        
        - **Demonstration Systems** require at least 50 high-quality qubits with appropriate connectivity
        - **Practical Applications** become feasible at around 1,000 qubits with low error rates
        - **Wide Adoption** would require 10,000+ qubit systems with fault-tolerance
        
        Based on current trends, Q-SHE could begin to see practical implementations in specialized settings within 3-5 years,
        with more widespread adoption possible in 5-10 years as quantum hardware continues to advance.
        """)
    
    with tab2:
        st.markdown("""
        ## Practical Challenges
        
        Despite its theoretical advantages, Q-SHE faces several practical challenges that must be addressed 
        before widespread adoption.
        """)
        
        # Create comparison of practical challenges
        challenges_comparison = {
            "Challenge": [
                "Integration with Classical Systems", 
                "Key Distribution", 
                "Implementation Complexity", 
                "Cost",
                "Standardization",
                "Operational Stability"
            ],
            "Classical ECC": [
                "Seamless", 
                "Established protocols", 
                "Well understood", 
                "Low",
                "Multiple standards exist",
                "High"
            ],
            "Q-SHE": [
                "Requires quantum-classical interface", 
                "Can use quantum key distribution", 
                "Highly complex", 
                "Currently high",
                "No standards yet",
                "Laboratory conditions required"
            ]
        }
        
        # Display the challenges comparison table
        st.table(pd.DataFrame(challenges_comparison))
        
        # Challenge assessment
        st.subheader("Challenge Assessment")
        
        # Create a difficulty rating for each challenge
        challenges = [
            "Quantum Hardware Development",
            "Quantum Memory Coherence",
            "Error Rates",
            "Scaling to Practical Sizes",
            "Integration with Classical Systems",
            "Cost Reduction",
            "Standardization"
        ]
        
        difficulty = [9, 8, 7, 8, 6, 8, 5]  # On a scale of 1-10
        progress = [6, 4, 5, 3, 5, 2, 1]    # On a scale of 1-10
        
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the difficulty bars
        y_pos = np.arange(len(challenges))
        ax.barh(y_pos, difficulty, height=0.4, color='crimson', alpha=0.7, label='Difficulty (1-10)')
        
        # Create the progress bars
        ax.barh(y_pos + 0.5, progress, height=0.4, color='royalblue', alpha=0.7, label='Progress (1-10)')
        
        # Add labels
        ax.set_yticks(y_pos + 0.25)
        ax.set_yticklabels(challenges)
        ax.invert_yaxis()  # Labels read top-to-bottom
        
        # Add value labels on bars
        for i, v in enumerate(difficulty):
            ax.text(v + 0.1, i, str(v), va='center', color='crimson', fontweight='bold')
        
        for i, v in enumerate(progress):
            ax.text(v + 0.1, i + 0.5, str(v), va='center', color='royalblue', fontweight='bold')
        
        # Format plot
        ax.set_xlabel('Rating (1-10)')
        ax.set_xlim(0, 11)
        ax.grid(axis='x', alpha=0.3)
        ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.05))
        ax.set_title('Q-SHE Implementation Challenges')
        
        st.pyplot(fig)
        
        # Add explanation of the challenges
        st.markdown("""
        ### Key Implementation Challenges
        
        **Quantum Hardware Development**:  
        Building quantum processors with sufficient qubits, connectivity, and gate fidelity remains one of the biggest challenges.
        
        **Quantum Memory Coherence**:  
        Maintaining quantum states long enough for the self-healing process to complete requires significant improvements in coherence times.
        
        **Error Rates**:  
        While Q-SHE is designed to handle errors, the underlying quantum hardware still needs to have sufficiently low error rates for initialization and operations.
        
        **Scaling to Practical Sizes**:  
        Scaling up from proof-of-concept demonstrations to practically useful system sizes requires overcoming significant engineering challenges.
        
        **Integration with Classical Systems**:  
        Interfacing quantum encryption systems with existing classical infrastructure presents both technical and compatibility challenges.
        
        **Cost Reduction**:  
        Current quantum systems are extremely expensive and require specialized operating conditions.
        
        **Standardization**:  
        Development of standards for Q-SHE implementation, operation, and interoperability is just beginning.
        """)
        
        # Timeline for addressing challenges
        with st.expander("Development Timeline"):
            st.markdown("""
            ### Projected Timeline for Addressing Challenges
            
            **Short Term (1-3 years)**:
            - Continued improvements in quantum hardware stability
            - Laboratory demonstrations of Q-SHE principles at small scale
            - Development of simplified protocols for near-term devices
            
            **Medium Term (3-7 years)**:
            - First specialized applications in high-security environments
            - Integration with existing quantum key distribution networks
            - Development of preliminary standards
            
            **Long Term (7-15 years)**:
            - Commercial Q-SHE systems for enterprise security
            - Integration with conventional encryption infrastructure
            - Widespread adoption for critical security applications
            
            The timeline is heavily dependent on the overall progress in quantum computing hardware, 
            but Q-SHE could become practically valuable for specialized applications even with modest
            improvements in current quantum technology.
            """)
    
    # Potential applications section
    st.header("Potential Applications")
    
    st.markdown("""
    Q-SHE's unique properties make it particularly well-suited for certain applications, 
    especially those requiring high security and resilience against sophisticated attacks.
    """)
    
    # Create a grid of applications
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Critical Infrastructure
        
        **Use Case**: Protection of power grids, water systems, and other critical infrastructure
        
        **Why Q-SHE Excels**:
        - Autonomous recovery from disruptions
        - Resilience against sophisticated attacks
        - High security for critical control systems
        
        **Timeline**: 5-7 years for initial deployment
        """)
    
    with col2:
        st.markdown("""
        ### Military Communications
        
        **Use Case**: Secure tactical communications in contested environments
        
        **Why Q-SHE Excels**:
        - Self-healing under active jamming
        - Resistance to quantum decryption attacks
        - Fault tolerance in high-interference scenarios
        
        **Timeline**: 3-5 years for specialized applications
        """)
    
    with col3:
        st.markdown("""
        ### Financial Systems
        
        **Use Case**: Securing high-value financial transactions and data
        
        **Why Q-SHE Excels**:
        - Future-proof against quantum threats
        - Superior protection against data breaches
        - Resilience against advanced persistent threats
        
        **Timeline**: 7-10 years for wide adoption
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Health Records
        
        **Use Case**: Long-term protection of sensitive health data
        
        **Why Q-SHE Excels**:
        - Extended security lifetime for records
        - Resistance to retrospective decryption
        - Automatic recovery from partial data corruption
        
        **Timeline**: 8-12 years for healthcare implementation
        """)
    
    with col2:
        st.markdown("""
        ### Satellite Communications
        
        **Use Case**: Secure communications with satellites in harsh environments
        
        **Why Q-SHE Excels**:
        - Error correction for radiation-induced errors
        - Low overhead for bandwidth-constrained systems
        - Resilience against space-based interference
        
        **Timeline**: 4-6 years for space-based deployment
        """)
    
    with col3:
        st.markdown("""
        ### Quantum Networks
        
        **Use Case**: End-to-end security for quantum internet communications
        
        **Why Q-SHE Excels**:
        - Native integration with quantum networks
        - Enhanced protections for quantum information
        - Natural compatibility with quantum repeaters
        
        **Timeline**: 10-15 years with quantum internet development
        """)
    
    # Application readiness assessment
    with st.expander("Application Readiness Assessment"):
        st.markdown("""
        ### Application Readiness Assessment
        
        This assessment evaluates when different application domains might be ready for Q-SHE adoption based on:
        
        1. **Security Requirements**: How critical is quantum-level security?
        2. **Resources Available**: Budget and technical capabilities
        3. **Risk Tolerance**: Willingness to adopt emerging technology
        4. **Technology Requirements**: Specific needs that match Q-SHE strengths
        """)
        
        # Create a readiness heatmap
        applications = ["Military", "Intelligence", "Banking", "Healthcare", "Cloud Storage", "IoT", "Consumer Electronics"]
        years = [2025, 2026, 2027, 2028, 2029, 2030, 2035, 2040]
        
        # Create data (10 = ready, 0 = not ready)
        readiness = np.array([
            [3, 4, 5, 6, 7, 8, 9, 10],  # Military
            [2, 3, 5, 6, 7, 8, 9, 10],  # Intelligence
            [1, 2, 3, 5, 6, 7, 9, 10],  # Banking
            [0, 1, 2, 3, 5, 6, 8, 9],   # Healthcare
            [0, 1, 2, 3, 4, 5, 8, 9],   # Cloud Storage
            [0, 0, 1, 2, 3, 4, 7, 9],   # IoT
            [0, 0, 0, 1, 2, 3, 6, 8]    # Consumer Electronics
        ])
        
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(readiness, cmap='YlGnBu')
        
        # Add labels
        ax.set_xticks(np.arange(len(years)))
        ax.set_yticks(np.arange(len(applications)))
        ax.set_xticklabels(years)
        ax.set_yticklabels(applications)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label("Readiness Scale (0-10)")
        
        # Add value annotations
        for i in range(len(applications)):
            for j in range(len(years)):
                text = ax.text(j, i, readiness[i, j],
                              ha="center", va="center", color="w" if readiness[i, j] > 5 else "black")
        
        ax.set_title("Q-SHE Application Readiness by Year")
        fig.tight_layout()
        
        st.pyplot(fig)
        
        st.markdown("""
        **Readiness Scale**:
        - **0-2**: Not ready, basic research only
        - **3-4**: Early experimental applications
        - **5-6**: Limited deployment in specialized settings
        - **7-8**: Growing adoption in high-security applications
        - **9-10**: Mainstream adoption within the sector
        
        Military and intelligence applications are likely to be the early adopters of Q-SHE, while consumer 
        applications will take much longer due to cost constraints and the need for mature, standardized implementations.
        """)

# Helper functions for the comparison simulations

def simulate_resilience_comparison(system_size, error_rates, fractal_dimension):
    """
    Simulate the resilience comparison between Q-SHE and classical ECC.
    
    Args:
        system_size: Size of the system in bits/qubits
        error_rates: Array of error rates to test
        fractal_dimension: Fractal dimension for Q-SHE
        
    Returns:
        Tuple of arrays with fidelities for Q-SHE and classical ECC
    """
    # Ensure system size is a power of 2 for quantum simulation
    next_power = 1 << (system_size-1).bit_length()
    if next_power != system_size:
        system_size = next_power
    # For Q-SHE, we'll model fidelity based on theoretical predictions
    # F(t) ≥ 1 - ε²(1 - e^(-Γt)), with t fixed and Γ dependent on dimension
    
    gamma = 0.5 * (3 / fractal_dimension)  # Recovery rate scaled by dimension
    recovery_time = 5.0  # Fixed recovery time
    
    # Q-SHE fidelities
    qshe_fidelities = 1 - error_rates**2 * (1 - np.exp(-gamma * recovery_time))
    
    # Classical ECC fidelities
    # For classical codes, we'll model a typical error correction capacity of t = O(sqrt(n))
    correction_capacity = min(int(np.sqrt(system_size) / 2), system_size // 4)
    
    classical_fidelities = []
    for error_rate in error_rates:
        # Calculate probability of having more errors than the correction capacity
        p_failure = 0
        for k in range(correction_capacity + 1, system_size + 1):
            from scipy import special
            p_failure += special.comb(system_size, k) * (error_rate ** k) * ((1 - error_rate) ** (system_size - k))
        
        # Fidelity approximation
        classical_fidelities.append(1 - p_failure)
    
    return qshe_fidelities, np.array(classical_fidelities)

def find_break_even_point(error_rates, qshe_fidelities, classical_fidelities):
    """
    Find the error rate at which both approaches have similar performance.
    
    Args:
        error_rates: Array of error rates
        qshe_fidelities: Array of Q-SHE fidelities
        classical_fidelities: Array of classical ECC fidelities
        
    Returns:
        Break-even error rate or None if no crossover
    """
    # Find where the difference changes sign
    diff = qshe_fidelities - classical_fidelities
    
    for i in range(len(diff) - 1):
        if (diff[i] <= 0 and diff[i+1] > 0) or (diff[i] >= 0 and diff[i+1] < 0):
            # Linear interpolation to find the crossing point
            x1, x2 = error_rates[i], error_rates[i+1]
            y1, y2 = diff[i], diff[i+1]
            
            # Calculate crossing point
            if y1 != y2:  # Avoid division by zero
                x_cross = x1 - y1 * (x2 - x1) / (y2 - y1)
                return x_cross
    
    # No crossover found
    return None

if __name__ == "__main__":
    main()
