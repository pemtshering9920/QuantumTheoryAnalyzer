import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
from utils.database import (
    save_simulation_result,
    get_simulation_results,
    get_simulation_result_by_id,
    get_simulation_stats,
    delete_simulation_result
)

st.set_page_config(
    page_title="Q-SHE Results Database",
    page_icon="💾",
    layout="wide"
)

def main():
    st.title("Quantum Self-Healing Encryption Results Database")
    
    st.markdown("""
    This page allows you to view, analyze, and manage your saved simulation results.
    Results from error correction, encryption demos, and resilience comparisons are 
    automatically saved to the database for future reference.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Results Explorer", "Statistics & Analysis", "Save Test Result"])
    
    with tab1:
        display_results_explorer()
    
    with tab2:
        display_statistics()
    
    with tab3:
        display_save_result_form()

def display_results_explorer():
    st.header("Results Explorer")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        simulation_type = st.selectbox(
            "Simulation Type", 
            ["All", "error_correction", "encryption", "comparison"],
            help="Filter results by simulation type"
        )
    
    with col2:
        limit = st.slider(
            "Results Limit", 
            min_value=5, 
            max_value=50, 
            value=10, 
            step=5,
            help="Maximum number of results to display"
        )
    
    with col3:
        offset = st.slider(
            "Results Offset", 
            min_value=0, 
            max_value=100, 
            value=0, 
            step=10,
            help="Number of results to skip"
        )
        
    # Get results based on filters
    type_filter = None if simulation_type == "All" else simulation_type
    results = get_simulation_results(simulation_type=type_filter, limit=limit, offset=offset)
    
    if not results:
        st.info("No simulation results found. Try running some simulations in the other sections first.")
        return
    
    # Display results table
    df = pd.DataFrame(results)
    
    # Format timestamp column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Select columns for display
    display_columns = [
        'id', 'timestamp', 'simulation_type', 'key_size', 'error_rate', 
        'recovery_time', 'initial_fidelity', 'corrupted_fidelity', 
        'recovered_fidelity', 'error_type'
    ]
    display_columns = [col for col in display_columns if col in df.columns]
    
    st.dataframe(df[display_columns], use_container_width=True)
    
    # Detail view for selected result
    st.subheader("Result Details")
    
    # Use selectbox for result selection
    if not df.empty:
        result_ids = df['id'].tolist()
        result_labels = [f"Result #{id} - {sim_type} ({time})" 
                        for id, sim_type, time in zip(df['id'], df['simulation_type'], df['timestamp'])]
        
        selected_label = st.selectbox("Select Result", result_labels)
        selected_index = result_labels.index(selected_label)
        selected_id = result_ids[selected_index]
        
        # Get full result
        result = get_simulation_result_by_id(selected_id)
        
        if result:
            # Display basic details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**ID:** {result['id']}")
                st.markdown(f"**Timestamp:** {result['timestamp']}")
                st.markdown(f"**Simulation Type:** {result['simulation_type']}")
                st.markdown(f"**Key Size:** {result['key_size']} bits")
                st.markdown(f"**Error Rate:** {result['error_rate']:.2%}")
                
                if result['error_type']:
                    st.markdown(f"**Error Type:** {result['error_type']}")
                
                if result['fractal_dimension']:
                    st.markdown(f"**Fractal Dimension:** {result['fractal_dimension']:.2f}")
            
            with col2:
                st.markdown("### Fidelity Metrics")
                
                # Create fidelity chart
                fig, ax = plt.subplots(figsize=(8, 4))
                
                stages = ['Initial', 'Corrupted', 'Recovered']
                fidelities = [
                    result['initial_fidelity'],
                    result['corrupted_fidelity'],
                    result['recovered_fidelity']
                ]
                
                # Bar colors based on values
                colors = ['green', 'red', 'blue']
                
                ax.bar(stages, fidelities, color=colors, alpha=0.7)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel('Fidelity')
                ax.set_title('Fidelity at Different Stages')
                ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Failure Threshold')
                ax.grid(axis='y', alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
            
            # Display description if available
            if result['description']:
                st.markdown("### Description")
                st.markdown(result['description'])
            
            # Display additional parameters if available
            if result['parameters']:
                st.markdown("### Additional Parameters")
                st.json(result['parameters'])
            
            # Delete button
            if st.button(f"Delete Result #{result['id']}", key=f"delete_{result['id']}"):
                success = delete_simulation_result(result['id'])
                if success:
                    st.success(f"Result #{result['id']} deleted successfully!")
                    st.rerun()  # Refresh the page
                else:
                    st.error("Failed to delete the result.")

def display_statistics():
    st.header("Statistics & Analysis")
    
    # Get statistics
    stats = get_simulation_stats()
    
    if stats['total_count'] == 0:
        st.info("No simulation results available for analysis. Try running some simulations first.")
        return
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Total Simulations:** {stats['total_count']}")
        
        # Display counts by type
        if stats['by_type']:
            st.markdown("**Simulation Types:**")
            for sim_type, count in stats['by_type'].items():
                st.markdown(f"- {sim_type}: {count}")
    
    with col2:
        st.markdown("**Average Fidelities:**")
        st.markdown(f"- Initial: {stats['avg_fidelities']['initial']:.4f}")
        st.markdown(f"- After Corruption: {stats['avg_fidelities']['corrupted']:.4f}")
        st.markdown(f"- After Recovery: {stats['avg_fidelities']['recovered']:.4f}")
        
        # Calculate average improvement
        avg_improvement = ((stats['avg_fidelities']['recovered'] - stats['avg_fidelities']['corrupted']) / 
                          max(0.001, 1 - stats['avg_fidelities']['corrupted'])) * 100
        
        st.markdown(f"**Average Recovery Improvement:** {avg_improvement:.1f}%")
    
    # Get all results for detailed analysis
    all_results = get_simulation_results(limit=100)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Analysis by error rate
        st.subheader("Recovery Performance by Error Rate")
        
        # Plot recovery performance vs error rate
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert error_rate to numeric and sort by it
        df['error_rate'] = pd.to_numeric(df['error_rate'])
        df = df.sort_values('error_rate')
        
        ax.scatter(df['error_rate'], df['corrupted_fidelity'], 
                  color='red', alpha=0.6, label='After Corruption')
        ax.scatter(df['error_rate'], df['recovered_fidelity'], 
                  color='blue', alpha=0.6, label='After Recovery')
        
        # Add trend lines
        if len(df) > 1:
            try:
                from scipy import stats as scipy_stats
                
                # Trend for corrupted fidelity
                slope, intercept, r, p, std_err = scipy_stats.linregress(
                    df['error_rate'], df['corrupted_fidelity']
                )
                ax.plot(df['error_rate'], intercept + slope * df['error_rate'], 
                      'r--', alpha=0.4, label=f'Corruption Trend (r={r:.2f})')
                
                # Trend for recovered fidelity
                slope, intercept, r, p, std_err = scipy_stats.linregress(
                    df['error_rate'], df['recovered_fidelity']
                )
                ax.plot(df['error_rate'], intercept + slope * df['error_rate'], 
                      'b--', alpha=0.4, label=f'Recovery Trend (r={r:.2f})')
            except:
                # Skip trend lines if error occurs
                pass
        
        ax.set_xlabel('Error Rate')
        ax.set_ylabel('Fidelity')
        ax.set_xlim(0, max(df['error_rate']) * 1.1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title('Recovery Performance vs Error Rate')
        
        st.pyplot(fig)
        
        # Group by different categories
        st.subheader("Analysis by Categories")
        
        analysis_by = st.selectbox(
            "Analyze By", 
            ["simulation_type", "key_size", "error_type"],
            help="Select category for analysis"
        )
        
        if analysis_by in df.columns:
            # Group results
            grouped = df.groupby(analysis_by).agg({
                'corrupted_fidelity': 'mean',
                'recovered_fidelity': 'mean',
                'id': 'count'
            }).reset_index()
            
            grouped = grouped.rename(columns={'id': 'count'})
            
            # Display grouped data
            st.dataframe(grouped, use_container_width=True)
            
            # Plot grouped data
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = range(len(grouped))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], grouped['corrupted_fidelity'], 
                  width, label='After Corruption', color='red', alpha=0.7)
            ax.bar([i + width/2 for i in x], grouped['recovered_fidelity'], 
                  width, label='After Recovery', color='blue', alpha=0.7)
            
            ax.set_xlabel(analysis_by.replace('_', ' ').title())
            ax.set_ylabel('Average Fidelity')
            ax.set_xticks(x)
            ax.set_xticklabels(grouped[analysis_by], rotation=45)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            ax.set_title(f'Average Fidelities by {analysis_by.replace("_", " ").title()}')
            
            # Add count annotations
            for i, count in enumerate(grouped['count']):
                ax.annotate(f'n={count}', 
                           xy=(i, min(grouped['corrupted_fidelity'][i], 0.1)),
                           ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)

def display_save_result_form():
    st.header("Save Test Result")
    
    st.markdown("""
    This form allows you to manually save a test simulation result to the database.
    Most simulation results are saved automatically when you run simulations in other sections.
    """)
    
    # Form inputs
    col1, col2 = st.columns(2)
    
    with col1:
        simulation_type = st.selectbox(
            "Simulation Type", 
            ["error_correction", "encryption", "comparison"],
            help="Type of simulation"
        )
        
        key_size = st.slider(
            "Key Size (bits)", 
            min_value=2, 
            max_value=64, 
            value=8,
            help="Size of the key in bits"
        )
        
        error_rate = st.slider(
            "Error Rate", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.05,
            help="Error rate used in the simulation"
        )
        
        recovery_time = st.slider(
            "Recovery Time", 
            min_value=0.0, 
            max_value=10.0, 
            value=5.0, 
            step=0.5,
            help="Recovery time used in the simulation"
        )
    
    with col2:
        initial_fidelity = st.slider(
            "Initial Fidelity", 
            min_value=0.0, 
            max_value=1.0, 
            value=1.0, 
            step=0.01,
            help="Initial fidelity value"
        )
        
        corrupted_fidelity = st.slider(
            "Corrupted Fidelity", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.01,
            help="Fidelity after error introduction"
        )
        
        recovered_fidelity = st.slider(
            "Recovered Fidelity", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.9, 
            step=0.01,
            help="Fidelity after recovery"
        )
        
        fractal_dimension = st.slider(
            "Fractal Dimension", 
            min_value=1.0, 
            max_value=3.0, 
            value=2.0, 
            step=0.1,
            help="Fractal dimension used (if applicable)"
        )
    
    # Additional fields
    error_type = st.selectbox(
        "Error Type", 
        ["Random Bit Flips", "Localized Burst Error", "Targeted Attack", "None"],
        help="Type of error introduced (if applicable)"
    )
    
    description = st.text_area(
        "Description", 
        value="Test simulation result",
        help="Text description of the simulation"
    )
    
    parameters_json = st.text_area(
        "Additional Parameters (JSON)", 
        value='{"custom_param": "value"}',
        help="Additional parameters as JSON"
    )
    
    # Validate and save
    if st.button("Save Test Result"):
        try:
            # Parse JSON
            parameters = json.loads(parameters_json)
            
            # Prepare error type
            error_type_value = None if error_type == "None" else error_type
            
            # Save result
            result_id = save_simulation_result(
                simulation_type=simulation_type,
                key_size=key_size,
                error_rate=error_rate,
                recovery_time=recovery_time,
                initial_fidelity=initial_fidelity,
                corrupted_fidelity=corrupted_fidelity,
                recovered_fidelity=recovered_fidelity,
                fractal_dimension=fractal_dimension,
                error_type=error_type_value,
                description=description,
                parameters=parameters
            )
            
            st.success(f"Test result saved successfully with ID: {result_id}")
        except json.JSONDecodeError:
            st.error("Invalid JSON in Additional Parameters field.")
        except Exception as e:
            st.error(f"Error saving result: {str(e)}")

if __name__ == "__main__":
    main()