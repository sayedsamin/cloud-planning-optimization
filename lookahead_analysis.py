"""
Lookahead Sensitivity Analysis.
Compares algorithm performance with different lookahead windows (1, 3, 6, 12 months).
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from config import SimConfig
from topology import SERVICE_TOPOLOGY, DECISION_CATALOG
from cost_model import generate_gbm_demand, propagate_demand, calculate_complex_cost
from algorithms import solve_da_ma_complex, solve_reactive, solve_global_optimal_dp, solve_iterative_dp
from data_generator import load_trace_from_csv

def solve_rolling_iterative_dp(services, full_trace, horizon, lookahead):
    """
    Rolling-window Iterative DP: At each time step, solve a lookahead-length DP
    and take the first step's decision.
    """
    current_states = [('Greenfield', 'None', 'None', 0) for _ in services]
    cumulative = 0.0
    
    for t in range(horizon):
        # Create a slice of the trace for the lookahead window
        slice_trace = {}
        for s in services:
            future = full_trace[s][t : t + lookahead]
            if len(future) < lookahead:
                future = np.pad(future, (0, lookahead - len(future)), 'edge')
            slice_trace[s] = future
        
        # Solve Iterative DP for this window
        plan = solve_iterative_dp(services, slice_trace, lookahead)
        
        # Take the first step's decision (index 0)
        next_states = [plan[s][0] for s in services]
        
        # Calculate cost
        for i, svc in enumerate(services):
            vol = full_trace[svc][t]
            cumulative += calculate_complex_cost(svc, vol, next_states[i], current_states[i])
        
        current_states = next_states
    
    return cumulative

def solve_dama_with_lookahead(services, full_trace, horizon, lookahead):
    """
    DA-MA (Genetic Algorithm) with a specific lookahead window.
    """
    current_states = [('Greenfield', 'None', 'None', 0) for _ in services]
    cumulative = 0.0
    
    for t in range(horizon):
        slice_data = {}
        for s in services:
            future = full_trace[s][t : t + lookahead]
            if len(future) < lookahead:
                future = np.pad(future, (0, lookahead - len(future)), 'edge')
            slice_data[s] = future
        
        next_states = solve_da_ma_complex(services, current_states, slice_data, lookahead=lookahead)
        
        for i, svc in enumerate(services):
            vol = full_trace[svc][t]
            cumulative += calculate_complex_cost(svc, vol, next_states[i], current_states[i])
        
        current_states = next_states
    
    return cumulative

def run_lookahead_analysis():
    """
    Compare algorithm performance across different lookahead windows.
    """
    print("=" * 60)
    print("  LOOKAHEAD SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    horizon = SimConfig.SCENARIO_HORIZON
    services = list(SERVICE_TOPOLOGY.keys())
    
    # Load trace
    print("\n[1/4] Loading demand trace...")
    trace_path = 'simulation_data.csv'
    if os.path.exists(trace_path):
        full_trace = load_trace_from_csv(trace_path)
        print(f"      Loaded from {trace_path}")
    else:
        root_trace = generate_gbm_demand(horizon + 12, SimConfig.GBM_MU, SimConfig.GBM_SIGMA, 
                                         SimConfig.GBM_START, SimConfig.GBM_SEED)
        full_trace = propagate_demand(root_trace, SERVICE_TOPOLOGY)
        print("      Generated new trace (GBM)")
    
    # Lookahead values to test
    lookaheads = [1, 3, 6, 12]
    results = {}
    
    print(f"\n[2/4] Computing baselines...")
    
    start_time = time.time()
    
    # Reactive (lookahead=1, by definition)
    current_states = [('Greenfield', 'None', 'None', 0) for _ in services]
    reactive_tco = 0.0
    for t in range(horizon):
        slice_data = {s: [full_trace[s][t]] for s in services}
        next_states = solve_reactive(services, current_states, slice_data)
        for i, svc in enumerate(services):
            vol = full_trace[svc][t]
            reactive_tco += calculate_complex_cost(svc, vol, next_states[i], current_states[i])
        current_states = next_states
    results['Reactive'] = reactive_tco
    print(f"    Reactive:           ${reactive_tco:,.2f}")
    
    # Decoupled DP (Lower Bound - full horizon, ignores coupling)
    optimal_plan = solve_global_optimal_dp(services, full_trace, horizon)
    current_states = [('Greenfield', 'None', 'None', 0) for _ in services]
    optimal_tco = 0.0
    for t in range(horizon):
        next_states = [optimal_plan[s][t] for s in services]
        for i, svc in enumerate(services):
            vol = full_trace[svc][t]
            optimal_tco += calculate_complex_cost(svc, vol, next_states[i], current_states[i])
        current_states = next_states
    results['Decoupled DP (Full)'] = optimal_tco
    print(f"    Decoupled DP (Full): ${optimal_tco:,.2f}")
    
    # Test DA-MA with different lookaheads
    print(f"\n[3/4] Testing DA-MA with lookaheads: {lookaheads}")
    for la in tqdm(lookaheads, desc="  DA-MA", unit="config"):
        tco = solve_dama_with_lookahead(services, full_trace, horizon, la)
        results[f'DA-MA (L={la})'] = tco
        tqdm.write(f"    DA-MA (L={la:2d}):        ${tco:,.2f}")
    
    # Test Iterative DP with different lookaheads
    print(f"\n[4/4] Testing Iterative DP with lookaheads: {lookaheads}")
    for la in tqdm(lookaheads, desc="  Iter-DP", unit="config"):
        tco = solve_rolling_iterative_dp(services, full_trace, horizon, la)
        results[f'Iterative DP (L={la})'] = tco
        tqdm.write(f"    Iter-DP (L={la:2d}):      ${tco:,.2f}")
    
    elapsed = time.time() - start_time
    print(f"\n      Completed in {elapsed/60:.1f} minutes")
    
    # Plot Graph 5: Lookahead Sensitivity
    print("\nGenerating Graph 5...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot DA-MA results
    dma_tcos = [results[f'DA-MA (L={la})'] for la in lookaheads]
    ax.plot(lookaheads, dma_tcos, 'g-o', linewidth=2, markersize=10, label='DA-MA (Genetic)')
    
    # Plot Iterative DP results
    iter_tcos = [results[f'Iterative DP (L={la})'] for la in lookaheads]
    ax.plot(lookaheads, iter_tcos, 'c-s', linewidth=2, markersize=10, label='Iterative DP')
    
    # Add horizontal baselines
    ax.axhline(y=results['Reactive'], color='m', linestyle='--', linewidth=1.5, label='Reactive')
    ax.axhline(y=results['Decoupled DP (Full)'], color='k', linestyle=':', linewidth=1.5, label='Decoupled DP (Lower Bound)')
    
    ax.set_xlabel('Lookahead Window (Months)', fontsize=12)
    ax.set_ylabel('Total Cost of Ownership ($)', fontsize=12)
    ax.set_title('Graph 5: Impact of Lookahead Window on Algorithm Performance', fontsize=14)
    ax.set_xticks(lookaheads)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/graph5_lookahead.png', dpi=150)
    print("      Saved: output/graph5_lookahead.png")
    
    try:
        plt.show()
    except:
        pass
    
    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  Decoupled DP (Lower Bound):   ${results['Decoupled DP (Full)']:>15,.2f}")
    print(f"  Reactive (Myopic):            ${results['Reactive']:>15,.2f}")
    print()
    print("  --- DA-MA (Genetic Algorithm) ---")
    for la in lookaheads:
        tco = results[f'DA-MA (L={la})']
        gap = ((tco - results['Decoupled DP (Full)']) / results['Decoupled DP (Full)']) * 100
        print(f"    L={la:2d}:  ${tco:>15,.2f}  ({gap:+.1f}% vs LB)")
    print()
    print("  --- Iterative DP ---")
    for la in lookaheads:
        tco = results[f'Iterative DP (L={la})']
        gap = ((tco - results['Decoupled DP (Full)']) / results['Decoupled DP (Full)']) * 100
        print(f"    L={la:2d}:  ${tco:>15,.2f}  ({gap:+.1f}% vs LB)")
    
    print("\n" + "=" * 60)
    
    return results

if __name__ == "__main__":
    run_lookahead_analysis()


