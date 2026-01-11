"""
Monte Carlo Sensitivity Analysis for Transition Costs.
Run this script to generate Graph 4 showing TCO distribution across random transition cost samples.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from config import SimConfig
from topology import SERVICE_TOPOLOGY
from cost_model import generate_gbm_demand, propagate_demand, calculate_complex_cost
from algorithms import solve_da_ma_complex, solve_reactive, solve_global_optimal_dp, solve_iterative_dp
from data_generator import load_trace_from_csv

def generate_random_transition_matrix(services, modes=['SaaS', 'PaaS', 'IaaS', 'Greenfield'], low=10, high=10000):
    """
    Generate a random transition cost matrix with log-uniform sampling.
    Returns: dict of (service_name, from_mode, to_mode) -> cost
    """
    matrix = {}
    for svc in services:
        complexity = SERVICE_TOPOLOGY[svc]['complexity']
        for f_mode in modes:
            for t_mode in modes:
                if f_mode != t_mode:
                    # Log-uniform: sample uniformly in log-space
                    log_cost = np.random.uniform(np.log(low), np.log(high))
                    base_cost = np.exp(log_cost)
                    # Scale by service complexity
                    matrix[(svc, f_mode, t_mode)] = base_cost * complexity
    return matrix

def solve_strategy_with_matrix(name, full_trace, horizon, transition_cost_matrix=None):
    """
    Run a strategy and return final TCO, passing custom transition cost matrix.
    """
    services = list(SERVICE_TOPOLOGY.keys())
    current_states = [('Greenfield', 'None', 'None', 0) for _ in services]
    cumulative = 0.0
    
    # Pre-computation for Optimal Strategy
    optimal_plan = None
    if name == 'Optimal':
        optimal_plan = solve_global_optimal_dp(services, full_trace, horizon)
    elif name == 'Iterative DP':
        optimal_plan = solve_iterative_dp(services, full_trace, horizon)

    for t in range(horizon):
        # 1. Decide
        if name == 'Proposed':
            LOOKAHEAD = 12
            slice_data = {}
            for s in services:
                future = full_trace[s][t : t+LOOKAHEAD]
                if len(future) < LOOKAHEAD: future = np.pad(future, (0, LOOKAHEAD-len(future)), 'edge')
                slice_data[s] = future
            next_states = solve_da_ma_complex(services, current_states, slice_data, lookahead=LOOKAHEAD)
        elif name == 'Reactive':
            slice_data = {s: [full_trace[s][t]] for s in services}
            next_states = solve_reactive(services, current_states, slice_data)
        elif name == 'Static SaaS':
            next_states = [('SaaS', 'Standard', 'us-east-1', 0)] * len(services)
        elif name == 'Static IaaS':
            next_states = [('IaaS', 'OD', 'us-east-1', 0)] * len(services)
        elif name == 'Optimal':
            next_states = [optimal_plan[s][t] for s in services]
        elif name == 'Iterative DP':
            next_states = [optimal_plan[s][t] for s in services]
        else: 
            next_states = current_states
        
        # 2. Calculate Cost with custom matrix
        monthly_total = 0.0
        for i, svc in enumerate(services):
            vol = full_trace[svc][t]
            cost = calculate_complex_cost(svc, vol, next_states[i], current_states[i], 
                                           transition_cost_matrix=transition_cost_matrix)
            monthly_total += cost
            
        cumulative += monthly_total
        current_states = next_states
        
    return cumulative  # Return final TCO

def run_monte_carlo_sensitivity(n_trials=100, verbose=True):
    """
    Run Monte Carlo sensitivity analysis on transition costs.
    For each trial, sample random transition costs and run simulation.
    
    Args:
        n_trials: Number of Monte Carlo trials
        verbose: If True, show detailed progress
    """
    print("=" * 60)
    print("  MONTE CARLO SENSITIVITY ANALYSIS")
    print("  Transition Costs ~ Log-Uniform(10, 10000)")
    print(f"  Trials: {n_trials}")
    print("=" * 60)
    
    horizon = SimConfig.SCENARIO_HORIZON
    services = list(SERVICE_TOPOLOGY.keys())
    
    # Load trace
    print("\n[1/3] Loading demand trace...")
    trace_path = 'simulation_data.csv'
    if os.path.exists(trace_path):
        full_trace = load_trace_from_csv(trace_path)
        print(f"      Loaded from {trace_path}")
    else:
        root_trace = generate_gbm_demand(horizon + 12, SimConfig.GBM_MU, SimConfig.GBM_SIGMA, SimConfig.GBM_START, SimConfig.GBM_SEED)
        full_trace = propagate_demand(root_trace, SERVICE_TOPOLOGY)
        print(f"      Generated new trace (GBM)")
    
    algorithms = ['Static SaaS', 'Static IaaS', 'Reactive', 'Proposed']
    results = {alg: [] for alg in algorithms}
    
    # Count total iterations for progress bar
    total_iterations = n_trials * len(algorithms)
    
    print(f"\n[2/3] Running {n_trials} trials x {len(algorithms)} algorithms = {total_iterations} iterations")
    print("      (Each 'Proposed' run uses GA, which is slow)")
    print()
    
    start_time = time.time()
    
    # Main progress bar for trials
    trial_pbar = tqdm(range(n_trials), 
                      desc="Trials", 
                      unit="trial",
                      ncols=100,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for trial in trial_pbar:
        # Generate random transition cost matrix
        trans_matrix = generate_random_transition_matrix(services)
        
        # Run each algorithm with this matrix
        for alg in algorithms:
            trial_pbar.set_postfix_str(f"Running: {alg}")
            tco = solve_strategy_with_matrix(alg, full_trace, horizon, trans_matrix)
            results[alg].append(tco)
    
    elapsed = time.time() - start_time
    print(f"\n      Completed in {elapsed/60:.1f} minutes ({elapsed:.1f}s)")
    
    # Plot Graph 4: Box Plot of TCO Distribution
    print("\n[3/3] Generating Graph 4...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    data = [results[alg] for alg in algorithms]
    bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
    
    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Total Cost of Ownership ($)', fontsize=12)
    ax.set_title(f'Graph 4: Monte Carlo Sensitivity Analysis (N={n_trials})\nTransition Costs ~ Log-Uniform(10, 10000)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/graph4_monte_carlo.png', dpi=150)
    print("      Saved: output/graph4_monte_carlo.png")
    
    try:
        plt.show()
    except:
        pass
    
    # Print Summary Statistics
    print("\n" + "=" * 60)
    print("  SUMMARY STATISTICS")
    print("=" * 60)
    for alg in algorithms:
        arr = np.array(results[alg])
        print(f"\n  {alg}:")
        print(f"    Mean TCO:    ${arr.mean():>15,.2f}")
        print(f"    Std Dev:     ${arr.std():>15,.2f}")
        print(f"    Min:         ${arr.min():>15,.2f}")
        print(f"    5th %:       ${np.percentile(arr, 5):>15,.2f}")
        print(f"    Median:      ${np.percentile(arr, 50):>15,.2f}")
        print(f"    95th %:      ${np.percentile(arr, 95):>15,.2f}")
        print(f"    Max:         ${arr.max():>15,.2f}")
    
    print("\n" + "=" * 60)
    
    return results

if __name__ == "__main__":
    run_monte_carlo_sensitivity(n_trials=100)

