"""
Advanced Lookahead Visualization.
Creates a combined visualization showing:
1. Cumulative TCO over time (like Graph 1) for each lookahead configuration
2. 3D surface plot showing Lookahead x Time x TCO
3. Heatmap showing algorithm x lookahead performance
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import time
from tqdm import tqdm
from config import SimConfig
from topology import SERVICE_TOPOLOGY
from cost_model import generate_gbm_demand, propagate_demand, calculate_complex_cost
from algorithms import solve_da_ma_complex, solve_reactive, solve_global_optimal_dp, solve_iterative_dp
from data_generator import load_trace_from_csv

def solve_with_history(algorithm, services, full_trace, horizon, lookahead):
    """
    Run an algorithm and return BOTH final TCO and cumulative history over time.
    """
    current_states = [('Greenfield', 'None', 'None', 0) for _ in services]
    cumulative = 0.0
    history = []
    
    # For Iterative DP with rolling window
    if algorithm == 'Iterative DP':
        for t in range(horizon):
            slice_trace = {}
            for s in services:
                future = full_trace[s][t : t + lookahead]
                if len(future) < lookahead:
                    future = np.pad(future, (0, lookahead - len(future)), 'edge')
                slice_trace[s] = future
            
            plan = solve_iterative_dp(services, slice_trace, lookahead)
            next_states = [plan[s][0] for s in services]
            
            for i, svc in enumerate(services):
                vol = full_trace[svc][t]
                cumulative += calculate_complex_cost(svc, vol, next_states[i], current_states[i])
            
            history.append(cumulative)
            current_states = next_states
    
    # For DA-MA
    elif algorithm == 'DA-MA':
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
            
            history.append(cumulative)
            current_states = next_states
    
    # For Reactive
    elif algorithm == 'Reactive':
        for t in range(horizon):
            slice_data = {s: [full_trace[s][t]] for s in services}
            next_states = solve_reactive(services, current_states, slice_data)
            
            for i, svc in enumerate(services):
                vol = full_trace[svc][t]
                cumulative += calculate_complex_cost(svc, vol, next_states[i], current_states[i])
            
            history.append(cumulative)
            current_states = next_states
    
    # For Decoupled DP (full horizon solve once)
    elif algorithm == 'Decoupled DP':
        optimal_plan = solve_global_optimal_dp(services, full_trace, horizon)
        for t in range(horizon):
            next_states = [optimal_plan[s][t] for s in services]
            for i, svc in enumerate(services):
                vol = full_trace[svc][t]
                cumulative += calculate_complex_cost(svc, vol, next_states[i], current_states[i])
            history.append(cumulative)
            current_states = next_states
    
    return history

def run_advanced_visualization():
    """
    Create advanced multi-panel visualization combining Graph 1 and Graph 5.
    """
    print("=" * 60)
    print("  ADVANCED LOOKAHEAD VISUALIZATION")
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
        print("      Generated new trace")
    
    lookaheads = [1, 3, 6, 9, 12, 15, 18]
    
    # Collect all histories
    histories = {}
    
    print("\n[2/4] Computing baselines...")
    histories['Reactive'] = solve_with_history('Reactive', services, full_trace, horizon, 1)
    print(f"    Reactive: ${histories['Reactive'][-1]:,.0f}")
    
    histories['Decoupled DP'] = solve_with_history('Decoupled DP', services, full_trace, horizon, horizon)
    print(f"    Decoupled DP: ${histories['Decoupled DP'][-1]:,.0f}")
    
    print("\n[3/4] Testing algorithms with different lookaheads...")
    
    # DA-MA with different lookaheads
    for la in tqdm(lookaheads, desc="  DA-MA"):
        key = f'DA-MA (L={la})'
        histories[key] = solve_with_history('DA-MA', services, full_trace, horizon, la)
        tqdm.write(f"    {key}: ${histories[key][-1]:,.0f}")
    
    # Iterative DP with different lookaheads
    for la in tqdm(lookaheads, desc="  Iter-DP"):
        key = f'Iter-DP (L={la})'
        histories[key] = solve_with_history('Iterative DP', services, full_trace, horizon, la)
        tqdm.write(f"    {key}: ${histories[key][-1]:,.0f}")
    
    print("\n[4/4] Generating advanced visualization...")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # === Panel 1: Cumulative TCO over time for DA-MA ===
    ax1 = fig.add_subplot(gs[0, 0])
    months = np.arange(1, horizon + 1)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(lookaheads)))
    for i, la in enumerate(lookaheads):
        ax1.plot(months, histories[f'DA-MA (L={la})'], color=colors[i], 
                 linewidth=2, label=f'L={la}')
    
    ax1.plot(months, histories['Reactive'], 'm--', linewidth=1.5, alpha=0.7, label='Reactive')
    ax1.plot(months, histories['Decoupled DP'], 'k:', linewidth=1.5, alpha=0.7, label='Lower Bound')
    
    ax1.set_xlabel('Month', fontsize=11)
    ax1.set_ylabel('Cumulative TCO ($)', fontsize=11)
    ax1.set_title('DA-MA: Cumulative TCO by Lookahead', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # === Panel 2: Cumulative TCO over time for Iterative DP ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, la in enumerate(lookaheads):
        ax2.plot(months, histories[f'Iter-DP (L={la})'], color=colors[i], 
                 linewidth=2, label=f'L={la}')
    
    ax2.plot(months, histories['Reactive'], 'm--', linewidth=1.5, alpha=0.7, label='Reactive')
    ax2.plot(months, histories['Decoupled DP'], 'k:', linewidth=1.5, alpha=0.7, label='Lower Bound')
    
    ax2.set_xlabel('Month', fontsize=11)
    ax2.set_ylabel('Cumulative TCO ($)', fontsize=11)
    ax2.set_title('Iterative DP: Cumulative TCO by Lookahead', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # === Panel 3: Final TCO comparison (bar chart) ===
    ax3 = fig.add_subplot(gs[0, 2])
    
    x = np.arange(len(lookaheads))
    width = 0.35
    
    dma_finals = [histories[f'DA-MA (L={la})'][-1] for la in lookaheads]
    iter_finals = [histories[f'Iter-DP (L={la})'][-1] for la in lookaheads]
    
    bars1 = ax3.bar(x - width/2, dma_finals, width, label='DA-MA', color='#2ca02c', alpha=0.8)
    bars2 = ax3.bar(x + width/2, iter_finals, width, label='Iterative DP', color='#17becf', alpha=0.8)
    
    ax3.axhline(y=histories['Reactive'][-1], color='m', linestyle='--', linewidth=1.5, label='Reactive')
    ax3.axhline(y=histories['Decoupled DP'][-1], color='k', linestyle=':', linewidth=1.5, label='Lower Bound')
    
    ax3.set_xlabel('Lookahead Window (Months)', fontsize=11)
    ax3.set_ylabel('Final TCO ($)', fontsize=11)
    ax3.set_title('Final TCO by Algorithm & Lookahead', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(la) for la in lookaheads])
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # === Panel 4: Heatmap of % Gap from Lower Bound ===
    ax4 = fig.add_subplot(gs[1, 0])
    
    lb = histories['Decoupled DP'][-1]
    algorithms = ['Reactive', 'DA-MA', 'Iter-DP']
    gap_data = []
    
    # Reactive (no lookahead variation)
    reactive_gap = ((histories['Reactive'][-1] - lb) / lb) * 100
    gap_data.append([reactive_gap] * len(lookaheads))
    
    # DA-MA
    dma_gaps = [((histories[f'DA-MA (L={la})'][-1] - lb) / lb) * 100 for la in lookaheads]
    gap_data.append(dma_gaps)
    
    # Iter-DP
    iter_gaps = [((histories[f'Iter-DP (L={la})'][-1] - lb) / lb) * 100 for la in lookaheads]
    gap_data.append(iter_gaps)
    
    gap_array = np.array(gap_data)
    im = ax4.imshow(gap_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=150)
    
    ax4.set_xticks(np.arange(len(lookaheads)))
    ax4.set_yticks(np.arange(len(algorithms)))
    ax4.set_xticklabels([str(la) for la in lookaheads])
    ax4.set_yticklabels(algorithms)
    ax4.set_xlabel('Lookahead Window (Months)', fontsize=11)
    ax4.set_title('% Gap from Lower Bound', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(lookaheads)):
            text = ax4.text(j, i, f'{gap_array[i, j]:.0f}%',
                           ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Gap (%)', fontsize=10)
    
    # === Panel 5: Line chart showing improvement with lookahead ===
    ax5 = fig.add_subplot(gs[1, 1:])
    
    ax5.plot(lookaheads, dma_finals, 'g-o', linewidth=2.5, markersize=12, label='DA-MA')
    ax5.plot(lookaheads, iter_finals, 'c-s', linewidth=2.5, markersize=12, label='Iterative DP')
    
    ax5.axhline(y=histories['Reactive'][-1], color='m', linestyle='--', linewidth=2, label='Reactive')
    ax5.axhline(y=histories['Decoupled DP'][-1], color='k', linestyle=':', linewidth=2, label='Lower Bound')
    
    # Add annotations for best values
    best_dma_idx = np.argmin(dma_finals)
    best_iter_idx = np.argmin(iter_finals)
    
    ax5.annotate(f'Best: ${dma_finals[best_dma_idx]/1000:.0f}K', 
                xy=(lookaheads[best_dma_idx], dma_finals[best_dma_idx]),
                xytext=(lookaheads[best_dma_idx] + 0.5, dma_finals[best_dma_idx] + 50000),
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    ax5.annotate(f'Best: ${iter_finals[best_iter_idx]/1000:.0f}K', 
                xy=(lookaheads[best_iter_idx], iter_finals[best_iter_idx]),
                xytext=(lookaheads[best_iter_idx] - 2, iter_finals[best_iter_idx] - 80000),
                fontsize=10, color='#17becf', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#17becf', lw=1.5))
    
    ax5.set_xlabel('Lookahead Window (Months)', fontsize=12)
    ax5.set_ylabel('Final TCO ($)', fontsize=12)
    ax5.set_title('Impact of Lookahead on Final TCO', fontsize=14, fontweight='bold')
    ax5.set_xticks(lookaheads)
    ax5.legend(loc='upper right', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # Fill between to show gap from lower bound
    ax5.fill_between(lookaheads, histories['Decoupled DP'][-1], iter_finals, 
                     alpha=0.2, color='cyan', label='_nolegend_')
    
    plt.suptitle('Graph 5: Lookahead Sensitivity Analysis\n(Combined View)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/graph5_advanced.png', dpi=150, bbox_inches='tight')
    print("      Saved: output/graph5_advanced.png")
    
    try:
        plt.show()
    except:
        pass
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  Lower Bound: ${lb:,.0f}")
    print(f"  Reactive:    ${histories['Reactive'][-1]:,.0f}")
    print()
    print(f"  Best DA-MA:      L={lookaheads[best_dma_idx]:2d} -> ${dma_finals[best_dma_idx]:,.0f}")
    print(f"  Best Iter-DP:    L={lookaheads[best_iter_idx]:2d} -> ${iter_finals[best_iter_idx]:,.0f}")
    print("=" * 60)

if __name__ == "__main__":
    run_advanced_visualization()
