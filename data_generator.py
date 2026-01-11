import numpy as np
import pandas as pd
import random
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ==========================================
# MODULE 1: GROUNDED INFRASTRUCTURE (AWS 2025)
# ==========================================
# Sources: AWS EC2, EKS, and VPC Pricing Pages (US-East-1)

# ==========================================
# MODULE 1: GROUNDED INFRASTRUCTURE (AWS 2025)
# ==========================================
# Sources: AWS EC2 On-Demand, RI, and Spot Advisor pages (US-East-1)

VM_CATALOG = {
    # Burstable (Dev/Test)
    't3.micro': {
        'cpu': 2, 'mem': 1.0, 'capacity_rps': 10.0,
        'price_od': 0.0104,         # On-Demand
        'price_ri': 0.0063,         # 1-Yr Standard No Upfront (~39% savings)
        'price_spot': 0.0031,       # Avg Spot Price (~70% savings)
        'spot_risk': 0.05           # 5% chance of interruption (Low risk)
    },
    't3.medium': {
        'cpu': 2, 'mem': 4.0, 'capacity_rps': 30.0,
        'price_od': 0.0416, 
        'price_ri': 0.0250, 
        'price_spot': 0.0125,
        'spot_risk': 0.05
    },
    
    # General Purpose (Production APIs)
    'm5.large': {
        'cpu': 2, 'mem': 8.0, 'capacity_rps': 80.0,
        'price_od': 0.0960, 
        'price_ri': 0.0610,         # ~36% savings
        'price_spot': 0.0350,       # ~63% savings
        'spot_risk': 0.10           # 10% Interruption risk (Medium)
    },
    'm5.xlarge': {
        'cpu': 4, 'mem': 16.0, 'capacity_rps': 160.0,
        'price_od': 0.1920, 
        'price_ri': 0.1220, 
        'price_spot': 0.0750,
        'spot_risk': 0.12           # Higher demand = Higher preemption risk
    },
    
    # Memory Optimized (Databases/Cache)
    # Note: Spot is rarely viable here due to state recovery time
    'r5.large': {
        'cpu': 2, 'mem': 16.0, 'capacity_rps': 60.0,
        'price_od': 0.1260, 
        'price_ri': 0.0790,         # ~37% savings
        'price_spot': 0.0450,       # ~64% savings
        'spot_risk': 0.15           # High risk for memory optimized
    },
}

# The "Friction" costs remain the same
INFRA_TAXES = {
    'EKS_Cluster_Hourly': 0.10,   
    'NAT_Gateway_Hourly': 0.045,  
    'NAT_Processing_GB': 0.045,   
    'ALB_Hourly': 0.0225          
}

# ==========================================
# MODULE 1.1: PAAS / SERVERLESS CATALOG (AWS Lambda)
# ==========================================
# Source: AWS Lambda Pricing (US-East-1)
# Physics: Costs scale with Memory Size and Execution Duration

PAAS_CATALOG = {
    'standard': {
        'price_per_request': 0.20 / 1_000_000, # $0.20 per 1M requests
        'price_per_gb_sec': 0.0000166667,      # Cost per GB-second
        'ephemeral_storage': 0.0000000309      # Cost per GB-second of /tmp
    },
    'provisioned': { # "Warm" Lambdas to avoid Cold Starts
        'price_per_request': 0.20 / 1_000_000,
        'price_per_gb_sec': 0.0000041667,      # Lower run cost...
        'provisioned_concurrency': 0.0000015   # ...but you pay to keep them warm
    }
}

# ==========================================
# MODULE 2: SERVICE TOPOLOGY (Online Boutique)
# ==========================================
# Purpose: Defines the dependency graph G(V,E) for the simulation.
# Structure: Key = Service, Value = Attributes & Dependencies
# Dependency Format: (Target, Prob_Call, Amplification, Payload_KB)

SERVICE_TOPOLOGY = {
    'frontend': {
        'type': 'stateless', 'complexity': 3,
        'dependencies': [
            ('ad', 1.0, 1.0, 5.0),
            ('productcatalog', 1.0, 4.0, 50.0), # Heavy fan-out
            ('recommendation', 0.8, 1.0, 10.0),
            ('cart', 0.4, 2.0, 5.0),
            ('shipping', 0.4, 1.0, 2.0),
            ('currency', 1.0, 1.0, 1.0),
            ('checkout', 0.1, 1.0, 10.0) # 10% Conversion Rate
        ]
    },
    'checkout': {
        'type': 'stateless', 'complexity': 4,
        'dependencies': [
            ('cart', 1.0, 1.0, 10.0),
            ('productcatalog', 1.0, 5.0, 20.0),
            ('shipping', 1.0, 1.0, 5.0),
            ('currency', 1.0, 1.0, 1.0),
            ('payment', 1.0, 1.0, 5.0),     # Critical Path
            ('email', 1.0, 1.0, 50.0)
        ]
    },
    'recommendation': {
        'type': 'stateless', 'complexity': 5,
        'dependencies': [('productcatalog', 1.0, 5.0, 5.0)]
    },
    # Leaf Nodes (Databases / External APIs)
    'productcatalog': {'type': 'stateful', 'complexity': 2, 'dependencies': []},
    'cart':           {'type': 'stateful', 'complexity': 2, 'dependencies': []},
    'payment':        {'type': 'stateful', 'complexity': 5, 'dependencies': []},
    'shipping':       {'type': 'stateless', 'complexity': 2, 'dependencies': []},
    'email':          {'type': 'stateless', 'complexity': 1, 'dependencies': []},
    'currency':       {'type': 'stateless', 'complexity': 1, 'dependencies': []},
    'ad':             {'type': 'stateless', 'complexity': 1, 'dependencies': []}
}

@dataclass
class ServiceProfile:
    name: str
    is_stateful: bool
    complexity: int

# ==========================================
# MODULE 3: PHYSICS ENGINE (Demand & Resources)
# ==========================================

def generate_gbm_demand(months: int, mu: float, sigma: float, start_val: float, seed: int):
    """
    Purpose: Generates stochastic User Demand (The Root Trace).
    Implements Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW
    """
    np.random.seed(seed)
    dt = 1
    t = np.linspace(0, months, months)
    W = np.cumsum(np.random.standard_normal(size=months)) * np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    demand = start_val * np.exp(drift + diffusion)
    return np.maximum(demand, 1000) # Floor at 1k requests

def propagate_demand(root_trace: np.ndarray, topology: dict):
    """
    Purpose: Cascades user demand down the microservice graph.
    Ensures backend traffic is mathematically correlated to frontend traffic.
    """
    traces = {name: np.zeros(len(root_trace)) for name in topology}
    traces['frontend'] = root_trace.copy()
    
    # Topological processing order (Manual for this specific graph)
    # Tier 1 -> Tier 2 -> Tier 3
    order = ['frontend', 'checkout', 'recommendation']
    
    for parent in order:
        parent_vol = traces[parent]
        for child, prob, amp, _ in topology[parent]['dependencies']:
            # Physics: Child = Parent * Prob * Amp
            flow = parent_vol * prob * amp
            
            # Add network jitter (+/- 5%)
            noise = np.random.normal(1.0, 0.05, len(flow))
            traces[child] += (flow * noise)
            
    return traces

def calculate_resources(service_name: str, monthly_vol: float):
    """
    Purpose: Converts abstract Volume -> Concrete VM Units.
    Uses Peak-to-Average Ratio (PAR) logic from literature.
    """
    SECONDS_PER_MONTH = 30 * 24 * 3600
    
    # CITATION: Guenter et al. (INFOCOM 2011) - Peak is 1.7x to 6.0x of average
    PEAK_TO_AVG_RATIO = 3.0 
    
    rps_avg = monthly_vol / SECONDS_PER_MONTH
    rps_peak = rps_avg * PEAK_TO_AVG_RATIO
    
    profile = SERVICE_TOPOLOGY[service_name]
    
    # 1. Select Best VM (Heuristic: Use burstable for low, m5 for high)
    vm_type = 't3.medium'
    if rps_peak > 50: vm_type = 'm5.large'
    if profile['type'] == 'stateful': vm_type = 'r5.large' # Memory optimized
    
    capacity = VM_CATALOG[vm_type]['capacity_rps']
    
    # 2. Provisioning Formula
    # N = ceil(Peak_Load / Node_Capacity)
    count = np.ceil(rps_peak / capacity)
    
    # 3. HA Constraint (Stateful must have standby)
    if profile['type'] == 'stateful':
        count = max(count, 2)
        
    return int(count), vm_type

# ==========================================
# MODULE 4: SCENARIO FACTORY & EXPORT
# ==========================================

def generate_full_dataset(horizon_months=36):
    """
    Generates the 'Multiverse' of 3 scenarios for all services.
    """
    scenarios = {}
    
    # 1. Unicorn (High Growth, High Volatility)
    # Drift 15%, Volatility 40%
    root_unicorn = generate_gbm_demand(horizon_months, 0.15, 0.40, 50000, 101)
    scenarios['Unicorn'] = propagate_demand(root_unicorn, SERVICE_TOPOLOGY)
    
    # 2. Zombie (Stagnation)
    # Drift 2%, Volatility 10%, Higher starting volume to cross SaaS/IaaS threshold
    root_zombie = generate_gbm_demand(horizon_months, 0.02, 0.10, 500000, 202)
    scenarios['Zombie'] = propagate_demand(root_zombie, SERVICE_TOPOLOGY)
    
    # 3. Pivot (Crash & Recovery)
    # Higher starting volume ensures IaaS is optimal before crash
    root_pivot = generate_gbm_demand(horizon_months, 0.15, 0.40, 300000, 303)
    # Inject 70% crash at Month 12 (drops into SaaS-favorable territory)
    root_pivot[12:18] *= 0.30 
    scenarios['Pivot'] = propagate_demand(root_pivot, SERVICE_TOPOLOGY)
    
    return scenarios

def save_trace_to_csv(trace_data: dict, filename='simulation_data.csv'):
    """
    Saves the trace dictionary (Service -> np.array) to a CSV file.
    """
    df = pd.DataFrame(trace_data)
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

def load_trace_from_csv(filename='simulation_data.csv'):
    """
    Loads the trace data from CSV back into a dictionary (Service -> np.array).
    """
    if not os.path.exists(filename):
        return None
    
    df = pd.read_csv(filename)
    # Convert back to dict of numpy arrays
    trace_data = {col: df[col].values for col in df.columns}
    print(f"Dataset loaded from {filename}")
    return trace_data

if __name__ == "__main__":
    print("Generating Cloud Lifecycle Dataset...")
    data = generate_full_dataset(36)
    
    # Save all three scenarios as separate CSV files
    scenario_files = {
        'Unicorn': 'simulation_data_unicorn.csv',   # Scenario A: Hyper-growth
        'Zombie': 'simulation_data_zombie.csv',     # Scenario B: Stagnation
        'Pivot': 'simulation_data_pivot.csv'        # Scenario C: Crash & Recovery
    }
    
    for scenario_name, filename in scenario_files.items():
        save_trace_to_csv(data[scenario_name], filename)
    
    # Print summary for Unicorn scenario
    print(f"\n{'Service':<15} | {'Month 1 Req':<12} | {'Month 36 Req':<12} | {'M36 Peak RPS':<12} | {'M36 VMs'}")
    print("-" * 75)
    
    unicorn_data = data['Unicorn']
    for svc, trace in unicorn_data.items():
        m1 = trace[0]
        m36 = trace[-1]
        vms, vtype = calculate_resources(svc, m36)
        peak_rps = (m36 / (30*24*3600)) * 3.0
        print(f"{svc:<15} | {m1:,.0f} | {m36:,.0f} | {peak_rps:,.1f} | {vms}x {vtype}")
        
    print("\nDataset generation complete. Ready for Optimization Solver.")