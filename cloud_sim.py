import numpy as np
import pandas as pd
import random
import copy
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# 1.1 IAAS CATALOG (Compute)
# Sources: AWS EC2 Pricing (OD, RI, Spot)
# spot_risk parameter acts as a "Cost Penalty Multiplier" in your objective function
VM_CATALOG = {
    't3.micro': {
        'cpu': 2, 'mem': 1.0, 'capacity_rps': 10.0,
        'price_od': 0.0104, 'price_ri': 0.0063, 'price_spot': 0.0031, 'spot_risk': 0.05
    },
    't3.medium': {
        'cpu': 2, 'mem': 4.0, 'capacity_rps': 30.0,
        'price_od': 0.0416, 'price_ri': 0.0250, 'price_spot': 0.0125, 'spot_risk': 0.05
    },
    'm5.large': {
        'cpu': 2, 'mem': 8.0, 'capacity_rps': 80.0,
        'price_od': 0.0960, 'price_ri': 0.0610, 'price_spot': 0.0350, 'spot_risk': 0.10
    },
    'm5.xlarge': {
        'cpu': 4, 'mem': 16.0, 'capacity_rps': 160.0,
        'price_od': 0.1920, 'price_ri': 0.1220, 'price_spot': 0.0750, 'spot_risk': 0.12
    },
    'r5.large': { # Memory Optimized (DBs)
        'cpu': 2, 'mem': 16.0, 'capacity_rps': 60.0,
        'price_od': 0.1260, 'price_ri': 0.0790, 'price_spot': 0.0450, 'spot_risk': 0.15
    },
}

# 1.2 PAAS CATALOG (Serverless / Lambda)
PAAS_CATALOG = {
    'standard': {
        'price_req': 0.20 / 1_000_000, # $0.20 per 1M
        'price_gb_sec': 0.0000166667,  # Duration cost
    }
}

# 1.3 SAAS CATALOG (Managed Services)
# Simple linear proxy for Auth0 / MongoDB pricing
SAAS_CATALOG = {
    'identity': {'base': 23.0, 'per_unit': 0.002}, # Auth0
    'database': {'base': 57.0, 'per_unit': 0.10},  # Atlas
    'generic':  {'base': 0.0,  'per_unit': 0.50 / 1000} # $0.50 per 1k
}


# 1.4 HIDDEN TAXES (The "Friction")
INFRA_TAXES = {
    'EKS_Cluster': 72.0,     # $/month (Control Plane)
    'NAT_Gateway': 0.045 * 730, # $/month (Hourly rate * 730)
    'ALB_Base': 0.0225 * 730,   # $/month
    'GB_Egress': 0.09        # $/GB
}

# ==============================================================================
# SECTION 2: APPLICATION TOPOLOGY (Online Boutique)
# ==============================================================================

# Structure: Key = Service Name
# Value: Metadata + Dependencies [(Target, Prob, Amp, PayloadKB)]
# use of complexity - cost = complexity * 40 hours * $80/hr
SERVICE_TOPOLOGY = {
    'frontend': {
        'type': 'stateless', 'complexity': 3,
        'paas_spec': {'ms': 50, 'mb': 1024},
        'dependencies': [
            ('ad', 1.0, 1.0, 5.0),
            ('productcatalog', 1.0, 4.0, 50.0),
            ('recommendation', 0.8, 1.0, 10.0),
            ('cart', 0.4, 2.0, 5.0), # 40% of requests, 2x Amplification, 5KB payload
            ('shipping', 0.4, 1.0, 2.0),
            ('currency', 1.0, 1.0, 1.0),
            ('checkout', 0.1, 1.0, 10.0) # 10% Conversion
        ]
    },
    'checkout': {
        'type': 'stateless', 'complexity': 4,
        'paas_spec': {'ms': 200, 'mb': 1024},
        'dependencies': [
            ('cart', 1.0, 1.0, 10.0),
            ('productcatalog', 1.0, 5.0, 20.0),
            ('shipping', 1.0, 1.0, 5.0),
            ('currency', 1.0, 1.0, 1.0),
            ('payment', 1.0, 1.0, 5.0),
            ('email', 1.0, 1.0, 50.0)
        ]
    },
    'recommendation': {
        'type': 'stateless', 'complexity': 5,
        'paas_spec': {'ms': 800, 'mb': 3008}, # ML Inference (Expensive on Lambda)
        'dependencies': [('productcatalog', 1.0, 5.0, 5.0)]
    },
    'productcatalog': {'type': 'stateful', 'complexity': 2, 'dependencies': [], 'paas_spec': {'ms': 20, 'mb': 256}},
    'cart':           {'type': 'stateful', 'complexity': 2, 'dependencies': [], 'paas_spec': {'ms': 20, 'mb': 256}},
    'payment':        {'type': 'stateful', 'complexity': 5, 'dependencies': [], 'paas_spec': {'ms': 100,'mb': 512}},
    'shipping':       {'type': 'stateless', 'complexity': 2, 'dependencies': [], 'paas_spec': {'ms': 50, 'mb': 512}},
    'email':          {'type': 'stateless', 'complexity': 1, 'dependencies': [], 'paas_spec': {'ms': 50, 'mb': 128}},
    'currency':       {'type': 'stateless', 'complexity': 1, 'dependencies': [], 'paas_spec': {'ms': 10, 'mb': 128}},
    'ad':             {'type': 'stateless', 'complexity': 1, 'dependencies': [], 'paas_spec': {'ms': 10, 'mb': 128}}
}

# ==============================================================================
# SECTION 3: PHYSICS ENGINE (Workload & Costing)
# ==============================================================================

def generate_gbm_demand(months, mu, sigma, start_val, seed):
    """ Generates Stochastic Demand (Geometric Brownian Motion) """
    np.random.seed(seed)
    dt = 1
    t = np.linspace(0, months, months)
    W = np.cumsum(np.random.standard_normal(size=months)) * np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    demand = start_val * np.exp(drift + diffusion)
    return np.maximum(demand, 1000)

def propagate_demand(root_trace, topology):
    """ Cascades demand down the dependency graph """
    traces = {name: np.zeros(len(root_trace)) for name in topology}
    traces['frontend'] = root_trace.copy()
    
    # Processing Order (Topological)
    order = ['frontend', 'checkout', 'recommendation']
    
    for parent in order:
        parent_vol = traces[parent]
        if 'dependencies' not in topology[parent]: continue
        
        for child, prob, amp, _ in topology[parent]['dependencies']:
            # Physics: Child = Parent * Prob * Amp
            flow = parent_vol * prob * amp
            noise = np.random.normal(1.0, 0.05, len(flow))
            traces[child] += (flow * noise)
            
    return traces

def calculate_step_cost(service_name, demand_vol, mode, prev_mode):
    """ 
    Calculates TCO for a single service for one month.
    Includes: Infra Bill, Labor, Transition Penalty.
    """
    meta = SERVICE_TOPOLOGY[service_name]
    cost_infra = 0.0
    cost_labor = 0.0
    cost_trans = 0.0
    
    # 1. INFRASTRUCTURE COST
    if mode == 'SaaS':
        # Simple Proxy: $2 per 10k requests (Generic)
        cost_infra = (demand_vol / 10000.0) * 2.0
        cost_labor = 0.5 * 80.0 # 0.5 hrs labor
        
    elif mode == 'PaaS':
        # Serverless Physics
        spec = meta.get('paas_spec', {'ms':100, 'mb':512})
        gb_sec = demand_vol * (spec['ms']/1000.0) * (spec['mb']/1024.0)
        cost_compute = gb_sec * PAAS_CATALOG['standard']['price_gb_sec']
        cost_req = demand_vol * PAAS_CATALOG['standard']['price_req']
        cost_infra = cost_compute + cost_req
        cost_labor = 5.0 * 80.0 # 5 hrs labor
        
    elif mode == 'IaaS':
        # Capacity Planning (PAR=3.0)
        rps_avg = demand_vol / (30*24*3600)
        rps_peak = rps_avg * 3.0
        
        # Select VM (Heuristic)
        vm_type = 't3.medium'
        if rps_peak > 50: vm_type = 'm5.large'
        if meta['type'] == 'stateful': vm_type = 'r5.large'
        
        capacity = VM_CATALOG[vm_type]['capacity_rps']
        count = max(1, np.ceil(rps_peak / capacity))
        if meta['type'] == 'stateful': count = max(count, 2) # HA Constraint
        
        # Pricing (Assume RI for DBs, OD for others for simplicity in this func)
        price_unit = VM_CATALOG[vm_type]['price_od']
        if meta['type'] == 'stateful': price_unit = VM_CATALOG[vm_type]['price_ri']
        
        cost_vm = count * price_unit * 730
        
        # Add Taxes (EKS, NAT, ALB)
        cost_taxes = INFRA_TAXES['EKS_Cluster'] + INFRA_TAXES['NAT_Gateway']
        if count > 1: cost_taxes += INFRA_TAXES['ALB_Base']
        
        cost_infra = cost_vm + cost_taxes
        cost_labor = 20.0 * 80.0 # 20 hrs labor
        
    # 2. TRANSITION COST
    if mode != prev_mode:
        # Scale Up (SaaS -> IaaS): Labor intensive
        if prev_mode in ['SaaS', 'PaaS'] and mode == 'IaaS':
            cost_trans = meta['complexity'] * 40 * 80.0 # 40hrs * $80
        # Scale Down (IaaS -> SaaS): Data Gravity intensive
        elif prev_mode == 'IaaS':
            cost_trans = 500.0 # Flat penalty for data egress/migration
            
    return cost_infra + cost_labor + cost_trans

# ==========================================
# 3.1. SOPHISTICATED COST LOGIC (The Fix)
# ==========================================

# CONSTANTS FOR PHYSICS
HOURS_TOTAL = 730
HOURS_PEAK = 12 * 30  # Assume 12 hours of high load per day
HOURS_OFF_PEAK = HOURS_TOTAL - HOURS_PEAK
REQS_PER_USER = 50.0  # Proxy to convert Reqs -> MAU (for Auth0 pricing)

TRANSITION_MATRIX = {
    # (From, To): Complexity Multiplier
    ('SaaS', 'PaaS'): 1.0,  # Moderate refactor (API wrapper)
    ('PaaS', 'IaaS'): 2.0,  # Heavy refactor (Dockerize + K8s manifests)
    ('SaaS', 'IaaS'): 3.0,  # Total rewrite (The "Big Bang")
    ('IaaS', 'PaaS'): 1.5,  # Migration logic required
    ('IaaS', 'SaaS'): 0.5,  # Data Egress & Shutdown only
    ('PaaS', 'SaaS'): 0.5,
}

def calculate_complex_cost(
    service_name, 
    demand_vol, 
    current_state,  # Tuple: (Mode, Strategy, Region) e.g., ('IaaS', 'Spot', 'us-east-1')
    prev_state,     # Tuple: (Mode, Strategy, Region)
    parent_region=None # Passed from the dependency graph logic
):
    """
    Calculates precise TCO handling Autoscaling, Spot/RI, and Cross-Region Egress.
    """
    mode, strategy, region = current_state
    p_mode, p_strat, p_region = prev_state
    
    meta = SERVICE_TOPOLOGY[service_name]
    cost_infra = 0.0
    cost_labor = 0.0
    cost_network = 0.0
    
    # ---------------------------------------------------
    # 1. SAAS LOGIC (User vs Request Pricing)
    # ---------------------------------------------------
    if mode == 'SaaS':
        # FIX 1: Distinguish Identity (MAU) from API (Requests)
        if service_name == 'auth' or service_name == 'identity':
            # Pricing: Auth0 style (Active Users)
            est_mau = demand_vol / REQS_PER_USER
            # Step function logic (e.g., first 7k free, then $23)
            if est_mau <= 7000: cost_infra = 0
            elif est_mau <= 50000: cost_infra = 23.0
            else: cost_infra = 23.0 + (est_mau - 50000) * 0.02
        else:
            # Pricing: Standard API (per 10k requests)
            cost_infra = (demand_vol / 10000.0) * 2.0
            
        cost_labor = 0.5 * 80.0 # Just paying bills

    # ---------------------------------------------------
    # 2. PAAS LOGIC (Serverless Physics)
    # ---------------------------------------------------
    elif mode == 'PaaS':
        spec = meta.get('paas_spec', {'ms': 100, 'mb': 512})
        gb_sec = demand_vol * (spec['ms']/1000.0) * (spec['mb']/1024.0)
        
        # Base Compute
        cost_compute = gb_sec * PAAS_CATALOG['standard']['price_gb_sec']
        cost_req = demand_vol * PAAS_CATALOG['standard']['price_req']
        
        cost_infra = cost_compute + cost_req
        cost_labor = 5.0 * 80.0

    # ---------------------------------------------------
    # 3. IAAS LOGIC (Autoscaling + Pricing Strategy)
    # ---------------------------------------------------
    elif mode == 'IaaS':
        # FIX 2: Autoscaling Group (ASG) Simulation
        # We model two states: Peak and Off-Peak
        rps_avg = demand_vol / (HOURS_TOTAL * 3600)
        rps_peak = rps_avg * 3.0
        rps_off = rps_avg * 0.5 # Trough traffic
        
        # Select Instance Type
        vm_type = 't3.medium'
        if rps_peak > 50: vm_type = 'm5.large'
        if meta['type'] == 'stateful': vm_type = 'r5.large'
        
        capacity = VM_CATALOG[vm_type]['capacity_rps']
        
        # Calculate Counts for ASG
        n_peak = max(1, np.ceil(rps_peak / capacity))
        n_off  = max(1, np.ceil(rps_off / capacity))
        
        # FIX 3: Pricing Strategy (Spot vs RI vs OD)
        price_hourly = 0.0
        cat_entry = VM_CATALOG[vm_type]
        
        if strategy == 'RI':
            # RI you pay 24/7 regardless of usage (Commitment)
            price_hourly = cat_entry['price_ri']
            # With RI, autoscaling saves nothing on the BILL (sunk cost)
            # You pay for Peak capacity 100% of time
            weighted_vms = n_peak 
            
        elif strategy == 'Spot':
            # Risk Logic: Add expected failure cost
            risk_penalty = cat_entry['spot_risk'] * 50.0 # $50 penalty per interruption
            price_hourly = cat_entry['price_spot'] + risk_penalty
            # Spot scales perfectly
            weighted_vms = ((n_peak * HOURS_PEAK) + (n_off * HOURS_OFF_PEAK)) / HOURS_TOTAL
            
        else: # On-Demand
            price_hourly = cat_entry['price_od']
            # OD scales perfectly
            weighted_vms = ((n_peak * HOURS_PEAK) + (n_off * HOURS_OFF_PEAK)) / HOURS_TOTAL

        # HA Constraint for Stateful (Min 2 nodes always)
        if meta['type'] == 'stateful':
            weighted_vms = max(weighted_vms, 2.0)

        cost_vm = weighted_vms * price_hourly * HOURS_TOTAL
        
        # Base Infra Taxes
        cost_taxes = INFRA_TAXES['EKS_Cluster'] + INFRA_TAXES['NAT_Gateway']
        if n_peak > 1: cost_taxes += INFRA_TAXES['ALB_Base']
        
        cost_infra = cost_vm + cost_taxes
        cost_labor = 20.0 * 80.0

    # ---------------------------------------------------
    # 4. NETWORK LATENCY TAX (FIX 4)
    # ---------------------------------------------------
    # If parent is in Region A and we are in Region B, pay egress
    if parent_region and region != parent_region:
        # 5KB average payload * Demand * Cross-Region Price
        # $0.02/GB is approximate inter-region transfer
        gb_transfer = (demand_vol * 5.0) / 1_000_000 # KB -> GB
        cost_network = gb_transfer * 0.02 
        
        # Plus Latency Penalty (soft cost or SLA violation cost)
        cost_network += 500.0 # Flat penalty for SLA risk

    # ---------------------------------------------------
    # 5. TRANSITION COST (FIX 5)
    # ---------------------------------------------------
    cost_trans = 0.0
    if mode != p_mode:
        # Lookup complexity multiplier
        mult = TRANSITION_MATRIX.get((p_mode, mode), 1.0)
        
        base_effort = meta['complexity'] * 40 * 80.0 # Base refactor cost
        cost_trans = base_effort * mult

    return cost_infra + cost_labor + cost_trans + cost_network


# ==========================================
# 3.1. SOPHISTICATED COST LOGIC (The Fix)
# ==========================================

# CONSTANTS FOR PHYSICS
HOURS_TOTAL = 730
HOURS_PEAK = 12 * 30  # Assume 12 hours of high load per day
HOURS_OFF_PEAK = HOURS_TOTAL - HOURS_PEAK
REQS_PER_USER = 50.0  # Proxy to convert Reqs -> MAU (for Auth0 pricing)

TRANSITION_MATRIX = {
    # (From, To): Complexity Multiplier
    ('SaaS', 'PaaS'): 1.0,  # Moderate refactor (API wrapper)
    ('PaaS', 'IaaS'): 2.0,  # Heavy refactor (Dockerize + K8s manifests)
    ('SaaS', 'IaaS'): 3.0,  # Total rewrite (The "Big Bang")
    ('IaaS', 'PaaS'): 1.5,  # Migration logic required
    ('IaaS', 'SaaS'): 0.5,  # Data Egress & Shutdown only
    ('PaaS', 'SaaS'): 0.5,
}

def calculate_complex_cost(
    service_name, 
    demand_vol, 
    current_state,  # Tuple: (Mode, Strategy, Region) e.g., ('IaaS', 'Spot', 'us-east-1')
    prev_state,     # Tuple: (Mode, Strategy, Region)
    parent_region=None # Passed from the dependency graph logic
):
    """
    Calculates precise TCO handling Autoscaling, Spot/RI, and Cross-Region Egress.
    """
    mode, strategy, region = current_state
    p_mode, p_strat, p_region = prev_state
    
    meta = SERVICE_TOPOLOGY[service_name]
    cost_infra = 0.0
    cost_labor = 0.0
    cost_network = 0.0
    
    # ---------------------------------------------------
    # 1. SAAS LOGIC (User vs Request Pricing)
    # ---------------------------------------------------
    if mode == 'SaaS':
        # FIX 1: Distinguish Identity (MAU) from API (Requests)
        if service_name == 'auth' or service_name == 'identity':
            # Pricing: Auth0 style (Active Users)
            est_mau = demand_vol / REQS_PER_USER
            # Step function logic (e.g., first 7k free, then $23)
            if est_mau <= 7000: cost_infra = 0
            elif est_mau <= 50000: cost_infra = 23.0
            else: cost_infra = 23.0 + (est_mau - 50000) * 0.02
        else:
            # Pricing: Standard API (per 10k requests)
            cost_infra = (demand_vol / 10000.0) * 2.0
            
        cost_labor = 0.5 * 80.0 # Just paying bills

    # ---------------------------------------------------
    # 2. PAAS LOGIC (Serverless Physics)
    # ---------------------------------------------------
    elif mode == 'PaaS':
        spec = meta.get('paas_spec', {'ms': 100, 'mb': 512})
        gb_sec = demand_vol * (spec['ms']/1000.0) * (spec['mb']/1024.0)
        
        # Base Compute
        cost_compute = gb_sec * PAAS_CATALOG['standard']['price_gb_sec']
        cost_req = demand_vol * PAAS_CATALOG['standard']['price_req']
        
        cost_infra = cost_compute + cost_req
        cost_labor = 5.0 * 80.0

    # ---------------------------------------------------
    # 3. IAAS LOGIC (Autoscaling + Pricing Strategy)
    # ---------------------------------------------------
    elif mode == 'IaaS':
        # FIX 2: Autoscaling Group (ASG) Simulation
        # We model two states: Peak and Off-Peak
        rps_avg = demand_vol / (HOURS_TOTAL * 3600)
        rps_peak = rps_avg * 3.0
        rps_off = rps_avg * 0.5 # Trough traffic
        
        # Select Instance Type
        vm_type = 't3.medium'
        if rps_peak > 50: vm_type = 'm5.large'
        if meta['type'] == 'stateful': vm_type = 'r5.large'
        
        capacity = VM_CATALOG[vm_type]['capacity_rps']
        
        # Calculate Counts for ASG
        n_peak = max(1, np.ceil(rps_peak / capacity))
        n_off  = max(1, np.ceil(rps_off / capacity))
        
        # FIX 3: Pricing Strategy (Spot vs RI vs OD)
        price_hourly = 0.0
        cat_entry = VM_CATALOG[vm_type]
        
        if strategy == 'RI':
            # RI you pay 24/7 regardless of usage (Commitment)
            price_hourly = cat_entry['price_ri']
            # With RI, autoscaling saves nothing on the BILL (sunk cost)
            # You pay for Peak capacity 100% of time
            weighted_vms = n_peak 
            
        elif strategy == 'Spot':
            # Risk Logic: Add expected failure cost
            risk_penalty = cat_entry['spot_risk'] * 50.0 # $50 penalty per interruption
            price_hourly = cat_entry['price_spot'] + risk_penalty
            # Spot scales perfectly
            weighted_vms = ((n_peak * HOURS_PEAK) + (n_off * HOURS_OFF_PEAK)) / HOURS_TOTAL
            
        else: # On-Demand
            price_hourly = cat_entry['price_od']
            # OD scales perfectly
            weighted_vms = ((n_peak * HOURS_PEAK) + (n_off * HOURS_OFF_PEAK)) / HOURS_TOTAL

        # HA Constraint for Stateful (Min 2 nodes always)
        if meta['type'] == 'stateful':
            weighted_vms = max(weighted_vms, 2.0)

        cost_vm = weighted_vms * price_hourly * HOURS_TOTAL
        
        # Base Infra Taxes
        cost_taxes = INFRA_TAXES['EKS_Cluster'] + INFRA_TAXES['NAT_Gateway']
        if n_peak > 1: cost_taxes += INFRA_TAXES['ALB_Base']
        
        cost_infra = cost_vm + cost_taxes
        cost_labor = 20.0 * 80.0

    # ---------------------------------------------------
    # 4. NETWORK LATENCY TAX (FIX 4)
    # ---------------------------------------------------
    # If parent is in Region A and we are in Region B, pay egress
    if parent_region and region != parent_region:
        # 5KB average payload * Demand * Cross-Region Price
        # $0.02/GB is approximate inter-region transfer
        gb_transfer = (demand_vol * 5.0) / 1_000_000 # KB -> GB
        cost_network = gb_transfer * 0.02 
        
        # Plus Latency Penalty (soft cost or SLA violation cost)
        cost_network += 500.0 # Flat penalty for SLA risk

    # ---------------------------------------------------
    # 5. TRANSITION COST (FIX 5)
    # ---------------------------------------------------
    cost_trans = 0.0
    if mode != p_mode:
        # Lookup complexity multiplier
        mult = TRANSITION_MATRIX.get((p_mode, mode), 1.0)
        
        base_effort = meta['complexity'] * 40 * 80.0 # Base refactor cost
        cost_trans = base_effort * mult

    return cost_infra + cost_labor + cost_trans + cost_network
    # ==============================================================================
# SECTION 4: THE SOLVER (Dependency-Aware Memetic Algorithm)
# ==============================================================================

# ==========================================
# 4.1. DECISION SPACE ENCODING
# ==========================================
# The solver will pick an index from this list for each service.
# We limit Region to 'us-east-1' for simplicity, but you can expand this list.

DECISION_CATALOG = [
    # Index 0: SaaS (Identity/Standard)
    ('SaaS', 'Standard', 'us-east-1'),
    
    # Index 1: PaaS (Serverless)
    ('PaaS', 'Standard', 'us-east-1'),
    
    # Index 2: IaaS (On-Demand) - Maximum flexibility
    ('IaaS', 'OD', 'us-east-1'),
    
    # Index 3: IaaS (Reserved Instance) - Low cost, high commit
    ('IaaS', 'RI', 'us-east-1'),
    
    # Index 4: IaaS (Spot) - Lowest cost, high risk
    ('IaaS', 'Spot', 'us-east-1')
]

def solve_da_ma(services_list, current_modes, trace_slice, lookahead=6):
    """
    The Proposed Algorithm.
    Optimizes the next 6 months using Genetic Algorithm + Dependency Repair.
    """
    MODE_MAP = ['SaaS', 'PaaS', 'IaaS']
    POP_SIZE = 20
    GENERATIONS = 10
    
    # Helper: Fitness Function (Minimize Cost)
    def get_fitness(genome):
        total_cost = 0
        p_modes = current_modes[:]
        
        for t in range(lookahead):
            step_modes = [MODE_MAP[g] for g in genome[t]]
            # Cost for this month
            for i, svc in enumerate(services_list):
                vol = trace_slice[svc][t]
                total_cost += calculate_step_cost(svc, vol, step_modes[i], p_modes[i])
                
                # COUPLING PENALTY (The Local Search Heuristic)
                # If I am IaaS and my dependency is SaaS -> Penalty!
                meta = SERVICE_TOPOLOGY[svc]
                if 'dependencies' in meta:
                    for target, _, _, _ in meta['dependencies']:
                        t_idx = services_list.index(target)
                        if step_modes[i] == 'IaaS' and step_modes[t_idx] != 'IaaS':
                            total_cost += 1000.0 # Latency Tax
                            
            p_modes = step_modes
        return total_cost

    # Init Population (Random)
    pop = [[[random.randint(0,2) for _ in services_list] for _ in range(lookahead)] for _ in range(POP_SIZE)]
    
    # Evolution
    for gen in range(GENERATIONS):
        pop.sort(key=get_fitness)
        pop = pop[:10] # Elitism
        
        # Crossover & Mutation
        while len(pop) < POP_SIZE:
            parent = random.choice(pop[:5])
            child = copy.deepcopy(parent)
            
            # Mutate
            if random.random() < 0.3:
                t = random.randint(0, lookahead-1)
                s = random.randint(0, len(services_list)-1)
                child[t][s] = random.randint(0, 2)
                
                # MEMETIC REPAIR (The "Secret Sauce")
                # If we mutated a parent, force children to match to avoid coupling penalty
                svc_name = services_list[s]
                if 'dependencies' in SERVICE_TOPOLOGY[svc_name]:
                    for target, _, _, _ in SERVICE_TOPOLOGY[svc_name]['dependencies']:
                        t_idx = services_list.index(target)
                        child[t][t_idx] = child[t][s] # Sync modes
            
            pop.append(child)
            
    # Return best next step
    best_genome = pop[0]
    return [MODE_MAP[i] for i in best_genome[0]]

def solve_da_ma_complex(services_list, current_states, trace_slice, lookahead=6):
    """
    Advanced Memetic Algorithm.
    Optimizes: Mode (SaaS/IaaS) + Strategy (RI/Spot) + Region.
    """
    POP_SIZE = 30
    GENERATIONS = 15
    ELITISM_COUNT = 5
    
    # Helper: Check if a genome is "Illegal" (e.g., DB on Spot)
    def is_valid_gene(service, gene_idx):
        mode, strat, _ = DECISION_CATALOG[gene_idx]
        meta = SERVICE_TOPOLOGY[service]
        
        # CONSTRAINT 1: Stateful services cannot use Spot
        if meta['type'] == 'stateful' and strat == 'Spot':
            return False
            
        return True

    # Helper: Fitness Function (Minimize TCO)
    def get_fitness(genome):
        total_cost = 0
        p_states = current_states[:] # Copy previous state
        
        for t in range(lookahead):
            # Decode the genome for this month
            # genome[t] is a list of indices [2, 4, 1, ...]
            step_states = [DECISION_CATALOG[idx] for idx in genome[t]]
            
            for i, svc in enumerate(services_list):
                vol = trace_slice[svc][t]
                
                # Check Validity Penalty (Soft Constraint)
                if not is_valid_gene(svc, genome[t][i]):
                    total_cost += 10_000 # Massive penalty for breaking physics
                
                # Calculate Cost
                # We assume parent_region is 'us-east-1' for simplicity here, 
                # or you can look up the parent's region dynamically.
                cost = calculate_complex_cost(svc, vol, step_states[i], p_states[i])
                total_cost += cost
                
                # COUPLING PENALTY (Local Search Heuristic)
                # If I am IaaS and my dependency is SaaS -> Penalty
                meta = SERVICE_TOPOLOGY[svc]
                if 'dependencies' in meta:
                    for target, _, _, _ in meta['dependencies']:
                        t_idx = services_list.index(target)
                        target_mode = step_states[t_idx][0] # Get 'SaaS'/'IaaS'
                        my_mode = step_states[i][0]
                        
                        if my_mode == 'IaaS' and target_mode != 'IaaS':
                            total_cost += 500.0 # Latency Tax
            
            p_states = step_states
        return total_cost

    # 1. INITIALIZATION
    # Random valid population
    pop = []
    for _ in range(POP_SIZE):
        ind = []
        for t in range(lookahead):
            step = []
            for s in services_list:
                # Randomly pick a valid configuration
                valid_choices = [x for x in range(len(DECISION_CATALOG)) if is_valid_gene(s, x)]
                step.append(random.choice(valid_choices))
            ind.append(step)
        pop.append(ind)
            
    # 2. EVOLUTION LOOP
    for gen in range(GENERATIONS):
        pop.sort(key=get_fitness)
        new_pop = pop[:ELITISM_COUNT]
        
        while len(new_pop) < POP_SIZE:
            # Tournament Selection
            p1 = random.choice(pop[:10])
            p2 = random.choice(pop[:10])
            child = copy.deepcopy(p1)
            
            # Crossover (Uniform)
            for t in range(lookahead):
                if random.random() < 0.5:
                    child[t] = p2[t][:]
            
            # Mutation
            if random.random() < 0.4:
                t = random.randint(0, lookahead-1)
                s = random.randint(0, len(services_list)-1)
                
                # Pick a new valid gene
                valid_choices = [x for x in range(len(DECISION_CATALOG)) if is_valid_gene(services_list[s], x)]
                new_val = random.choice(valid_choices)
                child[t][s] = new_val
                
                # MEMETIC REPAIR (The "Cluster Fix")
                # If we moved Service S to IaaS, check its children
                mode_new = DECISION_CATALOG[new_val][0]
                if mode_new == 'IaaS':
                    svc_name = services_list[s]
                    if 'dependencies' in SERVICE_TOPOLOGY[svc_name]:
                        for target, _, _, _ in SERVICE_TOPOLOGY[svc_name]['dependencies']:
                            t_idx = services_list.index(target)
                            # 50% chance to drag the dependency along to IaaS
                            if random.random() < 0.5:
                                # Force dependency to IaaS (OD as default safe choice)
                                child[t][t_idx] = 2 # Index 2 is IaaS/OD
            
            new_pop.append(child)
        pop = new_pop

    # Return best next step (Decoded)
    best_genome_step0 = pop[0][0]
    return [DECISION_CATALOG[idx] for idx in best_genome_step0]

# ==============================================================================
# SECTION 5: MAIN EXECUTION
# ==============================================================================

# ==============================================================================
# SECTION 5: MAIN EXECUTION ENGINE
# ==============================================================================

def run_simulation():
    print("--- 1. Generating 'Unicorn' Scenario Traces (3 Years) ---")
    horizon = 36
    # Generate 48 months to ensure we have buffer for the 6-month lookahead
    root_trace = generate_gbm_demand(horizon + 12, 0.15, 0.40, 50000, 101) 
    full_trace = propagate_demand(root_trace, SERVICE_TOPOLOGY)
    
    services = list(SERVICE_TOPOLOGY.keys())
    
    # Define Strategies to Compare
    strategies = ['Static SaaS', 'Static IaaS (OD)', 'Proposed (DA-MA)']
    
    # Store results for plotting/analysis
    results = {s: {'cost_cumulative': [], 'monthly_bills': [], 'decisions': []} for s in strategies}
    
    print("--- 2. Running Simulations ---")
    
    for strat in strategies:
        print(f"   > Simulating Strategy: {strat}...")
        
        # A. INITIALIZATION
        # Everyone starts as a "Greenfield" startup on SaaS
        # State Format: (Mode, Strategy, Region)
        current_states = [('SaaS', 'Standard', 'us-east-1') for _ in services]
        
        # Exception: Static IaaS forces a migration on Day 1
        if strat == 'Static IaaS (OD)':
            current_states = [('IaaS', 'OD', 'us-east-1') for _ in services]
            
        cumulative_cost = 0.0
        
        # B. TIME LOOP (Month 0 to 35)
        for t in range(horizon):
            
            # 1. OPTIMIZATION STEP (Make Decisions)
            if strat == 'Proposed (DA-MA)':
                # Prepare Forecast Slice (Next 6 months)
                slice_data = {}
                for s in services:
                    # Extract future demand (handle array bounds)
                    future = full_trace[s][t : t+6]
                    if len(future) < 6: # Pad with last value if near end
                        future = np.pad(future, (0, 6-len(future)), 'edge')
                    slice_data[s] = future
                
                # RUN SOLVER
                # It returns the optimal states for the *next* month
                next_states = solve_da_ma_complex(services, current_states, slice_data)
                
            elif strat == 'Static SaaS':
                next_states = [('SaaS', 'Standard', 'us-east-1')] * len(services)
                
            elif strat == 'Static IaaS (OD)':
                next_states = [('IaaS', 'OD', 'us-east-1')] * len(services)
            
            # 2. COST CALCULATION STEP (Pay Bills)
            monthly_total = 0.0
            month_decisions = {} # Log what happened this month
            
            for i, svc in enumerate(services):
                vol = full_trace[svc][t]
                
                # Calculate Cost based on the Move: Prev_State -> Next_State
                # Note: We pass current_states[i] as 'prev' and next_states[i] as 'current'
                # because the solver decided the configuration for *this* month t.
                cost = calculate_complex_cost(svc, vol, next_states[i], current_states[i])
                
                monthly_total += cost
                month_decisions[svc] = next_states[i]
            
            # 3. UPDATE HISTORY
            cumulative_cost += monthly_total
            current_states = next_states # Advance time
            
            # Log Data
            results[strat]['cost_cumulative'].append(cumulative_cost)
            results[strat]['monthly_bills'].append(monthly_total)
            results[strat]['decisions'].append(month_decisions)

    # ==========================================================================
    # SECTION 6: RESULTS & REPORTING
    # ==========================================================================
    print("\n" + "="*60)
    print(f"{'FINAL RESULTS (36 Months TCO)':^60}")
    print("="*60)
    print(f"{'Strategy':<25} | {'Total Cost ($)':<15} | {'Savings %'}")
    print("-" * 60)
    
    # Baseline for savings is Static IaaS (industry standard for mature apps)
    base_cost = results['Static IaaS (OD)']['cost_cumulative'][-1]
    
    for strat in strategies:
        cost = results[strat]['cost_cumulative'][-1]
        savings = 0.0
        if base_cost > 0:
            savings = ((base_cost - cost) / base_cost) * 100
            
        print(f"{strat:<25} | ${cost:,.0f}       | {savings:5.1f}%")
        
    # --------------------------------------------------------------------------
    # DETAILED LIFECYCLE LOG (Proposed Strategy)
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"{'PROPOSED ALGORITHM: MIGRATION LOG':^60}")
    print("="*60)
    print("Analyzing key architectural shifts for 'Frontend' service...")
    
    prop_decisions = results['Proposed (DA-MA)']['decisions']
    prev_mode = 'SaaS'
    
    for t, snapshot in enumerate(prop_decisions):
        curr_state = snapshot['frontend']
        curr_mode = curr_state[0]    # 'SaaS', 'PaaS', 'IaaS'
        curr_strat = curr_state[1]   # 'Standard', 'OD', 'Spot'
        
        # Detect Changes
        if curr_mode != prev_mode:
            print(f"[Month {t+1}] MIGRATION: Frontend switched {prev_mode} -> {curr_mode} ({curr_strat})")
            print(f"           Reason: Monthly Demand hit {full_trace['frontend'][t]:,.0f} reqs")
            prev_mode = curr_mode
            
        # Detect Strategy Changes (e.g. OD -> Spot)
        # (You can expand this to log other interesting events)

    print("\nSimulation Complete.")

if __name__ == "__main__":
    run_simulation()