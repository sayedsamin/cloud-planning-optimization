"""
Physics engines, demand generation, and cost calculations.
"""
import numpy as np
from config import SimConfig
from topology import SERVICE_TOPOLOGY, VM_CATALOG, PAAS_CATALOG, TRANSITION_MATRIX

def generate_gbm_demand(months, mu, sigma, start_val, seed):
    np.random.seed(seed)
    dt = 1
    t = np.linspace(0, months, months)
    W = np.cumsum(np.random.standard_normal(size=months)) * np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    demand = start_val * np.exp(drift + diffusion)
    return np.maximum(demand, 1000)

def propagate_demand(root_trace, topology):
    traces = {name: np.zeros(len(root_trace)) for name in topology}
    traces['frontend'] = root_trace.copy()
    
    # Topological Sort logic integrated
    visited = set()
    order = []
    
    def dfs(node):
        visited.add(node)
        if 'dependencies' in topology[node]:
            for child, _, _, _ in topology[node]['dependencies']:
                if child not in visited:
                    dfs(child)
        order.insert(0, node) # Post-order reversed

    if 'frontend' in topology:
        dfs('frontend')
    
    calc_order = order
    
    for parent in calc_order:
        parent_vol = traces[parent]
        if 'dependencies' not in topology[parent]: continue
        for child, prob, amp, _ in topology[parent]['dependencies']:
            if child not in traces: continue
            flow = parent_vol * prob * amp
            noise = np.random.normal(1.0, 0.05, len(flow))
            traces[child] += (flow * noise)
    return traces

def calculate_complex_cost(service_name, demand_vol, current_state, prev_state, parent_region=None):
    mode, strategy, region = current_state
    p_mode, p_strat, p_region = prev_state
    
    meta = SERVICE_TOPOLOGY[service_name]
    cost_infra = 0.0
    cost_labor = 0.0
    cost_network = 0.0
    
    # 1. SAAS
    if mode == 'SaaS':
        if service_name in ['auth', 'identity']: # MAU Pricing
            est_mau = demand_vol / SimConfig.REQS_PER_USER
            if est_mau <= 7000: cost_infra = 0
            elif est_mau <= 50000: cost_infra = 23.0
            else: cost_infra = 23.0 + (est_mau - 50000) * 0.02
        else: # Request Pricing
            cost_infra = (demand_vol / 10000.0) * 2.0
        cost_labor = SimConfig.LABOR_SAAS * SimConfig.LABOR_RATE_HOURLY
        
    # 2. PAAS
    elif mode == 'PaaS':
        spec = meta.get('paas_spec', {'ms': 100, 'mb': 512})
        gb_sec = demand_vol * (spec['ms']/1000.0) * (spec['mb']/1024.0)
        cost_compute = gb_sec * PAAS_CATALOG['standard']['price_gb_sec']
        cost_req = demand_vol * PAAS_CATALOG['standard']['price_req']
        cost_infra = cost_compute + cost_req
        cost_labor = SimConfig.LABOR_PAAS * SimConfig.LABOR_RATE_HOURLY

    # 3. IAAS
    elif mode == 'IaaS':
        rps_avg = demand_vol / (SimConfig.HOURS_TOTAL * 3600)
        rps_peak = rps_avg * 3.0
        rps_off = rps_avg * 0.5
        
        vm_type = 't3.medium'
        if rps_peak > 50: vm_type = 'm5.large'
        if meta['type'] == 'stateful': vm_type = 'r5.large'
        
        capacity = VM_CATALOG[vm_type]['capacity_rps']
        n_peak = max(1, np.ceil(rps_peak / capacity))
        n_off  = max(1, np.ceil(rps_off / capacity))
        
        cat_entry = VM_CATALOG[vm_type]
        if strategy == 'RI':
            price_hourly = cat_entry['price_ri']
            weighted_vms = n_peak 
        elif strategy == 'Spot':
            risk_penalty = cat_entry['spot_risk'] * 50.0 
            price_hourly = cat_entry['price_spot'] + risk_penalty
            weighted_vms = ((n_peak * SimConfig.HOURS_PEAK) + (n_off * SimConfig.HOURS_OFF_PEAK)) / SimConfig.HOURS_TOTAL
        else: # OD
            price_hourly = cat_entry['price_od']
            weighted_vms = ((n_peak * SimConfig.HOURS_PEAK) + (n_off * SimConfig.HOURS_OFF_PEAK)) / SimConfig.HOURS_TOTAL

        if meta['type'] == 'stateful': weighted_vms = max(weighted_vms, 2.0)

        cost_vm = weighted_vms * price_hourly * SimConfig.HOURS_TOTAL
        cost_taxes = SimConfig.TAX_EKS + SimConfig.TAX_NAT_GB_MONTH
        if n_peak > 1: cost_taxes += SimConfig.TAX_ALB_MONTH
        cost_infra = cost_vm + cost_taxes
        cost_labor = SimConfig.LABOR_IAAS * SimConfig.LABOR_RATE_HOURLY

    # 4. NETWORK
    if parent_region and region != parent_region:
        gb_transfer = (demand_vol * 5.0) / 1_000_000 
        cost_network = gb_transfer * SimConfig.cost_NETWORK_GB + SimConfig.COST_NETWORK_BASE 

    # 5. TRANSITION
    cost_trans = 0.0
    if mode != p_mode:
        mult = TRANSITION_MATRIX.get((p_mode, mode), 1.0)
        base_effort = meta['complexity'] * SimConfig.LABOR_TRANSITION * SimConfig.LABOR_RATE_HOURLY 
        cost_trans = base_effort * mult

    return cost_infra + cost_labor + cost_trans + cost_network
