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

    cost_trans = 0.0
    if mode != p_mode:
        mult = TRANSITION_MATRIX.get((p_mode, mode), 1.0)
        base_effort = meta['complexity'] * SimConfig.COST_TRANSITION_FIXED 
        cost_trans = base_effort * mult

    return cost_infra + cost_labor + cost_trans + cost_network + cost_reliability + cost_penalty

def calculate_complex_cost(service_name, demand_vol, current_state, prev_state, all_states_map=None, transition_cost_matrix=None):
    # Unpack 4-tuple (Mode, Strat, Reg, Age)
    # If legacy 3-tuple is passed (unlikely if we updated everything), handle it?
    # No, we assume clean switch.
    
    if len(current_state) == 4:
        mode, strategy, region, age = current_state
    else:
        # Fallback for careful migration, though we should restart sim
        mode, strategy, region = current_state
        age = 0
        
    if len(prev_state) == 4:
        p_mode, p_strat, p_region, p_age = prev_state
    else:
        p_mode, p_strat, p_region = prev_state
        p_age = 0
    
    meta = SERVICE_TOPOLOGY[service_name]
    cost_infra = 0.0
    cost_labor = 0.0
    cost_network = 0.0
    cost_penalty = 0.0
    
    # --- COST CALCULATION (UNCHANGED LOGIC, JUST RE-INDENTED IF NEEDED) ---
    # 1. SAAS
    if mode == 'SaaS':
        if service_name in ['auth', 'identity']: # MAU Pricing
            est_mau = demand_vol / SimConfig.REQS_PER_USER
            if est_mau <= 7000: cost_infra = 0
            elif est_mau <= 50000: cost_infra = 23.0
            else: cost_infra = 23.0 + (est_mau - 50000) * 0.02
        else: # Request Pricing
            cost_infra = (demand_vol / 10000.0) * 2.0
        cost_labor = SimConfig.COST_MAINTENANCE_MONTHLY
        
    # 2. PAAS
    elif mode == 'PaaS':
        spec = meta.get('paas_spec', {'ms': 100, 'mb': 512})
        gb_sec = demand_vol * (spec['ms']/1000.0) * (spec['mb']/1024.0)
        cost_compute = gb_sec * PAAS_CATALOG['standard']['price_gb_sec']
        cost_req = demand_vol * PAAS_CATALOG['standard']['price_req']
        cost_infra = cost_compute + cost_req
        cost_labor = SimConfig.COST_MAINTENANCE_MONTHLY
    
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
        cost_labor = SimConfig.COST_MAINTENANCE_MONTHLY

    # --- RI CONTRACT LOGIC ---
    # Check for Contract Breakage
    # If we were in RI(k) where k < 12, and we did NOT move to RI(k+1), we broke it.
    # Note: Transition RI(12) -> Anything is OK.
    if p_strat == 'RI' and 1 <= p_age < 12:
        # Expected next state: RI, Age = p_age + 1
        is_valid_continuation = (strategy == 'RI' and age == p_age + 1)
        
        if not is_valid_continuation:
            # PENALTY: Remaining months * Monthly Cost
            # We need to re-calculate what the monthly cost *was* (or is).
            # We use the current volume as a proxy or just the theoretical monthly RI price.
            # Using current volume is fair enough (opportunity cost).
            
            # Re-calc RI price for this volume
            # To minimize code duplication, we assume cost_infra calculated above is correct IF we were in RI.
            # But we might be in 'SaaS' now. So we need to calculate 'What RI would have cost'.
            
            # Fast calc for penalty base: use cost_infra if in IaaS/RI, else re-estimate.
            # Actually, let's just use the 'price_ri' from catalog and current demand.
            # (Matches the "remaining balance" idea).
            
            p_rps_avg = demand_vol / (SimConfig.HOURS_TOTAL * 3600)
            p_rps_peak = p_rps_avg * 3.0
            
            p_vm_type = 't3.medium'
            if p_rps_peak > 50: p_vm_type = 'm5.large'
            if meta['type'] == 'stateful': p_vm_type = 'r5.large'
            
            p_capacity = VM_CATALOG[p_vm_type]['capacity_rps']
            p_n_peak = max(1, np.ceil(p_rps_peak / p_capacity))
            
            # RI is always paid for Peak/Reserved capacity (n_peak)
            p_price_hourly = VM_CATALOG[p_vm_type]['price_ri']
            p_monthly_cost = p_n_peak * p_price_hourly * SimConfig.HOURS_TOTAL
            
            months_remaining = 12 - p_age
            penalty_amount = months_remaining * p_monthly_cost
            cost_penalty += penalty_amount
            
    # 4. NETWORK COST (Data Gravity) & COUPLING PENALTY
    if all_states_map and 'dependencies' in meta:
        for dep_svc, prob, amp, _ in meta['dependencies']:
            if dep_svc in all_states_map:
                target_state = all_states_map[dep_svc]
                target_mode = target_state[0]
                
                # Volume (GB) = Demand * Prob * Amp * Payload
                out_reqs = demand_vol * prob * amp
                total_gb = (out_reqs * SimConfig.AVG_PAYLOAD_KB) / 1_000_000.0
                
                rate = 0.0
                if mode == 'IaaS':
                    if target_mode == 'SaaS':
                        rate = SimConfig.COST_NAT_GATEWAY_GB + SimConfig.COST_INTERNET_EGRESS
                    elif target_mode == 'PaaS':
                        rate = SimConfig.COST_INTERNET_EGRESS 
                    elif target_mode == 'IaaS':
                        rate = SimConfig.COST_INTRA_REGION
                
                cost_network += (total_gb * rate)

                # COUPLING PENALTY
                if mode == 'IaaS' and target_mode != 'IaaS':
                    cost_penalty += SimConfig.PENALTY_COUPLING

    # 5. PHYSICS PENALTY (Stateful on Spot)
    if meta['type'] == 'stateful' and strategy == 'Spot':
        cost_penalty += SimConfig.PENALTY_PHYSICS_VIOLATION

    # 6. RELIABILITY COST
    availability = 1.0
    if mode == 'SaaS': availability = 0.999
    elif mode == 'PaaS': availability = 0.9995
    elif mode == 'IaaS':
        if strategy == 'Spot': availability = SimConfig.AVAILABILITY_SPOT
        else: availability = SimConfig.AVAILABILITY_OD
        
    lost_reqs = demand_vol * (1.0 - availability)
    cost_reliability = lost_reqs * SimConfig.REVENUE_PER_REQ

    # 7. TRANSITION
    cost_trans = 0.0
    if mode != p_mode:
        mult = TRANSITION_MATRIX.get((p_mode, mode), 1.0)
        
        # Use per-service transition cost matrix if provided (Monte Carlo)
        if transition_cost_matrix and (service_name, p_mode, mode) in transition_cost_matrix:
            base_cost = transition_cost_matrix[(service_name, p_mode, mode)]
        else:
            # Default: complexity * fixed cost
            base_cost = meta['complexity'] * SimConfig.COST_TRANSITION_FIXED
        
        cost_trans = base_cost * mult

    return cost_infra + cost_labor + cost_trans + cost_network + cost_reliability + cost_penalty
