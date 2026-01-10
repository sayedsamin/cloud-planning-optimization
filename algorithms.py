"""
Optimization strategies and solvers.
"""
import random
import copy
from config import SimConfig
from topology import SERVICE_TOPOLOGY, DECISION_CATALOG
from cost_model import calculate_complex_cost

def solve_reactive(services_list, current_states, slice_data):
    """
    Greedy/Reactive Solver: Picking the best move for t=0 only.
    """
    next_states = []
    
    # Pre-calculate valid choices 
    def is_valid_local(meta, mode, strategy):
        if meta['type'] == 'stateful' and strategy == 'Spot': return False
        return True

    for i, svc in enumerate(services_list):
        vol = slice_data[svc][0]
        meta = SERVICE_TOPOLOGY[svc]
        curr_state = current_states[i]
        
        best_state = curr_state
        best_cost = float('inf')
        
        # Evaluate all valid moves
        for dec in DECISION_CATALOG:
            mode, strat, reg = dec
            if not is_valid_local(meta, mode, strat): continue
            
            # Simple Greedy: Calculate cost if we move to 'dec'
            # (Independent optimization assumption)
            cost = calculate_complex_cost(svc, vol, dec, curr_state)
            
            if cost < best_cost:
                best_cost = cost
                best_state = dec
        
        next_states.append(best_state)
        
    return next_states

def solve_da_ma_complex(services_list, current_states, trace_slice, lookahead=6):
    """
    Optimized Memetic Algorithm with O(1) lookups and fast copying.
    """
    POP_SIZE = SimConfig.POP_SIZE
    GENERATIONS = SimConfig.GENERATIONS
    ELITISM_COUNT = SimConfig.ELITISM_COUNT
    
    # Helper: Check constraint (Moved up for scope)
    def is_valid_gene(service, gene_idx):
        mode, strat, _ = DECISION_CATALOG[gene_idx]
        meta = SERVICE_TOPOLOGY[service]
        if meta['type'] == 'stateful' and strat == 'Spot': return False
        return True

    # PRE-COMPUTATION FOR SPEED
    # Map service names to indices to avoid .index() calls in loops
    svc_idx_map = {name: i for i, name in enumerate(services_list)}
    
    # Pre-calculate valid choices for every service to avoid re-checking
    valid_genes_map = {}
    for i, svc in enumerate(services_list):
        valid_genes_map[i] = [x for x in range(len(DECISION_CATALOG)) if is_valid_gene(svc, x)]

    def get_fitness(genome):
        total_cost = 0
        p_states = current_states[:]
        
        for t in range(lookahead):
            # Decode genome indices to Decision Tuples
            step_states = [DECISION_CATALOG[idx] for idx in genome[t]]
            
            for i, svc in enumerate(services_list):
                vol = trace_slice[svc][t]
                meta = SERVICE_TOPOLOGY[svc]
                
                # 1. Physics Penalty
                if meta['type'] == 'stateful' and step_states[i][1] == 'Spot':
                     total_cost += SimConfig.PENALTY_PHYSICS_VIOLATION

                # 2. Financial Cost
                # Pass pre-calculated 'p_states[i]' to avoid re-decoding
                cost = calculate_complex_cost(svc, vol, step_states[i], p_states[i])
                total_cost += cost
                
                # 3. Coupling Penalty (Dependency Check)
                if 'dependencies' in meta:
                    for target, _, _, _ in meta['dependencies']:
                        t_idx = svc_idx_map[target]
                        if step_states[i][0] == 'IaaS' and step_states[t_idx][0] != 'IaaS':
                            total_cost += SimConfig.PENALTY_COUPLING
            
            p_states = step_states
        return total_cost

    # Pre-calculate current state indices for seeding (Optimized)
    current_indices = []
    for s_idx, s_name in enumerate(services_list):
        curr_mode, curr_strat, curr_reg = current_states[s_idx]
        # Find index in DECISION_CATALOG
        match_idx = 0
        for d_idx, dc in enumerate(DECISION_CATALOG):
            if dc == (curr_mode, curr_strat, curr_reg):
                match_idx = d_idx
                break
        current_indices.append(match_idx)

    # Initialization
    pop = []
    for i in range(POP_SIZE):
        ind = []
        is_seed_stable = (i < POP_SIZE // 2) # 50% seeded with stability
        
        for t in range(lookahead):
            step = []
            for s_idx, s in enumerate(services_list):
                # Use pre-calculated valid genes
                valid_choices = valid_genes_map[s_idx]
                
                if is_seed_stable:
                    # Prefer staying in current state if valid
                    target = current_indices[s_idx]
                    if target in valid_choices:
                        step.append(target)
                    else:
                        step.append(random.choice(valid_choices))
                else:
                    step.append(random.choice(valid_choices))
            ind.append(step)
        pop.append(ind)
    
    # EVOLUTION LOOP
    for gen in range(GENERATIONS):
        pop.sort(key=get_fitness)
        new_pop = pop[:ELITISM_COUNT] # Keep Elites
        
        while len(new_pop) < POP_SIZE:
            # Tournament
            p1 = random.choice(pop[:15]) # Pick from top 15
            p2 = random.choice(pop[:15])
            
            # FAST CROSSOVER (No deepcopy)
            child = [row[:] for row in p1] # 100x faster than deepcopy
            for t in range(lookahead):
                if random.random() < 0.5: 
                    child[t] = p2[t][:]
            
            # MUTATION
            if random.random() < 0.4:
                t = random.randint(0, lookahead-1)
                s = random.randint(0, len(services_list)-1)
                
                # Fast valid choice lookup
                new_val = random.choice(valid_genes_map[s])
                child[t][s] = new_val
                
                # MEMETIC REPAIR
                if DECISION_CATALOG[new_val][0] == 'IaaS':
                    svc_name = services_list[s]
                    if 'dependencies' in SERVICE_TOPOLOGY[svc_name]:
                        for target, _, _, _ in SERVICE_TOPOLOGY[svc_name]['dependencies']:
                            t_idx = svc_idx_map[target] # Fast lookup
                            # 50% chance to drag dependency to IaaS/OD (Index 2)
                            if random.random() < 0.5: 
                                child[t][t_idx] = 2 
            
            new_pop.append(child)
        pop = new_pop

    best_genome_step0 = pop[0][0]
    return [DECISION_CATALOG[idx] for idx in best_genome_step0]
