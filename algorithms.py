"""
Optimization strategies and solvers.
"""
import random
import copy
from config import SimConfig
from topology import SERVICE_TOPOLOGY, DECISION_CATALOG, TRANSITION_MATRIX
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
            mode, strat, reg, age = dec
            if not is_valid_local(meta, mode, strat): continue
            
            # Simple Greedy: Calculate cost if we move to 'dec'
            # (Independent optimization assumption)
            cost = calculate_complex_cost(svc, vol, dec, curr_state)
            
            if cost < best_cost:
                best_cost = cost
                best_state = dec
        
        next_states.append(best_state)
        
    return next_states

def solve_da_ma_complex(services_list, current_states, trace_slice, lookahead=12):
    """
    Optimized Memetic Algorithm with O(1) lookups and fast copying.
    """
    POP_SIZE = SimConfig.POP_SIZE
    GENERATIONS = SimConfig.GENERATIONS
    ELITISM_COUNT = SimConfig.ELITISM_COUNT
    
    # Helper: Check constraint (Moved up for scope)
    # Helper: Check constraint (Moved up for scope)
    def is_valid_gene(service, gene_idx):
        # Unpack 4-tuple
        mode, strat, _, _ = DECISION_CATALOG[gene_idx]
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
            
            # Create map for cost model to see other services
            current_step_map = {services_list[k]: step_states[k] for k in range(len(services_list))}

            for i, svc in enumerate(services_list):
                vol = trace_slice[svc][t]
                # Pass pre-calculated 'p_states[i]' to avoid re-decoding
                cost = calculate_complex_cost(svc, vol, step_states[i], p_states[i], all_states_map=current_step_map)
                total_cost += cost
            
            p_states = step_states        
        return total_cost

    # Pre-calculate current state indices for seeding (Optimized)
    # Pre-calculate current state indices for seeding (Optimized)
    current_indices = []
    for s_idx, s_name in enumerate(services_list):
        curr_mode, curr_strat, curr_reg, curr_age = current_states[s_idx]
        # Find index in DECISION_CATALOG
        match_idx = 0
        for d_idx, dc in enumerate(DECISION_CATALOG):
            if dc == (curr_mode, curr_strat, curr_reg, curr_age):
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
            ind.append(step)
        pop.append(ind)
        
    # HYBRID INITIALIZATION: Inject Greedy Solution
    # Calculate what the Reactive strategy would do for the entire lookahead window
    greedy_ind = []
    
    # We need a temporary 'current_state' tracker for the greedy simulation
    temp_states = current_states[:]
    
    for t in range(lookahead):
        # Create a slice for just this month t
        t_slice = {s: [trace_slice[s][t]] for s in services_list}
        
        # Get greedy decision for this month
        # Note: solve_reactive returns decision tuples, we need indices
        greedy_decs = solve_reactive(services_list, temp_states, t_slice)
        
        # Convert greedy decisions to genome indices
        step_indices = []
        for d in greedy_decs:
            # Find index in DECISION_CATALOG
            try:
                idx = DECISION_CATALOG.index(d)
            except ValueError:
                idx = 0 # Default fallback
            step_indices.append(idx)
        
        greedy_ind.append(step_indices)
        temp_states = greedy_decs # Update state for next greedy step
        
    # Replace first few random individuals with Greedy One
    for k in range(3): # Inject 3 copies to survive early tournaments
        pop[k] = copy.deepcopy(greedy_ind)
    
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

def solve_global_optimal_dp(services_list, full_trace, horizon):
    """
    Decoupled Exact DP Solver:
    Solves the optimal trajectory for each service INDEPENDENTLY using Viterbi/Dijkstra.
    This provides a 'Lower Bound' cost because it ignores coupling penalties (simulates infinite flexibility).
    """
    final_states = []
    # Start from Greenfield (Age=0)
    start_state = ('Greenfield', 'None', 'None', 0)
    
    # Pre-calculate Decision Catalog Indices
    n_states = len(DECISION_CATALOG)
    
    # Helper for Transition Validity
    def is_valid_transition(prev_state, curr_state):
        # Unpack
        p_mode, p_strat, p_reg, p_age = prev_state
        c_mode, c_strat, c_reg, c_age = curr_state
        
        # 1. RI Continuity Rule
        if p_strat == 'RI':
            if p_age < 12:
                # MUST go to RI(p_age + 1) to be a "valid continuance"
                # BUT we allow "breaking" the contract (RI -> OD or RI -> RI(1))
                # So physically, we can go anywhere.
                # WAIT. We can NOT go to RI(k) where k != p_age + 1 AND k != 1.
                # Example: RI(5) -> RI(7) is IMPOSSIBLE.
                # RI(5) -> RI(6) is OK.
                # RI(5) -> RI(1) is OK (Break + New).
                # RI(5) -> OD is OK (Break).
                
                if c_strat == 'RI':
                    if c_age == p_age + 1: return True # Normal flow
                    if c_age == 1: return True         # New Contract (Break old)
                    return False                       # Impossible jump
            
            # If p_age == 12, we are free.
            # If c_strat == 'RI', it MUST be c_age == 1 (New contract)
            if p_age == 12 and c_strat == 'RI' and c_age != 1: return False
            
        # 2. Entry Rule
        # If we enter RI from Non-RI, it MUST be Age=1
        if p_strat != 'RI' and c_strat == 'RI':
            if c_age != 1: return False
            
        return True
    
    # Let's implement the core DP logic here to return the FULL PLAN for a single service
    
    plan_per_service = {}
    
    for svc in services_list:
        # Viterbi Table: dp[t][state_idx] = min_cumulative_cost
        # Path Table: path[t][state_idx] = prev_state_idx
        
        # Initialize t=0
        # Cost from Greenfield to any state at t=0
        dp = [[float('inf')] * n_states for _ in range(horizon)]
        path = [[-1] * n_states for _ in range(horizon)]
        
        vol0 = full_trace[svc][0]
        
        for i_curr, curr in enumerate(DECISION_CATALOG):
            # Cost to enter this state from Greenfield
            # Note: We use calculate_complex_cost but need to be careful about transition logic
            # Explicit transition cost calculation
            cost = calculate_complex_cost(svc, vol0, curr, start_state)
            dp[0][i_curr] = cost
            
        # Iterate Forward
        for t in range(1, horizon):
            vol = full_trace[svc][t]
            for i_curr, curr in enumerate(DECISION_CATALOG):
                 # Find best previous state
                 # Min( dp[t-1][prev] + transition(prev->curr) + operate(curr) )
                 
                 best_prev_cost = float('inf')
                 best_prev_idx = -1
                 
                 # Cost of operations is same regardless of previous state (except transition)
                 # But calculate_complex_cost bundles them.
                 # Let's decouple: OpCost + TransCost
                 
                 # Optimization: Current state cost (without transition)
                 # We assume transition cost is 0 if prev==curr
                 # This loop is N^2 = 25 transitions. Small.
                 
                 for i_prev, prev in enumerate(DECISION_CATALOG):
                     prev_score = dp[t-1][i_prev]
                     if prev_score == float('inf'): continue
                     
                     # Check Validity first
                     if not is_valid_transition(prev, curr): continue

                     
                     step_cost = calculate_complex_cost(svc, vol, curr, prev)
                     total = prev_score + step_cost
                     
                     if total < best_prev_cost:
                         best_prev_cost = total
                         best_prev_idx = i_prev
                
                 dp[t][i_curr] = best_prev_cost
                 path[t][i_curr] = best_prev_idx
        
        # Backtrack
        final_min_cost = float('inf')
        final_idx = -1
        for i in range(n_states):
            if dp[horizon-1][i] < final_min_cost:
                final_min_cost = dp[horizon-1][i]
                final_idx = i
                
        # Reconstruct
        trajectory = []
        curr_idx = final_idx
        for t in range(horizon-1, -1, -1):
            trajectory.append(DECISION_CATALOG[curr_idx])
            curr_idx = path[t][curr_idx]
            
        trajectory.reverse() # 0..Horizon
        plan_per_service[svc] = trajectory
        
    return plan_per_service

def solve_isolated_strategies(players, environment_signal, game_horizon):
    """
    Calculates the 'Isolated Best Response' for each player.
    Game Theory equivalent: Finding the optimal strategy assuming no interaction with other players.
    (Used as the initialization step s^(0)).
    """
    initial_strategy_profile = {}
    
    # Pre-calculate Decision Space Indices
    n_strategies = len(DECISION_CATALOG)
    
    # Helper for Transition Validity (Rules of the Game)
    def is_valid_move(prev_move, curr_move):
        p_mode, p_strat, p_reg, p_age = prev_move
        c_mode, c_strat, c_reg, c_age = curr_move
        
        # 1. Contract Continuity Constraint
        if p_strat == 'RI':
            if p_age < 12:
                # Must continue contract OR break it (start new or go OD)
                # Cannot jump to arbitrary contract age
                if c_strat == 'RI':
                    if c_age == p_age + 1: return True # Continuity
                    if c_age == 1: return True         # Breach & Re-sign
                    return False                       # Invalid jump
            # If p_age == 12, contract ends. Freedom.
            if p_age == 12 and c_strat == 'RI' and c_age != 1: return False
            
        # 2. Contract Entry Constraint
        if p_strat != 'RI' and c_strat == 'RI':
            if c_age != 1: return False
            
        return True
    
    for player in players:
        # Cost Matrix: [Time x Strategy_Index]
        min_cost_matrix = [[float('inf')] * n_strategies for _ in range(game_horizon)]
        best_response_path = [[-1] * n_strategies for _ in range(game_horizon)]
        
        # t=0: Initialization
        demand_0 = environment_signal[player][0]
        start_state = ('Greenfield', 'None', 'None', 0)
        
        for i_curr, strategy in enumerate(DECISION_CATALOG):
            # Calculate Local Cost (Self-Cost only)
            cost = calculate_complex_cost(player, demand_0, strategy, start_state)
            min_cost_matrix[0][i_curr] = cost
            
        # Forward Pass (Viterbi / Bellman Equation)
        for t in range(1, game_horizon):
            demand_t = environment_signal[player][t]
            for i_curr, current_strategy in enumerate(DECISION_CATALOG):
                 
                 best_prev_cost = float('inf')
                 best_prev_idx = -1
                 
                 for i_prev, prev_strategy in enumerate(DECISION_CATALOG):
                     prev_accumulated_cost = min_cost_matrix[t-1][i_prev]
                     if prev_accumulated_cost == float('inf'): continue
                     
                     if not is_valid_move(prev_strategy, current_strategy): continue

                     # Marginal Cost of this step
                     step_cost = calculate_complex_cost(player, demand_t, current_strategy, prev_strategy)
                     total_cost = prev_accumulated_cost + step_cost
                     
                     if total_cost < best_prev_cost:
                         best_prev_cost = total_cost
                         best_prev_idx = i_prev
                
                 min_cost_matrix[t][i_curr] = best_prev_cost
                 best_response_path[t][i_curr] = best_prev_idx
        
        # Backward Pass (Strategy Reconstruction)
        final_min_cost = float('inf')
        final_idx = -1
        for i in range(n_strategies):
            if min_cost_matrix[game_horizon-1][i] < final_min_cost:
                final_min_cost = min_cost_matrix[game_horizon-1][i]
                final_idx = i
                
        trajectory = []
        curr_idx = final_idx
        for t in range(game_horizon-1, -1, -1):
            trajectory.append(DECISION_CATALOG[curr_idx])
            curr_idx = best_response_path[t][curr_idx]
            
        trajectory.reverse()
        initial_strategy_profile[player] = trajectory
        
    return initial_strategy_profile

def solve_nash_equilibrium_ibr(players, environment_signal, game_horizon, max_rounds=5):
    """
    Finds the Nash Equilibrium using Iterative Best Response (IBR).
    AKA: 'Coordinate Descent' on the Global Potential Function.
    
    Algorithm:
    1. Initialize s^(0) with Isolated Best Strategies.
    2. Round k:
       - Pick player 'i'.
       - Fix strategies of all opponents s_(-i).
       - Player 'i' updates s_i to minimize Cost(s_i, s_(-i)).
    3. Repeat until no player changes strategy (Nash Equilibrium).
    """
    # 1. Initialization (s^0)
    strategy_profile = solve_isolated_strategies(players, environment_signal, game_horizon)
    
    n_strategies = len(DECISION_CATALOG)
    
    # 2. Negotiation Rounds
    for round_idx in range(max_rounds):
        unstable_players_count = 0
        
        # Iterate through agents (Asynchronous Update)
        for active_player in players:
            
            # The 'Best Response' Calculation (Viterbi with Coupling)
            min_cost_matrix = [[float('inf')] * n_strategies for _ in range(game_horizon)]
            path_pointer = [[-1] * n_strategies for _ in range(game_horizon)]
            
            # Snapshot of Opponent Strategies (Context)
            def get_opponent_profile(t):
                return {p: strategy_profile[p][t] for p in players}

            # t=0
            demand_0 = environment_signal[active_player][0]
            start_state = ('Greenfield', 'None', 'None', 0)
            opponents_0 = get_opponent_profile(0)
            
            for i_curr, strategy in enumerate(DECISION_CATALOG):
                # Validity Check (Start Constraints)
                _, strat_type, _, age = strategy
                if strat_type == 'RI' and age != 1: 
                     min_cost_matrix[0][i_curr] = float('inf')
                     continue

                # Calculate Cost (Self + Coupling Penalty)
                cost = calculate_complex_cost(active_player, demand_0, strategy, start_state, all_states_map=opponents_0)
                min_cost_matrix[0][i_curr] = cost
                
            # Forward Pass
            for t in range(1, game_horizon):
                demand_t = environment_signal[active_player][t]
                opponents_t = get_opponent_profile(t)
                
                for i_curr, current_strategy in enumerate(DECISION_CATALOG):
                    best_prev_cost = float('inf')
                    best_prev_idx = -1
                    
                    for i_prev, prev_strategy in enumerate(DECISION_CATALOG):
                        prev_cost_accum = min_cost_matrix[t-1][i_prev]
                        if prev_cost_accum == float('inf'): continue
                        
                        # Validity check inline
                        p_mode, p_strat, _, p_age = prev_strategy
                        c_mode, c_strat, _, c_age = current_strategy
                        
                        is_valid = True
                        if p_strat == 'RI':
                            if p_age < 12:
                                if c_strat == 'RI':
                                    if not (c_age == p_age + 1 or c_age == 1): is_valid = False
                            else: # p_age == 12
                                if c_strat == 'RI' and c_age != 1: is_valid = False
                        elif p_strat != 'RI' and c_strat == 'RI':
                            if c_age != 1: is_valid = False
                            
                        if not is_valid: continue
                        
                        # Cost Calculation (Key: all_states_map passes neighbor data)
                        step_cost = calculate_complex_cost(active_player, demand_t, current_strategy, prev_strategy, all_states_map=opponents_t)
                        total_cost = prev_cost_accum + step_cost
                        
                        if total_cost < best_prev_cost:
                            best_prev_cost = total_cost
                            best_prev_idx = i_prev
                    
                    min_cost_matrix[t][i_curr] = best_prev_cost
                    path_pointer[t][i_curr] = best_prev_idx

            # Backward Pass (Best Response Extraction)
            final_idx = -1
            min_global_cost = float('inf')
            for i in range(n_strategies):
                if min_cost_matrix[game_horizon-1][i] < min_global_cost:
                    min_global_cost = min_cost_matrix[game_horizon-1][i]
                    final_idx = i
            
            new_best_response = []
            curr_idx = final_idx
            for t in range(game_horizon-1, -1, -1):
                new_best_response.append(DECISION_CATALOG[curr_idx])
                curr_idx = path_pointer[t][curr_idx]
            new_best_response.reverse()
            
            # Check for Equilibrium Stability
            if new_best_response != strategy_profile[active_player]:
                unstable_players_count += 1
                strategy_profile[active_player] = new_best_response
                
        if unstable_players_count == 0:
            # print(f"Nash Equilibrium reached in round {round_idx+1}")
            break 
            
    return strategy_profile

# Backward-compatible alias
def solve_iterative_dp(services_list, full_trace, horizon, max_iterations=5):
    """Alias for solve_nash_equilibrium_ibr for backward compatibility."""
    return solve_nash_equilibrium_ibr(services_list, full_trace, horizon, max_iterations)

