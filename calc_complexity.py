from config import SimConfig
from topology import SERVICE_TOPOLOGY, DECISION_CATALOG

def calc_complexity():
    # PARAMETERS
    N_SERVICES = len(SERVICE_TOPOLOGY)
    N_STATES = len(DECISION_CATALOG)
    HORIZON = SimConfig.SCENARIO_HORIZON
    
    # 1. OPTIMAL (DP) COMPLEXITY -> Edges Evaluated
    # t=0: From 'Greenfield' (1 state) to Any N_STATES
    t0_evals = 1 * N_STATES
    # t=1..H: From Any N_STATES to Any N_STATES
    step_evals = N_STATES * N_STATES
    
    dp_per_service = t0_evals + ((HORIZON - 1) * step_evals)
    dp_total_evals = dp_per_service * N_SERVICES
    
    # Implicit Search Space (Number of possible paths)
    # The DP implicitly checks all of these by pruning sub-optimal ones.
    implicit_paths_per_service = N_STATES ** HORIZON
    
    # 2. PROPOSED (GA) COMPLEXITY
    # Approaches problem as rolling window lookahead
    # Evaluations = Steps * Generations * PopSize
    ga_evals = HORIZON * SimConfig.GENERATIONS * SimConfig.POP_SIZE
    
    print(f"--- COMPLEXITY ANALYSIS ---")
    print(f"Services: {N_SERVICES}")
    print(f"Horizon: {HORIZON} months")
    print(f"States per Service: {N_STATES}")
    print(f"---------------------------")
    print(f"OPTIMAL (DP) Algorithm:")
    print(f"  Explicit Transitions Evaluated: {dp_total_evals:,}")
    print(f"  Implicit Scenarios Covered:     10 * 5^{HORIZON} (~1.45e25)")
    print(f"---------------------------")
    print(f"PROPOSED (GA) Algorithm:")
    print(f"  Approximate Evaluations:        {ga_evals:,}")

if __name__ == "__main__":
    calc_complexity()
