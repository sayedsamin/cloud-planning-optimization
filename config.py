"""
Configuration parameters for the Cloud Simulation.
"""

class SimConfig:
    # PHYSICS
    HOURS_TOTAL = 730
    HOURS_PEAK = 12 * 30
    HOURS_OFF_PEAK = 730 - (12 * 30)
    REQS_PER_USER = 50.0
    
    # TAXES
    TAX_EKS = 72.0
    TAX_NAT_GB_MONTH = 0.045 * 730
    TAX_ALB_MONTH = 0.0225 * 730
    TAX_EGRESS_GB = 0.09
    
    # LABOR (Hourly Rates)
    LABOR_SAAS = 0.5
    LABOR_PAAS = 5.0
    LABOR_IAAS = 20.0
    LABOR_TRANSITION = 40.0
    LABOR_RATE_HOURLY = 80.0
    
    # COST PENALTIES
    PENALTY_PHYSICS_VIOLATION = 10000.0
    PENALTY_COUPLING = 500.0
    COST_NETWORK_BASE = 500.0
    cost_NETWORK_GB = 0.02
    
    # GENETIC ALGORITHM
    POP_SIZE = 50
    GENERATIONS = 30
    ELITISM_COUNT = 5
    
    # SCENARIO
    SCENARIO_HORIZON = 36
    GBM_MU = 0.15
    GBM_SIGMA = 0.40
    GBM_START = 50000
    GBM_SEED = 101
