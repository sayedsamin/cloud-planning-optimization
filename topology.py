"""
Static data definitions: Catalogs, Topologies, and Matrices.
"""

# 1.1 IAAS CATALOG (Compute)
VM_CATALOG = {
    't3.micro':  {'cpu': 2, 'mem': 1.0, 'capacity_rps': 10.0, 'price_od': 0.0104, 'price_ri': 0.0063, 'price_spot': 0.0031, 'spot_risk': 0.05},
    't3.medium': {'cpu': 2, 'mem': 4.0, 'capacity_rps': 30.0, 'price_od': 0.0416, 'price_ri': 0.0250, 'price_spot': 0.0125, 'spot_risk': 0.05},
    'm5.large':  {'cpu': 2, 'mem': 8.0, 'capacity_rps': 80.0, 'price_od': 0.0960, 'price_ri': 0.0610, 'price_spot': 0.0350, 'spot_risk': 0.10},
    'm5.xlarge': {'cpu': 4, 'mem': 16.0,'capacity_rps': 160.0,'price_od': 0.1920, 'price_ri': 0.1220, 'price_spot': 0.0750, 'spot_risk': 0.12},
    'r5.large':  {'cpu': 2, 'mem': 16.0,'capacity_rps': 60.0, 'price_od': 0.1260, 'price_ri': 0.0790, 'price_spot': 0.0450, 'spot_risk': 0.15},
}

# 1.2 PAAS CATALOG (Serverless)
PAAS_CATALOG = {
    'standard': {'price_req': 0.20 / 1_000_000, 'price_gb_sec': 0.0000166667}
}

# 1.3 TRANSITION MATRIX
TRANSITION_MATRIX = {
    ('SaaS', 'PaaS'): 1.0, 
    ('PaaS', 'IaaS'): 2.0, 
    ('SaaS', 'IaaS'): 3.0, 
    ('IaaS', 'PaaS'): 1.5, 
    ('IaaS', 'SaaS'): 0.5, 
    ('PaaS', 'SaaS'): 0.5,
    ('SaaS', 'SaaS'): 0.0, ('PaaS', 'PaaS'): 0.0, ('IaaS', 'IaaS'): 0.0 # No cost to stay
}

# 2. APPLICATION TOPOLOGY
SERVICE_TOPOLOGY = {
    'frontend': {
        'type': 'stateless', 'complexity': 3, 'paas_spec': {'ms': 50, 'mb': 1024},
        'dependencies': [
            ('ad', 1.0, 1.0, 5.0), ('productcatalog', 1.0, 4.0, 50.0), ('recommendation', 0.8, 1.0, 10.0),
            ('cart', 0.4, 2.0, 5.0), ('shipping', 0.4, 1.0, 2.0), ('currency', 1.0, 1.0, 1.0), ('checkout', 0.1, 1.0, 10.0)
        ]
    },
    'checkout': {
        'type': 'stateless', 'complexity': 4, 'paas_spec': {'ms': 200, 'mb': 1024},
        'dependencies': [
            ('cart', 1.0, 1.0, 10.0), ('productcatalog', 1.0, 5.0, 20.0), ('shipping', 1.0, 1.0, 5.0),
            ('currency', 1.0, 1.0, 1.0), ('payment', 1.0, 1.0, 5.0), ('email', 1.0, 1.0, 50.0)
        ]
    },
    'recommendation': {'type': 'stateless', 'complexity': 5, 'paas_spec': {'ms': 800, 'mb': 3008}, 'dependencies': [('productcatalog', 1.0, 5.0, 5.0)]},
    'productcatalog': {'type': 'stateful', 'complexity': 2, 'dependencies': [], 'paas_spec': {'ms': 20, 'mb': 256}},
    'cart':           {'type': 'stateful', 'complexity': 2, 'dependencies': [], 'paas_spec': {'ms': 20, 'mb': 256}},
    'payment':        {'type': 'stateful', 'complexity': 5, 'dependencies': [], 'paas_spec': {'ms': 100,'mb': 512}},
    'shipping':       {'type': 'stateless', 'complexity': 2, 'dependencies': [], 'paas_spec': {'ms': 50, 'mb': 512}},
    'email':          {'type': 'stateless', 'complexity': 1, 'dependencies': [], 'paas_spec': {'ms': 50, 'mb': 128}},
    'currency':       {'type': 'stateless', 'complexity': 1, 'dependencies': [], 'paas_spec': {'ms': 10, 'mb': 128}},
    'ad':             {'type': 'stateless', 'complexity': 1, 'dependencies': [], 'paas_spec': {'ms': 10, 'mb': 128}}
}

# 3. DECISION SPACE
DECISION_CATALOG = [
    ('SaaS', 'Standard', 'us-east-1'),
    ('PaaS', 'Standard', 'us-east-1'),
    ('IaaS', 'OD', 'us-east-1'),
    ('IaaS', 'RI', 'us-east-1'),
    ('IaaS', 'Spot', 'us-east-1')
]
