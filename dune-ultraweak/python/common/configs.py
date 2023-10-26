default_config = {
    'rescaling': False,
    'rescalingOut': 1.0,
    'grid': {
        'dim': 2,
        'refinement': 0,
        'yasp_x':  128,
        'yasp_y':  128
    },
    'problem': {
        'discontinuousInflow': False,
        'nInflowBumps':  4,
        'eta': 0.2,
        'openingHeight': 0.25,
        'coatingHeight': 0.125,
        'non-parametric': {}
    },
    'darcy': {
        'useDarcy':  True,
        'extraRefinements': 0,
        'reduction': 1e-12,
        'min_permeability': 0.2,
        'coatingPermeability': 0.05
    },
    'visualization': {
        'subsampling': 8,
        'subsampling_velocity': 5,
        'subsampling_dg': 5
    },
    'testcase': 'P2', # either {'P1', 'P2', 'P3'}
    'reduction': {
        'nTrain_reaction': 500,
        'nTrain_inflow': 10,
        'nTest': 500,
        'pod_tol': 1e-10,
        'greedy_tol': 1e-15,
        'greedy_max_extensions': 30,
        'orthonormalization': 'H1b', # either {'L2', 'H1b', 'fixedMu'}
        'fixedMu': None
    }
}

default_solver_config = {
    'type': 'cgsolver',
    'verbose': 0,
    'maxit': 200,
    'reduction': 1e-20,
    'preconditioner': {
        'type': 'amg',
        'iterations': 1,
        'relaxation': 1,
        'maxLevel': 15,
        'coarsenTarget': 2000,
        'criterionSymmetric': True,
        'smoother': 'sor',
        'smootherIterations': 1,
        'smootherRelaxation': 1
    }
}
