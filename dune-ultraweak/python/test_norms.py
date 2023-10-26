# -*- tab-width: 4; indent-tabs-mode: nil  -*-

from common.configs import default_config, default_solver_config
from common.discretizer import discretize
from common.pod import run_pod
from common.testing_utility import get_parameter_sets

if __name__ == "__main__":
    config = default_config
    fom = discretize(config, default_solver_config)
    red_config = config['reduction']

    # precompute snapshots once
    snapshots = fom.solution_space.empty()
    config['testcase'] = 'P2'
    training_set, _ = get_parameter_sets(config['testcase'], fom, red_config)
    for mu in training_set:
        snapshots.append(fom.solve(mu))

    for method, mu in zip(['L2', 'H1b', 'fixedMu', 'fixedMu'],
                          [None, None, 0.0, 1.0]):
        red_config['orthonormalization'] = method
        red_config['fixedMu'] = mu
        red_config['nTrain_inflow'] = 1
        run_pod(fom, red_config, config['testcase'], snapshots)
