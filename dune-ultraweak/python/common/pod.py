# -*- tab-width: 4; indent-tabs-mode: nil  -*-

from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np

from pymor.algorithms.pod import pod
from pymor.operators.constructions import LincombOperator

from configs import default_config, default_solver_config
from discretizer import discretize
from pymor_classes import CustomReductor
from testing_utility import test_incrementally, get_parameter_sets

def run_pod(fom, reduction_config, testcase,
            snapshots=None,
            test_snapshots=None):
    assert testcase in ['P1', 'P2', 'P3']

    nTest = reduction_config['nTest']
    if snapshots is None:
        training_set, test_set = get_parameter_sets(testcase, fom, reduction_config)
        snapshots = fom.solution_space.empty()
        for mu in training_set:
            snapshots.append(fom.solve(mu))
    else:
        _, test_set = get_parameter_sets(testcase, fom, reduction_config)

    method = reduction_config['orthonormalization']
    assert method in ['L2', 'H1b', 'fixedMu']
    if method == 'L2':
        from pymor_classes import DuneL2Product
        product = DuneL2Product(fom.solver)
    elif method == 'H1b':
        from pymor_classes import DuneH1bProduct
        product = DuneH1bProduct(fom.solver)
    elif method == 'fixedMu':
        from pymor.operators.constructions import FixedParameterOperator
        fixedMu = fom.operator.parameters.parse({'Acw': reduction_config['fixedMu'],
                                                 'Bcc': reduction_config['fixedMu']})
        product = FixedParameterOperator(fom.operator, fixedMu)
    else:
        raise NotImplementedError(f'method {method} is not implemented')

    print(f'Running POD with {method}-orthonormalization...')


    rb, svals = pod(snapshots, product=product, rtol=reduction_config['pod_tol'])

    import csv
    suffix = f'_POD_ntrain_{reduction_config["nTrain_reaction"]}'
    suffix += f'_{testcase}_{method}'

    if method == 'fixedMu':
        suffix += f'_{reduction_config["fixedMu"]}'
    with open(f'singular_values{suffix}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['n', 'sigma'])
        for i in range(svals.shape[0]):
            writer.writerow((i+1, svals[i]))

    # reduce and test the (sub)models
    suffix += f'_ntest_{nTest}'
    reductor = CustomReductor(fom, rb, product=product)
    test_incrementally(reductor, test_set, filespecifier=suffix)

if __name__ == "__main__":
    config = default_config
    fom = discretize(config, default_solver_config)
    run_pod(fom, config['reduction'], config['testcase'])
