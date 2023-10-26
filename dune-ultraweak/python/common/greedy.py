# -*- tab-width: 4; indent-tabs-mode: nil  -*-

from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np

from pymor.algorithms.greedy import rb_greedy
from pymor.operators.constructions import FixedParameterOperator
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor

from configs import default_config, default_solver_config
from discretizer import discretize
from pymor_classes import CustomReductor
from testing_utility import test_incrementally, get_parameter_sets

def run_greedy(fom, reduction_config, testcase):
    assert testcase in ['P1', 'P2', 'P3']

    nTest = reduction_config['nTest']
    training_set, test_set = get_parameter_sets(testcase, fom, reduction_config)

    # Experimentally obtained values for the chosen example
    Tmin = 0.4
    Tmax = 1.2
    cinfty = 1
    # TODO: check whether these are the current constants
    Cp = 2*Tmax + (1-Tmin) # = 3.0
    coercivityConstant = ExpressionParameterFunctional('(2+9*(2*Acw**2 + 1))**(-0.5)',
                                                       fom.parameters)

    method = reduction_config['orthonormalization']
    assert method in ['L2', 'H1b', 'fixedMu']
    if method == 'L2':
        from pymor_classes import DuneL2Product
        product = DuneL2Product(fom.solver)
        reductor = CoerciveRBReductor(fom, product=product,
                                      coercivity_estimator=coercivityConstant)
        filespecifier = ''
    elif method == 'H1b':
        from pymor_classes import DuneH1bProduct
        product = DuneH1bProduct(fom.solver)
        reductor = CoerciveRBReductor(fom, product=product,
                                      coercivity_estimator=coercivityConstant)
    elif method == 'fixedMu':
        from pymor_classes import FixedParameterProduct
        fixedMu = fom.operator.parameters.parse({'Acw': reduction_config['fixedMu'],
                                                 'Bcc': reduction_config['fixedMu'],
                                                 'Cgin': -1})
        fixedMuproduct = FixedParameterOperator(fom.operator, fixedMu)
        reductor = CoerciveRBReductor(fom, product=product)
    else:
        raise NotImplementedError(f'method {method} is not implemented')

    # TODO: write norm-aware Gram-Schmidt
    extension_params = {'method': 'gram_schmidt'}

    suffix = f'_ntrain_{reduction_config["nTrain_reaction"]}'
    suffix += f'_{testcase}_{method}'
    reduced_basis = rb_greedy(fom=fom, reductor=reductor, training_set=training_set,
                              use_error_estimator=True, error_norm=None,
                              rtol=reduction_config['greedy_tol'],
                              max_extensions=reduction_config['greedy_max_extensions'],
                              extension_params=extension_params)

    errors = reduced_basis['max_errs']

    import csv
    with open(f'greedy_error{suffix}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['n', 'error'])
        for i in range(len(errors)):
            writer.writerow((i+1, errors[i]))

    plt.figure()
    plt.semilogy(np.arange(1,len(errors)+1), errors)
    plt.title('decay of the greedy-error')
    plt.show()

    suffix += f'_ntest_{nTest}'
    test_incrementally(reductor, test_set, filespecifier=suffix)

if __name__ == "__main__":
    config = default_config
    fom = discretize(config, default_solver_config)
    run_greedy(fom, config['reduction'], config['testcase'])
