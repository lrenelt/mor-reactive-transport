# -*- tab-width: 4; indent-tabs-mode: nil  -*-

from matplotlib import pyplot as plt
import numpy as np
import time

def get_parameter_sets(testcase, fom, reduction_config):
    nTest = reduction_config['nTest']
    if testcase == 'P1':
        parameter_space = fom.parameters.space({'Acw': [0., 1.],
                                                'Bcc': [0., 0.],
                                                'Cgin': [1., 1.]})
        training_set = parameter_space.sample_uniformly(
            {'Acw': reduction_config['nTrain_reaction'],
             'Bcc': 1,
             'Cgin': 1})
        test_set = parameter_space.sample_randomly(nTest)
    elif testcase == 'P2':
        parameter_space = fom.parameters.space({'Acw': [0., 1.],
                                                'Bcc': [0., 1.],
                                                'Cgin': [1., 1.]})
        training_set = parameter_space.sample_uniformly(
            {'Acw': reduction_config['nTrain_reaction'],
             'Bcc': reduction_config['nTrain_reaction'],
             'Cgin': 1})
        test_set = parameter_space.sample_randomly(2*nTest) #heuristic
        # enforce constrain
        training_set = [m for m in training_set if m['Acw'][0] >= m['Bcc'][0]]
        test_set = [m for i,m in enumerate(test_set)
                    if m['Acw'][0] >= m['Bcc'][0]][:nTest]
    elif testcase == 'P3':
        parameter_space = fom.parameters.space({'Acw': [0., 1.],
                                                'Bcc': [0., 1.],
                                                'Cgin': [1., 10.]})
        training_set = parameter_space.sample_uniformly(
            {'Acw': reduction_config['nTrain_reaction'],
             'Bcc': reduction_config['nTrain_reaction'],
             'Cgin': reduction_config['nTrain_inflow']})
        # enforce constrain
        test_set = parameter_space.sample_randomly(2*nTest) #heuristic
        training_set = [m for m in training_set if m['Acw'][0] >= m['Bcc'][0]]
        test_set = [m for i,m in enumerate(test_set)
                    if m['Acw'][0] >= m['Bcc'][0]][:nTest]
    else:
        raise NotImplementedError(f'Unknown testcase ({testcase})')
    return training_set, test_set


def evaluate_reduced_model(reductor, rom, fomSols, parameters, error_norm,
                           evaluate_estimator=False):
    nTest = len(parameters)
    errors = np.empty(nTest)
    estimatedErrors = -np.ones(nTest)

    if evaluate_estimator:
        estimator = reductor.assemble_error_estimator_for_subbasis({'RB': rom.order})

    romTimings = np.empty(nTest)

    for i,mu in enumerate(parameters):
        t1 = time.perf_counter()
        u_rom = rom.solve(mu)
        t2 = time.perf_counter()
        romTimings[i] = t2-t1

        diff = reductor.reconstruct(u_rom)-fomSols[i]
        errors[i] = error_norm(diff, mu)
        if evaluate_estimator:
            estimatedErrors[i] = estimator.estimate_error(u_rom, mu, m=None)

    return errors, estimatedErrors, romTimings

def test_incrementally(reductor, test_set,
                       evaluate_estimator=False, filespecifier=''):
    fom = reductor.fom

    # TODO: move outside
    def fullErrorNorm(u,mu):
        return np.sqrt(fom.operator.apply2(u,u,mu)).item()

    rbSize = len(reductor.bases['RB'])
    nTest = len(test_set)

    fomTimings = np.empty(nTest)

    fomSols = []
    for i,mu in enumerate(test_set):
        t1 = time.perf_counter()
        fomSols.append(fom.solve(mu))
        t2 = time.perf_counter()
        fomTimings[i] = t2-t1

    errors = np.empty((nTest,rbSize))
    estimatedErrors = np.empty((nTest,rbSize))
    romTimings = np.empty((nTest,rbSize))

    for n in range(rbSize):
      rom_n = reductor.reduce(dims=n+1)

      data = evaluate_reduced_model(
          reductor, rom_n, fomSols, test_set, fullErrorNorm,
          evaluate_estimator)

      errors[:,n] = data[0]
      estimatedErrors[:,n] = data[1]
      romTimings[:,n] = data[2]

    import csv
    header = ['cw', 'cc', 'gin', 'fom_time']
    header += [f'error_dim_{n+1}' for n in range(rbSize)]
    header += [f'rom_time_dim_{n+1}' for n in range(rbSize)]

    with open('rb_evaluation' + filespecifier + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, p in enumerate(test_set):
            row = [p.to_numpy()[0], p.to_numpy()[1], p.to_numpy()[2], fomTimings[i]]
            row += [errors[i,n] for n in range(rbSize)]
            row += [romTimings[i,n] for n in range(rbSize)]
            writer.writerow(tuple(row))

def test_full_rom(reductor, parameter_space, nTest,
                  filespecifier='',
                  visualize_max_error=True):
    fom = reductor.fom
    rom = reductor.reduce()

    def fullErrorNorm(u,mu):
        return np.sqrt(fom.operator.apply2(u,u,mu)).item()

    parameters = parameter_space.sample_randomly(nTest)

    fomTimings = np.empty(nTest)

    fomSols = []
    for i,mu in enumerate(parameters):
        t1 = time.perf_counter()
        fomSols.append(fom.solve(mu))
        t2 = time.perf_counter()
        fomTimings[i] = t2-t1

    errors, _, romTimings = evaluate_reduced_model(reductor, rom, fomSols,
            parameters, fullErrorNorm)

    imax = np.argmax(errors)
    mu_max = parameters[imax]
    print(f'max error at mu={mu_max}: {errors[imax]: .2e}')

    if visualize_max_error:
        u_rom = reductor.reconstruct(rom.solve(mu_max))
        u_fom = fom.solve(mu_max)
        diff = u_rom-u_fom
        fom.visualize(u_fom, mu_max, f'fom_max_err')
        fom.visualize(u_rom, mu_max, f'sol_max_err')
        fom.visualize(diff, mu_max, f'diff_max_err')

    cw = [p.to_numpy()[0] for p in parameters]
    cc = [p.to_numpy()[1] for p in parameters]
    gin = [p.to_numpy()[2] for p in parameters]

    import csv
    with open('fullROM' + filespecifier + '.csv',
              'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['cw', 'cc', 'gin', 'error', 'time_FOM', 'time_ROM'])
        for data in zip(cw,cc,gin,errors,fomTimings,romTimings):
            writer.writerow(data)
