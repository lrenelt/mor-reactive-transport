# -*- tab-width: 4; indent-tabs-mode: nil  -*-

from mpi4py import MPI

from common.configs import default_config, default_solver_config
from common.discretizer import discretize

if __name__ == "__main__":

    config = default_config
    config['grid']['yasp_x'] = 128
    config['grid']['yasp_y'] = 128
    solver_config = default_solver_config

    fom = discretize(config, solver_config, order=1)
    print('FOM model: ', fom)
    print('Parameters: ', fom.parameters)

    filenames = ["solution_darcy_fo_0-0-1",
                 "solution_darcy_fo_1-0-1",
                 "solution_darcy_fo_03-01-1"]
    parameters = [{'Acw': 0., 'Bcc': 0., 'Cgin': 1.0},
                  {'Acw': 1., 'Bcc': 0., 'Cgin': 1.0},
                  {'Acw': 0.3, 'Bcc': 0.1, 'Cgin': 1.0}]
    for mu,f in zip(parameters,filenames):
        mu = fom.parameters.parse(mu)
        fom.visualize(fom.solve(mu), mu, filename=f)
