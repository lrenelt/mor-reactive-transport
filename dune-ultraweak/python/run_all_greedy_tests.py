# -*- tab-width: 4; indent-tabs-mode: nil  -*-

from common.configs import default_config, default_solver_config
from common.discretizer import discretize
from common.greedy import run_greedy

if __name__ == "__main__":
    config = default_config
    # make sure some parameters are set as intended
    config['reduction']['orthonormalization'] = 'H1b'

    fom = discretize(config, default_solver_config)
    for tc, ntrain in zip(['P1', 'P2', 'P3'], [500, 35, 35]):
        config['reduction']['ntrain_reaction'] = ntrain
        run_greedy(fom, config['reduction'], tc)
