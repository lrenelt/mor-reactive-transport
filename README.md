This repository contains the code to reproduce the results from 'Model order reduction of an ultraweak and optimally stable variational formulation for parametrized reactive transport problems' by Christian Engwer, Mario Ohlberger and Lukas Renelt.

The code is based on Dune-PDELab and its dependencies which are all included as submodules. After you have cloned this repository you need to invoke ```git submodule init``` and ```git submodule update``` to fetch the submodules contents. Alternatively you can clone this repository with the option ```--recurse-submodules```.

## Setup of the build system
The location of the build-directory is specified in the *.opts*-file, consider changing the path beforehand. Per default, the directory is set to ```~/build_gcc_10```. Please adapt the following commands accordingly if you choose any other directory.

In order to build Dune-PDELab call (from the base directory)

    ./dune-common/bin/dunecontrol --opts=default.opts configure

This will initialise the build directory for all modules. Afterwards, you can build Dune-PDELab via

    ./dune-common/bin/dunecontrol --opts=default.opts --module=dune-pdelab make

## Reproduction of the nonparametric results

To reproduce the results from Figure 3 (convergence under h-refinement), build the test with

    ./dune-common/bin/dunecontrol --opts=default.opts --only=dune-ultraweak make runAllConvergenceTests

Then, switch to the build directory (``` cd ~/build_gcc_10/dune-ultraweak/dune/ultraweak/test```) and execute the test (```./runAllConvergenceTests```). The results are stored in the same folder as '.csv'-files.

## Setup of the python environment for the parametric tests

The tests for the parametrized problem are implemented in python using python-bindings for the DUNE code. To build the python-module change to the base directory and run

    ./dune-common/bin/dunecontrol --opts=default.opts --only=dune-ultraweak make ipyultraweak'

Afterwards, you can activate the python environment via

    source ~/build_gcc_10/dune-python-env/bin/activate

Now, change to folder with the python scripts (```cd ~/build_gcc_10/dune-ultraweak/python```) and run the shell script ```pyenv.sh``` i.e.

    chmod +x ./pyenv.sh
    source ./pyenv.sh

This sets the correct ```PYTHONPATH``` environmental variable.

## Reproduction of the parametric test results

To compute the solutions in Figure 4, run

    python3 compute_example_solutions.py

The resulting .vtu-files can for example be visualized with the software *paraview*.

To compute the data for Figure 5, run

    python3 test_norms.py

The results are stored as .csv-data.

To compute the data for Figure 6, run

    python3 run_all_greedy_tests.py

The results are also stored as .csv-data. Please remember that the measured runtimes depend on your local machine and may vary form our published data.





