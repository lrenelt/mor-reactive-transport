// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

// always include the config file
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

// C++ includes
#include<math.h>
#include<iostream>

#include <dune/pdelab.hh>

#include "solveNormalEq.hh"

int main(int argc, char** argv) {
  try {
    Dune::MPIHelper::instance(argc, argv);

    // Read parameters from ini file
    Dune::ParameterTree pTree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readINITree("debug_parameters.ini",pTree);
    ptreeparser.readOptions(argc,argv,pTree);

    solveTransportNormalEq<1>(pTree);
    solveTransportNormalEq<2>(pTree);
  }
  catch (Dune::Exception &e) {
    std::cerr << "Dune reported error: " << e << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "Unknown exception thrown!" << std::endl;
    return 1;
  }
}
