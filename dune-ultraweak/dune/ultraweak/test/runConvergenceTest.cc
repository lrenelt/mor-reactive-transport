// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

// always include the config file
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

// C++ includes
#include<iostream>

#include<dune/pdelab.hh>

#include "problems.hh"
#include "../testutil.hh"

int main(int argc, char** argv) {
  try {
    Dune::MPIHelper::instance(argc, argv);

    const int dim = 2;
    using RF = double;

    // Read parameters from ini file
    Dune::ParameterTreeParser ptreeparser;

    Dune::ParameterTree pTree;
    ptreeparser.readINITree("debug_parameters.ini",pTree);
    ptreeparser.readOptions(argc,argv,pTree);

    Dune::ParameterTree pTreeConvTest;
    ptreeparser.readINITree("conv_test_parameters.ini",pTreeConvTest);
    ptreeparser.readOptions(argc,argv,pTreeConvTest);

    // make grid
    typedef Dune::YaspGrid<dim> Grid;
    Dune::FieldVector<double, dim> domain(1.0);
    std::array<int, dim> domainDims;

    if (dim==2) {
      domain[0] = 1.0;
      domain[1] = 1.0;
      domainDims[0] = pTreeConvTest.get<int>("grid.yasp_x");
      domainDims[1] = pTreeConvTest.get<int>("grid.yasp_y");
    }
    else
      std::cerr << "Currently only 2-dimensional version implemented" << std::endl;

    auto gridp = std::make_shared<Grid>(domain, domainDims);

    PureTransportProblem<typename Grid::LevelGridView, RF> baseProblem(pTree);
    if (pTree.get<bool>("darcy.useDarcy")) {
      if (pTreeConvTest.get<bool>("firstOrder"))
        doTestWithRefSolDarcy<1>(gridp, pTree, pTreeConvTest, baseProblem);
      if (pTreeConvTest.get<bool>("secondOrder"))
        doTestWithRefSolDarcy<2>(gridp, pTree, pTreeConvTest, baseProblem);
    }
    else {
      if (pTreeConvTest.get<bool>("firstOrder"))
        doTestWithRefSol<1>(gridp, pTree, pTreeConvTest, baseProblem);
      if (pTreeConvTest.get<bool>("secondOrder"))
        doTestWithRefSol<2>(gridp, pTree, pTreeConvTest, baseProblem);
    }
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
