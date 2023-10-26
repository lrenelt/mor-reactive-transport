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

#include "problems.hh"
#include "dune/ultraweak/darcy-velocity/catalysatorProblems.hh"
#include "dune/ultraweak/darcy-velocity/darcyMixedSolver.hh"
#include "dune/ultraweak/darcy-velocity/darcyProblem.hh"
#include "../darcy-velocity/darcyVelocityAdapter.hh"

#include "dune/ultraweak/transport/defaultParametrization.hh"
#include "dune/ultraweak/transport/parametrizedSolver.hh"

int main(int argc, char** argv) {
  try {
    Dune::MPIHelper::instance(argc, argv);

    const int dim = 2;
    using RF = double;
    static const std::size_t order = 2;

    // Read parameters from ini file
    Dune::ParameterTree pTree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readINITree("debug_parameters.ini",pTree);
    ptreeparser.readOptions(argc,argv,pTree);

    // make grid
    const int refinement = pTree.get<int>("grid.refinement");

    typedef Dune::YaspGrid<dim> Grid;
    Dune::FieldVector<double, dim> domain(1.0);
    std::array<int, dim> domainDims;

    domain[0] = 1.0;
    domain[1] = 1.0;
    domainDims[0] = pTree.get<int>("grid.yasp_x");
    domainDims[1] = pTree.get<int>("grid.yasp_y");

    auto gridp = std::make_shared<Grid>(domain, domainDims);
    gridp->globalRefine(refinement);
    auto gv = gridp->leafGridView();

    Dune::ParameterTree solverpTree;
    ptreeparser.readINITree("solver_config.ini", solverpTree);

    using GV = typename Grid::LeafGridView;
    PureTransportProblem<GV, RF> baseProblem(pTree);
    ExtendedTransportParametrization parametrization;
    const double mu0 = pTree.get<double>("problem.non-parametric.fixedInflowScaling");
    const double mu1 = pTree.get<double>("problem.non-parametric.fixedInflowOffset");
    const double mu2 = pTree.get<double>("problem.non-parametric.fixedReactionWashcoat");
    const double mu3 = pTree.get<double>("problem.non-parametric.fixedReactionCoating");

    if (pTree.get<bool>("darcy.useDarcy")) {
      using DarcyProblem = CatalysatorProblem<GV, RF>;
      DarcyProblem darcyProblem(pTree);
      DarcyMixedSolver darcysolver(gv, darcyProblem, pTree);
      darcysolver.solve();

      using DGFType = DarcyMixedSolver<GV,DarcyProblem>::DiscreteGridFunction;
      std::unique_ptr<DGFType> darcyVelocity = darcysolver.getDiscreteGridFunctionPtr();

      DarcyVelocityAdapter problem(baseProblem, *darcyVelocity);
      using Solver = ParametrizedSolver<order, GV, decltype(problem), decltype(parametrization)>;
      Solver solver(gv, problem, parametrization, pTree);
      const auto sol = solver.solve({mu0, mu1, mu2, mu3}, solverpTree);
      solver.visualize(sol, {mu0, mu1, mu2, mu3}, "separated_normal_eq_sol");
    }
    else {
      using Solver = ParametrizedSolver<order, GV, decltype(baseProblem), decltype(parametrization)>;
      Solver solver(gv, baseProblem, parametrization, pTree);
      const auto sol = solver.solve({mu0, mu1, mu2, mu3}, solverpTree);
      solver.visualize(sol, {mu0, mu1, mu2, mu3}, "separated_normal_eq_sol");
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
