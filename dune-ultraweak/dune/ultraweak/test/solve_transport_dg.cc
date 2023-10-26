// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

// C++ includes
#include<math.h>
#include<iostream>

#include <dune/pdelab.hh>

#include "problems.hh"
#include "../darcy-velocity/catalysatorProblems.hh"
#include "../darcy-velocity/darcyMixedSolver.hh"
#include "../darcy-velocity/darcyProblem.hh"
#include "../darcy-velocity/darcySIPGSolver.hh"
#include "../darcy-velocity/darcyVelocityAdapter.hh"
#include "../darcy-velocity/utility.hh"
#include "../upwindDGsolver.hh"
#include "../transport/defaultParametrization.hh"

/**
   Solves the pure transport problem using DG.
*/
int main(int argc, char** argv)
{
  try{
     // Maybe initialize mpi
    Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
    if (!Dune::MPIHelper::isFake)
      std::cout << "Code is running on " << helper.size() << " processes." << std::endl;

    const int dim = 2;
    using RF = double;
    static const std::size_t order = 1;
    static constexpr bool solveMixed = true;

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

    if (dim==2) {
      domain[0] = 1.0;
      domain[1] = 1.0;
      domainDims[0] = pTree.get<int>("grid.yasp_x");
      domainDims[1] = pTree.get<int>("grid.yasp_y");
    }
    else {
      std::cerr << "Currently only 2-dimensional version implemented" << std::endl;
    }

    auto gridp = std::make_shared<Grid>(domain, domainDims);
    gridp->globalRefine(refinement);
    auto gv = gridp->leafGridView();

    using GV = typename Grid::LeafGridView;

    PureTransportProblem<GV,RF> baseProblem(pTree);
    ExtendedTransportParametrization parametrization;
    parametrization.initializeFromConfig(pTree);
    ParametrizationDecorator paramProblem(baseProblem, parametrization);

    if (pTree.get<bool>("darcy.useDarcy")) {
      if constexpr (solveMixed) {
        using DarcyProblem = CatalysatorProblem<GV, RF, BoundaryConditionType::mixed>;
        DarcyProblem darcyProblem(pTree);

        DarcyMixedSolver darcysolver(gv, darcyProblem, pTree);
        darcysolver.solve();
        darcysolver.writeVTK();

        const auto darcyVelocity = darcysolver.getDiscreteGridFunction();

        // from here on its the same
        DarcyVelocityAdapter problem(paramProblem, darcyVelocity);
        problem.setRefinement(false);
        using DGSolver = UpwindDGSolver<order,GV,decltype(problem)>;
        DGSolver solver(gv, problem, pTree);
        solver.solve();

        writeVelocityFieldVTK(darcyVelocity, pTree.get<int>("visualization.subsampling_velocity"));
        solver.writeVTK();
      }
      else {
        using DarcyProblem = CatalysatorProblem<GV, RF, BoundaryConditionType::pressure>;
        DarcyProblem darcyProblem(pTree);

        static const std::size_t orderDarcy = 2;
        DarcySIPGSolver<orderDarcy> darcysolver(gv, darcyProblem, pTree);
        darcysolver.solve();
        darcysolver.writeVTK("pressureSolutionDG");

        const auto gradientDgf = Dune::PDELab::DiscreteGridFunctionGradient(
          darcysolver.getGfs(), darcysolver.getCoefficientVector());
        // no matrix-valued discrete grid functions!
        auto permLambda = [darcyProblem](const auto& el, const auto& x){
          return -darcyProblem.A(el,x)[0][0]; };
        const auto permDgf = Dune::PDELab::makeGridFunctionFromCallable(gv, permLambda);
        const auto darcyVelocity = Dune::PDELab::ProductGridFunctionAdapter(permDgf,gradientDgf);

        writeVelocityFieldVTK(darcyVelocity, pTree.get<int>("visualization.subsampling_velocity"));

        // from here on its the same
        DarcyVelocityAdapter problem(paramProblem, darcyVelocity);
        problem.setRefinement(false);
        using DGSolver = UpwindDGSolver<order,GV,decltype(problem)>;
        DGSolver solver(gv, problem, pTree);
        solver.solve();
        solver.writeVTK();
      }
    }
    else {
      using DGSolver = UpwindDGSolver<order,GV,decltype(paramProblem)>;
      DGSolver solver(gv, paramProblem, pTree);
      solver.solve();
      solver.writeVTK();
    }
  }
  catch (Dune::Exception &e){
    std::cerr << "Dune reported error: " << e << std::endl;
        return 1;
  }
  catch (...){
    std::cerr << "Unknown exception thrown!" << std::endl;
        return 1;
  }
}
