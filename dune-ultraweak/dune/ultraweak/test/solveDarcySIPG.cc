// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune/pdelab.hh>

#include "../darcy-velocity/catalysatorProblems.hh"
#include "../darcy-velocity/darcySIPGSolver.hh"
#include "../darcy-velocity/utility.hh"

int main(int argc, char** argv)
{
  try{
     // Maybe initialize mpi
    Dune::MPIHelper::instance(argc, argv);

    const int dim = 2;
    static constexpr std::size_t order = 2;
    using RF = double;

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
    const auto gv = gridp->leafGridView();

    using GV = typename Grid::LeafGridView;

    using Problem = CatalysatorProblem<GV,RF,BoundaryConditionType::pressure>;
    Problem problem(pTree);
    using Solver = DarcySIPGSolver<order,GV,Problem>;
    Solver solver(gv, problem, pTree);
    solver.solve();
    solver.writeVTK("darcySolutionSIPG");

    const auto fluxDgf = solver.getDiscreteGridFunctionFlux();
    writeVelocityFieldVTK(fluxDgf, "velocityFieldSIPG");
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
