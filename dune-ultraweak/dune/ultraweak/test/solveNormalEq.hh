// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_TEST_SOLVE_NORMAL_EQ_HH
#define DUNE_ULTRAWEAK_TEST_SOLVE_NORMAL_EQ_HH

#include "dune/ultraweak/darcy-velocity/catalysatorProblems.hh"
#include "dune/ultraweak/darcy-velocity/darcySIPGSolver.hh"
#include "dune/ultraweak/darcy-velocity/darcyVelocityAdapter.hh"

#include "dune/ultraweak/test/problems.hh"
#include "dune/ultraweak/transport/normalEqSolver.hh"

/**
   Solve the normal equations for the linear advection-reaction
   problem. Directly asssembles and solves for one given parameter.
*/
template<typename std::size_t order>
void solveTransportNormalEq(Dune::ParameterTree& pTree) {
    const int dim = 2;
    using RF = double;

    typedef Dune::YaspGrid<dim> Grid;
    Dune::FieldVector<double, dim> domain(1.0);
    std::array<int, dim> domainDims;

    domain[0] = 1.0;
    domain[1] = 1.0;
    domainDims[0] = pTree.get<int>("grid.yasp_x");
    domainDims[1] = pTree.get<int>("grid.yasp_y");

    auto gridp = std::make_shared<Grid>(domain, domainDims);
    int refinements = 0;
    if (pTree.get<bool>("darcy.useDarcy")) {
      refinements = pTree.get<int>("darcy.extraRefinements");
      if (refinements > 0)
        gridp->globalRefine(refinements);
    }

    using GV = typename Grid::LevelGridView;
    const auto gv = gridp->levelGridView(0);

    Dune::ParameterTree solverpTree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readINITree("solver_config.ini", solverpTree);


    PureTransportProblem<GV,RF> baseProblem(pTree);
    if (pTree.get<bool>("darcy.useDarcy")) {
      using DarcyProblem = CatalysatorProblem<GV,RF,BoundaryConditionType::pressure>;
      DarcyProblem darcyProblem(pTree);
      const auto gvRef = gridp->levelGridView(refinements);
      static const std::size_t orderDarcy = order+2;
      auto darcysolver = DarcySIPGSolver<orderDarcy,GV,DarcyProblem>(gvRef, darcyProblem, pTree);
      darcysolver.solve();
      const auto darcy_velocity = darcysolver.getDiscreteGridFunctionFluxPtr();

      DarcyVelocityAdapter problem(baseProblem, *darcy_velocity);
      problem.setRefinement(refinements > 0);
      using Solver = NormalEqSolver<order,GV,decltype(problem)>;
      Solver solver(gv, problem, pTree);
      solver.solve(solverpTree);
      solver.writeVTK();
    }
    else {
      using Solver = NormalEqSolver<order,GV,decltype(baseProblem)>;
      Solver solver(gv, baseProblem, pTree);
      solver.solve(solverpTree);
      solver.writeVTK();
    }
}

#endif  // DUNE_ULTRAWEAK_TEST_SOLVE_NORMAL_EQ_HH
