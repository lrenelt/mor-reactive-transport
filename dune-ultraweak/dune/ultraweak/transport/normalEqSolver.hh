// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_TRANSPORT_NORMAL_EQ_SOLVER_HH
#define DUNE_ULTRAWEAK_TRANSPORT_NORMAL_EQ_SOLVER_HH

#include <dune/pdelab.hh>

#include <dune/istl/solverfactory.hh>

#include "defaultParametrization.hh"
#include "discreteGridFunctionReconstruction.hh"
#include "normalEqLocalOperator.hh"
#include "traits.hh"

#include "../solvingManager.hh"

// Wrapper class for all solving aspects
template< std::size_t order, typename GV, typename Problem>
class NormalEqSolver : public SolvingManager<GV, Problem, NormalEqTraits<GV,order>> {
public:
  using Traits = NormalEqTraits<GV,order>;

private:
  using Base = SolvingManager<GV, Problem, Traits>;

  using Parametrization = ExtendedTransportParametrization;

  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using NormalEqOperatorType = SymmetricTransportOperator<Problem, Parametrization, typename Traits::FEM>;
  using NormalEqGO = Dune::PDELab::GridOperator<
    typename Traits::GridFunctionSpace, typename Traits::GridFunctionSpace,
    NormalEqOperatorType, MBE,
    typename Traits::RF, typename Traits::RF, typename Traits::RF,
    typename Base::CC, typename Base::CC>;

private:
  using SchurMatType = Dune::PDELab::Backend::Native<typename NormalEqGO::Jacobian>;
  using Y = Dune::PDELab::Backend::Native<typename Traits::CoefficientVector>;
  using SchurOperatorType = Dune::MatrixAdapter<SchurMatType, Y,Y>;

public:
  NormalEqSolver(const GV& gv, Problem& problem, Dune::ParameterTree& pTree) :
    Base(gv, typename Traits::FEM(gv), problem, pTree)
  {
    gfs.name("Numerical solution of the normal equations");

    // Make a global operator
    const int dim = Traits::dim;
    MBE mbe(1<<(dim+1)); // guess nonzeros per row

    using std::pow;
    const int extraIntOrders = pTree.get<int>("extraIntOrders");
    const int intorder = pow(Traits::order + extraIntOrders, 2) + 1;

    parametrization_.initializeFromConfig(pTree);

    lop_ = std::make_shared<NormalEqOperatorType>(problem, parametrization_,
      intorder, pTree.template get<bool>("rescaling"),
      pTree.template get<double>("rescalingOut"));

    normaleqgo = std::make_shared<NormalEqGO>(gfs, cc, gfs, cc, *lop_, mbe);

    // initialize solvers
    Dune::initSolverFactories<SchurOperatorType>();
  }

  Dune::InverseOperatorResult solvingStats;

  /**
     Solves linear transport using the 'normal equations'
  */
  void solve(Dune::ParameterTree& solverConfig) {
    std::cout << "Solving the normal equation with order " << std::to_string(Traits::order) << "..." << std::endl;
    // assemble the system matrix
    typename NormalEqGO::Jacobian wrappedSchurMat(*normaleqgo, 0.0);
    typename NormalEqGO::Domain tempEvalPoint(gfs, 0.0);
    normaleqgo->jacobian(tempEvalPoint, wrappedSchurMat);

    // assemble the rhs
    typename NormalEqGO::Range rhs(gfs, 0.0);
    normaleqgo->residual(tempEvalPoint, rhs);
    rhs *= -1;
    auto natRHS = Dune::PDELab::Backend::native(rhs);

    // create linear operator
    SchurMatType schurMat = Dune::PDELab::Backend::native(wrappedSchurMat);
    const auto schurop = std::make_shared<SchurOperatorType>(schurMat);

    // create solver
    const auto solver = getSolverFromFactory(schurop, solverConfig);

    // solve
    auto solution = std::make_shared<Y>(schurMat.N());
    *solution = 0.0;
    solver->apply(*solution, natRHS, solvingStats);
    coeffs.attach(solution); // attach the underlying container

    std::cout << "Done!" << std::endl;

    if (solvingStats.condition_estimate > 0)
      std::cout << "Convergence rate: " << solvingStats.conv_rate << ", condition (estimate): " << solvingStats.condition_estimate << std::endl;
  }

  void writeVTK(std::string filename = "normalEqSol") const {
    // write output files
    using GFS = typename Traits::GridFunctionSpace;
    using CoefficientVector = typename Traits::CoefficientVector;

    using DGF = typename Traits::DiscreteGridFunction;
    using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGF>;

    const int subsampling = pTree.template get<int>("visualization.subsampling");
    Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, Dune::RefinementIntervals(subsampling), false);
    DGF dgf = this->getDiscreteGridFunction();
    vtkwriter.addVertexData(std::make_shared<VTKF>(dgf, "test_sol"));

    using DGFGradient = Dune::PDELab::DiscreteGridFunctionGradient<GFS,CoefficientVector>;
    using DGFReconstruction = DiscreteGridFunctionReconstruction<
      DGF, DGFGradient, Problem, Parametrization>;
    using VTKF_REC = Dune::PDELab::VTKGridFunctionAdapter<DGFReconstruction>;

    const auto dgfScalar = this->getDiscreteGridFunction();
    const auto dgfGradient = DGFGradient(gfs,coeffs);
    const auto dgfReconstruction = DiscreteGridFunctionReconstruction(
             dgfScalar, dgfGradient, problem, parametrization_,
             pTree.template get<bool>("rescaling"));
    vtkwriter.addVertexData(std::make_shared<VTKF_REC>(dgfReconstruction, "reconstruction_subsampling"));

    if (Traits::order == 2)
      filename += "_so";
    vtkwriter.write(filename, Dune::VTK::appendedraw);
  }

public:
  using Base::problem;
  using Base::gv;
  using Base::gfs;
  using Base::cc;
  using Base::coeffs;
  using Base::pTree;

private:
  Parametrization parametrization_;
  std::shared_ptr<NormalEqOperatorType> lop_;
  std::shared_ptr<NormalEqGO> normaleqgo;
};

#endif  // DUNE_ULTRAWEAK_TRANSPORT_NORMAL_EQ_SOLVER_HH
