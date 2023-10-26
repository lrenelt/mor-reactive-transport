// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_DARCYSIPGSOLVER_HH
#define DUNE_ULTRAWEAK_DARCYSIPGSOLVER_HH

#include "dune/pdelab.hh"

#include "dune/ultraweak/solvingManager.hh"
#include "dune/ultraweak/transport/traits.hh"

#include "discreteGridFunctionFlux.hh"

// Solves the Darcy equation using an SIPG-formulation
template<std::size_t order, typename GV, typename Problem, typename Traits_ = DGSolverTraits<GV,order>>
class DarcySIPGSolver : public SolvingManager<GV, Problem, Traits_> {

  using Traits = Traits_;

private:
  using Base = SolvingManager<GV, Problem, Traits>;

public:
  DarcySIPGSolver(const GV& gv_, Problem& problem_, Dune::ParameterTree& pTree_) :
    Base(gv_, typename Traits::FEM(), problem_, pTree_) {
    gfs.name("dg_sol");
  }

  Dune::InverseOperatorResult solvingStats;

  void solve() {
    std::cout << "Starting to solve the Darcy-equation (SIPG formulation, order="
      + std::to_string(order) + ")" << std::endl;

    static const int dim = Traits::dim;
    using DF = typename Traits::DF;
    using RF = typename Traits::RF;

    // Local operator
    using LocalOperator = Dune::PDELab::ConvectionDiffusionDG<Problem, typename Traits::FEM>;
    LocalOperator localOperator(problem, Dune::PDELab::ConvectionDiffusionDGMethod::SIPG,
                                Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn, 1.0);

    // constraints
    using GridFunctionSpace = typename Traits::GridFunctionSpace;
    using ConstraintsContainer = typename GridFunctionSpace::template ConstraintsContainer<RF>::Type;
    ConstraintsContainer constraintsContainer;
    constraintsContainer.clear();

    // Grid operator
    gfs.update();
    using std::pow;
    const int dofestimate = pow(2, dim) * gfs.maxLocalSize();
    using MatrixBackend = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
    MatrixBackend matrixBackend(dofestimate);
    using GridOperator = Dune::PDELab::GridOperator<GridFunctionSpace,
                                                    GridFunctionSpace,
                                                    LocalOperator,
                                                    MatrixBackend,
                                                    DF,
                                                    RF,
                                                    RF,
                                                    ConstraintsContainer,
                                                    ConstraintsContainer>;
    GridOperator gridOperator(gfs, constraintsContainer,
                              gfs, constraintsContainer,
                              localOperator, matrixBackend);

    std::cout << "gfs with " << gfs.size() << " dofs generated  "<< std::endl;
    std::cout << "cc with " << constraintsContainer.size() << " dofs generated  "<< std::endl;

    // Solution vector
    coeffs = 0.0;

    // Solve matrix-based with PDELab preset
    const int verbosity = 2;
    const int maxiter = 5000;

    // set up projection CG space
    using CGFEM = Dune::PDELab::PkQkLocalFiniteElementMap<DF,RF,dim>;
    static const int cgorder = 1;
    const CGFEM cgfem(cgorder);

    using CGBCType = Dune::PDELab::DirichletConstraintsParameters;
    using VectorBackend = Dune::PDELab::ISTL::VectorBackend<>;
    using CGGFS = Dune::PDELab::GridFunctionSpace<GV, CGFEM, CGBCType, VectorBackend>;
    CGGFS cggfs(gv,cgfem);

    std::cout << "\nUsing sequential CG-solver with AMG preconditioning... \n" << std::endl;
    using LinearSolver = typename Dune::PDELab::ISTLBackend_SEQ_AMG_4_DG<
      GridOperator, CGGFS,Dune::PDELab::CG2DGProlongation, Dune::SeqSSOR, Dune::CGSolver>;

    LinearSolver linearSolver(gridOperator, cggfs, maxiter, verbosity);

    auto params = linearSolver.parameters();
    params.setCoarsenTarget(2000);
    params.setMaxLevel(20);
    params.setProlongationDampingFactor(1.8);
    params.setNoPreSmoothSteps(2);
    params.setNoPostSmoothSteps(2);
    params.setGamma(1);
    params.setAdditive(false);
    linearSolver.setParameters(params);

    using CoefficientVector = typename Traits::CoefficientVector;
    using Solver = Dune::PDELab::StationaryLinearProblemSolver<GridOperator, LinearSolver, CoefficientVector>;

    const double reduction = pTree.template get<double>("darcy.reduction");
    Solver solver(gridOperator, linearSolver, coeffs, reduction);
    solver.apply();

    const auto res = linearSolver.result();
    solvingStats.converged = res.converged;
    solvingStats.iterations = res.iterations;
    solvingStats.elapsed = res.elapsed;
    solvingStats.reduction = res.reduction;
    solvingStats.conv_rate = res.conv_rate;
  }

  // special grid function for the flux
  using DGFFlux = DiscreteGridFunctionFlux<
    typename Traits::GridFunctionSpace,
    typename Traits::CoefficientVector,
    Problem>;

  DGFFlux getDiscreteGridFunctionFlux() const {
    return DGFFlux(gfs,coeffs,problem);
  }

  auto getDiscreteGridFunctionFluxPtr() const {
    return std::make_unique<DGFFlux>(gfs,coeffs,problem);
  }

  // Visualization
  void writeVTK(std::string filename = "solutionDG") const {
    using VTKWriter = Dune::SubsamplingVTKWriter<GV>;
    Dune::RefinementIntervals subsampling(pTree.template get<double>("visualization.subsampling_dg"));
    VTKWriter vtkwriter(gv, subsampling);
    std::string vtkfile(filename);
    using DGF = typename Traits::DiscreteGridFunction;
    using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGF>;
    auto dgf = this->getDiscreteGridFunction();
    vtkwriter.addVertexData(std::make_shared<VTKF>(dgf, "dg_solution"));

    using VTKFFlux = Dune::PDELab::VTKGridFunctionAdapter<DGFFlux>;
    auto dgfFlux = this->getDiscreteGridFunctionFlux();
    vtkwriter.addCellData(std::make_shared<VTKFFlux>(dgfFlux, "flux_solution"));

    vtkwriter.write(vtkfile, Dune::VTK::appendedraw);
  }

protected:
  using Base::problem;
  using Base::gv;
  using Base::gfs;
  using Base::coeffs;
  using Base::pTree;
};

#endif // DUNE_ULTRAWEAK_DARCYSIPGSOLVER_HH
