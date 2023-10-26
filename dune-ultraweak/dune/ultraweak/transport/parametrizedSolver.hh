// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_TRANSPORT_PARAMETRIZED_SOLVER_HH
#define DUNE_ULTRAWEAK_TRANSPORT_PARAMETRIZED_SOLVER_HH

#include <dune/pdelab.hh>

#include "defaultParametrization.hh"
#include "discreteGridFunctionReconstruction.hh"
#include "normalEqLocalOperator.hh"
#include "traits.hh"

#include "../H1bnorm.hh"
#include "../solvingManager.hh"

// Wrapper class for all solving aspects
// TODO: adapt the SolvingManager
template<std::size_t order, typename GV, typename Problem, typename Parametrization>
class ParametrizedSolver : public SolvingManager<GV, Problem, NormalEqTraits<GV,order>> {
public:
  using Traits = NormalEqTraits<GV,order>;

private:
  using Base = SolvingManager<GV, Problem, Traits>;

  using GFS = typename Traits::GridFunctionSpace;
  using CoefficientVector = typename Traits::CoefficientVector;

public:
  using RF = typename Traits::RF;
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;

  using NormalEqOperatorType = SymmetricTransportOperator<Problem, Parametrization, typename Traits::FEM>;
  using NormalEqGO = Dune::PDELab::GridOperator<
    typename Traits::GridFunctionSpace, typename Traits::GridFunctionSpace,
    NormalEqOperatorType, MBE, RF, RF , RF,
    typename Base::CC, typename Base::CC>;

  // parametric typedefs
  using ParameterType = typename Parametrization::ParameterType;
  static constexpr std::size_t Qa = Parametrization::Qa;
  static constexpr std::size_t Qasquared = Qa * Qa;
  static constexpr std::size_t Qf = Parametrization::Qf;

private:
  using DGF = typename Traits::DiscreteGridFunction;
  using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGF>;
  using DGFGradient = Dune::PDELab::DiscreteGridFunctionGradient<GFS,CoefficientVector>;
  using DGFReconstruction = DiscreteGridFunctionReconstruction<
    DGF, DGFGradient, Problem, Parametrization>;
  using VTKF_REC = Dune::PDELab::VTKGridFunctionAdapter<DGFReconstruction>;

public:
  using MatrixType = Dune::PDELab::Backend::Native<typename NormalEqGO::Jacobian>;
  using RHSVectorType = Dune::PDELab::Backend::Native<typename NormalEqGO::Range>;
  using Y = Dune::PDELab::Backend::Native<typename NormalEqGO::Domain>;

private:
  using L2MassOperatorType = Dune::PDELab::L2;
  using L2MassGO = Dune::PDELab::GridOperator<
    typename Traits::GridFunctionSpace, typename Traits::GridFunctionSpace,
    L2MassOperatorType, MBE, RF, RF, RF,
    typename Base::CC, typename Base::CC>;

  using H1bMassOperatorType = H1bnorm<Problem, typename Traits::FEM>;
  using H1bMassGO = Dune::PDELab::GridOperator<
    typename Traits::GridFunctionSpace, typename Traits::GridFunctionSpace,
    H1bMassOperatorType, MBE, RF, RF, RF,
    typename Base::CC, typename Base::CC>;

public:
  using L2MatrixType = Dune::PDELab::Backend::Native<typename L2MassGO::Jacobian>;
  using H1bMatrixType = Dune::PDELab::Backend::Native<typename H1bMassGO::Jacobian>;

  using OperatorType = Dune::MatrixAdapter<MatrixType,Y,Y>;

  ParametrizedSolver(const GV gv, Problem& problem,
                     Parametrization& parametrization,
                     Dune::ParameterTree& pTree) :
    Base(gv, typename Traits::FEM(gv), problem, pTree), parametrization_(parametrization)
  {
    gfs.name("Numerical solution of the normal equations");

    // make parameter-independent operators
    const int dim = Traits::dim;
    MBE mbe(1<<(dim+1)); // guess nonzeros per row
    const int intorder = Traits::order + 2;
    const bool rescaling = pTree.template get<bool>("rescaling");

    typename NormalEqGO::Domain tempEvalPoint(gfs, 0.0);

    lop_ = std::make_shared<NormalEqOperatorType>(problem, parametrization_, intorder, rescaling);
    NormalEqGO normalGo(gfs, cc, gfs, cc, *lop_, mbe);

    // assembly
    assembleAllParameterIndependentParts(normalGo, parametrization_, matrices_, rhsVectors_);

    // wrap into an operator
    for (const auto& mat : matrices_)
      ops_.emplace_back(mat);

    // assemble L2-product on the test space
    L2MassOperatorType mass;
    auto l2massgo = std::make_shared<L2MassGO>(gfs, cc, gfs, cc, mass, mbe);
    typename L2MassGO::Jacobian l2mat(*l2massgo, 0.0);
    l2massgo->jacobian(tempEvalPoint, l2mat);
    l2matrix_ = Dune::PDELab::Backend::native(l2mat);

    // assemble H1b-product on the test space
    H1bMassOperatorType h1bmass(problem);
    auto h1bmassgo = std::make_shared<H1bMassGO>(gfs, cc, gfs, cc, h1bmass, mbe);
    typename H1bMassGO::Jacobian h1bmat(*h1bmassgo, 0.0);
    h1bmassgo->jacobian(tempEvalPoint, h1bmat);
    h1bmatrix_ = Dune::PDELab::Backend::native(h1bmat);

    // initialize solvers
    Dune::initSolverFactories<OperatorType>();
}

  Dune::InverseOperatorResult solvingStats;

  /**
     Solves linear transport using the 'normal equations'
  */
  Y solve(const ParameterType mu, Dune::ParameterTree& solverConfig) {
    parametrization_.setParameter(mu);
    // assemble parameter-dependent matrix
    MatrixType mat = matrices_[0]; // TODO: different initialization?
    mat *= 0.0;
    for (std::size_t i=0; i<Qasquared; i++)
      mat.axpy(parametrization_.bilinear().theta(i), matrices_[i]);

    // assemble parameter-dependent RHS
    RHSVectorType rhsVector = rhsVectors_[0]; // TODO: different initialization?
    rhsVector *= 0.0;
    for (std::size_t i=0; i<Qf; i++)
      rhsVector.axpy(parametrization_.rhs().theta(i), rhsVectors_[i]);

    const auto linOp = std::make_shared<OperatorType>(mat);

    const auto solver = getSolverFromFactory(linOp, solverConfig);

    Y solution(mat.N());
    solution *= 0.0;
    solver->apply(solution, rhsVector, solvingStats);

    coeffs = CoefficientVector(gfs, solution); // attach the underlying container

    if (solvingStats.condition_estimate > 0)
      std::cout << "Convergence rate: " << solvingStats.conv_rate << ", condition (estimate): " << solvingStats.condition_estimate << std::endl;

    return solution;
  }

  void visualize(Y rawCoeffs, const ParameterType mu, std::string filename = "normal_eq_sol") {
    coeffs = CoefficientVector(gfs, rawCoeffs);
    parametrization_.setParameter(mu);
    writeVTK(filename);
  }

  void writeVTK(std::string filename) const {
    const int subsampling = pTree.template get<int>("visualization.subsampling");
    Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, Dune::RefinementIntervals(subsampling), false);

    DGF dgf = this->getDiscreteGridFunction();
    vtkwriter.addVertexData(std::make_shared<VTKF>(dgf, "test_sol"));

    const auto dgfGradient = DGFGradient(gfs, coeffs);
    const auto dgfReconstruction = DiscreteGridFunctionReconstruction(
             dgf, dgfGradient, problem, parametrization_,
             pTree.template get<bool>("rescaling"));
    vtkwriter.addVertexData(std::make_shared<VTKF_REC>(
      dgfReconstruction, "reconstruction_subsampling"));

    if (Traits::order == 2)
      filename += "_so";
    vtkwriter.write(filename, Dune::VTK::appendedraw);

  }

  const auto& getMatrices() const {
    return matrices_;
  }

  const L2MatrixType& getL2MassMatrix() const {
    return l2matrix_;
  }

  const H1bMatrixType& getH1bMassMatrix() const {
    return h1bmatrix_;
  }

  const auto& getOperators() const {
    return ops_;
  }

  const auto& getRhsVectors() const {
    return rhsVectors_;
  }

private:
  using Base::problem;
  using Base::gv;
  using Base::gfs;
  using Base::cc;
  using Base::coeffs;
  using Base::pTree;

protected:
  L2MatrixType l2matrix_;
  H1bMatrixType h1bmatrix_;
  std::array<MatrixType,Qasquared> matrices_;
  std::vector<OperatorType> ops_;

  std::array<RHSVectorType,Qf> rhsVectors_;

  Parametrization& parametrization_;
  std::shared_ptr<NormalEqOperatorType> lop_;
};

#endif  // DUNE_ULTRAWEAK_TRANSPORT_PARAMETRIZED_SOLVER_HH
