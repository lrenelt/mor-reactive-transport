// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_DARCY_VELOCITY_DARCY_MIXED_SOLVER_HH
#define DUNE_ULTRAWEAK_DARCY_VELOCITY_DARCY_MIXED_SOLVER_HH

#include <dune/pdelab.hh>

#include "darcyProblem.hh"
#include "constraints.hh"
#include "masslumping.hh"
#include "reconstruction.hh"
#include "schurcomplement.hh"
#include "../solvingManager.hh"


template<typename GV>
struct DarcyMixedTraits {
  static const int dim = GV::dimension;

  using DF = typename GV::Grid::ctype;
  using RF = double;

  using VBE = Dune::PDELab::ISTL::VectorBackend<>;

  static const int order_rt = 0;
  using FEM = Dune::PDELab::RaviartThomasLocalFiniteElementMap<GV,DF,RF, order_rt>;
  using RT0CON = Dune::PDELab::RT0Constraints;
  using GridFunctionSpace = Dune::PDELab::GridFunctionSpace<GV,FEM,RT0CON,VBE>;
  using CoefficientVector = Dune::PDELab::Backend::Vector<GridFunctionSpace, DF>;
  using DiscreteGridFunction = Dune::PDELab::DiscreteGridFunctionPiola<GridFunctionSpace, CoefficientVector>;
};

// Wrapper class for all solving aspects
template<typename GV, typename Problem, typename Traits = DarcyMixedTraits<GV>>
class DarcyMixedSolver : public SolvingManager<GV, Problem, Traits> {
private:
  using Base = SolvingManager<GV, Problem, Traits>;

private:
  // defines for the pressure space
  static const int dim = Traits::dim;

  using DF = typename Traits::DF;
  using RF = typename Traits::RF;

  static const int order_dg = 0;
  using FEM_DG = Dune::PDELab::QkDGLocalFiniteElementMap<DF,RF,order_dg, dim, Dune::PDELab::QkDGBasisPolynomial::lagrange>;
  using NOCON = Dune::PDELab::NoConstraints;
  using GFS_DG = Dune::PDELab::GridFunctionSpace<GV,FEM_DG,NOCON,typename Traits::VBE>;
  using CoeffsDG = Dune::PDELab::Backend::Vector<GFS_DG, DF>;
  using DGFDG = Dune::PDELab::DiscreteGridFunction<GFS_DG, CoeffsDG>;

  const FEM_DG fem_dg;
  GFS_DG gfs_dg;
  CoeffsDG coeffs_dg;

  // defines for the tensor space
  using TensorVBE = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::bcrs>;
  using TensorGFS = Dune::PDELab::CompositeGridFunctionSpace<TensorVBE, Dune::PDELab::LexicographicOrderingTag, typename Traits::GridFunctionSpace,GFS_DG>;

public:
  DarcyMixedSolver(const GV& gv_, Problem& problem_, Dune::ParameterTree& pTree_) :
    Base(gv_, typename Traits::FEM(gv_), problem_, pTree_),
    gfs_dg(gv, fem_dg), coeffs_dg(gfs_dg)
  {
    gfs.name("RT0");
    gfs_dg.name("DG");
  }

  void solve() {
    std::cout << "Starting to solve the Darcy-equation (mixed formulation)" << std::endl;

    // make tensor gfs
    typename Traits::GridFunctionSpace gfs_rt_copy(gv,fem);
    GFS_DG gfs_dg_copy(gv,fem_dg);
    TensorGFS tensor_gfs(gfs_rt_copy,gfs_dg_copy);

    // Local operator
    using LocalOperator = Dune::PDELab::DiffusionMixed<Problem>;
    LocalOperator localOperator(problem);

    // Assemble constraints
    Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<Problem> bctype(gv,problem);
    using CC = typename TensorGFS::template ConstraintsContainer<RF>::Type;
    CC cc;
    cc.clear();
    Dune::PDELab::constraints(bctype, tensor_gfs, cc);

    // Grid operator
    tensor_gfs.update();
    using std::pow;
    const int dofestimate = pow(2, dim) * tensor_gfs.maxLocalSize();
    using MatrixBackend = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
    MatrixBackend matrixBackend(dofestimate);
    using GridOperator = Dune::PDELab::GridOperator<TensorGFS,
                                                    TensorGFS,
                                                    LocalOperator,
                                                    MatrixBackend,
                                                    DF, RF, RF,
                                                    CC, CC>;
    GridOperator gridOperator(tensor_gfs, cc, tensor_gfs, cc,
                              localOperator, matrixBackend);

    std::cout << "gfs with " << tensor_gfs.size() << " dofs generated  "<< std::endl;
    std::cout << "cc with " << cc.size() << " dofs generated  "<< std::endl;

    // Solve matrix-based
    using XGFS = typename TensorGFS::template Child<0>::Type;
    using WrappedX = Dune::PDELab::Backend::Vector<XGFS, DF>;
    using X = Dune::PDELab::Backend::Native<WrappedX>;
    using YGFS = typename TensorGFS::template Child<1>::Type;
    using WrappedY = Dune::PDELab::Backend::Vector<YGFS, DF>;
    using Y = Dune::PDELab::Backend::Native<WrappedY>;

    using RhsType = typename GridOperator::Range;
    typename GridOperator::Domain temp(tensor_gfs, 0.0);
    RhsType rhs(tensor_gfs, 0.0);
    gridOperator.residual(temp, rhs);

    typename GridOperator::Jacobian fullMat(gridOperator, 0.0);
    gridOperator.jacobian(temp, fullMat);

    const auto flowLambda = [this](const auto& is, const auto& x){ return problem.j(is,x); };
    auto func1 = Dune::PDELab::makeGridFunctionFromCallable(gv, flowLambda);
    // Dirichlet conditions will be weakly imposed
    auto func2 = Dune::PDELab::makeGridFunctionFromCallable(gv,[](const auto& x){ return 0.0; });
    const auto boundaryGridFunction = Dune::PDELab::CompositeGridFunction(func1, func2);

    // assemble strong constraints
    RhsType rhsConstraints(tensor_gfs, 0.0);
    Dune::PDELab::interpolate(boundaryGridFunction, tensor_gfs, rhsConstraints);
    Dune::PDELab::copy_constrained_dofs(cc, rhsConstraints, rhs);
    const auto rhsInit = rhsConstraints;

    // correct rhs by strong nonzero Dirichlet contributions
    using Dune::PDELab::Backend::native;
    Dune::PDELab::set_nonconstrained_dofs(cc, 0.0, rhsConstraints);
    RhsType rhsCorrection(tensor_gfs, 0.0);
    native(fullMat).mv(native(rhsConstraints), native(rhsCorrection));
    Dune::PDELab::set_constrained_dofs(cc, 0.0, rhsCorrection);
    rhs -= rhsCorrection;

    // switch to native matrices
    auto fullMatNative = native(fullMat);
    auto aMat = fullMatNative[0][0];
    auto bMat = fullMatNative[0][1];
    auto cMat = fullMatNative[1][1];

    eliminateColumns(aMat, cc);

    // Solve Schurcomplement
    using AMatrixType = decltype(aMat);
    using BMatrixType = decltype(bMat);

    Dune::MatrixAdapter<AMatrixType,X,X> a_op(aMat);

    // set up preconditioner for inner iteration
    const RF relaxation = 1.0;
    using ILU = Dune::SeqILU<AMatrixType,X,X>;
    auto prec_inner = std::make_shared<ILU>(aMat, relaxation);

    // use CG solver to approximate inverse upper left block
    const RF reduction_inner = 1e-8;
    const int maxit_inner = 10;
    const int verbose_inner = 0;
    Dune::CGSolver<X> inner_solver(a_op, *prec_inner, reduction_inner, maxit_inner, verbose_inner);

    // perform mass lumping
    using MatrixType = typename MassLumpedMatrixType<AMatrixType, BMatrixType>::type;
    MatrixType aApprox;
    getMassLumpedMatrix(aMat, bMat, aApprox);
    aApprox += cMat;
    using OperatorType = Dune::MatrixAdapter<MatrixType,Y,Y>;
    auto aApproxOp = std::make_shared<OperatorType>(aApprox);

    // set up AMG
    using Smoother = Dune::SeqSOR<MatrixType,Y,Y>;
    using AMG = Dune::Amg::AMG<OperatorType,Y,Smoother>;

    using SmootherArgs = typename Dune::Amg::SmootherTraits<Smoother>::Arguments;
    SmootherArgs smootherArgs;
    smootherArgs.iterations = 1;
    smootherArgs.relaxationFactor = 1;
    Dune::Amg::Parameters params(15, 2000);

    using Criterion = Dune::Amg::CoarsenCriterion<Dune::Amg::SymmetricCriterion<MatrixType, Dune::Amg::FirstDiagonal>>;
    Criterion criterion(params);

    auto prec_outer = std::make_shared<AMG>(*aApproxOp, criterion, smootherArgs);

    auto schurop = init_schurcomplement<Y>(inner_solver, bMat, cMat);
    const RF reduction_outer = 1e-20;
    const int maxit_outer = 2000;
    const int verbose_outer = 1;
    Dune::CGSolver<Y> outer_solver(schurop, *prec_outer, reduction_outer, maxit_outer, verbose_outer);

    auto solution = std::make_shared<Y>(bMat.M());
    *solution = 0.0;
    Dune::InverseOperatorResult res;

    // Additional CG-solver with higher accuracy for single A-matrix inversions
    const RF reduction_high_acc = 1e-12;
    const int maxit_high_acc = 100;
    Dune::CGSolver<X> a_inv_solver(a_op, *prec_inner, reduction_high_acc, maxit_high_acc, verbose_inner);

    // compute adapted RHS
    auto rhs0 = native(rhs.block(0));
    const auto rhs1 = native(rhs.block(1));

    X rhs_proj(aMat.N());
    X rhs0copy = rhs0;
    a_inv_solver.apply(rhs_proj, rhs0copy, res);
    auto schurrhs = rhs1;
    schurrhs *= -1.0;
    bMat.umtv(rhs_proj, schurrhs);

    // solve the schurcomplement
    outer_solver.apply(*solution, schurrhs, res);
    coeffs_dg.attach(solution);

    // reconstruct velocity solution
    auto reconstructionOp = init_reconstruction<Y>(a_inv_solver, bMat, rhs0);
    auto reconstruction = reconstructionOp.getReconstructionVector();
    reconstructionOp.apply(*solution, *reconstruction);

    typename Traits::CoefficientVector rec_coeffs(gfs);
    rec_coeffs.attach(reconstruction);
    coeffs = rec_coeffs;
  }

  void writeVTK(std::string filename = "darcyMixedSolution") const {
    using VTKWriter = Dune::SubsamplingVTKWriter<GV>;
    Dune::RefinementIntervals subsampling(pTree.template get<double>("visualization.subsampling_velocity"));
    VTKWriter vtkwriter(gv, subsampling);
    std::string vtkfile(filename);

    // plot velocity
    auto rt0dgf = this->getDiscreteGridFunction();
    using RT0_VTKF = Dune::PDELab::VTKGridFunctionAdapter<typename Traits::DiscreteGridFunction>;
    vtkwriter.addCellData(std::make_shared<RT0_VTKF>(rt0dgf, "Velocity"));

    // plot pressure
    DGFDG dgdgf(gfs_dg, coeffs_dg);
    using DG_VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGFDG>;
    vtkwriter.addCellData(std::make_shared<DG_VTKF>(dgdgf, "Pressure"));

    // plot permeability field
    auto Alambda = [this](const auto& el, const auto& x){ return this->problem.A(el, x)[0][0]; };
    auto Amag = Dune::PDELab::makeGridFunctionFromCallable(gv,Alambda);
    using VTKFA = Dune::PDELab::VTKGridFunctionAdapter<decltype(Amag)>;
    vtkwriter.addCellData(std::make_shared<VTKFA>(Amag, "Permeability tensor"));

    vtkwriter.write(vtkfile, Dune::VTK::appendedraw);
  }

protected:
  using Base::problem;
  using Base::gv;
  using Base::fem;
  using Base::gfs;
  using Base::coeffs;
  using Base::pTree;
};


#endif // DUNE_ULTRAWEAK_DARCY_VELOCITY_DARCY_MIXED_SOLVER_HH
