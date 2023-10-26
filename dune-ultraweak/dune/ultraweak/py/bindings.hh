#ifndef DUNE_ULTRAWEAK_PY_BINDINGS_HH
#define DUNE_ULTRAWEAK_PY_BINDINGS_HH

#include "dune/ultraweak/darcy-velocity/catalysatorProblems.hh"
#include "dune/ultraweak/darcy-velocity/darcyMixedSolver.hh"
#include "dune/ultraweak/darcy-velocity/darcySIPGSolver.hh"
#include "dune/ultraweak/darcy-velocity/darcyVelocityAdapter.hh"

#include "dune/ultraweak/transport/defaultParametrization.hh"
#include "dune/ultraweak/transport/normalEqLocalOperator.hh"
#include "dune/ultraweak/transport/parametrizedSolver.hh"
#include "dune/ultraweak/test/problems.hh"

#include "dune/ultraweak/py/parameter_tree.hh"

namespace py = pybind11;

namespace Dune {
  namespace Ultraweak {
    namespace Python {

      template<typename Impl, std::size_t order>
      class Operator {
      public:
        using DomainType = typename Impl::domain_type;
        using RangeType = typename Impl::range_type;

        Operator(const Impl& op) : op_(op),
          dimDomain(op.getmat().N()), dimRange(op.getmat().M())
        {}

        void apply(const DomainType& x, RangeType& y) const {
          op_.apply(x,y);
        }

      protected:
        static const std::size_t order_ = order;
        const Impl& op_;
      public:
        std::size_t dimDomain;
        std::size_t dimRange;
      };

      template<std::size_t order, typename RF, int dim, typename BaseProblem, typename Parametrization>
      class TransportSolver {

      private:
        using Grid = Dune::YaspGrid<dim>;
        using GV = typename Grid::LeafGridView;

        // Darcy-problem specific defines
        // TODO: move outside or do some template magic
        using DarcyProblem = CatalysatorProblem<GV,RF, BoundaryConditionType::pressure>;
        static const std::size_t orderDarcy = 3;
        using DarcySolver = DarcySIPGSolver<orderDarcy,GV,DarcyProblem>;
        using DGFType = typename DarcySolver::DGFFlux;
        using Problem = DarcyVelocityAdapter<BaseProblem,DGFType>;

        using Base = ParametrizedSolver<order, GV, Problem, Parametrization>;
        using WrappedVectorType = typename Base::Traits::CoefficientVector;

      public:
        using VectorType = typename Dune::PDELab::Backend::Native<WrappedVectorType>;
        using ParameterType = typename Parametrization::ParameterType;
        using OperatorType =  typename Dune::Ultraweak::Python::Operator<
          typename Base::OperatorType,order>;

        TransportSolver(py::dict config) :
          pTree_(toParameterTree(config))
        {
          // grid setup
          Dune::FieldVector<double, dim> domain({1.0, 1.0});
          std::array<int, dim> domainDims = {
            domainDims[0] = pTree_.get<int>("grid.yasp_x"),
            domainDims[1] = pTree_.get<int>("grid.yasp_y")};

          grid_ = std::make_unique<Grid>(domain, domainDims);
          grid_->globalRefine(pTree_.get<int>("grid.refinement"));
          const auto gv = grid_->leafGridView();

          // precompute velocity field
          darcyProblem_ = std::make_unique<DarcyProblem>(pTree_);
          darcySolver_ = std::make_unique<DarcySolver>(gv, *darcyProblem_, pTree_);
          darcySolver_->solve();
          dgf_ = darcySolver_->getDiscreteGridFunctionFluxPtr();

          // define the problem
          baseProblem_ = std::make_unique<BaseProblem>(pTree_);
          problem_ = std::make_unique<Problem>(*baseProblem_, *dgf_);
          parametrization_ = std::make_unique<Parametrization>();

          // make solver object
          solver_ = std::make_unique<Base>(gv, *problem_, *parametrization_, pTree_);

          // set public members
          dimSource = solver_->getGfs().globalSize();
          dimRange = dimSource;

          // define the separated operators
          const auto& matrixOperators = solver_->getOperators();
          for(const auto& op : matrixOperators)
            operators_.emplace_back(op);
        }

        TransportSolver(TransportSolver&& other) = delete;

        auto solve(const ParameterType mu, py::dict solverConfig) {
          auto pTree = toParameterTree(solverConfig);
          return solver_->solve(mu, pTree);
        }

        void visualize(VectorType vec, const ParameterType mu, const std::string filename="default_filename") {
          std::cout << "Writing output to " << filename << ".vtk ..." << std::endl;
          solver_->visualize(vec, mu, filename);
        }

        const auto& getMatrices() const {
          return solver_->getMatrices();
        }

        const auto& getL2MassMatrix() const {
          return solver_->getL2MassMatrix();
        }

        const auto& getH1bMassMatrix() const {
          return solver_->getH1bMassMatrix();
        }

        const auto& getOperators() const {
          return operators_;
        }

        const auto& getRhsVectors() const {
          return solver_->getRhsVectors();
        }

      protected:
        Dune::ParameterTree pTree_;
        std::unique_ptr<DarcyProblem> darcyProblem_;
        std::unique_ptr<BaseProblem> baseProblem_;
        std::unique_ptr<DGFType> dgf_;
        std::unique_ptr<Problem> problem_;
        std::unique_ptr<Parametrization> parametrization_;
        std::unique_ptr<DarcySolver> darcySolver_;
        std::unique_ptr<Base> solver_;
        std::unique_ptr<Grid> grid_;

        std::vector<OperatorType> operators_;

      public:
        std::size_t dimSource;
        std::size_t dimRange;
      };
    }
  }
}

template<std::size_t order, typename GV, typename RF>
void registrateTransportSolver(py::module m) {
  constexpr int dim = 2;
  using Problem = PureTransportProblem<GV,RF>;
  using Parametrization = ExtendedTransportParametrization;
  using T = Dune::Ultraweak::Python::TransportSolver<order,RF,dim,Problem,Parametrization>;

  const std::string clsName = "TransportSolverOrder"+std::to_string(order);
  auto cls = py::class_<T, std::shared_ptr<T>>(m, clsName.c_str());

  cls.def(py::init<py::dict>(), py::arg("config"));
  cls.def_readonly("dim_source", &T::dimSource);
  cls.def_readonly("dim_range", &T::dimRange);

  cls.def("visualize", &T::visualize, "write graphic output for a given vector");
  cls.def("getMatrices", &T::getMatrices, "get the raw matrices");
  cls.def("getL2MassMatrix", &T::getL2MassMatrix, "get the l2 mass matrix");
  cls.def("getH1bMassMatrix", &T::getH1bMassMatrix, "get the H1b mass matrix");
  cls.def("getRhsVectors", &T::getRhsVectors, "get the raw rhs vector");

  cls.def("solve", &T::solve, "solve for given parameter",
          py::arg("mu"), py::arg("solverConfig"));

  // registrate the parameter-independent operators
  using OperatorT = typename T::OperatorType;
  const std::string opClsName = "OperatorOrder"+std::to_string(order);
  auto cls_op = py::class_<OperatorT, std::shared_ptr<OperatorT>>(m, opClsName.c_str());
  cls_op.def("apply", &OperatorT::apply, py::arg("x"), py::arg("y"));
  cls_op.def_readonly("dim_source", &OperatorT::dimDomain);
  cls_op.def_readonly("dim_range", &OperatorT::dimRange);

  cls.def("getOperators", &T::getOperators, "get the matrix operators");
}

#endif  // DUNE_ULTRAWEAK_PY_BINDINGS_HH<
