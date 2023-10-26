#ifndef DUNE_ULTRAWEAK_TRANSPORT_TRAITS_HH
#define DUNE_ULTRAWEAK_TRANSPORT_TRAITS_HH

template<typename GV, std::size_t order_>
class NormalEqTraits {
public:
  using GridView = GV;
  static const int dim = GV::dimension;
  static const int order = order_;
  using DF = typename GV::Grid::ctype; // type for ccordinates
  using RF = double; //Dune::Float128;           // type for computations

  using FEM = Dune::PDELab::QkLocalFiniteElementMap<GV,DF,RF,order>;
  using CON = Dune::PDELab::ConformingDirichletConstraints;
  using VBE = Dune::PDELab::ISTL::VectorBackend<>;
  using GridFunctionSpace = Dune::PDELab::GridFunctionSpace<GV,FEM,CON,VBE>;
  using CoefficientVector = Dune::PDELab::Backend::Vector<GridFunctionSpace, RF>;
  using DiscreteGridFunction = Dune::PDELab::DiscreteGridFunction<GridFunctionSpace, CoefficientVector>;
};

template<class GV, std::size_t order_>
struct DGSolverTraits {
  static const int dim = GV::dimension;
  using DF = typename GV::Grid::ctype;
  using RF = double;
  static const int order = order_;

  using FEM = Dune::PDELab::QkDGLocalFiniteElementMap<DF, RF, order, dim>;
  using Constraints = Dune::PDELab::NoConstraints;
  using VectorBackend = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::fixed, Dune::QkStuff::QkSize<order, dim>::value>;
  using GridFunctionSpace = Dune::PDELab::GridFunctionSpace<GV, FEM, Constraints, VectorBackend>;
  using CoefficientVector = Dune::PDELab::Backend::Vector<GridFunctionSpace, DF>;
  using DiscreteGridFunction = Dune::PDELab::DiscreteGridFunction<GridFunctionSpace, CoefficientVector>;
};

#endif  // DUNE_ULTRAWEAK_TRANSPORT_TRAITS_HH
