// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_CATALYSATOR_PROBLEMS_HH
#define DUNE_ULTRAWEAK_CATALYSATOR_PROBLEMS_HH

#include <dune/pdelab.hh>

enum BoundaryConditionType {
  mixed = 0,
  pressure = 1
};

/**
   Pressure gradient from top left to bottom right with a horizontal
   low permeability block in the middle
*/
template <typename GV, typename RF, BoundaryConditionType bc>
class CatalysatorProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF>
{
private:
  const double tol = 1e-10;

public:
  using Base = Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF>;
  using Traits = typename Base::Traits;

  CatalysatorProblem(Dune::ParameterTree& pTree) : Base(),
    pTree_(pTree), I(0.0),
    minPerm_(pTree_.get<RF>("darcy.min_permeability")),
    coatingPerm_(pTree_.get<RF>("darcy.coatingPermeability")),
    openingHeight_(pTree_.get<RF>("problem.openingHeight")),
    coatingHeight_(pTree_.get<RF>("problem.coatingHeight")),
    halfReactionBlockHeight_(0.5 - openingHeight_ - coatingHeight_)
  {
    // precompute unity tensor
    for (std::size_t i=0; i<Traits::dimDomain; i++)
      I[i][i] = 1.0;
  }

  // Boundary condition type
  template<typename Element, typename X>
  auto bctype(const Element& el, const X& x) const
  {
    auto global = el.geometry().global(x);
    using std::abs;
    if (((global[0] < tol) and (global[1] > 1-openingHeight_ - tol)) or
        ((global[0] > 1-tol) and (global[1] < openingHeight_ + tol))){
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    }
    else
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
  }

  // Permeability tensor (isotropic)
  template<typename Element, typename X>
  auto A (const Element& el, const X& x) const
  {
    const auto& global = el.geometry().center();

    using std::abs;
    if (abs(global[1] - 0.5) < halfReactionBlockHeight_ + tol)
      return minPerm_ * I;
    else if (abs(global[1] - 0.5) < halfReactionBlockHeight_ + coatingHeight_ + tol)
      return coatingPerm_ * I;
    else
      return I;
  }

  // Dirichlet condition
  template<typename Element, typename X>
  auto g (const Element& el, const X& x) const
  {
    const auto global = el.geometry().global(x);
    const auto gval = 1.0 - global[0];
    if constexpr (bc == BoundaryConditionType::mixed)
      return -gval;
    else if constexpr (bc == BoundaryConditionType::pressure)
      return gval;
    else {
      DUNE_THROW(Dune::RangeError, "CatalysatorProblem: unknown BoundaryConditionType");
      return -1;
    }
  }

  //! Neumann boundary condition
  template<typename Intersection, typename X>
  auto j (const Intersection& is, const X& x) const
  {
    const auto global = is.geometry().global(x);

    using Vec = Dune::FieldVector<double,2>;
    const double eta = pTree_.template get<double>("problem.eta");

    // Poiseuille profile
    auto profile = [eta, this](const auto& r){
      using std::pow;
      return Vec({(pow(0.5*openingHeight_,2) - pow(r,2)) / (4*eta), 0.0});
    };

    using std::abs;
    Vec ret(0.0);
    if ((global[0] < tol and global[1] > 1-openingHeight_ - tol)) {
      const auto& radius = abs(global[1] - (1-0.5*openingHeight_));
      ret = profile(radius);
    }
    else if ((global[0] > 1-tol and global[1] < openingHeight_ + tol)) {
      const auto& radius = abs(global[1] - (0.5*openingHeight_));
      ret = profile(radius);
    }

    if constexpr (bc == BoundaryConditionType::mixed)
      return ret;
    else if constexpr (bc == BoundaryConditionType::pressure)
      return ret.dot(is.unitOuterNormal(x));
    else {
      DUNE_THROW(Dune::RangeError, "CatalysatorProblem: unknown BoundaryConditionType");
      return -1;
    }
  }

private:
  Dune::ParameterTree& pTree_;
  typename Traits::PermTensorType I;
  const RF minPerm_;
  const RF coatingPerm_;
  const RF openingHeight_;
  const RF coatingHeight_;;
  const RF halfReactionBlockHeight_;
};

#endif  // DUNE_ULTRAWEAK_CATALYSATOR_PROBLEMS_HH
