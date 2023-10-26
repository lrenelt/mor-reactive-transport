// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_DARCY_VELOCITY_DARCY_PROBLEM_HH
#define DUNE_ULTRAWEAK_DARCY_VELOCITY_DARCY_PROBLEM_HH

#include "dune/pdelab.hh"

template <typename GV, typename RF>
class DarcyProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF>
{
public:
  using Base = Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF>;
  using Traits = typename Base::Traits;

  DarcyProblem(Dune::ParameterTree& pTree) : Base(), pTree_(pTree), I(0.0) {
    // precompute unity tensor
    for (std::size_t i=0; i<Traits::dimDomain; i++)
      I[i][i] = 1.0;
  }

  // Boundary condition type
  template<typename Element, typename X>
  auto bctype(const Element& el, const X& x) const
  {
    auto global = el.geometry().global(x);
    if (global[0] < 1e-10 or global[0] > 1-1e-10)
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    else
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
  }

  // Permeability tensor (isotropic)
  template<typename Element, typename X>
  auto A (const Element& el, const X& x) const
  {
    auto global = el.geometry().global(x);

    const RF perm_min = pTree_.get<double>("darcy.min_permeability");

    if (pTree_.get<bool>("darcy.discontinuous")) {
      using std::abs;
      if (abs(global[0]-0.5) < 0.25 and abs(global[1]-0.5) < 0.25)
        return perm_min*I;
    }
    else {
      const RF width = pTree_.template get<double>("darcy.width");
      const typename Traits::RangeType midpoint({0.5, 0.5});
      const RF dist = (global-midpoint).two_norm();
      using std::exp;
      if (pTree_.template get<bool>("darcy.compact")) {
        const RF r1 = width / 4;
        const RF r2 = width / 2;
        if (dist <= r1)
          return perm_min * I;
        else if (dist <= r2)
          return (perm_min + (1-perm_min)/exp(-1.)*exp(-1./(1-pow((r2-dist)/(r2-r1), 2)))) * I;
      }
      else {
        return (1 - (1-perm_min)*exp(-dist/width)) * I;
      }
    }
    return I;
  }

  // Dirichlet condition
  template<typename Element, typename X>
  auto g (const Element& el, const X& x) const
  {
    const auto gval = 1.0 - el.geometry().global(x)[0];
    return -gval;
  }

  //! Neumann boundary condition
  template<typename Intersection, typename X>
  auto j (const Intersection& is, const X& x) const
  {
    return 0.0;
  }

private:
  Dune::ParameterTree& pTree_;
  typename Traits::PermTensorType I;
};

#endif // DUNE_ULTRAWEAK_DARCY_VELOCITY_DARCY_PROBLEM_HH
