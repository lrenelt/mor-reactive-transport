#ifndef DUNE_ULTRAWEAK_DARCY_VELOCITY_DARCY_VELOCITY_ADAPTER_HH
#define DUNE_ULTRAWEAK_DARCY_VELOCITY_DARCY_VELOCITY_ADAPTER_HH

#include <dune/pdelab.hh>

/**
   Replaces the velocity in an existing problem by the evaluation
   of a precomputed gridfunction. Also adapts boundary parts accordingly.
*/
template<typename BaseProblem, typename DGFType>
class DarcyVelocityAdapter : public BaseProblem
{
public:
  using Traits = typename BaseProblem::Traits;

  // TODO: solve in here?
  DarcyVelocityAdapter(const BaseProblem& baseProblem, const DGFType& velocityDgf)
    : BaseProblem(baseProblem), baseProblem_(baseProblem), velocityDgf_(velocityDgf) {}

  template<typename Element, typename X>
  auto bctype(const Element& el, const X& x) const
  {
    const auto& insidePos = el.geometryInInside().global(x);

    const auto& vel = b(el.inside(), insidePos);
    const auto val = vel.dot(el.unitOuterNormal(x));
    const double tol = 1e-10;
    if (val > tol)
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Outflow;
    else if (val < -tol)
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    else
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::None;
  }

  template<typename Element, typename X>
  auto b (const Element& el, const X& x) const
  {
    // The dgf might be defined on a refined grid
    // -> search for the corresponding entity
    typename Traits::RangeType ret(0.0);
    if (isRefined_) {
      using GV = typename DGFType::GridViewType;
      using Search = Dune::HierarchicSearch<typename GV::Grid, typename GV::IndexSet>;
      const auto xg = el.geometry().global(x);
      const GV& gv = velocityDgf_.getGridView();
      const auto subelement = Search(gv.grid(), gv.indexSet()).findEntity(xg);
      const auto xl = subelement.geometry().local(xg);
      velocityDgf_.evaluate(subelement, xl, ret);
    }
    else
      velocityDgf_.evaluate(el, x, ret);
    return ret;
  }

  // TODO: can we infer this automatically?
  void setRefinement(const bool isRefined) {
    isRefined_ = isRefined;
  }

protected:
  bool isRefined_ = false;

private:
  const BaseProblem& baseProblem_;
  const DGFType& velocityDgf_;
};

#endif  // DUNE_ULTRAWEAK_DARCY_VELOCITY_DARCY_VELOCITY_ADAPTER_HH
