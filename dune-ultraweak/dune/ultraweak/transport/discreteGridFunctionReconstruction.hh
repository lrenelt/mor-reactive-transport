#ifndef DUNE_ULTRAWEAK_TRANSPORT_DISCRETEGRIDFUNCTIONRECONSTRUCTION_HH
#define DUNE_ULTRAWEAK_TRANSPORT_DISCRETEGRIDFUNCTIONRECONSTRUCTION_HH

#include <vector>

#include <dune/pdelab.hh>

template<typename DGFScalar, typename DGFGradient, typename Problem, typename Parametrization>
class DiscreteGridFunctionReconstruction :
  public Dune::PDELab::GridFunctionBase<typename DGFScalar::Traits,
  DiscreteGridFunctionReconstruction<DGFScalar,DGFGradient,Problem,Parametrization>>
{
public:
  using Traits = typename DGFScalar::Traits;
  using RangeType = double;
private:
  using BaseT = Dune::PDELab::GridFunctionBase<
  Traits, DiscreteGridFunctionReconstruction<
            DGFScalar,DGFGradient,Problem,Parametrization>>;

public:
  DiscreteGridFunctionReconstruction(const DGFScalar& dgfScalar, const DGFGradient& dgfGradient, const Problem& problem, const Parametrization& param, const bool rescaling = false) :
    dgfScalar_(dgfScalar),
    dgfGradient_(dgfGradient),
    problem_(problem),
    param_(param),
    rescaling_(rescaling) {}


  inline void evaluate(const typename DGFScalar::Traits::ElementType& e,
                       const typename DGFScalar::Traits::DomainType& x,
                       RangeType& y) const
  {
    dgfScalar_.evaluate(e,x,phi);
    dgfGradient_.evaluate(e,x,gradphi);

    //evaluate data functions
    const auto velocity = problem_.b(e, x);
    const auto c = problem_.c(e, x);

    //using RangeType = typename DGFScalar::Traits::RangeType;
    y = param_.bilinear().left().makeParametricLincomb(
          std::array<RangeType,3>({-velocity.dot(gradphi), c[0] * phi, c[1] * phi}));
    if (rescaling_)
      y /= velocity.two_norm();
  }

  //! get a reference to the GridView
  const typename Traits::GridViewType& getGridView() const
  { return dgfScalar_.getGridView(); }


private:
  const DGFScalar& dgfScalar_;
  const DGFGradient& dgfGradient_;
  const Problem& problem_;
  const Parametrization& param_;
  const bool rescaling_; // TODO: remove?

  mutable typename DGFScalar::Traits::RangeType phi;
  mutable typename DGFGradient::Traits::RangeType gradphi;
};

#endif  // DUNE_ULTRAWEAK_TRANSPORT_DISCRETEGRIDFUNCTIONRECONSTRUCTION_HH
