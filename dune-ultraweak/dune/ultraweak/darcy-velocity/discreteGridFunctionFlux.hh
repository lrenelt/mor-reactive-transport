#ifndef DUNE_ULTRAWEAK_TRANSPORT_DISCRETEGRIDFUNCTIONFLUX_HH
#define DUNE_ULTRAWEAK_TRANSPORT_DISCRETEGRIDFUNCTIONFLUX_HH

#include <vector>

#include <dune/pdelab.hh>

template<typename GFS, typename Coeffs, typename Problem, int dim=2>
class DiscreteGridFunctionFlux :
  public Dune::PDELab::GridFunctionBase<
    Dune::PDELab::GridFunctionTraits<typename GFS::Traits::GridView,
                                     double,dim,Dune::FieldVector<double,dim>>,
    DiscreteGridFunctionFlux<GFS,Coeffs,Problem>>
{
public:
  using Traits = Dune::PDELab::GridFunctionTraits<typename GFS::Traits::GridView,
                                                  double,dim,Dune::FieldVector<double,dim>>;
private:
  using BaseT = Dune::PDELab::GridFunctionBase<
    Traits, DiscreteGridFunctionFlux<GFS,Coeffs,Problem>>;
  using DGFGradient = Dune::PDELab::DiscreteGridFunctionGradient<GFS,Coeffs>;

public:
  DiscreteGridFunctionFlux(const GFS& gfs, const Coeffs& coeffs, const Problem& problem) :
    gfs_(gfs),
    coeffs_(coeffs),
    dgfGradient_(DGFGradient(gfs_,coeffs_)),
    problem_(problem) {}


  inline void evaluate(const typename Traits::ElementType& e,
                       const typename Traits::DomainType& x,
                       Traits::RangeType& y) const
  {
    dgfGradient_.evaluate(e,x,gradphi);
    problem_.A(e,x).mv(-gradphi,y);
  }

  //! get a reference to the GridView
  const typename Traits::GridViewType& getGridView() const
  { return dgfGradient_.getGridView(); }


private:
  const GFS& gfs_;
  const Coeffs& coeffs_;
  const DGFGradient dgfGradient_;
  const Problem& problem_;

  mutable typename Traits::RangeType gradphi;
};

#endif  // DUNE_ULTRAWEAK_TRANSPORT_DISCRETEGRIDFUNCTIONFLUX_HH
