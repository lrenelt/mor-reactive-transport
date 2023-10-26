#ifndef DUNE_ULTRAWEAK_H1BNORM_HH
#define DUNE_ULTRAWEAK_H1BNORM_HH

template<typename Problem, typename FEM>
class H1bnorm :
  public Dune::PDELab::FullVolumePattern,
  public Dune::PDELab::LocalOperatorDefaultFlags
{
public:
  enum { doPatternVolume = true };
  enum { doAlphaVolume = true };

  using LocalBasisType = typename
    FEM::Traits::FiniteElementType::Traits::LocalBasisType;
  using CacheType = Dune::PDELab::LocalBasisCache<LocalBasisType>;

  using VelocityType = typename Problem::Traits::RangeType;

  H1bnorm(const Problem& problem, const std::size_t intorder = 2) :
    problem_(problem), intorder_(intorder) {}

  template<typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_volume(const EG& eg, const LFSU& lfsu,
                    const X& x, const LFSV& lfsv, R& r) const
  {
    using RF = typename
      LFSU::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeFieldType;
    using RangeType = typename
      LFSU::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeType;
    using size_type = typename LFSU::Traits::SizeType;

    const auto& cell = eg.entity();
    const auto geo = eg.geometry();

    RangeType u;
    RangeType bgradu;
    std::vector<RangeType> phi(lfsu.size());
    auto gradphi = makeJacobianContainer(lfsu);
    std::vector<RangeType> bgradv(lfsu.size());

    VelocityType velocity;
    RF factor;

    for (const auto& qp: quadratureRule(geo, intorder_)) {
      phi = cache.evaluateFunction(qp.position(), lfsu.finiteElement().localBasis());
      auto& js = cache.evaluateJacobian(qp.position(), lfsu.finiteElement().localBasis());
      const auto S = geo.jacobianInverseTransposed(qp.position());
      for (size_type i=0; i<lfsu.size(); ++i)
        S.mv(js[i][0], gradphi[i][0]);

      velocity = problem_.b(cell, qp.position());
      for (size_type i=0; i<lfsu.size(); i++)
        bgradv[i] = velocity.dot(gradphi[i][0]);

      u = 0.0;
      bgradu = 0.0;
      for (size_type i=0; i<lfsu.size(); i++) {
        u += x(lfsu,i)*phi[i];
        bgradu += x(lfsu,i)*bgradv[i];
      }

      factor = qp.weight() * geo.integrationElement(qp.position());
      for (size_type j=0; j<lfsv.size(); j++)
        r.accumulate(lfsv, j, factor * (bgradu * bgradv[j] + u * phi[j]));
    }
  }

  template<typename EG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_volume(const EG& eg, const LFSU& lfsu,
                       const X& x, const LFSV& lfsv, M& mat) const
  {
    using RF = typename
      LFSU::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeFieldType;
    using RangeType = typename
      LFSU::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeType;
    using size_type = typename LFSU::Traits::SizeType;

    const auto& cell = eg.entity();
    const auto geo = eg.geometry();

    std::vector<RangeType> phi(lfsu.size());
    auto gradphi = makeJacobianContainer(lfsu);
    std::vector<RangeType> bgradv(lfsu.size());

    VelocityType velocity;
    RF factor;

    for (const auto& qp: quadratureRule(geo, intorder_)) {
      phi = cache.evaluateFunction(qp.position(), lfsu.finiteElement().localBasis());
      auto& js = cache.evaluateJacobian(qp.position(), lfsu.finiteElement().localBasis());
      const auto S = geo.jacobianInverseTransposed(qp.position());
      for (size_type i=0; i<lfsu.size(); ++i)
        S.mv(js[i][0], gradphi[i][0]);

      velocity = problem_.b(cell, qp.position());
      for (size_type i=0; i<lfsu.size(); i++)
        bgradv[i] = velocity.dot(gradphi[i][0]);

      factor = qp.weight() * geo.integrationElement(qp.position());
      for (size_type i=0; i<lfsu.size(); i++)
        for (size_type j=0; j<lfsv.size(); j++)
          mat.accumulate(lfsv, j, lfsu, i, factor * (bgradv[i] * bgradv[j] + phi[i] * phi[j]));
    }
  }

  template<typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void jacobian_apply_volume (const EG& eg, const LFSU& lfsu,
                              const X& z, const LFSV& lfsv,
                              R& r) const {
    alpha_volume(eg,lfsu,z,lfsv,r);
  }

private:
  const Problem& problem_;
  const std::size_t intorder_;
  CacheType cache;
};



#endif  // DUNE_ULTRAWEAK_H1BNORM_HH
