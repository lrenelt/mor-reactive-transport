// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_RECONSTRUCTION_HH
#define DUNE_ULTRAWEAK_RECONSTRUCTION_HH

template<typename Y, typename X, typename BMatrixType>
class Reconstruction : public Dune::LinearOperator<Y,X> {
public:
  using DomainType = Y;
  using RangeType = X;
  using field_type = typename Y::field_type;

  Reconstruction(Dune::InverseOperator<X,X>& aInvOp, const BMatrixType& bMat,
                 const X& rhsX, const int pm) : aInvOp_(aInvOp), bMat_(bMat),
                                                   rhsX_(rhsX), pm_(pm) {}

  void apply(const Y& y, X& x) const {
    X temp(rhsX_);
    bMat_.usmv(pm_, y, temp);

    Dune::InverseOperatorResult res;
    aInvOp_.apply(x, temp, res);
  }

  void applyscaleadd(field_type alpha, const Y& y, X& x) const {
    X temp1(rhsX_);
    bMat_.usmv(pm_, y, temp1);

    X temp2(bMat_.N());
    temp2 = 0.0;
    Dune::InverseOperatorResult res;
    aInvOp_.apply(temp2, temp1, res);
    x.axpy(alpha,temp2);
  }

  std::shared_ptr<X> getReconstructionVector() const {
    return std::make_shared<X>(bMat_.N());
  }

  Dune::SolverCategory::Category category() const {
    return Dune::SolverCategory::sequential;
  }

private:
  Dune::InverseOperator<X,X>& aInvOp_;
  const BMatrixType bMat_;
  const X rhsX_;
  const int pm_;
};

template<typename Y, typename X, typename BMatrixType>
auto init_reconstruction(Dune::InverseOperator<X,X>& aInvOp, const BMatrixType& bMat) {
  X dummy(bMat.N());
  dummy = 0.0;
  return Reconstruction<Y, X, BMatrixType>(aInvOp, bMat, std::move(dummy), 1);
}

template<typename Y, typename X, typename BMatrixType>
auto init_reconstruction(Dune::InverseOperator<X,X>& aInvOp, const BMatrixType& bMat, const X& rhsX) {
  return Reconstruction<Y, X, BMatrixType>(aInvOp, bMat, rhsX, -1);
}

#endif  // DUNE_ULTRAWEAK_RECONSTRUCTION_HH
