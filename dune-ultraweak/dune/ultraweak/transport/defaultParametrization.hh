#ifndef DUNE_ULTRAWEAK_TRANSPORT_DEFAULT_PARAMETRIZATION_HH
#define DUNE_ULTRAWEAK_TRANSPORT_DEFAULT_PARAMETRIZATION_HH

#include "dune/ultraweak/parametricCoefficientsWrapper.hh"

// TODO: write an interface here

class DefaultTransportParametrization {
public:
  static constexpr std::size_t nParams = 3;
  using ParameterValueType = double;
  using ParameterType = std::array<ParameterValueType, nParams>;

  static constexpr std::size_t Qa = 2;
  using ParametrizationOnesided = ParametricCoefficientsWrapper<ParameterType, Qa>;
  using ParametrizationBilinear = NormalEqParametricCoefficientWrapper<ParametrizationOnesided>;

  static constexpr std::size_t Qf = 3;
  using ParametrizationRhs = ParametricCoefficientsWrapper<ParameterType, Qf>;

  using ParameterFunctionalType =
    typename ParametrizationOnesided::ParameterFunctionalType;

  DefaultTransportParametrization() {
    std::array<ParameterFunctionalType,Qa> thetas = {
      [](const ParameterType& mu){ return 1.0; },
      [](const ParameterType& mu){ return mu[0]; }
    };
    ParametrizationOnesided paramLeft(thetas);
    ParametrizationOnesided paramRight(thetas);
    parametrizationBilinear_ = std::make_shared<ParametrizationBilinear>(
                                 std::move(paramLeft), std::move(paramRight));

    std::array<ParameterFunctionalType,Qf> thetasRhs = {
      [](const ParameterType& mu){ return 1.0; },
      [](const ParameterType& mu){ return mu[1]; },
      [](const ParameterType& mu){ return mu[2]; }
    };
    parametrizationRhs_ = std::make_shared<ParametrizationRhs>(std::move(thetasRhs));
  }

  DefaultTransportParametrization(DefaultTransportParametrization& other) = delete;
  DefaultTransportParametrization(const DefaultTransportParametrization& other) = delete;

  const ParametrizationBilinear& bilinear() const {
    return *parametrizationBilinear_;
  }

  std::shared_ptr<ParametrizationBilinear> bilinearPtr() const {
    return parametrizationBilinear_;
  }

  const ParametrizationRhs& rhs() const {
    return *parametrizationRhs_;
  }

  std::shared_ptr<ParametrizationRhs> rhsPtr() const {
    return parametrizationRhs_;
  }

  // convenience function
  void setParameter(const ParameterType& mu) const {
    parametrizationBilinear_->setParameter(mu);
    parametrizationRhs_->setParameter(mu);
  }

  void initializeFromConfig(Dune::ParameterTree pTree) const {
    const double mu0 = pTree.template get<double>("problem.non-parametric.fixedReaction");
    const double mu1 = pTree.template get<double>("problem.non-parametric.fixedSource");
    const double mu2 = pTree.template get<double>("problem.non-parametric.fixedInflow");

    this->setParameter({mu0, mu1, mu2});
  }

private:
  std::shared_ptr<ParametrizationBilinear> parametrizationBilinear_;
  std::shared_ptr<ParametrizationRhs> parametrizationRhs_;
};

// TODO: merge with above
/**
   Parametrization with:
   mu0: reaction in washcoat
   mu1: reaction in coating
   mu2: scaling of the inflow condition
*/
class ExtendedTransportParametrization {
public:
  static constexpr std::size_t nParams = 3;
  using ParameterValueType = double;
  using ParameterType = std::array<ParameterValueType, nParams>;

  static constexpr std::size_t Qa = 3;
  using ParametrizationOnesided = ParametricCoefficientsWrapper<ParameterType, Qa>;
  using ParametrizationBilinear = NormalEqParametricCoefficientWrapper<ParametrizationOnesided>;

  static constexpr std::size_t Qf = 2;
  using ParametrizationRhs = ParametricCoefficientsWrapper<ParameterType, Qf>;

  using ParameterFunctionalType =
    typename ParametrizationOnesided::ParameterFunctionalType;

  ExtendedTransportParametrization() {
    std::array<ParameterFunctionalType,Qa> thetas = {
      [](const ParameterType& mu){ return 1.0; },
      [](const ParameterType& mu){ return mu[0]; },
      [](const ParameterType& mu){ return mu[1]; }
    };
    ParametrizationOnesided paramLeft(thetas);
    ParametrizationOnesided paramRight(thetas);
    parametrizationBilinear_ = std::make_shared<ParametrizationBilinear>(
                                 std::move(paramLeft), std::move(paramRight));

    std::array<ParameterFunctionalType,Qf> thetasRhs = {
      [](const ParameterType& mu){ return 1.0; },
      [](const ParameterType& mu){ return mu[2]; }
    };
    parametrizationRhs_ = std::make_shared<ParametrizationRhs>(std::move(thetasRhs));
  }

  ExtendedTransportParametrization(ExtendedTransportParametrization& other) = delete;
  ExtendedTransportParametrization(const ExtendedTransportParametrization& other) = delete;

  const ParametrizationBilinear& bilinear() const {
    return *parametrizationBilinear_;
  }

  std::shared_ptr<ParametrizationBilinear> bilinearPtr() const {
    return parametrizationBilinear_;
  }

  const ParametrizationRhs& rhs() const {
    return *parametrizationRhs_;
  }

  std::shared_ptr<ParametrizationRhs> rhsPtr() const {
    return parametrizationRhs_;
  }

  // convenience function
  void setParameter(const ParameterType& mu) const {
    parametrizationBilinear_->setParameter(mu);
    parametrizationRhs_->setParameter(mu);
  }

  void initializeFromConfig(Dune::ParameterTree pTree) const {
    const double mu0 = pTree.get<double>("problem.non-parametric.fixedReactionWashcoat");
    const double mu1 = pTree.get<double>("problem.non-parametric.fixedReactionCoating");
    const double mu2 = pTree.get<double>("problem.non-parametric.fixedInflowScaling");

    this->setParameter({mu0, mu1, mu2});
  }

private:
  std::shared_ptr<ParametrizationBilinear> parametrizationBilinear_;
  std::shared_ptr<ParametrizationRhs> parametrizationRhs_;
};

#endif  // DUNE_ULTRAWEAK_TRANSPORT_DEFAULT_PARAMETRIZATION_HH
