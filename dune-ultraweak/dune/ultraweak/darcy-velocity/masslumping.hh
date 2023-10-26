#ifndef DUNE_ULTRAWEAK_MASSLUMPING_HH
#define DUNE_ULTRAWEAK_MASSLUMPING_HH


// TODO: do we want the A-Matrix dependency here?
template<typename AMatrixType, typename BMatrixType>
struct MassLumpedMatrixType {
  using type = typename Dune::TransposedMatMultMatResult<BMatrixType,BMatrixType>::type;
};

// TODO: what about a lower right block?
template<typename AMatrixType, typename BMatrixType>
auto getMassLumpedMatrix(const AMatrixType& aMat, const BMatrixType& bMat,
                         typename MassLumpedMatrixType<AMatrixType, BMatrixType>::type& lumpedMat) {
  using field_type = typename AMatrixType::field_type;

  BMatrixType scaledBMat(bMat);

  // perform mass lumping
  for (auto row_it = aMat.begin(); row_it != aMat.end(); ++row_it) {
    field_type row_sum = 0.0;
    for (auto col_it = row_it->begin(); col_it != row_it->end(); ++col_it) {
      using std::abs;
      // TODO: allow other norms here
      row_sum += abs(*col_it); // l1-Norm of row
    }
    scaledBMat[row_it.index()] /= row_sum;
  }

  Dune::transposeMatMultMat(lumpedMat, bMat, scaledBMat);
}

#endif // DUNE_ULTRAWEAK_MASSLUMPING_HH
