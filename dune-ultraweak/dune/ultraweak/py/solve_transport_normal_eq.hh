#ifndef DUNE_ULTRAWEAK_PY_SOLVE_TRANSPORT_NORMAL_EQ_HH
#define DUNE_ULTRAWEAK_PY_SOLVE_TRANSPORT_NORMAL_EQ_HH

#include "../test/solve_transport_normal_eq.hh"
#include "parameter_tree.hh"

void pySolveTransportNormalEq(pybind11::dict config) {
  auto pTree = toParameterTree(config);
  solveTransportNormalEq(pTree);
}

#endif  // DUNE_ULTRAWEAK_PY_SOLVE_TRANSPORT_NORMAL_EQ_HH
