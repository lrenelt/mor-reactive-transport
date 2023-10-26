#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <dune/python/pybind11/pybind11.h>
#include <dune/python/pybind11/stl.h>
#include <dune/python/pybind11/numpy.h>

#include "dune/ultraweak/py/bindings.hh"

namespace py = pybind11;

PYBIND11_MODULE(ipyultraweak, m)
{
  m.doc() = "pybind11 dune-ultraweak plugin";

  constexpr int dim = 2;

  using Grid = Dune::YaspGrid<dim>;
  using GV = typename Grid::LeafGridView;
  using RF = double;

  registrateTransportSolver<1,GV,RF>(m);
  registrateTransportSolver<2,GV,RF>(m);
}
